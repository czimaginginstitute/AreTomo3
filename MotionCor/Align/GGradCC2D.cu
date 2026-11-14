#include "CAlignInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::MotionCor::Align;

//--------------------------------------------------------------------
// Let F_k be the Fourier transform of frame k and s_k its shift. The
// shift-corrected frame is G_k = F_k * exp(-i * 2PI * (x*sx/Nx +
// y*sy/Ny)) and S = sum_k G_k. The target function is
//
//    E = sum_{i!=j} sum_q W(q) Re[G_i(q) conj(G_j(q))]
//      = sum_q W(q) (|S(q)|^2 - sum_k |G_k(q)|^2)
//
// where W(q) is the B-factor weighting. Since sum_k |G_k|^2 does not
// depend on the shifts, the derivatives with respect to s_k are
//
//    dE/dsx  =  2 sum_q W(q) ax Im[conj(S) G_k]
//    d2E/dsx2 = -2 sum_q W(q) ax^2 Re[conj(S - G_k) G_k]
//
// with ax = 2PI x / Nx, ay = 2PI y / Ny. The sum over q is over the
// full plane. With the half-complex layout, columns 0 < x < Nx/2 are
// counted twice for their Hermitian mates.
//
// For each frame mGGrad accumulates per block the 7 terms below and
// mGReduce sums the blocks in a fixed order to keep the results
// reproducible from run to run (no atomics).
//    0: sum W ax Im[conj(S)G]    1: sum W ay Im[conj(S)G]
//    2: sum W ax^2 Re[conj(R)G]  3: sum W ay^2 Re[conj(R)G]
//    4: sum W ax ay Re[conj(R)G] 5: sum W Re[conj(R)G]
//    6: sum W |G|^2
// where R = S - G. When gCmpFrm is null, only term 6 is calculated
// for gCmpSum, i.e. sum W |S|^2.
//--------------------------------------------------------------------
#define GRAD_TERMS 7
#define GRAD_THREADS 256
#define GRAD_BLOCKS 256

static __global__ void mGGrad
(	cufftComplex* gCmpFrm,
	cufftComplex* gCmpSum,
	int iCmpX,
	int iCmpY,
	float fShiftX,   // phase shift per index along x
	float fShiftY,   // phase shift per index along y
	float fBFactor,
	float* gfPartial // GRAD_BLOCKS * GRAD_TERMS
)
{	__shared__ float s_afSum[GRAD_TERMS][GRAD_THREADS];
	float afSum[GRAD_TERMS] = {0.0f};
	//-----------------
	int iNx = (iCmpX - 1) * 2;
	float fAx = 6.2831853f / iNx;
	float fAy = 6.2831853f / iCmpY;
	float fFilt = -0.25f * fBFactor / ((iCmpX - 1) * iCmpY);
	int iCmpSize = iCmpX * iCmpY;
	//-----------------
	for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<iCmpSize;
	   i+=gridDim.x * blockDim.x)
	{	if(i == 0) continue;
		int x = i % iCmpX;
		int y = i / iCmpX;
		if(y > (iCmpY / 2)) y -= iCmpY;
		//----------------
		float fW = expf(fFilt * (x * x + y * y));
		if(x > 0 && x < (iCmpX - 1)) fW *= 2.0f;
		//----------------
		cufftComplex cS = gCmpSum[i];
		if(gCmpFrm == 0L)
		{	afSum[6] += fW * (cS.x * cS.x + cS.y * cS.y);
			continue;
		}
		//----------------
		float fCos, fSin;
		sincosf(x * fShiftX + y * fShiftY, &fSin, &fCos);
		cufftComplex cF = gCmpFrm[i];
		float fGx = fCos * cF.x - fSin * cF.y;
		float fGy = fCos * cF.y + fSin * cF.x;
		//----------------
		float fRe = cS.x * fGx + cS.y * fGy;
		float fIm = cS.x * fGy - cS.y * fGx;
		float fP = fGx * fGx + fGy * fGy;
		float fRR = fRe - fP;
		float fX = fAx * x;
		float fY = fAy * y;
		//----------------
		afSum[0] += fW * fX * fIm;
		afSum[1] += fW * fY * fIm;
		afSum[2] += fW * fX * fX * fRR;
		afSum[3] += fW * fY * fY * fRR;
		afSum[4] += fW * fX * fY * fRR;
		afSum[5] += fW * fRR;
		afSum[6] += fW * fP;
	}
	//-----------------
	for(int t=0; t<GRAD_TERMS; t++)
	{	s_afSum[t][threadIdx.x] = afSum[t];
	}
	__syncthreads();
	//-----------------
	for(int iOffset=blockDim.x/2; iOffset>0; iOffset/=2)
	{	if(threadIdx.x < iOffset)
		{	for(int t=0; t<GRAD_TERMS; t++)
			{	s_afSum[t][threadIdx.x] +=
				   s_afSum[t][threadIdx.x + iOffset];
			}
		}
		__syncthreads();
	}
	//-----------------
	if(threadIdx.x < GRAD_TERMS)
	{	gfPartial[blockIdx.x * GRAD_TERMS + threadIdx.x] =
		   s_afSum[threadIdx.x][0];
	}
}

//--------------------------------------------------------------------
// One thread per term. Blocks are summed sequentially in double
// precision so that the result does not depend on scheduling.
//--------------------------------------------------------------------
static __global__ void mGReduce
(	float* gfPartial,
	int iNumBlocks,
	double* gdRes
)
{	int t = threadIdx.x;
	if(t >= GRAD_TERMS) return;
	double dSum = 0.0;
	for(int b=0; b<iNumBlocks; b++)
	{	dSum += gfPartial[b * GRAD_TERMS + t];
	}
	gdRes[t] = dSum;
}

GGradCC2D::GGradCC2D(void)
{
	m_fBFactor = 100.0f;
	m_iMaxFrames = 0;
	m_gfPartial = 0L;
	m_gdRes = 0L;
	m_pdRes = 0L;
}

GGradCC2D::~GGradCC2D(void)
{
	this->Clean();
}

void GGradCC2D::Clean(void)
{
	if(m_gfPartial != 0L) cudaFree(m_gfPartial);
	if(m_gdRes != 0L) cudaFree(m_gdRes);
	if(m_pdRes != 0L) cudaFreeHost(m_pdRes);
	m_gfPartial = 0L;
	m_gdRes = 0L;
	m_pdRes = 0L;
	m_iMaxFrames = 0;
}

void GGradCC2D::SetSize(int* piCmpSize, int iNumFrames)
{
	m_aiCmpSize[0] = piCmpSize[0];
	m_aiCmpSize[1] = piCmpSize[1];
	if(iNumFrames <= m_iMaxFrames) return;
	//-----------------
	this->Clean();
	m_iMaxFrames = iNumFrames;
	//-----------------
	size_t tBytes = sizeof(float) * GRAD_BLOCKS * GRAD_TERMS;
	cudaMalloc(&m_gfPartial, tBytes);
	//-----------------
	tBytes = sizeof(double) * GRAD_TERMS * (m_iMaxFrames + 1);
	cudaMalloc(&m_gdRes, tBytes);
	cudaMallocHost(&m_pdRes, tBytes);
}

void GGradCC2D::SetBFactor(float fBFactor)
{
	m_fBFactor = fBFactor;
}

//--------------------------------------------------------------------
// pfShift: the frame shift as stored in CStackShift. The frame is
//    corrected by the opposite shift as done in CAlignedSum.
// The results are stored in slot iFrame and available after Wait().
//--------------------------------------------------------------------
void GGradCC2D::DoFrame
(	cufftComplex* gCmpFrm,
	cufftComplex* gCmpSum,
	float* pfShift,
	int iFrame,
	cudaStream_t stream
)
{	float f2PI = (float)(8 * atan(1.0));
	int iNx = (m_aiCmpSize[0] - 1) * 2;
	float fShiftX = -pfShift[0] * (f2PI / iNx);
	float fShiftY = -pfShift[1] * (f2PI / m_aiCmpSize[1]);
	//-----------------
	mGGrad<<<GRAD_BLOCKS, GRAD_THREADS, 0, stream>>>(gCmpFrm,
	   gCmpSum, m_aiCmpSize[0], m_aiCmpSize[1], fShiftX, fShiftY,
	   m_fBFactor, m_gfPartial);
	mGReduce<<<1, 32, 0, stream>>>(m_gfPartial, GRAD_BLOCKS,
	   m_gdRes + iFrame * GRAD_TERMS);
}

//--------------------------------------------------------------------
// Returns sum_q W(q) |S(q)|^2 over the full plane excluding DC.
//--------------------------------------------------------------------
double GGradCC2D::DoPower
(	cufftComplex* gCmpSum,
	cudaStream_t stream
)
{	mGGrad<<<GRAD_BLOCKS, GRAD_THREADS, 0, stream>>>(0L,
	   gCmpSum, m_aiCmpSize[0], m_aiCmpSize[1], 0.0f, 0.0f,
	   m_fBFactor, m_gfPartial);
	double* gdRes = m_gdRes + m_iMaxFrames * GRAD_TERMS;
	mGReduce<<<1, 32, 0, stream>>>(m_gfPartial, GRAD_BLOCKS, gdRes);
	double* pdRes = m_pdRes + m_iMaxFrames * GRAD_TERMS;
	cudaMemcpyAsync(pdRes, gdRes, sizeof(double) * GRAD_TERMS,
	   cudaMemcpyDefault, stream);
	cudaStreamSynchronize(stream);
	return pdRes[6];
}

//--------------------------------------------------------------------
// Copies the results of the first iNumFrames frames to host. Each
// frame has GRAD_TERMS (7) doubles, see the comments at the top.
//--------------------------------------------------------------------
double* GGradCC2D::Wait(int iNumFrames, cudaStream_t stream)
{
	size_t tBytes = sizeof(double) * GRAD_TERMS * iNumFrames;
	cudaMemcpyAsync(m_pdRes, m_gdRes, tBytes,
	   cudaMemcpyDefault, stream);
	cudaStreamSynchronize(stream);
	return m_pdRes;
}
