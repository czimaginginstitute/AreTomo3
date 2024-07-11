#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

//----------------------------------------------------------
// s_gfCtfParam[0]: wavelength in pixel
// s_gfCtfParam[1]: Cs in pixel
//----------------------------------------------------------
static __device__ __constant__ float s_gfCtfParam[2];

//--------------------------------------------------------------------
// 1. fDfMean, fDfSigma are in pixel, not angstrom.
//--------------------------------------------------------------------
static __device__ float mGCalcPhase
(	float fDfMean,
	float fDfSigma,
	float fAzimuth,
	float fExtPhase,
	float fY
)
{	float fX = blockIdx.x * 0.5f / (gridDim.x - 1);
	//-----------------
	float fS2 = fX * fX + fY * fY;
	float fW2 = s_gfCtfParam[0] * s_gfCtfParam[0];
	//-----------------
	fX = atanf(fY / (fX + (float)1e-30));
	fX = fDfMean + fDfSigma * cosf(2.0f * (fX - fAzimuth));
	//-----------------
	fX = fExtPhase + 3.1415926f * s_gfCtfParam[0] * fS2
	   * (fX - 0.5f * fW2 * fS2 * s_gfCtfParam[1]);
	return fX;
}

//--------------------------------------------------------------------
// 1. Flip the phase of image Fourier transform (gCmp) when CTF is
//    positive. This keeps particles dark.
//--------------------------------------------------------------------
static __global__ void mGPhaseFlip
(	float fDfMean,
	float fDfSigma,
	float fAzmuth,
	float fExtPhase,
	cufftComplex* gCmp,
	int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(blockIdx.x == 0 && y == 0) return;
	if(y >= iCmpY) return;
	//-----------------
	float fY = y / (float)iCmpY;
	if(fY > 0.5f) fY = fY - 1.0f;
	//-----------------------------------------------
	// fY is the phase now.
	//-----------------------------------------------
	fY = mGCalcPhase(fDfMean, fDfSigma, 
	   fAzmuth, fExtPhase, fY);
	fY = -sinf(fY);
	if(fY <= 0) return;
	//-----------------
	int i = y * gridDim.x + blockIdx.x;
	gCmp[i].x = -gCmp[i].x;
	gCmp[i].y = -gCmp[i].y;
}

static __global__ void mGWeinerFilter
(	float fDfMean,
	float fDfSigma,
	float fAzmuth,
	float fExtPhase,
	float fBFactor,
	cufftComplex* gCmp,
	int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(blockIdx.x == 0 && y == 0) return;
	if(y >= iCmpY) return;
	//-----------------
	float fX = blockIdx.x * 0.5f / (gridDim.x - 1.0f);
	float fY = y / (float)iCmpY;
	if(fY > 0.5f) fY = fY - 1.0f;
	float fR2 = (fX * fX + fY * fY);
	//-----------------
	float fCTF = mGCalcPhase(fDfMean, fDfSigma,
	   fAzmuth, fExtPhase, fY);
	fCTF = -sinf(fCTF);
	//-----------------
	float fSign = (fCTF <= 0) ? 1.0f : -1.0f; // dark particles
	fX = 9.0f * expf(fR2 * 4.0f);
	fCTF = (fabsf(fCTF) + fX) / (fX + 1.0f) * fSign;
	fCTF = expf(-fBFactor * sqrtf(fR2)) / fCTF;
	//-----------------
	int i = y * gridDim.x + blockIdx.x;
	gCmp[i].x *= fCTF;
	gCmp[i].y *= fCTF;
}

/*
static __global__ void mGWeinerFilter
(       float fDfMean,
        float fDfSigma,
        float fAzmuth,
        float fExtPhase,
        float* gfNoise2,
        cufftComplex* gCmp,
        int iCmpY
)
{       int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(blockIdx.x == 0 && y == 0) return;
        if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
        //-----------------
        float fY = y / (float)iCmpY;
        if(fY > 0.5f) fY = fY - 1.0f;
        //-----------------
        float fCTF = mGCalcPhase(fDfMean, fDfSigma,
           fAzmuth, fExtPhase, fY);
        fCTF = sinf(fCTF);
        //-----------------
        float fFilter = blockIdx.x * 0.5f / (gridDim.x - 1);
        fFilter = expf(-10.0 * (fFilter * fFilter + fY * fY));
	//-----------------
	float fAmp2 = gCmp[i].x * gCmp[i].x + gCmp[i].y * gCmp[i].y;
       	fCTF = fAmp2 * fCTF / (fAmp2 * fCTF * fCTF + gfNoise2[0] + 1.0f);
	fCTF *= fFilter;
        //-----------------
        gCmp[i].x *= fCTF;
        gCmp[i].y *= fCTF;
}
*/
/*
static __global__ void mGCalcNoise2
(	cufftComplex* gCmp,
	int iCmpY,
	float* gfNoise2
)
{	extern __shared__ float s_afShared[];
	float* s_afCount = &s_afShared[blockDim.y];
	//-----------------
	float fNx = (gridDim.x - 1.0f) * 2.0f;
	float fSumAmp2 = 0.0f;
	float fCount = 0;
	cufftComplex aCmp;
	int i = 0;
	//-----------------
	for(int y=threadIdx.y; y<iCmpY; y+=blockDim.y)
	{	float fY = y / (float)iCmpY;
		if(fY > 0.5f) fY -= 1.0f;
		for(int x=0; x<gridDim.x; x++)
		{	float fX = x / fNx;
			fX = sqrtf(fX * fX + fY * fY);
			if(fX < 0.5f) continue;
			//---------------
			aCmp = gCmp[y * gridDim.x + blockIdx.x];
			fSumAmp2 += (aCmp.x * aCmp.x + aCmp.y * aCmp.y);
			fCount += 1;
		}
	}
	s_afShared[threadIdx.y] = fSumAmp2;
	s_afCount[threadIdx.y] = fCount;
	__syncthreads();
	//-----------------
	int iOffset = blockDim.y / 2;
	while(iOffset > 0)
	{	if(threadIdx.y < iOffset)
		{	i = iOffset + threadIdx.y;
			s_afShared[threadIdx.y] += s_afShared[i];
			s_afCount[threadIdx.y] += s_afCount[i];
		}
		__syncthreads();
		iOffset /= 2;
	}
	if(threadIdx.y != 0) return;
	//-----------------	
	if(s_afCount[0] == 0) gfNoise2[0] = 0.0f;
	else gfNoise2[0] = s_afShared[0] / s_afCount[0];	
}
*/

GCorrCTF2D::GCorrCTF2D(void)
{
	m_gfNoise2 = 0L;
	m_bPhaseFlip = false;
	m_fBFactor = 15.0f;
}

GCorrCTF2D::~GCorrCTF2D(void)
{
	if(m_gfNoise2 != 0L) cudaFree(m_gfNoise2);
	m_gfNoise2 = 0L;
}

void GCorrCTF2D::SetParam(MD::CCtfParam* pCtfParam)
{
	float afCtfParam[2] = {0.0f};
	afCtfParam[0] = pCtfParam->m_fWavelength;
	afCtfParam[1] = pCtfParam->m_fCs;
	cudaMemcpyToSymbol(s_gfCtfParam, afCtfParam, sizeof(float) * 2);
	//-----------------
	m_fAmpPhase = (float)atanf(pCtfParam->m_fAmpContrast / (1.0f 
	   - pCtfParam->m_fAmpContrast * pCtfParam->m_fAmpContrast));
	//-----------------
	if(m_gfNoise2 == 0L) cudaMalloc(&m_gfNoise2, sizeof(float));
}

void GCorrCTF2D::SetPhaseFlip(bool bValue)
{
	m_bPhaseFlip = bValue;
}

void GCorrCTF2D::SetLowpass(int iBFactor)
{
	m_fBFactor = (float)iBFactor;
	if(m_fBFactor < 0) m_fBFactor = 0.0f;
}

void GCorrCTF2D::DoIt
(	float fDfMin,   float fDfMax, 
	float fAzimuth, float fExtPhase,
	float fTilt, cufftComplex* gCmp, 
	int* piCmpSize, cudaStream_t stream
)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piCmpSize[0], 1);
	aGridDim.y = (piCmpSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//-----------------
	//size_t tSmBytes = sizeof(float) * aBlockDim.y * 2;
	//mGCalcNoise2<<<aGridDim, aBlockDim, tSmBytes, stream>>>(gCmp,
	//   piCmpSize[1], m_gfNoise2);
	//-----------------
	float fDfMean = 0.5f * (fDfMin + fDfMax);
        float fDfSigma = 0.5f * (fDfMax - fDfMin);
	float fAddPhase = m_fAmpPhase + fExtPhase;
	//-----------------
	if(m_bPhaseFlip)
        {	mGPhaseFlip<<<aGridDim, aBlockDim, 0, stream>>>(fDfMean,
		   fDfSigma, fAzimuth, fAddPhase, gCmp, piCmpSize[1]);
	}
	else	
	{	float fBFactor = m_fBFactor / 
		   (float)(cos(fTilt * 0.01745) + 0.001f);
		//----------------
		mGWeinerFilter<<<aGridDim, aBlockDim, 0, stream>>>(fDfMean, 
		   fDfSigma, fAzimuth, fAddPhase, fBFactor, 
		   gCmp, piCmpSize[1]);
	}
}

void GCorrCTF2D::DoIt
(	MD::CCtfParam* pCtfParam, 
	float fTilt,
	cufftComplex* gCmp, 
	int* piCmpSize,
	cudaStream_t stream
)
{	this->SetParam(pCtfParam);
	this->DoIt(pCtfParam->m_fDefocusMin, pCtfParam->m_fDefocusMax,
	   pCtfParam->m_fAstAzimuth, pCtfParam->m_fExtPhase,
	   fTilt, gCmp, piCmpSize);
}
