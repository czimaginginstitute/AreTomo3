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
	fX = 8.0f * expf(fR2 * 2.0f);
	fCTF = (fabsf(fCTF) + fX) / (fX + 1.0f) * fSign;
	fCTF = expf(-fBFactor * sqrtf(fR2)) / fCTF;
	//-----------------
	int i = y * gridDim.x + blockIdx.x;
	gCmp[i].x *= fCTF;
	gCmp[i].y *= fCTF;
}

static __global__ void mGMultiplyCTF
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
	//---------------------------
	float fY = y / (float)iCmpY;
	if(fY > 0.5f) fY = fY - 1.0f;
	//---------------------------
	float fX = blockIdx.x * 0.5f / (gridDim.x - 1.0f);
	float fR = sqrtf(fX * fX + fY * fY);
	//---------------------------
	float fCTF = mGCalcPhase(fDfMean, fDfSigma,
	   fAzmuth, fExtPhase, fY);
	fCTF = fmaxf(fCTF, 1.5708f); // 1 to the 1st peak
	fCTF = sinf(fCTF); // dark particles
	fCTF *= expf(-15.0f * fR);
	//---------------------------
	int i = y * gridDim.x + blockIdx.x;
	gCmp[i].x *= fCTF;
	gCmp[i].y *= fCTF;
}

//--------------------------------------------------------------------
// 1. m_iMode = 1: Wiener filter (SZ version), m_iMode = 2: phase
//    flipping, m_iMode = 3: multiplying CTF for template matching.
//--------------------------------------------------------------------
GCorrCTF2D::GCorrCTF2D(void)
{
	m_gfNoise2 = 0L;
	m_iMode = 1;  // Wiener filter
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

void GCorrCTF2D::SetMode(int iMode)
{
	m_iMode = iMode;
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
	//---------------------------
	float fDfMean = 0.5f * (fDfMin + fDfMax);
        float fDfSigma = 0.5f * (fDfMax - fDfMin);
	float fAddPhase = m_fAmpPhase + fExtPhase;
	//---------------------------
	if(m_iMode == 2)
        {	mGPhaseFlip<<<aGridDim, aBlockDim, 0, stream>>>(fDfMean,
		   fDfSigma, fAzimuth, fAddPhase, gCmp, piCmpSize[1]);
	}
	else if(m_iMode == 3)
	{	mGMultiplyCTF<<<aGridDim, aBlockDim, 0, stream>>>(fDfMean,
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
