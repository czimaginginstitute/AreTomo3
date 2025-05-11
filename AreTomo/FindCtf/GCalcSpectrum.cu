#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

//-----------------------------------------------------------------------------
// 1. Calculate amplitude spectrum given gComp, the Fourier
//    transform.
// 2. The spectrum is the half spectrum with x frequency ranging from 0 to
//    0.5 and y frequency form -0.5 to +0.5.
// 3. The zero frequency is at (0, iCmpY / 2).
//-----------------------------------------------------------------------------
static __global__ void mGCalculate
(	cufftComplex* gComp, 
	float* gfSpectrum,
	int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//--------------------------------------------
	// put DC at x = 0, y = iCmpY / 2
	//--------------------------------------------
	y = y + iCmpY / 2;
	if(y >= iCmpY) y = y - iCmpY;
	//---------------------------
	float fAmp2 = gComp[i].x * gComp[i].x + gComp[i].y * gComp[i].y;
	gfSpectrum[y * gridDim.x + blockIdx.x] = sqrtf(fAmp2); 
}

static __global__ void mGLogrithm
(	float* gfSpectrum,
	int iSizeY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * gridDim.x + blockIdx.x;
	gfSpectrum[i] = logf(gfSpectrum[i]);
}

//------------------------------------------------------------------------------
// 1. DC of gfHalfSpect is already at (0, iCmpY / 2)
// 2. DC of gfFullSpect will be at (iCmpX / 2, iCmpY / 2)
//------------------------------------------------------------------------------
static __global__ void mGenFullSpect
(	float* gfHalfSpect,
	float* gfFullSpect,
	int iHalfX,
	int iCmpY,
	int iFullSizeX
)
{	int xSrc, ySrc, xDst, yDst;
	yDst = blockIdx.y * blockDim.y + threadIdx.y;
	if(yDst >= iCmpY) return;
	//-----------------------
	xDst = blockIdx.x - iHalfX;
	if(xDst >= 0)
	{	xSrc = xDst;
		ySrc = yDst;
	}
	else
	{	xSrc = -xDst;
		ySrc = iCmpY - yDst;
	}
	gfFullSpect[yDst * iFullSizeX + blockIdx.x] = 
	   gfHalfSpect[ySrc * (iHalfX + 1) + xSrc];
}

GCalcSpectrum::GCalcSpectrum(void)
{
	m_pCufft2D = new MU::CCufft2D;
}

GCalcSpectrum::~GCalcSpectrum(void)
{
	if(m_pCufft2D != 0L) delete m_pCufft2D;
}

void GCalcSpectrum::Clean(void)
{
	m_pCufft2D->DestroyPlan();
}

void GCalcSpectrum::SetSize(int* piSize, bool bPadded)
{
	m_aiPadSize[0] = piSize[0];
	m_aiPadSize[1] = piSize[1];
	if(!bPadded) m_aiPadSize[0] = (piSize[0] / 2 + 1) * 2;
	//-----------------
	m_aiCmpSize[0] = m_aiPadSize[0] / 2;
	m_aiCmpSize[1] = m_aiPadSize[1];
	//-----------------
	m_pCufft2D->CreateForwardPlan(m_aiPadSize, true);
}

void GCalcSpectrum::DoPad
(	float* gfPadImg,
	float* gfSpectrum,
	bool bLog
)
{	m_pCufft2D->Forward(gfPadImg, false);
	cudaStreamSynchronize((cudaStream_t)0);
	//-----------------
	this->DoIt((cufftComplex*)gfPadImg, gfSpectrum, m_aiCmpSize, bLog);
}

void GCalcSpectrum::DoIt
(	cufftComplex* gCmp,
	float* gfSpectrum,
	int* piCmpSize,
	bool bLog
)
{	dim3 aBlockDim(1, 512);
	int iGridY = piCmpSize[1] / aBlockDim.y + 1;
	dim3 aGridDim(piCmpSize[0], iGridY);
	mGCalculate<<<aGridDim, aBlockDim>>>(gCmp, gfSpectrum, piCmpSize[1]);
	if(!bLog) return;
	//-----------------
        mGLogrithm<<<aGridDim, aBlockDim>>>(gfSpectrum, piCmpSize[1]);
}


void GCalcSpectrum::Logrithm
(	float* gfSpectrum,
	int* piSize
)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piSize[0], piSize[1]/aBlockDim.y+1);
	mGLogrithm<<<aGridDim, aBlockDim>>>(gfSpectrum, piSize[1]);
}

void GCalcSpectrum::GenFullSpect
(	float* gfHalfSpect, 
	int* piCmpSize,
	float* gfFullSpect,
	bool bFullPadded
)
{	int iHalfX = piCmpSize[0] - 1;
	int iNx = iHalfX * 2;
	int iFullSizeX = bFullPadded ? iNx + 2 : iNx;
	//-----------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iNx, 1);
	aGridDim.y = (piCmpSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//-----------------
	mGenFullSpect<<<aGridDim, aBlockDim>>>(gfHalfSpect,
	   gfFullSpect, iHalfX, piCmpSize[1], iFullSizeX);
}	
