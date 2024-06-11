#include "CMaUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define OPT_NVIDIA

using namespace McAreTomo::MaUtil;

//--------------------------------------------------------------------
// 1. gCmp1 and gCmp2 are the Fourier transforms of square images.
// 2. gCmp1 and gCmp2 are recommmended to be normalized by number
//    of pixels to prevent overflow.
// 3. gfFRC is a 1D array of size iCmpX. 
//--------------------------------------------------------------------
static __global__ void mGCalcFRC
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	float* gfFRC,
	int iCmpY,
	int iWidth	
)
{	extern __shared__ float s_gfConv[];
	float* s_gfPow1 = &s_gfConv[blockDim.y];
	float* s_gfPow2 = &s_gfConv[blockDim.y * 2];
	float* s_gfCount = &s_gfConv[blockDim.y * 3];
	//-----------------
	int iLow = blockIdx.x - iWidth / 2;
	int iHigh = iLow + iWidth;
	if(iLow < 0)
	{	iLow = 0;
		iHigh = iWidth;
	}
	else if(iHigh >= gridDim.x)
	{	iHigh = gridDim.x - 1;
		iLow = iHigh - iWidth;
	}
	//-----------------
	cufftComplex c1, c2;
	float fSumCC = 0.0f, fCount = 0.0f;
	float fSumP1 = 0.0f, fSumP2 = 0.0f;
	float fHalfY = 0.5f * iCmpY;
	float fR = 0.0f;
	int i = 0;
	//-----------------
	for(int y=threadIdx.y; y<iCmpY; y+=blockDim.y)
	{	float fY = (y < fHalfY) ? y : y - iCmpY;
		if(fabsf(fY) >= iHigh) continue;
		//----------------
		for(int x=0; x<gridDim.x; x++)
		{	if(x >= iHigh) continue;
			//---------------
			fR = sqrtf(x * x + fY * fY);
			if(fR < iLow || fR >= iHigh) continue;
			//---------------
			i = y * gridDim.x + x;
			c1 = gCmp1[i];
			c2 = gCmp2[i];
			fSumCC += (c1.x * c2.x + c1.y * c2.y);
			fSumP1 += (c1.x * c1.x + c1.y * c1.y);
			fSumP2 += (c2.x * c2.x + c2.y * c2.y);
			fCount += 1.0f;
		}	
	}
	s_gfConv[threadIdx.y] = fSumCC;
	s_gfPow1[threadIdx.y] = fSumP1;
	s_gfPow2[threadIdx.y] = fSumP2;
	s_gfCount[threadIdx.y] = fCount;
	__syncthreads();
	//-----------------------------------------------
	// iLow is used below for offset
	//-----------------------------------------------
	iLow = blockDim.y / 2;
	while(iLow > 0)
	{	if(threadIdx.y < iLow)
		{	i = iLow + threadIdx.y;
			s_gfConv[threadIdx.y] += s_gfConv[i];
			s_gfPow1[threadIdx.y] += s_gfPow1[i];
			s_gfPow2[threadIdx.y] += s_gfPow2[i];
			s_gfCount[threadIdx.y] += s_gfCount[i];
		}
		__syncthreads();
		iLow /= 2;
	}
	if(threadIdx.y != 0) return;
	//-----------------
	if(s_gfCount[0] == 0)
	{	gfFRC[blockIdx.x] = 0.0f;
	}
	else
	{	gfFRC[blockIdx.x] = s_gfConv[0] / 
		   sqrtf(s_gfPow1[0] * s_gfPow2[0]);
	}
}

//--------------------------------------------------------------------
// 1. m_fRes is the resolution in pixels at FRC = 0.143. Nyquist 
//    is at 2.0 pixels.
//--------------------------------------------------------------------

GCalcFRC::GCalcFRC(void)
{
	m_fRes = 0.0f;
}

GCalcFRC::~GCalcFRC(void)
{
}

//--------------------------------------------------------------------
// 1. gfFRC is a 1D array of piCmpSize[0].
// 2. Need to call GCalcFRC::Setup() beforehand.
//--------------------------------------------------------------------
void GCalcFRC::DoIt
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	float* gfFRC,
	int iRingWidth, // in pixel
	int* piCmpSize,
        cudaStream_t stream
)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piCmpSize[0], 1);
	aGridDim.y = (piCmpSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//-----------------
	size_t tSharedBytes = sizeof(float) * aBlockDim.y * 4; 
	//-----------------
	mGCalcFRC<<<aGridDim, aBlockDim, tSharedBytes, stream>>>(gCmp1, 
	   gCmp2, gfFRC, piCmpSize[1], iRingWidth);
}

float* GCalcFRC::DoIt
(	float* pfImg1, float* pfImg2,
	int* piImgSize,
	bool bPadded,
	int iRingWidth
)
{	m_aiCmpSize[0] = piImgSize[0] / 2 + 1;
	if(bPadded) m_aiCmpSize[0] = piImgSize[0] / 2;
	m_aiCmpSize[1] = piImgSize[1];
	//-----------------
	float* gfPadImg1 = mHost2Gpu(pfImg1, piImgSize);
	float* gfPadImg2 = mHost2Gpu(pfImg2, piImgSize);
	//-----------------
	mCalcFFT(gfPadImg1, gfPadImg2);
	cufftComplex* gCmp1 = (cufftComplex*)gfPadImg1;
	cufftComplex* gCmp2 = (cufftComplex*)gfPadImg2;
	//-----------------
	float *gfFRC = 0L;
	size_t tBytes = sizeof(float) * m_aiCmpSize[0];
	cudaMalloc(&gfFRC, tBytes);
	//-----------------
	this->DoIt(gCmp1, gCmp2, gfFRC, iRingWidth, 
	   m_aiCmpSize, (cudaStream_t)0);
	//-----------------
	if(gfPadImg1 != 0L) cudaFree(gfPadImg1);
	if(gfPadImg2 != 0L) cudaFree(gfPadImg2);
	//-----------------
	float* pfFRC = new float[m_aiCmpSize[0]];
	cudaMemcpy(pfFRC, gfFRC, sizeof(float) * m_aiCmpSize[0],
	   cudaMemcpyDefault);
	if(gfFRC != 0L) cudaFree(gfFRC);
	//-----------------
	return pfFRC;
}

float* GCalcFRC::mHost2Gpu(float* pfImg, int* piImgSize)
{
	float* gfPadImg = 0L;
	int iPadX = m_aiCmpSize[0] * 2;
	size_t tBytes = sizeof(float) * iPadX * m_aiCmpSize[1];
	cudaMalloc(&gfPadImg, tBytes);
	//-----------------
	tBytes = sizeof(float) * piImgSize[0];
	for(int y=0; y<piImgSize[1]; y++)
	{	cudaMemcpy(&gfPadImg[y * iPadX], &pfImg[y * piImgSize[0]],
		   tBytes, cudaMemcpyDefault);
	}
	return gfPadImg;
}

void GCalcFRC::mCalcFFT(float* gfPadImg1, float* gfPadImg2)
{
	CCufft2D cufft2D;
	int aiPadSize[] = {m_aiCmpSize[0] * 2, m_aiCmpSize[1]};
	cufft2D.CreateForwardPlan(aiPadSize, true);
	//-----------------
	cufft2D.Forward(gfPadImg1, (cufftComplex*)gfPadImg1, true);
	cufft2D.Forward(gfPadImg2, (cufftComplex*)gfPadImg2, true);
}

