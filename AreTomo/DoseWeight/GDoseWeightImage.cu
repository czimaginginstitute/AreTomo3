#include "CDoseWeightInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::AreTomo::DoseWeight;

static __device__ __constant__ float gfKvPixSize[2];

//===================================================================
// Weight scheme is based upon Niko lab's formula.
//===================================================================
static __device__ float mGCalcWeight(int y, int iCmpY, float fDose)
{	
	float fX = blockIdx.x * 0.5f / (gridDim.x - 1);
	float fY = y / (float)iCmpY;
	if(fY >= 0.5f) fY -= 1.0f;
	fX = sqrtf(fX * fX + fY * fY) / gfKvPixSize[1];
	//---------------------------------------------
	float fCritDose = 0.24499f * powf(fX, -1.6649f) + 2.8141f;
	fCritDose *= gfKvPixSize[0];
	float fWeight = expf(-0.5f * fDose / fCritDose);
	return fWeight; 
}

static __global__ void mGBuildWeightSum2
(	float fDose,
	int iCmpY,
	float* gfWeightSum
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	if(i == 0) return;
	//---------------------------------
	float fW = mGCalcWeight(y, iCmpY, fDose);
	gfWeightSum[i] += (fW * fW);
}

static __global__ void mGSqrt(int iCmpY, int iNumImgs, float* gfWeightSum)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
	if(i == 0) return;
	gfWeightSum[i] = sqrtf(gfWeightSum[i] / iNumImgs);
}

static __global__ void mGWeight
(	float fDose,
	int iCmpY,
	float* gfWeightSum,
	cufftComplex* gCmpFrame
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
	if(i == 0) return;
        //----------------
	float fW = mGCalcWeight(y, iCmpY, fDose);
	fW = fW / gfWeightSum[i];
	//-----------------------
	gCmpFrame[i].x *= fW;
	gCmpFrame[i].y *= fW;
}

GDoseWeightImage::GDoseWeightImage(void)
{
	m_gfWeightSum = 0L;
}

GDoseWeightImage::~GDoseWeightImage(void)
{
	this->Clean();
}

void GDoseWeightImage::Clean(void)
{
	if(m_gfWeightSum != 0L) cudaFree(m_gfWeightSum);
	m_gfWeightSum = 0L;
}

void GDoseWeightImage::BuildWeight
(	float fPixelSize,
	float fKv,
	float* pfImgDose, // accumulated dose
	int* piStkSize,
	cudaStream_t stream
)
{	this->Clean();
	if(pfImgDose == 0L) return;
	//------------------------
	float fKvFactor = 1.0f;
	if(fKv == 200) fKvFactor = 0.8f;
	else if(fKv == 120) fKvFactor = 0.45f;
	else if(fKv == 300) fKvFactor = 1.0f;
	else return;
	//----------
	m_aiCmpSize[0] = piStkSize[0] / 2 + 1;
	m_aiCmpSize[1] = piStkSize[1];
	//--------------------------
	size_t tBytes = sizeof(float) * m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gfWeightSum, tBytes);
	cudaMemset(m_gfWeightSum, 0, tBytes);
	//-----------------------------------
	float afKvPixSize[] = {fKvFactor, fPixelSize};
	cudaMemcpyToSymbolAsync(gfKvPixSize, afKvPixSize, sizeof(gfKvPixSize),
	   0, cudaMemcpyDefault, stream);
	//-------------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_aiCmpSize[0], 1);
	aGridDim.y = m_aiCmpSize[1] / aBlockDim.y + 1;
	for(int i=0; i<piStkSize[2]; i++)
	{	mGBuildWeightSum2<<<aGridDim, aBlockDim, 0, stream>>>(
		   pfImgDose[i], m_aiCmpSize[1], m_gfWeightSum);
	}
	mGSqrt<<<aGridDim, aBlockDim, 0, stream>>>(m_aiCmpSize[1], 
	   piStkSize[2], m_gfWeightSum);
}

void GDoseWeightImage::DoIt
( 	cufftComplex* gCmpFrame,
	float fDose,
	cudaStream_t stream
)
{	if(m_gfWeightSum == 0L) return;
	//-----------------------------
	dim3 aBlockDim(1, 128);
	dim3 aGridDim(m_aiCmpSize[0], 1);
	aGridDim.y = (m_aiCmpSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	mGWeight<<<aGridDim, aBlockDim, 0, stream>>>(fDose, 
	   m_aiCmpSize[1], m_gfWeightSum, gCmpFrame);
}

