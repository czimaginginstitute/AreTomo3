#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::Util;

static __constant__ float c_aiSizes[4]; // iImgX, iImgY, iTileX, iTileY
extern __shared__ char s_acArray[];

static __global__ void mGConv
(	float* gfImg1, 
	float* gfImg2,
	int iStartX,
	int iStartY,
	float* gfSums
)
{	float* sfSum0 = (float*)&s_acArray[0];
	float* sfSum1 = sfSum0 + blockDim.x;
	float* sfSum2 = sfSum1 + blockDim.x;
	float* sfSum3 = sfSum2 + blockDim.x;
	float* sfSum4 = sfSum3 + blockDim.x;
	//-----------------
	int i = 0;
	float afSum[5] = {0.0f};
	for(int y=blockIdx.x; i<c_aiSizes[3]; i+=gridDim.x)
	{	i = (y + iStartY) * c_aiSizes[0] + iStartX;
		float* gfTile1 = gfImg1 + i;
		float* gfTile2 = gfImg2 + i;
		//----------------
		for(int x=threadIdx.x; x<c_aiSizes[2]; x+=blockDim.x)
		{	float f1 = gfTile1[x];
			float f2 = gfTile2[x];
			afSum[0] += f1;
			afSum[1] += f2;
			afSum[2] += (f1 * f1);
			afSum[3] += (f2 * f2);
			afSum[4] += (f1 * f2);
		}
	}
	//-----------------
	sfSum0[threadIdx.x] = afSum[0];
	sfSum1[threadIdx.x] = afSum[1];
	sfSum2[threadIdx.x] = afSum[2];
	sfSum3[threadIdx.x] = afSum[3];
	sfSum4[threadIdx.x] = afSum[4];
	__syncthreads();
	//-----------------
	for(int iOffset = blockDim.x/2; iOffset > 0; iOffset = iOffset/2)
	{	if(threadIdx.x < iOffset)
		{	i = threadIdx.x + iOffset;
			sfSum0[threadIdx.x] += sfSum0[i];
			sfSum1[threadIdx.x] += sfSum1[i];
			sfSum2[threadIdx.x] += sfSum2[i];
			sfSum3[threadIdx.x] += sfSum3[i];
			sfSum4[threadIdx.x] += sfSum4[i];
		}
	}
	__syncthreads();
	//--------------
	if(threadIdx.x != 0) return;
	gfSums[blockIdx.x] = sfSum0[0];
	gfSums[blockIdx.x + gridDim.x] = sfSum1[0];
	gfSums[blockIdx.x + gridDim.x * 2] = sfSum2[0];
	gfSums[blockIdx.x + gridDim.x * 3] = sfSum3[0];
	gfSums[blockIdx.x + gridDim.x * 4] = sfSum4[0];
}

static __global__ void mGSum1D(float* gfSums)
{	
	float* sfSum0 = (float*)&s_acArray[0];
	float* sfSum1 = sfSum0 + blockDim.x;
	float* sfSum2 = sfSum1 + blockDim.x;
	float* sfSum3 = sfSum2 + blockDim.x;
	float* sfSum4 = sfSum3 + blockDim.x;
	sfSum0[threadIdx.x] = gfSums[threadIdx.x];
	sfSum1[threadIdx.x] = gfSums[threadIdx.x + blockDim.x];
	sfSum2[threadIdx.x] = gfSums[threadIdx.x + blockDim.x * 2];
	sfSum3[threadIdx.x] = gfSums[threadIdx.x + blockDim.x * 3];
	sfSum4[threadIdx.x] = gfSums[threadIdx.x + blockDim.x * 4];
	__syncthreads();
	//-----------------
	for(int iOffset = blockDim.x/2; iOffset > 0; iOffset = iOffset/2)
	{	if(threadIdx.x < iOffset)
		{	int i = threadIdx.x + iOffset;
			sfSum0[threadIdx.x] += sfSum0[i];
			sfSum1[threadIdx.x] += sfSum1[i];
			sfSum2[threadIdx.x] += sfSum2[i];
			sfSum3[threadIdx.x] += sfSum3[i];
			sfSum4[threadIdx.x] += sfSum4[i];
		}
	}
	//-----------------
	if(threadIdx.x != 0) return;
	gfSums[0] = sfSum0[0];
	gfSums[1] = sfSum1[0];
	gfSums[2] = sfSum2[0];
	gfSums[3] = sfSum3[0];
	gfSums[4] = sfSum4[0];
}

//--------------------------------------------------------------------
// 1. This class is implemented for correlating a pair of patches
//    from two adjacent xy slices in a tomogram.
//--------------------------------------------------------------------
GLocalCC2D::GLocalCC2D(void)
{
	m_aBlockDim.x = 256;
	m_aGridDim.x = 256;
	m_aBlockDim.y = 1;
	m_aGridDim.y = 1;
	m_gfSums = 0L;
}

GLocalCC2D::~GLocalCC2D(void)
{
	if(m_gfSums != 0L) cudaFree(m_gfSums);
}

void GLocalCC2D::SetSizes(int* piImgSize, int* piTileSize)
{	
	m_aiSizes[0] = piImgSize[0];
	m_aiSizes[1] = piImgSize[1];
	m_aiSizes[2] = piTileSize[0];
	m_aiSizes[3] = piTileSize[1];
	cudaMemcpyToSymbol(c_aiSizes, m_aiSizes, sizeof(m_aiSizes));
	//-----------------
	int iTilePixels = piTileSize[0] * piTileSize[1];
	int iGridDimX = (iTilePixels + m_aBlockDim.x - 1) / m_aBlockDim.x;
	iGridDimX = (iGridDimX / 32) * 32;
	if(iGridDimX < 32) iGridDimX = 32;
	else if(iGridDimX > 512) iGridDimX = 512;
	m_aGridDim.x = iGridDimX;
	//-----------------
	if(m_gfSums != 0L) cudaFree(m_gfSums);
	m_gfSums = 0L;
	cudaMalloc(&m_gfSums, m_aGridDim.x * 5 * sizeof(float));
}

float GLocalCC2D::DoIt
(	float* gfImg1, 
	float* gfImg2, 
	int* piStart
)
{	int iShmBytes = sizeof(float) * m_aGridDim.x * 5;
        mGConv<<<m_aGridDim, m_aBlockDim, iShmBytes>>>(gfImg1,
	   gfImg2, piStart[0], piStart[1], m_gfSums);
	//-----------------
	dim3 aBlockDim(m_aGridDim.x, 1);
	dim3 aGridDim(1, 1);
	iShmBytes = sizeof(float) * aBlockDim.x * 5;
	mGSum1D<<<aGridDim, aBlockDim, iShmBytes>>>(m_gfSums);
	//-----------------
	float afRes[5] = {0.0f};
	cudaMemcpy(afRes, m_gfSums, sizeof(afRes), cudaMemcpyDefault);
	//-----------------
	int iTilePixels = m_aiSizes[2] * m_aiSizes[3];
	for(int i=0; i<5; i++) afRes[i] /= iTilePixels;
	//-----------------
	afRes[2] = afRes[2] - afRes[0] * afRes[0];
	afRes[3] = afRes[3] - afRes[1] * afRes[1];
	if(afRes[2] > 0) afRes[2] = (float)sqrt(afRes[2]);
	else return 0.0f;
	if(afRes[3] > 0) afRes[3] = (float)sqrt(afRes[3]);
	else return 0.0f;
	//---------------
	float fCC = (afRes[4] - afRes[0] * afRes[1]) /(afRes[2] * afRes[3]);
	return fCC;
}

