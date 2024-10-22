#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::Util;

//------------------------------------------
// c_aiSizes: iImgX, iImgY, iTileX, iTileY
//------------------------------------------
static __device__ __constant__ int c_aiSizes[4]; 

static __global__ void mGCalcRms
(	float* gfImg1, 
	float* gfImg2,
	int iStartX,
	int iStartY,
	float* gfSums
)
{	extern __shared__ float sfSum0[];
	float* sfSum1 = sfSum0 + blockDim.x;
	//-----------------
	int i = 0;
	float afSum[5] = {0.0f};
	for(int y=blockIdx.x; y<c_aiSizes[3]; y+=gridDim.x)
	{	i = (y + iStartY) * c_aiSizes[0] + iStartX;
		float* gfTile1 = gfImg1 + i;
		float* gfTile2 = gfImg2 + i;
		//----------------
		for(int x=threadIdx.x; x<c_aiSizes[2]; x+=blockDim.x)
		{	float fV = gfTile1[x] - gfTile2[x];
			afSum[0] += fV;
			afSum[1] += (fV * fV);
		}
	}
	//-----------------
	sfSum0[threadIdx.x] = afSum[0];
	sfSum1[threadIdx.x] = afSum[1];
	__syncthreads();
	//-----------------
	for(int iOffset = blockDim.x/2; iOffset > 0; iOffset = iOffset/2)
	{	if(threadIdx.x < iOffset)
		{	i = threadIdx.x + iOffset;
			sfSum0[threadIdx.x] += sfSum0[i];
			sfSum1[threadIdx.x] += sfSum1[i];
		}
	}
	__syncthreads();
	//--------------
	if(threadIdx.x != 0) return;
	gfSums[blockIdx.x] = sfSum0[0];
	gfSums[blockIdx.x + gridDim.x] = sfSum1[0];
}

static __global__ void mGSum1D(float* gfSums)
{	
	extern __shared__ float sfSum0[];
	float* sfSum1 = sfSum0 + blockDim.x;
	sfSum0[threadIdx.x] = gfSums[threadIdx.x];
	sfSum1[threadIdx.x] = gfSums[threadIdx.x + blockDim.x];
	__syncthreads();
	//-----------------
	for(int iOffset = blockDim.x/2; iOffset > 0; iOffset = iOffset/2)
	{	if(threadIdx.x < iOffset)
		{	int i = threadIdx.x + iOffset;
			sfSum0[threadIdx.x] += sfSum0[i];
			sfSum1[threadIdx.x] += sfSum1[i];
		}
	}
	//-----------------
	if(threadIdx.x != 0) return;
	gfSums[0] = sfSum0[0];
	gfSums[1] = sfSum1[0];
}

//--------------------------------------------------------------------
// 1. This class is implemented for correlating a pair of patches
//    from two adjacent xy slices in a tomogram.
//--------------------------------------------------------------------
GLocalRms2D::GLocalRms2D(void)
{
	m_aBlockDim.x = 256;
	m_aGridDim.x = 256;
	m_aBlockDim.y = 1;
	m_aGridDim.y = 1;
	m_gfSums = 0L;
}

GLocalRms2D::~GLocalRms2D(void)
{
	if(m_gfSums != 0L) cudaFree(m_gfSums);
}

void GLocalRms2D::SetSizes(int* piImgSize, int* piTileSize)
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
	else if(iGridDimX > 256) iGridDimX = 256;
	m_aGridDim.x = iGridDimX;
	//-----------------
	if(m_gfSums != 0L) cudaFree(m_gfSums);
	cudaMalloc(&m_gfSums, m_aGridDim.x * 5 * sizeof(float));
}

float GLocalRms2D::DoIt
(	float* gfImg1, 
	float* gfImg2, 
	int* piStart
)
{	m_fRms = 0.0f;
	//-----------------
	int iShmBytes = sizeof(float) * m_aBlockDim.x * 2;
        mGCalcRms<<<m_aGridDim, m_aBlockDim, iShmBytes>>>(gfImg1,
	   gfImg2, piStart[0], piStart[1], m_gfSums);
	//-----------------
	dim3 aBlockDim(m_aGridDim.x, 1);
	dim3 aGridDim(1, 1);
	iShmBytes = sizeof(float) * aBlockDim.x * 3;
	mGSum1D<<<aGridDim, aBlockDim, iShmBytes>>>(m_gfSums);
	//-----------------
	float afRes[2] = {0.0f};
	cudaMemcpy(afRes, m_gfSums, sizeof(afRes), cudaMemcpyDefault);
	//-----------------
	int iTilePixels = m_aiSizes[2] * m_aiSizes[3];
	afRes[0] /= iTilePixels;
	afRes[1] /= iTilePixels;
	m_fRms = afRes[0] - afRes[1] * afRes[1];
	if(m_fRms <= 0) m_fRms = 0.0f;
	else m_fRms = (float)sqrt(m_fRms);
	return m_fRms;
}

