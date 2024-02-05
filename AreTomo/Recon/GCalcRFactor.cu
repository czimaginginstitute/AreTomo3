#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::Recon;

extern __shared__ char s_acArray[];

static __global__ void mGCalcRFactor
(	int iProjX,
	int iNumProjs,
	float* gfProj, // full projection, not y-slice
	float* gfSum,  // size of iProjY
	int* giCount   // size of iProjY
)
{	float* sfSum = (float*)&s_acArray[0];
	int* siCount = (int*)&sfSum[blockDim.x];
	sfSum[threadIdx.x] = 0.0f;
	siCount[threadIdx.x] = 0;
	__syncthreads();
	//--------------
	int i, j, k;
	int iPixels = iProjX * iNumProjs;
	j = blockDim.x * gridDim.x;
	int iSegments = (iPixels + j - 1) / j;
	j = blockIdx.x * blockDim.x + threadIdx.x;
	//----------------------------------------
	for(i=0; i<iSegments; i++)
	{	k = j * iSegments + i;
		if(k >= iPixels) break;
		float fVal = gfProj[k];
		if(fVal < (float)-1e10) continue;
		//-------------------------------
		sfSum[threadIdx.x] += fabs(fVal);
		siCount[threadIdx.x] += 1;
	}
	__syncthreads();
	//--------------
	i = blockDim.x / 2;
	while(i > 0)
	{	if(threadIdx.x < i)
		{	sfSum[threadIdx.x] += sfSum[i+threadIdx.x];
			siCount[threadIdx.x] += siCount[i+threadIdx.x];
		}
		__syncthreads();
		i /= 2;
	}
	//-------------
	if(threadIdx.x == 0)
	{	gfSum[blockIdx.x] = sfSum[0];
		giCount[blockIdx.x] = siCount[0];
	}
}

static __global__ void mGCalcRFactorSum
(	float* gfSum,
	int* giCount
)
{	float* sfSum = (float*)&s_acArray[0];
	int* siCount = (int*)&sfSum[blockDim.x];
	sfSum[threadIdx.x] = gfSum[threadIdx.x];
	siCount[threadIdx.x] = giCount[threadIdx.x];
	__syncthreads();
	//--------------
	int i = blockDim.x / 2;
	while(i > 0)
	{	if(threadIdx.x < i)
		{	sfSum[threadIdx.x] += sfSum[i+threadIdx.x];
			siCount[threadIdx.x] += siCount[i+threadIdx.x];
		}
		__syncthreads();
		i /= 2;
	}
	//-------------
	if(threadIdx.x == 0)
	{	gfSum[0] = sfSum[0];
		giCount[0] = siCount[0];
	}
}	

GCalcRFactor::GCalcRFactor(void)
{
	m_gfSum = 0L;
	m_giCount = 0L;
}

GCalcRFactor::~GCalcRFactor(void)
{
	this->Clean();
}

void GCalcRFactor::Clean(void)
{
	if(m_gfSum != 0L) cudaFree(m_gfSum);
	if(m_giCount != 0L) cudaFree(m_giCount);
	m_gfSum = 0L;
	m_giCount = 0L;
}

void GCalcRFactor::Setup(int iProjX, int iNumProjs)
{
	this->Clean();
	m_iProjX = iProjX;
	m_iNumProjs = iNumProjs;
	cudaMalloc(&m_gfSum, sizeof(float) * m_iNumProjs);
	cudaMalloc(&m_giCount, sizeof(int) * m_iNumProjs);
}

void GCalcRFactor::DoIt
(	float* gfProj,
	float* pfRfSum,
	int* piRfCount,
	cudaStream_t stream
)
{	dim3 aBlockDim(512, 1);
	dim3 aGridDim(512, 1);
	int iShmBytes = (sizeof(float) + sizeof(int)) * aBlockDim.x;
	mGCalcRFactor<<<aGridDim, aBlockDim, iShmBytes, stream>>>
	( m_iProjX, m_iNumProjs, gfProj, m_gfSum, m_giCount
	);
	//-------------------------------------------------
	aGridDim.x = 1;
	mGCalcRFactorSum<<<aGridDim, aBlockDim, iShmBytes, stream>>>
	(m_gfSum, m_giCount);
	//-------------------
	cudaMemcpyAsync
	( pfRfSum, m_gfSum, sizeof(float),
	  cudaMemcpyDefault, stream
	);
	cudaMemcpyAsync
	( piRfCount, m_giCount, sizeof(int), 
	  cudaMemcpyDefault, stream
	);
}
	
