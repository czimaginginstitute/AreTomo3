#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::Util;

extern __shared__ char s_acArray[];

static __global__ void mGConv2D
(	float* gfImg1,
	float* gfImg2,
	int iSizeX,
	int iSizeY,
	int iPadX,
	float fMissingVal,
	float* gfRes
)
{	extern __shared__ float shared[];
	//-------------------------------
	float afTemp[6] = {0.0f};
	for(int y=blockIdx.x; y<iSizeY; y+=gridDim.x)
	{	float* gfPtr1 = gfImg1 + y * iPadX;
		float* gfPtr2 = gfImg2 + y * iPadX;
		for(int x=threadIdx.x; x<iSizeX; x+=blockDim.x)
		{	float v1 = gfPtr1[x];
			if(v1 < fMissingVal) continue;
			float v2 = gfPtr2[x];
			if(v2 < fMissingVal) continue;
			//----------------------------
			afTemp[0] += v1;
			afTemp[1] += v2;
			afTemp[2] += (v1 * v1);
			afTemp[3] += (v2 * v2);
			afTemp[4] += (v1 * v2);
			afTemp[5] += 1.0f;
		}
	}
	//-------------------------------------
	for(int i=0; i<6; i++)
	{	shared[threadIdx.x + i * blockDim.x] = afTemp[i];
	}
	__syncthreads();
	//--------------
	for(int offset=blockIdx.x/2; offset>0; offset=offset/2)
	{	if(threadIdx.x < offset)
		{	for(int i=0; i<6; i++)
			{	int j = threadIdx.x + i * blockDim.x;
				shared[j] += shared[j + offset];
			}
		}
		__syncthreads();
	}
	//----------------------
	if(threadIdx.x == 0)
	{	for(int i=0; i<6; i++)
		{	gfRes[6 * blockIdx.x + i] = shared[i * blockDim.x];
		}
	}
}

static __global__ void mGSum1D(float* gfSums)
{	
	extern __shared__ float shared[];
	for(int i=0; i<6; i++)
	{	shared[i * blockDim.x+threadIdx.x] 
		= gfSums[6 * threadIdx.x + i];
	}
	__syncthreads(); 
	//--------------
	for(int offset=blockDim.x/2; offset>0; offset=offset/2)
	{	if(threadIdx.x < offset)
		{	for(int i=0; i<6; i++)
			{	int j = i * blockDim.x + threadIdx.x;
				shared[j] += shared[j + offset];
			}
			__syncthreads();
		}
	}
	//------------------------------
	if(threadIdx.x == 0)
	{	for(int i=0; i<6; i++)
		{	gfSums[i] = shared[i * blockDim.x];
		}
	}
}

GRealCC2D::GRealCC2D(void)
{
	m_fMissingVal = (float)-1e10;
}

GRealCC2D::~GRealCC2D(void)
{
}

void GRealCC2D::SetMissingVal(double dVal)
{
	m_fMissingVal = (float)dVal;
}

float GRealCC2D::DoIt
(	float* gfImg1, 
	float* gfImg2,
	float* gfBuf,
	int* piSize,
	bool bPad,
	cudaStream_t stream
)
{	int iSizeX = piSize[0];
	int iPadX = piSize[0];
	if(bPad) iSizeX = (iPadX / 2 - 1) * 2;
	//------------------------------------
	dim3 aBlockDim(128, 1, 1);
	dim3 aGridDim(128, 1, 1);
	int iShmBytes = sizeof(float) * aBlockDim.x * 6;
	//----------------------------------------------
	mGConv2D<<<aGridDim, aBlockDim, iShmBytes, stream>>>
	( gfImg1, gfImg2, iSizeX, piSize[1], iPadX,
	  m_fMissingVal, gfBuf );
	//-----------------------
	mGSum1D<<<1, aBlockDim, iShmBytes, stream>>>(gfBuf);
	float afVals[6] = {0.0f};
	cudaMemcpy(afVals, gfBuf, sizeof(afVals), cudaMemcpyDefault);
	//-----------------------------------------------------------
	for(int i=0; i<5; i++) afVals[i] /= afVals[5];
	double dStd1 = afVals[2] - afVals[0];
	double dStd2 = afVals[3] - afVals[1];
	double dCC = afVals[4] - afVals[0] * afVals[1];
	if(dStd1 < 0) dStd1 = 0;
	if(dStd2 < 0) dStd2 = 0;
	dStd1 = sqrt(dStd1);
	dStd2 = sqrt(dStd2);
	dCC = dCC / (dStd1 * dStd2 + 1e-30);
	if(dCC < -1) dCC = -1;
	else if(dCC > 1) dCC = 1;
	return (float)dCC;	
}

