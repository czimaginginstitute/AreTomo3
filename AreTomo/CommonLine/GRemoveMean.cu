#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::CommonLine;

#define BLOCK_SIZE_X 512

static __global__ void mGCalcMean
(	float* gfPadLine,
	int iSize
)
{	__shared__ float sfSum[BLOCK_SIZE_X];
	__shared__ int siCount[BLOCK_SIZE_X];
	sfSum[threadIdx.x] = 0.0f;
	siCount[threadIdx.x] = 0;
	__syncthreads();
	//--------------
	int iSegments = iSize / blockDim.x + 1;
	for(int i=0; i<iSegments; i++)
	{	int x = i * blockDim.x + threadIdx.x;
		if(x < iSize)
		{	float fVal = gfPadLine[x];
			if(fVal > (float)-1e10)
			{	sfSum[threadIdx.x] += fVal;
				siCount[threadIdx.x] += 1;
			}
		}
		__syncthreads();
	}
	//----------------------
	int iHalf = blockDim.x / 2;
	while(iHalf > 0)
	{	if(threadIdx.x < iHalf)
		{	int j = iHalf + threadIdx.x;
			sfSum[threadIdx.x] += sfSum[j];
			siCount[threadIdx.x] += siCount[j];
		}
		__syncthreads();
		iHalf = iHalf / 2;
	}
	//------------
	if(threadIdx.x == 0)
	{	if(siCount[0] == 0) gfPadLine[iSize+1] = 0;
		else gfPadLine[iSize+1] = sfSum[0] / siCount[0];
	}
}

static __global__ void mGRemoveMean
(	float* gfPadLine,
	int iSize
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iSize) return;
	//if(gfPadLine[i] < (float)-1e10) gfPadLine[i] = 0;
	if(gfPadLine[i] < 0) gfPadLine[i] = 0.0f;
	else gfPadLine[i] -= gfPadLine[iSize+1];
	//--------------------------------------
	float fHalf = 0.5f * iSize;
	float fR = fabsf(i - fHalf) / fHalf;
	fR = 0.5f * (1 - cosf(3.14159 * fR));
	fR = 1.0f - powf(fR, 100.0f);
	gfPadLine[i] *= (fR * fR);
}

GRemoveMean::GRemoveMean(void)
{
}

GRemoveMean::~GRemoveMean(void)
{
}

void GRemoveMean::DoIt
(	float* gfPadLine,
	int iPadSize
)
{	int iSize = (iPadSize / 2 - 1) * 2;
	dim3 aBlockDim(BLOCK_SIZE_X, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = iSize / aBlockDim.x + 1;
	mGCalcMean<<<aGridDim, aBlockDim>>>(gfPadLine, iSize);
	//----------------------------------------------------
	aGridDim.x = iSize / aBlockDim.x + 1;
	mGRemoveMean<<<aGridDim, aBlockDim>>>(gfPadLine, iSize);
}
