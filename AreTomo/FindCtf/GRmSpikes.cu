#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

//-----------------------------------------------------------------------------
// 1. gfSpect is half power spectrum of which the DC is at (0, iSizeY/2).
//-----------------------------------------------------------------------------
static __global__ void mGFindBad
(	float* gfSpect,
	bool* gbBad,
	int iSizeY,
	int iHalfBox
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * gridDim.x + blockIdx.x;
	//-----------------
	int iHalfX = gridDim.x - 1;
	int iHalfY = iSizeY / 2;
	//-----------------
	y = y - iHalfY;
	float fR = blockIdx.x * 0.5f / (gridDim.x - 1);
	fR = sqrtf(fR * fR + y * y / (float)(iSizeY * iSizeY));
	if(fR < 0.04f)
	{	gbBad[i] = true;
		return;
	}
	//-----------------
	int iX = 0, iY = 0;
	float fMean = 0.0f, fStd = 0.0f;
	for(int k=-iHalfBox; k<=iHalfBox; k++)
	{	int yy = k + y;
		for(int j=-iHalfBox; j<=iHalfBox; j++)
		{	int xx = j + blockIdx.x;
			if(xx >= iHalfX) xx = xx - 2 * iHalfX;
			//---------------
			if(xx >= 0)
			{	iX = xx;
				iY = iHalfY + yy;
			}
			else
			{	iX = -xx;
				iY = iHalfY - yy;
			}
			if(iY < 0) iY += iSizeY;
			else if(iY >= iSizeY) iY -= iSizeY;
			//---------------
			iX = iY * gridDim.x + iX;
			fR = gfSpect[iX];
			fMean += fR;
			fStd += (fR * fR);
          	}
	}
	y = 2 * iHalfBox + 1;
	fMean = fMean / y;
	fStd = fStd / (y * y) - fMean * fMean;
	if(fStd <= 0) fStd = 0.0f;
	else fStd = sqrtf(fStd);
	//-----------------
	if(fStd == 0) gbBad[i] = false;
	else if(gfSpect[i] < (fMean - 5.0f * fStd)) gbBad[i] = true;
	else if(gfSpect[i] > (fMean + 5.0f * fStd)) gbBad[i] = true;
	else gbBad[i] = false;
}

static __global__ void mGRemove
(	float* gfSpect,
	bool* gbBad,
	int iSizeY,
	int iBoxSize
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * gridDim.x + blockIdx.x;
	if(!gbBad[i]) return;
	//-----------------
	int iHalfBox = iBoxSize / 2;
	int iSize = iBoxSize * iBoxSize;
	//-----------------
	unsigned int next = i;
	//-----------------
	for(int j=0; j<20; j++)
	{	next = (next * 19 + 57) % iSize;
		int iX = (next % iSize) - iHalfBox + blockIdx.x;
		if(iX < 0 || iX >= gridDim.x) continue;
		//----------------
		int iY = (next / iSize) - iHalfBox + y;	
		if(iY < 0 || iY >= iSizeY) continue;
		//----------------
		int k = iY * gridDim.x + iX;
		if(gbBad[k]) continue;
		//----------------
		gfSpect[i] = gfSpect[k];
		break;
	}
}

GRmSpikes::GRmSpikes(void)
{
}

GRmSpikes::~GRmSpikes(void)
{
}

void GRmSpikes::DoIt
(	float* gfSpect,
	int* piSpectSize,
	int iBoxSize
)
{	bool* gbBad = 0L;
	int iBytes = piSpectSize[0] * piSpectSize[1] * sizeof(bool);
	cudaMalloc(&gbBad, iBytes);
	//-----------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piSpectSize[0], 1);
	aGridDim.y = (piSpectSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//-----------------
	mGFindBad<<<aGridDim, aBlockDim>>>(gfSpect, gbBad,
	   piSpectSize[1], iBoxSize / 2);
	//-----------------
	mGRemove<<<aGridDim, aBlockDim>>>(gfSpect, gbBad,
	   piSpectSize[1], 5);
	//-----------------
	cudaFree(gbBad);
}

