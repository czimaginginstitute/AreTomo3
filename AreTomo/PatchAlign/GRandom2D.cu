#include "CPatchAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::PatchAlign;

static __global__ void mGDoIt
(	float* gfInImg,
	unsigned int iPadX,
	unsigned int iSizeY,
	float* gfOutImg
)
{	unsigned int i, j, k;
	i =  blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= iSizeY) return;
	else i = i * iPadX + blockIdx.x;
	//------------------------------
	float fVal = gfInImg[i];
	if(fVal > (float)-1e10)
	{	gfOutImg[i] = gfInImg[i];
		return;
	}
	//-------------
	unsigned int iSize = gridDim.x * iSizeY;
	unsigned int iFact = iSize / 13;
	unsigned int iNext = (i * iFact + 37) % iSize;
	j = (iNext / gridDim.x) * iPadX + (iNext % gridDim.x);
	fVal = gfInImg[j];
	if(fVal > (float)-1e10)
	{	gfOutImg[i] = fVal;
		return;
	}
	//-------------
	for(k=0; k<iSize; k++)
	{	iNext = (iNext * iFact + 37) % iSize;
		j = (iNext / gridDim.x) * iPadX + (iNext % gridDim.x);
		fVal = gfInImg[j];
		if(fVal > (float)-1e10)
		{	gfOutImg[i] = fVal;
			return;
		}
	}
	gfOutImg[i] = 0.0f;
}

GRandom2D::GRandom2D(void)
{
}

GRandom2D::~GRandom2D(void)
{
}

void GRandom2D::DoIt
(	float* gfInImg,
	float* gfOutImg,
	int* piImgSize,
	bool bPadded,
	cudaStream_t stream
) 
{	int iImgX = piImgSize[0];
	int iImgY = piImgSize[1];
	if(bPadded) iImgX = (piImgSize[0] / 2 - 1) * 2;
	//---------------------------------------------	
	dim3 aBlockDim(1, 64);
	dim3 aGridDim(iImgX, 1);
	aGridDim.y = (iImgY + aBlockDim.y - 1) / aBlockDim.y;
	mGDoIt<<<aGridDim, aBlockDim, 0, stream>>>(gfInImg, 
	   piImgSize[0], piImgSize[1], gfOutImg);
}

