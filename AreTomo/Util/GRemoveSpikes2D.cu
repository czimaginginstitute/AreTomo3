#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Util;

static __global__ void mGMediumFilter
(	float* gfInImg,
	int iPadX,
	int iSizeY,
	int iWinSize,
	float* gfOutImg
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	//---------------------
	int i = y * iPadX + blockIdx.x;
	gfOutImg[i] = gfInImg[i];
	//-----------------------
	int iStartX = blockIdx.x - iWinSize / 2;
	if(iStartX < 0) iStartX = 0;
	//--------------------------
	int iStartY = y - iWinSize / 2;
	if(iStartY < 0) iStartY = 0;
	//--------------------------
	if((iStartX + iWinSize) >= gridDim.x) iStartX = gridDim.x - iWinSize;
	if((iStartY + iWinSize) >= iSizeY) iStartY = iSizeY - iWinSize;
	//-------------------------------------------------------------
	int ix, iy;
	float fMax = (float)-1e30, fMean = 0.0f, fVal = 0.0f;
	for(iy=0; iy<iWinSize; iy++)
	{	float* gfVal = gfInImg + (iy+iStartY) * iPadX + iStartX;
		for(ix=0; ix<iWinSize; ix++)
		{	fVal = gfVal[ix];
			fMean += fVal;
			if(fMax < fVal) fMax = fVal;
		}
	}
	int iSize = iWinSize * iWinSize;
	fMean = fMean / iSize;
	//--------------------
	fVal = gfInImg[i];
	if(fVal < fMax) return;
	if((fVal - fMean) < fmaxf(fMean, 5.0f)) return;
	//---------------------------------------------
	int iNext = (i * 7 + 17) % iSize;
	for(int j=0; j<iSize; j++)
	{	ix = iNext % iWinSize + iStartX;
		iy = iNext / iWinSize + iStartY;
		fVal = gfInImg[iy * iPadX + ix];
		if(fVal < fMax)
		{	gfOutImg[i] = fVal;
			return;
		}
		else iNext = (iNext * 7 + 17) % iSize;
	}
}

GRemoveSpikes2D::GRemoveSpikes2D(void)
{
}

GRemoveSpikes2D::~GRemoveSpikes2D(void)
{
}

void GRemoveSpikes2D::DoIt
(	float* gfInImg,
	int* piImgSize,
	bool bPadded,
	int iWinSize,
	float* gfOutImg,
	cudaStream_t stream
)
{	int iSizeX = piImgSize[0];
	if(bPadded) iSizeX = (piImgSize[0] / 2 - 1) * 2;
	//----------------------------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iSizeX, piImgSize[1] / aBlockDim.y + 1);
	mGMediumFilter<<<aGridDim, aBlockDim, 0, stream>>>
	( gfInImg, piImgSize[0], piImgSize[1], iWinSize, gfOutImg
	);
}

