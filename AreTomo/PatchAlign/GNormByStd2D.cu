#include "CPatchAlignInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::PatchAlign;

static __global__ void mGNormByStd2D
(	float* gfImg, 
	int iSizeX,
	int iPadX,
	int iWinX,
	int iWinY
)
{	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if(ix >= iSizeX) return;
	int iPixel = blockIdx.y * iPadX + ix;
	//-----------------------------------
	int iOffsetX = ix - iWinX / 2;
	int iOffsetY = blockIdx.y - iWinY / 2;
	//------------------------------------------------
	float fMean = 0.0f, fVar = 0.0f, fVal = 0.0f;
	int iCount = 0; 
	for(int j=0; j<iWinY; j++)
	{	int y = j + iOffsetY;
		if(y < 0) continue;
		else if(y >= gridDim.y) continue;
		for(int i=0; i<iWinX; i++)
		{	int x = i + iOffsetX;
			if(x < 0) continue;
			else if(x >= iSizeX) continue;
			fVal = gfImg[y * iPadX + x];
			if(fVal < (float)-1e10) continue;
			fMean += fVal;
			fVar += (fVal * fVal);
			iCount += 1;
		}
	}
	if(iCount == 0) 
	{	gfImg[iPixel] = (float)-1e30;
		return;
	}
	//-------------
	fMean /= iCount;
	fVar = fVar / iCount - fMean * fMean;
	if(fVar <= 0) gfImg[iPixel] = (float)-1e30;
	else gfImg[iPixel] /= fVar;
}

GNormByStd2D::GNormByStd2D(void)
{
}

GNormByStd2D::~GNormByStd2D(void)
{
}

void GNormByStd2D::DoIt(float* gfImg, int* piImgSize, bool bPadded, 
	int* piWinSize, cudaStream_t stream)
{
	if(gfImg == 0L) return;
	//---------------------
	int iImageX = bPadded ? (piImgSize[0] / 2 - 1) * 2 : piImgSize[0];
	dim3 aBlockDim(512, 1);
	dim3 aGridDim( (iImageX  + aBlockDim.x - 1)/ aBlockDim.x, 
	   piImgSize[1] );
	mGNormByStd2D<<<aGridDim, aBlockDim, 0, stream>>>(gfImg, iImageX,
	   piImgSize[0], piWinSize[0], piWinSize[1]);
}
