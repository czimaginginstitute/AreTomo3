#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

//-----------------------------------------------------------------------------
// 1. The DC is assumed at (0, iCmpY/2), see GCalcSpectrum.cu
//-----------------------------------------------------------------------------
static __global__ void mGRemove
(	float* gfInSpect,
	float* gfOutSpect,
	int iCmpY,
	float fMinFreq
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	int iHalfX = gridDim.x - 1;
	int iHalfY = iCmpY / 2;
	//-------------------------
	y -= iHalfY;
	float fR = blockIdx.x * 0.5f / (gridDim.x - 1);
	fR = sqrtf(fR * fR + y * y / (float)(iCmpY * iCmpY));
	if(fR < 0.04f)
	{	gfOutSpect[i] = 0.0f;
		return;
	}
	//----------------------------
	int iBoxSize = fMinFreq * iCmpY;
	if(iBoxSize > (0.3f * iCmpY)) iBoxSize = (int)(0.3f * iCmpY);
	iBoxSize = iBoxSize / 2 * 2 + 1;
	if(iBoxSize < 7) iBoxSize = 7;
	//-------------------------------------------------------
	// (iX, iY): origin at lower left corner
	// (xx, yy): origin at iX = 0, iY = iHalfY
	// (xx = 0, yy=0) and (iX = 0, iY=iHalfY) is DC component
	//-------------------------------------------------------
	int iX = 0, iY = 0;
	int iHalfBox = iBoxSize / 2;
	float fBackground = 0.0f;
	for(int k=-iHalfBox; k<=iHalfBox; k++)
	{	int yy = k + y;
		for(int j=-iHalfBox; j<=iHalfBox; j++)
		{	int xx = j + blockIdx.x;
			if(xx >= iHalfX) xx = xx - 2 * iHalfX;
			if(xx >= 0)
			{	iX = xx;
				iY = iHalfY + yy;
			}
			else
			{	iX = -xx;
				iY = iHalfY - yy;
			}
			if(iY < 0) iY += iCmpY;
			else if(iY >= iCmpY) iY -= iCmpY;
			//-------------------------------
			iX = iY * gridDim.x + iX;
			fBackground += sqrtf(gfInSpect[iX]);
          	}
	}
	fBackground /= (iBoxSize * iBoxSize);
	fBackground = (fBackground * fBackground);
	gfOutSpect[i] = gfInSpect[i] - fBackground;
}

GRmBackground2D::GRmBackground2D(void)
{
}

GRmBackground2D::~GRmBackground2D(void)
{
}

void GRmBackground2D::DoIt
(	float* gfInSpect,
	float* gfOutSpect,	
	int* piCmpSize,
	float fMinFreq
)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piCmpSize[0], 1);
	aGridDim.y = piCmpSize[1] / aBlockDim.y;
	if((piCmpSize[1] % aBlockDim.y) > 0) aGridDim.y++;
	//------------------------------------------------
	mGRemove<<<aGridDim, aBlockDim>>>(gfInSpect, gfOutSpect,
	   piCmpSize[1], fMinFreq);
}

