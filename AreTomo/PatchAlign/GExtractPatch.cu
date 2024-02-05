#include "CPatchAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::AreTomo::PatchAlign;

static __device__ __constant__ int giInSize[2];
static __device__ __constant__ int giOutSize[2];

static __global__ void mGExtract
(	float* gfInImg,
	int iInImgX,
	int iShiftX,
	int iShiftY,
	float* gfOutImg
)
{	int x = 0;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= giOutSize[1]) return;
	unsigned int i = y * giOutSize[0] + blockIdx.x;
	//---------------------------------------------
	int iOffsetX = iInImgX / 2 - gridDim.x / 2 - iShiftX;
	int iOffsetY = giInSize[1] / 2 - giOutSize[1] / 2 - iShiftY;
	x = blockIdx.x + iOffsetX;
	y = y + iOffsetY;
	//---------------
	if(x >= 0 && x < iInImgX && y >= 0 && y < giInSize[1]) 
	{	gfOutImg[i] = gfInImg[y * giInSize[0] + x];
		return;
	}
	//-------------
	unsigned int uiSize = gridDim.x * giOutSize[1];
	unsigned int uiFact = uiSize / 13;
	unsigned int uiNext = (i * uiFact + 37) % uiSize;
	for(int k=0; k<uiSize; k++)
	{	x = uiNext % gridDim.x + iOffsetX;
		y = uiNext / gridDim.x + iOffsetY;
		if(x >= 0 && x < iInImgX && y >= 0 && y < giInSize[1])
		{	gfOutImg[i] = gfInImg[y * giInSize[0] + x];
			return;
		}
		uiNext = (uiNext * uiFact + 37) % uiSize;
	}
	gfOutImg[i] = 0.0f;
}

GExtractPatch::GExtractPatch(void)
{
}

GExtractPatch::~GExtractPatch(void)
{
}

void GExtractPatch::SetSizes
(	int* piInSize,
	bool bInPadded, 
	int* piOutSize,
	bool bOutPadded
)
{	cudaMemcpyToSymbol(giInSize, piInSize, sizeof(giInSize));
	cudaMemcpyToSymbol(giOutSize, piOutSize, sizeof(giOutSize));
	//----------------------------------------------------------
	m_iInImgX = piInSize[0];
	if(bInPadded) m_iInImgX = (piInSize[0] / 2 - 1) * 2;
	//--------------------------------------------------
	memcpy(m_aiOutSize, piOutSize, sizeof(m_aiOutSize));
	m_iOutImgX = piOutSize[0];
	if(bOutPadded) m_iOutImgX = (piOutSize[0] / 2 - 1) * 2;
}

void GExtractPatch::DoIt
(	float* gfInImg,
	int* piShift,
	bool bRandomFill,
	float* gfOutImg
)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_iOutImgX, 1);
	aGridDim.y = (m_aiOutSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//-----------------
	mGExtract<<<aGridDim, aBlockDim>>>
	( gfInImg, m_iInImgX, piShift[0], piShift[1], gfOutImg 
	);
	if(!bRandomFill) return;
	//-----------------
	/*
	bool bPadded = (m_aiOutSize[0] == m_iOutImgX);
	GRandom2D aGRandom2D;
	aGRandom2D.DoIt(gfOutImg, gfOutImg, m_aiOutSize, bPadded); 
	*/
}

