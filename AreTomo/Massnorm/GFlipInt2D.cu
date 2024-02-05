#include "CMassNormInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::MassNorm;

static __global__ void mGFlipInt2D
(	float* gfImg,
	int iPadX,
	int iSizeY,
	float fOffset
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	//---------------------
	int i = y * iPadX + blockIdx.x;
	float fInt = gfImg[i];
	if(fInt < (float)-1e10) return;
	//-----------------------------
	gfImg[i] = fOffset - gfImg[i];
}


GFlipInt2D::GFlipInt2D(void)
{
}

GFlipInt2D::~GFlipInt2D(void)
{
}

void GFlipInt2D::DoIt
(	float* gfImg,
	int* piSize,
	bool bPadded,
	float fMin,
	float fMax,
	cudaStream_t stream
)
{	int iSizeX = bPadded ? (piSize[0] / 2 - 1) * 2 : piSize[0];
	int iSizeY = piSize[1];
	float fOffset = fMax * 2;
	//-----------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iSizeX, 1);
	aGridDim.y = (iSizeY + aBlockDim.y - 1) / aBlockDim.y;
	mGFlipInt2D<<<aGridDim, aBlockDim, 0, stream>>>
	( gfImg, piSize[0], iSizeY, fOffset );
}

