#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Util;

static __global__ void mGMutualMask2D
(	float* gfImg1,
	float* gfImg2,
	int iPadX,
	int iSizeY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * iPadX + blockIdx.x;
	if(gfImg1[i] < (float)-1e10) gfImg2[i] = (float)-1e30;
	else if(gfImg2[i] < (float)-1e10) gfImg1[i] = (float)-1e30; 
}

GMutualMask2D::GMutualMask2D(void)
{
}

GMutualMask2D::~GMutualMask2D(void)
{
}

void GMutualMask2D::DoIt
(	float* gfImg1,
	float* gfImg2,
	int* piSize,
	bool bPadded,
	cudaStream_t stream
)
{	int iSizeX = bPadded ? (piSize[0] / 2 - 1) * 2 : piSize[0];
	int iSizeY = piSize[1];
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iSizeX, 1);
	aGridDim.y = (iSizeY + aBlockDim.y - 1) / aBlockDim.y;
	mGMutualMask2D<<<aGridDim, aBlockDim, 0, stream>>>(gfImg1, 
	   gfImg2, piSize[0], iSizeY);
}

