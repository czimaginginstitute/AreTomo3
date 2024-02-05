#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::Recon;

static __global__ void mGDiffProj
(	float* gfRawProjs,
	float* gfForProjs,
	float* gfDifProjs, // at a given y location
	int iProjSizeX,    // proj size, can be padded
	int iNumProjs
)
{	int iProj = blockIdx.y * blockDim.y + threadIdx.y;
	if(iProj >= iNumProjs) return;
	//----------------------------
	int i = iProj * iProjSizeX + blockIdx.x;
	if(gfForProjs[i] < (float)-1e10) gfDifProjs[i] = (float)-1e30;
	else if(gfRawProjs[i] < (float)-1e10) gfDifProjs[i] = (float)-1e30;
	else gfDifProjs[i] = gfRawProjs[i] - gfForProjs[i];
}

GDiffProj::GDiffProj(void)
{
}

GDiffProj::~GDiffProj(void)
{
}

void GDiffProj::DoIt // projections are y-slice
(	float* gfRawProjs,
	float* gfForProjs,
	float* gfDiffProjs,
	int* piProjSize,
	bool bPadded,
	cudaStream_t stream
)
{	int iProjX = piProjSize[0];
	if(bPadded) iProjX = (piProjSize[0] / 2 - 1) * 2;
	//-----------------------------------------------
	dim3 aBlockDim(1, 32);
	dim3 aGridDim(iProjX, 1);
	aGridDim.y = (piProjSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//-----------------------------------------------------------
	mGDiffProj<<<aGridDim, aBlockDim, 0, stream>>>
	( gfRawProjs, gfForProjs, gfDiffProjs,
	  piProjSize[0], piProjSize[1]
	);
}

