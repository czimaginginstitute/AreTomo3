#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::Recon;

static __global__ void mGWeightMeaProjs
(	float* gfCosSin,
	int iProjSizeX, // padded size if gfProjs is padded 
	int iNumProjs,
	int iVolZ,
	float* gfProjs
)
{	int iProj = blockIdx.y * blockDim.y + threadIdx.y;
	if(iProj >= iNumProjs) return;
	//----------------------------
	int i = iProj * iProjSizeX + blockIdx.x;
	gfProjs[i] *= (gfCosSin[iProj * 2] / iVolZ);
} 
	

GWeightProjs::GWeightProjs(void)
{
}

GWeightProjs::~GWeightProjs(void)
{
}

void GWeightProjs::DoIt
(	float* gfProjs,
	float* gfCosSin,
	int* piProjSize,
	bool bPadded,
	int iVolZ,
	cudaStream_t stream
)
{	int iProjX = piProjSize[0];
	if(bPadded) iProjX = (piProjSize[0] / 2 - 1) * 2;
	//-----------------------------------------------
	dim3 aBlockDim(1, 32);
	dim3 aGridDim(iProjX, 1);
	aGridDim.y = (piProjSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//-----------------------------------------------------------
	mGWeightMeaProjs<<<aGridDim, aBlockDim, 0, stream>>>
	( gfCosSin, piProjSize[0], piProjSize[1], iVolZ, gfProjs
	);
}

