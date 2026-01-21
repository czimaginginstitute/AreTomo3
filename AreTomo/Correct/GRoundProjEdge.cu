#include "CCorrectInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Correct;

static __global__ void mGRoundEdge
(	float* gfProj, 
	int iImgX,
	int iPadX,
	float fMaskSizeX
)
{	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= iImgX) return;
	//---------------------------
	int i = blockIdx.y * iPadX + x;
	if(gfProj[i] < (float)-1e10)
	{	gfProj[i] = 0.0f;
		return;
	}
	//---------------------------
	float fCentX = iImgX * 0.5f;
	float fX = 2.0f * fabsf(blockIdx.x - fCentX) / fMaskSizeX - 1.0f;
	if(fX <= 0.0f) return;
	//---------------------------
	float fEdge = (iImgX - fMaskSizeX) * 0.5f;
	fX = (fabsf(blockIdx.x - fCentX) - 0.5f * fMaskSizeX) / fEdge;
	//---------------------------
	fX = 0.5f + 0.5f * cosf(3.1415926f * fX);
	gfProj[i] = gfProj[i] * fX;
}

GRoundProjEdge::GRoundProjEdge(void)
{
}

GRoundProjEdge::~GRoundProjEdge(void)
{
}

void GRoundProjEdge::DoIt
(	float* gfProj,
	int* piSize, 
	bool bPadded,
	float fTiltAngle,
	cudaStream_t stream
)
{	int iImgX = bPadded ? (piSize[0]/2 - 1) * 2 : piSize[0];
	//---------------------------
	float fMaskX = (float)cos(fTiltAngle * 3.1416f) * iImgX;
	fMaskX = fMaskX * 0.9f;
	//---------------------------
	dim3 aBlockDim(512, 1);
	int iGridX = (iImgX + aBlockDim.x - 1) / aBlockDim.x;
	dim3 aGridDim(iGridX, piSize[1]);
	//-----------------
	mGRoundEdge<<<aGridDim, aBlockDim, 0, stream>>>(gfProj,
	   iImgX, piSize[1], fMaskX);
}

