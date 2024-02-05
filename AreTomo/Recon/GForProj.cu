#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::Recon;

static __device__ __constant__ int giVolSize[3];  
// giVolSize: iVolX, iVolXPadded, iVolZ
//----------------------------------------------
static __global__ void mGForProjs
(	float* gfVol,
	float* gfCosSin,
	int iProjSizeX, // paddded if gfForProjs is padded
	int iNumProjs,
	float* gfForProjs // At a given y location
)
{	int iProj = blockIdx.y * blockDim.y + threadIdx.y;
	if(iProj >= iNumProjs) return;
	//----------------------------
	int i = 2 * iProj;
	float fCos = gfCosSin[i];
	float fSin = gfCosSin[i + 1];
	int iRayLength = (int)(giVolSize[2] / fCos + 1.5f); 
	//-------------------------------------------------
	float fXp = blockIdx.x + 0.5f - 0.5f * gridDim.x;
	float fTempX = fXp * fCos + giVolSize[0] * 0.5f;
	float fTempZ = fXp * fSin + giVolSize[2] * 0.5f;
	float fZStartp = -fXp * fSin / fCos - 0.5f * iRayLength;
	//------------------------------------------------------
	float fInt = 0.0f;
	int iCount = 0;
	int iEndX = giVolSize[0] - 1;
	int iEndZ = giVolSize[2] - 1;
	for(i=0; i<iRayLength; i++)
	{	float fZ = i + fZStartp;
		float fX = fTempX - fZ * fSin;
		fZ = fTempZ + fZ * fCos;
		//----------------------
		if(fX < 0 || fZ < 0 || fX > iEndX || fZ > iEndZ) continue;
		//-------------------------------------------
		fZ = gfVol[giVolSize[1] * (int)fZ + (int)fX];
		if(fZ < (float)-1e10) continue;
		//-----------------------------
		fInt += fZ;
		iCount += 1;
	}
	i = iProj * iProjSizeX + blockIdx.x;
	if(iCount == 0) gfForProjs[i] = (float)-1e30;
	else gfForProjs[i] = fInt / iCount;
}

GForProj::GForProj(void)
{
}

GForProj::~GForProj(void)
{
}

void GForProj::SetVolSize(int iVolX, bool bPadded, int iVolZ)
{
	m_aiVolSize[0] = iVolX;
	m_aiVolSize[1] = iVolX;
	m_aiVolSize[2] = iVolZ;
	if(bPadded) m_aiVolSize[0] = (iVolX / 2 - 1) * 2;
	cudaMemcpyToSymbol(giVolSize, m_aiVolSize, sizeof(int) * 3);
}

void GForProj::DoIt       // project to specified tilt angles
(	float* gfVol,
	float* gfCosSin,  
	int* piProjSize,
	bool bPadded,
	float* gfForProjs,
	cudaStream_t stream 
)
{	int iProjX = piProjSize[0];
	if(bPadded) iProjX = (iProjX / 2 - 1) * 2;
	//----------------------------------------
	dim3 aBlockDim(1, 32);
	dim3 aGridDim(iProjX, 1);
	aGridDim.y = (piProjSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	mGForProjs<<<aGridDim, aBlockDim, 0, stream>>>
	( gfVol, gfCosSin, piProjSize[0], piProjSize[1], gfForProjs
	);
}

