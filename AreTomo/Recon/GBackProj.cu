#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::Recon;

// iProjX, iPadProjX, iAllProjs, iVolX
static __device__ __constant__ int giSize[4]; 
//-------------------------------------------
static __global__ void mGBackProj
(	float* gfPadSinogram,
	float* gfCosSin,
	int iStartProj,
	int iNumProjs,
	bool bSart,
	float fRelax,
	float* gfVolXZ
)
{	int iX = blockIdx.x * blockDim.x + threadIdx.x;
	if(iX >= giSize[3]) return;
	//-------------------------
	float fX = iX + 0.5f - giSize[3] * 0.5f;
	float fZ = blockIdx.y + 0.5f - gridDim.y * 0.5f;
	float fProjCentX = giSize[0] / 2.0f;
	int iProjEndX = giSize[0] - 2.0f;
	//-------------------------------
        float fInt = 0.0f;
	int i, iProj = 0, iCount = 0;
	for(i=0; i<iNumProjs; i++)
	{	iProj = i + iStartProj;	
		float fXp = fX * gfCosSin[2 * iProj] 
			+ fZ * gfCosSin[2 * iProj +1] 
			+ fProjCentX;
		if(fXp < 0 || fXp > iProjEndX) continue;
		//--------------------------------------
		fXp = gfPadSinogram[iProj * giSize[1] + (int)fXp];
		if(fXp <= (float)-1e10) continue;
		//-------------------------------
		fInt += fXp;
		iCount += 1;
        }
	if(iCount <= 0) return;
	//---------------------
	i = blockIdx.y * giSize[3] + iX;
	fInt = fRelax * fInt / iCount + gfVolXZ[i];
	//-----------------------------------------
	if(bSart) gfVolXZ[i] = fmaxf(fInt, 0.0f);
	else gfVolXZ[i] = fInt;
}

GBackProj::GBackProj(void)
{
}

GBackProj::~GBackProj(void)
{
}

void GBackProj::SetSize(int* piPadProjSize, int* piVolSize)
{
	int iProjX = (piPadProjSize[0] / 2 - 1) * 2;
	int aiSize[] = {iProjX, piPadProjSize[0], 
	   piPadProjSize[1], piVolSize[0]};
	cudaMemcpyToSymbol(giSize, aiSize, sizeof(giSize));
	//-------------------------------------------------
	m_aBlockDim.x = 512;
	m_aBlockDim.y = 1;
	m_aGridDim.x = (piVolSize[0] + m_aBlockDim.x - 1) / m_aBlockDim.x;
	m_aGridDim.y = piVolSize[1];
} 

void GBackProj::DoIt
(	float* gfPadSinogram,
	float* gfCosSin,  // cosine and sine of all tilt angles
	int iStartProj,
	int iNumProjs,
	bool bSart,
	float fRelax,
	float* gfVolXZ,
	cudaStream_t stream
)
{	mGBackProj<<<m_aGridDim, m_aBlockDim, 0, stream>>>
	( gfPadSinogram, gfCosSin, iStartProj, iNumProjs, 
	  bSart, fRelax, gfVolXZ
	);
}
