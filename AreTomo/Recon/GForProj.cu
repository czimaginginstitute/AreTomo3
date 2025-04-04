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
	int iProjSizeX,   // padded if gfForProjs is padded.
	float* gfForProjs // At a given y location
)
{	extern __shared__ float s_pRayInt[];
	float* s_pCount = &s_pRayInt[blockDim.y];
	//-----------------------------------------------------
	// Note: gridDim.x = unpadded projX, 
	//       gridDim.y = iNumProjs
	//       iProjSizeX = padded projX if gfForProjs is
	//           padded.
	//-----------------------------------------------------
	int i = 2 * blockIdx.y;
	float fCos = gfCosSin[i];
	float fSin = gfCosSin[i + 1];
	int iRayLength = (int)(giVolSize[2] / fCos + 1.5f); 
	//---------------------------
	float fXp = blockIdx.x + 0.5f - 0.5f * gridDim.x;
	float fTempX = fXp * fCos + giVolSize[0] * 0.5f;
	float fTempZ = fXp * fSin + giVolSize[2] * 0.5f;
	//---------------------------
	int iEndX = giVolSize[0] - 2;
	int iEndZ = giVolSize[2] - 2;
	//---------------------------
	float fSum = 0.0f, fCount = 0.0f;
	for(i=threadIdx.y; i<iRayLength; i+=blockDim.y)
	{	float fZ = i - iRayLength * 0.5f;
		float fX = fTempX - fZ * fSin;
		fZ = fTempZ + fZ * fCos;
		//--------------------------
		if(fX < 0 || fZ < 0 || fX > iEndX || fZ > iEndZ) continue;
		//--------------------------
		int iX = (int)fX;
		int iZ = (int)fZ;
		fX -= iX;
		fZ -= iZ;
		//--------------------------
		int j = iZ * giVolSize[1] + iX;
		fXp = gfVol[j] * (1 - fX) * (1 - fZ) 
		   + gfVol[j+1] * fX * (1 - fZ)
		   + gfVol[j+giVolSize[1]] * (1 - fX) * fZ
		   + gfVol[j+giVolSize[1]+1] * fX * fZ;
		if(fXp < (float)-1e10) continue;
		//--------------------------
		fSum += fXp;
		fCount += 1.0f;
	}
	s_pRayInt[threadIdx.y] = fSum;
	s_pCount[threadIdx.y] = fCount;
	__syncthreads();
	//---------------------------
	for(i=blockDim.y/2; i>0; i=i/2)
	{	if(threadIdx.y < i)
		{	int j = threadIdx.y + i;
			s_pRayInt[threadIdx.y] += s_pRayInt[j];
			s_pCount[threadIdx.y] += s_pCount[j];
		}
		__syncthreads();
	}
	//---------------------------
	if(threadIdx.y != 0) return;
	i = blockIdx.y * iProjSizeX + blockIdx.x;
	if(s_pCount[0] <= 0) gfForProjs[i] = (float)-1e30;
	else gfForProjs[i] = s_pRayInt[0] / s_pCount[0];
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
	//---------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iProjX, piProjSize[1]);
	//---------------------------
	int iShmBytes = sizeof(float) * aBlockDim.y * 2;
	mGForProjs<<<aGridDim, aBlockDim, iShmBytes, stream>>>(gfVol, 
	   gfCosSin, piProjSize[0], gfForProjs);
}

