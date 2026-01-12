#include "CProjAlignInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>

#define s_fD2R 0.01745f

using namespace McAreTomo::AreTomo::ProjAlign;

static __device__ __constant__ int giProjSize[2];
static __device__ __constant__ int giVolSize[2];

static __device__ float mDIntProj(float* gfProj, float fX)
{
	int x = (int)fX;
	float fVal = gfProj[x];
	if(fVal < (float)-1e10) return (float)-1e30;
	else return fVal;
}

static __global__ void mGBackProj
(	float* gfSinogram,
	float* gfTiltAngles,
	float fProjAngle,
	int iStartIdx,
	int iEndIdx,
	float* gfVol // xz slice
)
{	int iX = blockIdx.x * blockDim.x + threadIdx.x;
	if(iX >= giVolSize[0]) return;
	float fX = iX +0.5f - giVolSize[0] * 0.5f;
	float fZ = blockIdx.y + 0.5f - giVolSize[1] * 0.5f;
	//-------------------------------------------------
	float fInt = 0.0f;
	float fCount = 0.0f;
	float fCentX = giProjSize[0] * 0.5f;
	int iEnd = giProjSize[0] - 1;
	//-----------------------------------
	for(int i=iStartIdx; i<=iEndIdx; i++)
	{	float fW = cosf((fProjAngle - gfTiltAngles[i]) * s_fD2R);
		float fV = gfTiltAngles[i] * s_fD2R;
		float fCos = cosf(fV);
		float fSin = sinf(fV); 
		//--------------------
		fV = fX * fCos + fZ * fSin + fCentX;
		if(fV < 0 || fV > iEnd) continue;
		//-------------------------------
		float* gfProj = gfSinogram + i * giProjSize[0];
		fV = mDIntProj(gfProj, fV);
		if(fV < (float)-1e20) continue;
		//------------------------------------------------
		// fV * fCos: distribute the projection intensity
		// along the back-projection ray.
		//------------------------------------------------
		fInt += (fV * fCos * fW);
		fCount += fW;
	}
	int i = blockIdx.y * giVolSize[0] + iX;
	if(fCount < 0.001f) gfVol[i] = (float)-1e30;
	else gfVol[i] = fInt / fCount;
}

extern __shared__ char s_cArray[];

static __global__ void mGForProj
(	float* gfVol,
	int iRayLength,
	float fCos,
	float fSin,
	float* gfReproj
)
{	float* sfSum = (float*)&s_cArray[0];
	int*  siCount = (int*)&sfSum[blockDim.y];
	sfSum[threadIdx.y] = 0.0f;
	siCount[threadIdx.y] = 0;
	__syncthreads();
	//--------------
	float fXp = blockIdx.x + 0.5f - gridDim.x * 0.5f;
	float fTempX = fXp * fCos + giVolSize[0] * 0.5f;
	float fTempZ = fXp * fSin + giVolSize[1] * 0.5f;
	float fZStartp = -fXp * fSin / fCos - 0.5f * iRayLength;
	//------------------------------------------------------
	int i = 0;
	int iEndX = giVolSize[0] - 1;
	int iEndZ = giVolSize[1] - 1;
	float fX = 0.0f, fZ = 0.0f, fV = 0.0f;
	int iSegments = iRayLength / blockDim.y + 1;
	for(i=0; i<iSegments; i++)
	{	fZ = i * blockDim.y + threadIdx.y;
		if(fZ >= iRayLength) continue;
		//----------------------------
		fZ = fZ + fZStartp;
		fX = fTempX - fZ * fSin;
		fZ = fTempZ + fZ * fCos;
		//----------------------
		if(fX >= 0 && fX < iEndX && fZ >= 0 && fZ < iEndZ)
		{	fV = gfVol[giVolSize[0] * (int)(fZ) + (int)(fX)];
			if(fV >= (float)-1e10)
			{	sfSum[threadIdx.y] += fV;
				siCount[threadIdx.y] += 1;
			}
		}
	}
	__syncthreads();
	//--------------
	i = blockDim.y / 2;
	while(i > 0)
	{	if(threadIdx.y < i)
		{	sfSum[threadIdx.y] += sfSum[threadIdx.y+i];
			siCount[threadIdx.y] += siCount[threadIdx.y+i];
		}
		__syncthreads();
		i /= 2;
	}
	//-------------
	if(threadIdx.y != 0) return;
	if(siCount[0] < 0.8f * iRayLength) 
	{	gfReproj[blockIdx.x] = (float)-1e30;
	}
	else 
	{	gfReproj[blockIdx.x] = sfSum[0] / siCount[0];
	}
}

GReproj::GReproj(void)
{
	m_gfVol = 0L;
	m_gfReproj = 0L;
}

GReproj::~GReproj(void)
{
	this->Clean();
}

void GReproj::Clean(void)
{
	if(m_gfVol != 0L) cudaFree(m_gfVol);
	if(m_gfReproj != 0L) cudaFree(m_gfReproj);
	m_gfVol = 0L;
	m_gfReproj = 0L;
}

void GReproj::SetSizes
(	int iProjX,
	int iNumProjs,
	int iVolX,
	int iVolZ
)
{	this->Clean();
	//------------
	m_aiProjSize[0] = iProjX;
	m_aiProjSize[1] = iNumProjs;
	m_aiVolSize[0] = iVolX;
	m_aiVolSize[1] = iVolZ;
	//---------------------
	int iBytes = sizeof(int) * 2;
	cudaMemcpyToSymbol(giProjSize, m_aiProjSize, iBytes);
	cudaMemcpyToSymbol(giVolSize, m_aiVolSize, iBytes);
	//-------------------------------------------------
	iBytes = m_aiProjSize[0] * sizeof(float);
	cudaMalloc(&m_gfReproj, iBytes);
	//------------------------------
	iBytes = m_aiVolSize[0] * m_aiVolSize[1] * sizeof(float);
	cudaMalloc(&m_gfVol, iBytes);
}

void GReproj::DoIt
(	float* gfSinogram,
	float* gfTiltAngles,
	int* piProjRange,
	float fProjAngle,
	cudaStream_t stream
)
{	m_gfTiltAngles = gfTiltAngles;
	m_gfSinogram = gfSinogram;
	m_stream = stream;
	//----------------
	mBackProj(fProjAngle, piProjRange);
	mForwardProj(fProjAngle);
}

void GReproj::mBackProj(float fProjAngle, int* piProjRange)
{
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, m_aiVolSize[1]);
	aGridDim.x = m_aiVolSize[0] / aBlockDim.x + 1;
	//--------------------------------------------
	mGBackProj<<<aGridDim, aBlockDim, 0, m_stream>>>
	( m_gfSinogram, m_gfTiltAngles, fProjAngle, 
	  piProjRange[0], piProjRange[1], m_gfVol
	);
}

void GReproj::mForwardProj(float fProjAngle)
{
	double dRad = 4.0 * atan(1.0) / 180.0;
	float fCos = (float)cos(dRad * fProjAngle);
	float fSin = (float)sin(dRad * fProjAngle);
	int iRayLength = (int)(m_aiVolSize[1] / fCos + 0.5f);
        //---------------------------------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_aiProjSize[0], 1);
	int iShmBytes = (sizeof(float) + sizeof(int)) * aBlockDim.y;
	//----------------------------------------------------------
	mGForProj<<<aGridDim, aBlockDim, iShmBytes, m_stream>>>
	( m_gfVol, iRayLength, fCos, fSin, m_gfReproj
	);
}

