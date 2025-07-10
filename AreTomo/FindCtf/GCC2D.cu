#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::FindCtf;

//-----------------------------------------------------------------------------
// 1. The zero-frequency component is at (x=0, y=iCmpY/2). The frequency
//    range in y direction is [-CmpY/2, CmpY/2).
// 2. fFreqLow, fFreqHigh are in the range of [0, 0.5f] of unit 1/pixel.
//-----------------------------------------------------------------------------
static __global__ void mGCalc2D
(	float* gfCTF2D, 
	float* gfSpectrum,
	int iCmpX,
	int iCmpY,
	float fFreqLow2,
	float fFreqHigh2,
	float fBFactor,
	float* gfRes
)
{	extern __shared__ float s_afSums[];
	//---------------------------
	int i = 0, iOffset = 0, x=0, y=0;
	float afSums[5] = {0.0f};
	float fX = 0.0f, fY = 0.0f;
	//---------------------------
	for(y=blockIdx.x; y<iCmpY; y+=gridDim.x)
	{	fY = (y - iCmpY * 0.5f) / iCmpY;
		iOffset = y * iCmpX;
		for(x=threadIdx.x; x<iCmpX; x+=blockDim.x)
		{	fX = (0.5f * x) / (iCmpX - 1);
			fX = fX * fX + fY * fY;
			if(fX <fFreqLow2 || fX > fFreqHigh2) continue;
			//--------------------------------------------
			i = iOffset + x;
			float fC = (fabsf(gfCTF2D[i]) - 0.5f) 
			   * expf(-fBFactor * fX);
			float fS = gfSpectrum[i];
			afSums[0] += (fC * fS);
			afSums[1] += fC;
			afSums[2] += fS;
			afSums[3] += (fC * fC);
			afSums[4] += (fS * fS);
		}
	}
	//---------------------------
	fX = 1.0f / (iCmpX * iCmpY);
	for(i=0; i<5; i++)
	{	iOffset = i * blockDim.x + threadIdx.x;
		s_afSums[iOffset] = afSums[i] * fX;
	}
	__syncthreads();
	//---------------------------
	iOffset = blockDim.x / 2;
	while(iOffset > 0)
	{	if(threadIdx.x < iOffset)
		{	for(i=0; i<5; i++)
			{	x = i * blockDim.x + threadIdx.x;
				s_afSums[x] += s_afSums[x+iOffset];
			}
		}
		__syncthreads();
		iOffset /= 2;
	}
	if(threadIdx.x != 0) return;
	//---------------------------
	x = blockIdx.x * 5;	
	for(i=0; i<5; i++)
	{	y = i * blockDim.x;
		gfRes[x+i] = s_afSums[i*blockDim.x];
	}
}

static __global__ void mGCalc1D(float* gfSum)
{
	extern __shared__ float s_afSums[];
	//---------------------------
	int i = 0, iOffset = 0;
	iOffset = threadIdx.x * 5;
	for(i=0; i<5; i++)
	{	s_afSums[i * blockDim.x + threadIdx.x] = gfSum[iOffset + i];
	}
	__syncthreads();
	//---------------------------
	iOffset = blockDim.x / 2;
	while(iOffset > 0)
	{	if(threadIdx.x < iOffset)
		{	for(i=0; i<5; i++)
			{	int j = i * blockDim.x + threadIdx.x;
				s_afSums[j] += s_afSums[j+iOffset];
			}
		}
		__syncthreads();
		iOffset /= 2;
	}
	if(threadIdx.x != 0) return;
	//---------------------------
	for(i=0; i<5; i++)
	{	gfSum[i] = s_afSums[blockDim.x * i];
	}
}

GCC2D::GCC2D(void)
{
	m_fBFactor = 1.0f;
	m_gfRes = 0L;
}

GCC2D::~GCC2D(void)
{
	if(m_gfRes != 0L) cudaFree(m_gfRes);
}

void GCC2D::Setup
(	float fFreqLow,  // [0, 0.5]
	float fFreqHigh, // [0, 0.5]
	float fBFactor
)
{	m_fFreqLow = fFreqLow;
	m_fFreqHigh = fFreqHigh;
	m_fBFactor = fBFactor;
}

void GCC2D::SetSize(int* piCmpSize)
{
	if(m_gfRes != 0L) cudaFree(m_gfRes);
	m_aiCmpSize[0] = piCmpSize[0];
	m_aiCmpSize[1] = piCmpSize[1];
	//----------------------------------
	int iSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	double dSize = sqrtf(iSize);
	if(dSize > 256) m_iBlockDimX = 256;
	else if(dSize > 128) m_iBlockDimX = 128;
	else m_iBlockDimX = 64;
	//---------------------------------------
	m_iGridDimX = (iSize + m_iBlockDimX - 1)/ m_iBlockDimX;
	if(m_iGridDimX > 512) m_iGridDimX = 512;
	else if(m_iGridDimX > 256) m_iGridDimX = 256;
	else if(m_iGridDimX > 128) m_iGridDimX = 128;
	else m_iGridDimX = 64;
	//-------------------------------------------
	cudaMalloc(&m_gfRes, 5 * m_iGridDimX * sizeof(float));
}

float GCC2D::DoIt
(	float* gfCTF, 
	float* gfSpectrum
)
{	dim3 aBlockDim(m_iBlockDimX, 1);
	dim3 aGridDim(m_iGridDimX, 1);
	size_t tSmBytes = sizeof(float) * aBlockDim.x * 5;
	//---------------------------
	float fFreqLow2 = m_fFreqLow / m_aiCmpSize[1];
	float fFreqHigh2 = m_fFreqHigh / m_aiCmpSize[1];
	if(fFreqHigh2 > 0.75f) fFreqHigh2 = 0.75f;
	fFreqLow2 *= fFreqLow2;
	fFreqHigh2 *= fFreqHigh2;
	//---------------------------
	mGCalc2D<<<aGridDim, aBlockDim, tSmBytes>>>(gfCTF, gfSpectrum, 
	   m_aiCmpSize[0], m_aiCmpSize[1], fFreqLow2, fFreqHigh2, 
	   m_fBFactor, m_gfRes);
        //---------------------------
	aBlockDim.x = aGridDim.x; aBlockDim.y = 1;
	aGridDim.x = 1; aGridDim.y = 1;
	tSmBytes = sizeof(float) * aBlockDim.x * 5;
	mGCalc1D<<<aGridDim, aBlockDim, tSmBytes>>>(m_gfRes);
	//---------------------------
	float afStats[5] = {0.0f};
	cudaMemcpy(afStats, m_gfRes, sizeof(afStats), cudaMemcpyDefault);
	float fCC   = afStats[0] - afStats[1] * afStats[2];
	float fStd1 = afStats[3] - afStats[1] * afStats[1];
	float fStd2 = afStats[4] - afStats[2] * afStats[2];
	if(fStd1 < 0) fStd1 = 0.0f;
	else fStd1 = (float)sqrt(fStd1);
	if(fStd2 < 0) fStd2 = 0.0f;
	else fStd2 = (float)sqrt(fStd2);
	if(fStd1 == 0 || fStd2 == 0) fCC = 0.0f;
	else fCC = fCC / (fStd1 * fStd2);
	return fCC;
}

