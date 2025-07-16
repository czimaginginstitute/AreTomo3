#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZEX 512

using namespace McAreTomo::AreTomo::FindCtf;

//-----------------------------------------------------------------------------
// 1. The zero-frequency component is at (x=0, y=iCmpY/2). The frequency
//    range in y direction is [-CmpY/2, CmpY/2).
// 2. fFreqLow, fFreqHigh are in the range of [0, 0.5f] of unit 1/pixel.
//-----------------------------------------------------------------------------
static __global__ void mGCalculate
(	float* gfCTF,
	float* gfSpectrum,
	int iSize,
	float fFreqLow,
	float fFreqHigh,
	float fBFactor,
	float* gfRes
)
{	extern __shared__ float s_gfSums[];
	int i = 0, iOffset = 0, x=0;
	//---------------------------
	for(i=0; i<6; i++)
	{	iOffset = i * blockDim.x;
		s_gfSums[iOffset + threadIdx.x] = 0.0f;
	}
	__syncthreads();
	//---------------------------
	x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= iSize) return;
	//---------------------------
	if(x >= fFreqLow && x < fFreqHigh)
	{	float fCTF = x * 0.5f / (iSize - 1);
		float fSpec = gfSpectrum[x]; 
		fCTF = (fabsf(gfCTF[x]) - 0.5f) * expf(-fBFactor * fCTF * fCTF);
		//--------------------------
		s_gfSums[threadIdx.x] = fCTF * fSpec;
		s_gfSums[blockDim.x + threadIdx.x] = fCTF;
		s_gfSums[2 * blockDim.x + threadIdx.x] = fSpec;
		s_gfSums[3 * blockDim.x + threadIdx.x] = fCTF * fCTF;
		s_gfSums[4 * blockDim.x + threadIdx.x] = fSpec * fSpec;
		s_gfSums[5 * blockDim.x + threadIdx.x] = 1.0f;
	}
	__syncthreads();
	//---------------------------
	iOffset = blockDim.x / 2;
	while(iOffset > 0)
	{	if(threadIdx.x < iOffset)
		{	for(i=0; i<6; i++)
			{	int k = i * blockDim.x + threadIdx.x;
				s_gfSums[k] += s_gfSums[k + iOffset];
			}
		}
		__syncthreads();
		iOffset /= 2;
	}
	//---------------------------
	if(threadIdx.x == 0)
	{	iOffset = 6 * blockIdx.x;
		for(i=0; i<6; i++)
		{	gfRes[iOffset + i] = s_gfSums[i * blockDim.x];
		}
	}
}

GCC1D::GCC1D(void)
{
	m_fBFactor = 1.0f;
	m_gfRes = 0L;
}

GCC1D::~GCC1D(void)
{
	if(m_gfRes != 0L) cudaFree(m_gfRes);
}

void GCC1D::Setup
(	float fFreqLow,  // pixel in Fourier domain
	float fFreqHigh, // pixel in Fourier domain
	float fBFactor
)
{	m_fFreqLow = fFreqLow;
	m_fFreqHigh = fFreqHigh;
	m_fBFactor = fBFactor;
}

void GCC1D::SetSize(int iSize)
{
	if(m_gfRes != 0L) cudaFree(m_gfRes);
	//----------------------------------
	m_iSize = iSize;
	cudaMalloc(&m_gfRes, sizeof(float) * m_iSize);
} 

float GCC1D::DoIt(float* gfCTF, float* gfSpectrum)
{    	
	dim3 aBlockDim(256, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = (m_iSize + aBlockDim.x - 1) / aBlockDim.x;
	//-----------------------------------------------------
	size_t tBytes = sizeof(float) * m_iSize;
	cudaMemset(m_gfRes, 0, tBytes);
	//----------------------------
	tBytes = sizeof(float) * aBlockDim.x * 6;
	mGCalculate<<<aGridDim, aBlockDim, tBytes>>>(gfCTF, gfSpectrum, 
	   m_iSize, m_fFreqLow, m_fFreqHigh, m_fBFactor, m_gfRes);
     	//-----------------------------------------------
	float* pfRes = new float[aGridDim.x * 6];
	tBytes = sizeof(float) * aGridDim.x * 6;
	cudaMemcpy(pfRes, m_gfRes, tBytes, cudaMemcpyDefault);
	//----------------------------------------------------
	double adVals[6] = {0.0};
	for(int i=0; i<aGridDim.x; i++)
	{	int j = 6 * i;
		for(int k=0; k<6; k++)
		{	adVals[k] += pfRes[j + k];
		}
	}
	for(int i=0; i<5; i++)
	{	adVals[i] /= (adVals[5] + 1e-30);
	}
	if(pfRes != 0L) delete[] pfRes;
	//-----------------------------
	double dCC = adVals[0] - adVals[1] * adVals[2];
	double dStd1 = adVals[3] - adVals[1] * adVals[1];
	double dStd2 = adVals[4] - adVals[2] * adVals[2];
	if(dStd1 > 0) dStd1 = sqrt(dStd1);
	if(dStd2 > 0) dStd2 = sqrt(dStd2);
	if(dStd1 > 0 && dStd2 > 0) dCC /= (dStd1 * dStd2);
	else dCC = 0.0;
	return (float)dCC;
}

float GCC1D::DoCPU
(	float* gfCTF,
	float* gfSpectrum,
	int iSize
)
{	float* pfCTF = new float[iSize];
	float* pfSpectrum = new float[iSize];
	size_t tBytes = sizeof(float) * iSize;
	cudaMemcpy(pfCTF, gfCTF, tBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(pfSpectrum, gfSpectrum, tBytes, cudaMemcpyDeviceToHost);
	//-----------------------------------------------------------------
	float fFreqLow = 2 * m_fFreqLow * iSize;
	float fFreqHigh = 2 * m_fFreqHigh * iSize;
	double dXY = 0, dStd1 = 0, dStd2 = 0;
	for(int i=0; i<iSize; i++)
	{	if(i < fFreqLow || i >= fFreqHigh) continue;
		float fX = i * 0.5f / (iSize - 1);
		dXY += (pfCTF[i] * pfSpectrum[i]) * exp(-m_fBFactor * fX * fX);
		dStd1 += (pfCTF[i] * pfCTF[i]);
		dStd2 += (pfSpectrum[i] * pfSpectrum[i]);
	}
	double dCC = dXY / sqrt(dStd1 * dStd2);
	delete[] pfCTF;
	delete[] pfSpectrum;
	return (float)dCC;
}
