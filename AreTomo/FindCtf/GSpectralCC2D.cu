#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::FindCtf;

//--------------------------------------------------------------------
// 1. The zero-frequency component is at (x=0, y=iSpectY/2). 
// 2. The frequency range in y direction is [-CmpY/2, CmpY/2).
//--------------------------------------------------------------------
static __global__ void mGCalc2D
(	float* gfCTF2D, 
	float* gfSpect,
	int iSpectX,
	int iSpectY,
	int iWidth,
	float* gfCC
)
{	extern __shared__ float s_afShared[];
	float* s_afSumStd1 = &s_afShared[blockDim.y];
	float* s_afSumStd2 = &s_afShared[blockDim.y * 2];
	float* s_afCount = &s_afShared[blockDim.y * 3];	
	//-----------------
	int iLow = blockIdx.x - iWidth / 2;
	int iHigh = iLow + iWidth;
	if(iLow < 0)
	{	iLow = 0;
		iHigh = iWidth;
	}
	else if(iHigh > iSpectX) 
	{	iLow = iSpectX - iWidth;
		iHigh = iSpectX; 
	}
	//-----------------
	float fSumCC = 0.0f; 
	float fSumStd1 = 0.0f;
	float fSumStd2 = 0.0f;
	float fCount = 0.0f;
	//-----------------
	for(int y=threadIdx.y; y<iSpectY; y+=blockDim.y)
	{	float fY = fabsf(y - iSpectY * 0.5f);
		if(fY >= iHigh) continue;
		//----------------
		for(int x=0; x<iSpectX; x++)
		{	if(x >= iHigh) continue;
			//---------------
			float fR = sqrtf(x * x + fY * fY);
			if(fR < iLow || fR >= iHigh) continue;
			//---------------
			int i = y * iSpectX + x;
			float fC = gfCTF2D[i];
			float fS = gfSpect[i];
			fSumCC += (fC * fS);
			fSumStd1 += (fC * fC);
			fSumStd2 += (fS * fS);
			fCount += 1.0f;
		}
	}
	s_afShared[threadIdx.y] = fSumCC;
	s_afSumStd1[threadIdx.y] = fSumStd1;
	s_afSumStd2[threadIdx.y] = fSumStd2;
	s_afCount[threadIdx.y] = fCount;
	__syncthreads();
	//-----------------
	int iOffset = blockDim.y / 2;
	while(iOffset > 0)
	{	if(threadIdx.y < iOffset)
		{	int i = iOffset + threadIdx.y;
			s_afShared[threadIdx.y] += s_afShared[i];
			s_afSumStd1[threadIdx.y] += s_afSumStd1[i];
			s_afSumStd2[threadIdx.y] += s_afSumStd2[i];
			s_afCount[threadIdx.y] += s_afCount[i];
		}
		__syncthreads();
		iOffset /= 2;
	}
	if(threadIdx.y != 0) return;
	//-----------------
	if(s_afCount[0] == 0)
	{	gfCC[blockIdx.x] = 0.0f;
	}
	else
	{	gfCC[blockIdx.x] = s_afShared[0] / 
		   sqrtf(s_afSumStd1[0] * s_afSumStd2[0]);
	}
}

GSpectralCC2D::GSpectralCC2D(void)
{
	m_pfCC = 0L;
}

GSpectralCC2D::~GSpectralCC2D(void)
{
	if(m_pfCC != 0L) cudaFreeHost(m_pfCC);
}

//--------------------------------------------------------------------
// 1. piSpectSize is the size of half spectrum.
//--------------------------------------------------------------------
void GSpectralCC2D::SetSize(int* piSpectSize)
{
	m_aiSpectSize[0] = piSpectSize[0];
	m_aiSpectSize[1] = piSpectSize[1];
	//-----------------
	if(m_pfCC != 0L) cudaFree(m_pfCC);
	cudaMallocHost(&m_pfCC, m_aiSpectSize[0] * sizeof(float));
}

int GSpectralCC2D::DoIt
(	float* gfCTF, 
	float* gfSpect
)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_aiSpectSize[0], 1);
	size_t tSmBytes = sizeof(float) * aBlockDim.y * 4;
	//-----------------
	mGCalc2D<<<aGridDim, aBlockDim, tSmBytes>>>(gfCTF, gfSpect, 
	   m_aiSpectSize[0], m_aiSpectSize[1], 3, m_pfCC);
        //-----------------
	int iMin = 0;
	float fMin = 1000.0f;
	for(int i=0; i<m_aiSpectSize[0]; i++)
	{	float fDif = fabsf(m_pfCC[i] - 0.143f);
		if(fDif >= fMin) continue;
		fMin = fDif;
		iMin = i;
	}
	return iMin;	
}

