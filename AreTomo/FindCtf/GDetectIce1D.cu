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
static __global__ void mGCalcRingAvg
(	float* gfSpectrum,
	int iSize,
	float fRingMin,
	float fRingMax,
	float* gfRes
)
{	extern __shared__ float s_afSum[];
	float* s_afCount = &s_afSum[blockDim.x];
	//---------------------------
	float fSum = 0.0f;
	int iCount = 0;
	float fN = (iSize - 1.0f) * 2;
	//---------------------------
	for(int x=threadIdx.x; x<iSize; x+=blockDim.x)
	{	float fX = x / fN;
		if(fX < fRingMin || fX > fRingMax) continue;
		//-------------------
		fSum += gfSpectrum[x];
		iCount += 1;
	}
	s_afSum[threadIdx.x] = fSum;
	s_afCount[threadIdx.x] = (float)iCount;
	__syncthreads();
	//---------------------------
	for (int offset=blockDim.x/2; offset>0; offset=offset/2)
	{	if(threadIdx.x < offset)
		{	int x = threadIdx.x + offset;
			s_afSum[threadIdx.x] += s_afSum[x];
			s_afCount[threadIdx.x] += s_afCount[x];
		}
		__syncthreads();
	}
	//-------------
	if(threadIdx.x != 0) return;
	if(s_afCount[0] == 0) gfRes[0] = 0.0f;
	else gfRes[0] = s_afSum[0] / s_afCount[0];
}

GDetectIce1D::GDetectIce1D(void)
{
}

GDetectIce1D::~GDetectIce1D(void)
{
}

void GDetectIce1D::DoIt
(	float* gfSpectrum,
	int iSize,
 	float fPixSize // angstrom
)
{	float afIceRange1[] = {4.0f, 3.8f}; // center at 3.9A
	afIceRange1[0] = fPixSize / afIceRange1[0];
	afIceRange1[1] = fPixSize / afIceRange1[1];
	float fIceAmp1 = mCalcAmp(gfSpectrum, iSize, afIceRange1);
	//---------------------------
	float afIceRange2[] = {3.6f, 3.4f}; // center at 3.5A
	afIceRange2[0] = fPixSize / afIceRange2[0];
	afIceRange2[1] = fPixSize / afIceRange2[1];
	float fIceAmp2 = mCalcAmp(gfSpectrum, iSize, afIceRange2);
	//---------------------------
	float afRefRange[] = {4.8f, 4.4f};  // center at 4.6A
	afRefRange[0] = fPixSize / afRefRange[0];
	afRefRange[1] = fPixSize / afRefRange[1];
	float fRefAmp = mCalcAmp(gfSpectrum, iSize, afRefRange);
	//---------------------------
	float fEps = (float)1e-20;
	float fIce1 = (fIceAmp1 - fRefAmp) / (fRefAmp + fEps);
	float fIce2 = (fIceAmp2 - fRefAmp) / (fRefAmp + fEps);
	//---------------------------
	m_bIceRing1 = (fIce1 > 0.80f);
	m_bIceRing2 = (fIce2 > 0.50f);
	
	printf("Ice rings: \n"
	   "  fIceAmp1   fIceAmp2   fRefAmp\n"
	   "  %.4e       %.4e       %.4e\n",
	   fIceAmp1, fIceAmp2, fRefAmp);
	printf("Ice detection: \n"
	   "  ring 1     ring 2\n"
	   "  %d         %d\n", m_bIceRing1, m_bIceRing2);
	
}

float GDetectIce1D::mCalcAmp(float* gfSpect, int iSize, float* pfRingRange)
{
	float* gfAvg = 0L;
	cudaMalloc(&gfAvg, sizeof(float));
	//---------------------------
	dim3 aBlockDim(256, 1);
	dim3 aGridDim(1, 1);
	int iSmBytes = sizeof(float) * aBlockDim.x * 2;
	mGCalcRingAvg<<<aGridDim, aBlockDim, iSmBytes>>>(
	   gfSpect, iSize,
	   pfRingRange[0], pfRingRange[1],
	   gfAvg);
	//---------------------------
	float fAmp = 0.0f;
	cudaMemcpy(&fAmp, gfAvg, sizeof(float), cudaMemcpyDefault);
	if(gfAvg != 0L) cudaFree(&gfAvg);
	fAmp = (float)fabs(fAmp);
	return fAmp;
}
