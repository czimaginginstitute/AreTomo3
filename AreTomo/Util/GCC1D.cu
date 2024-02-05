#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::Util;

extern __shared__ char s_acArray[];

static __global__ void mGConv
(	cufftComplex* gComp1, 
	cufftComplex* gComp2,
	int iCmpSize,
	float fBFactor,
	float* gfCC,
	float* gfStd
)
{	float* sfCC = (float*)&s_acArray[0];
	float* sfStd = &sfCC[blockDim.x];
	sfCC[threadIdx.x] = 0.0f;
	sfStd[threadIdx.x] = 0.0f;
	//------------------------
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < iCmpSize && i != 0)
	{	float fFilt = i / (2.0f * (iCmpSize - 1));
		fFilt = expf(-2.0f * fBFactor * fFilt * fFilt);
		//---------------------------------------------
		float fA1, fA2;
		fA1 = gComp1[i].x * gComp1[i].x + gComp1[i].y * gComp1[i].y;
		fA2 = gComp2[i].x * gComp2[i].x + gComp2[i].y * gComp2[i].y;
		fA1 = sqrtf(fA1);
		fA2 = sqrtf(fA2);
		//---------------
		sfCC[threadIdx.x] = (gComp2[i].x * gComp1[i].x 
			+ gComp2[i].y * gComp1[i].y) * fFilt 
			/ (fA1 * fA2 + (float)1e-20);
		sfStd[threadIdx.x] = fFilt;
	}
	__syncthreads();
	//--------------
	i = blockDim.x >> 1;
	while(i > 0)
	{	if(threadIdx.x < i)
		{	sfCC[threadIdx.x] += sfCC[i + threadIdx.x];
			sfStd[threadIdx.x] += sfStd[i + threadIdx.x];
		}
		__syncthreads();
		i = i >> 1;
	}
	//-------------
	if(threadIdx.x != 0) return;
	gfCC[blockIdx.x] = sfCC[0];
	gfStd[blockIdx.x] = sfStd[0];
}

static __global__ void mGSum
(	float* gfCC,
	float* gfStd,
	float* gfCCSum,
	float* gfStdSum,
	int iSize
)
{	float* sfCCSum = (float*)&s_acArray[0];
	float* sfStdSum = (float*)&sfCCSum[blockDim.x];
	sfCCSum[threadIdx.x] = 0.0f;
	sfStdSum[threadIdx.x] = 0.0f;
	//---------------------------
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < iSize)
	{	sfCCSum[threadIdx.x] = gfCC[i];
		sfStdSum[threadIdx.x] = gfStd[i];
	}
	__syncthreads();
	//--------------
	i = blockDim.x >> 1;
	while(i > 0)
	{	if(threadIdx.x < i)
		{	sfCCSum[threadIdx.x] += sfCCSum[threadIdx.x + i];
			sfStdSum[threadIdx.x] += sfStdSum[threadIdx.x + i];
		}
		__syncthreads();
		i = i >> 1;
	}
	//-----------------
	if(threadIdx.x != 0) return;
	gfCCSum[blockIdx.x] = sfCCSum[0];
	gfStdSum[blockIdx.x] = sfStdSum[0];
}

GCC1D::GCC1D(void)
{
	m_fBFactor = 500.0f;
}

GCC1D::~GCC1D(void)
{
}

void GCC1D::SetBFactor(float fBFactor)
{
	m_fBFactor = fBFactor;
}

float GCC1D::DoIt
(	cufftComplex* gCmp1, 
	cufftComplex* gCmp2, 
	int iCmpSize
)
{	int iWarps = mCalcWarps(iCmpSize, 32); // power of 2 and <=16
	dim3 aBlockDim(iWarps * 32, 1);
	dim3 aGridDim(iCmpSize / aBlockDim.x + 1, 1);
	int iShmBytes = sizeof(float) * 2 * aBlockDim.x;
	//----------------------------------------------
	int iBlocks = aGridDim.x;
	size_t tBytes = sizeof(float) * iBlocks * 2;
	float *gfCC = 0L, *gfStd = 0L;
	cudaMalloc(&gfCC, tBytes);
	cudaMalloc(&gfStd, tBytes);
	//-------------------------
        mGConv<<<aGridDim, aBlockDim, iShmBytes>>>
	( gCmp1, gCmp2, iCmpSize, m_fBFactor, gfCC, gfStd
	);
        //-----------------------------------------------
	float* gfCCSum = gfCC + iBlocks;
	float* gfStdSum = gfStd + iBlocks;
	iWarps = mCalcWarps(iBlocks, 32);
	aBlockDim.x = iWarps * 32;
	aGridDim.x = iBlocks / aBlockDim.x + 1;
	iShmBytes = sizeof(float) * aBlockDim.x * 2;
	//------------------------------------------
	mGSum<<<aGridDim, aBlockDim, iShmBytes>>>
	( gfCC, gfStd, gfCCSum, gfStdSum, iBlocks
	);
	//---------------------------------------
	iBlocks = aGridDim.x;
	tBytes = sizeof(float) * iBlocks;
	float* pfCCSum = new float[iBlocks];
	float* pfStdSum = new float[iBlocks];
	cudaMemcpy(pfCCSum, gfCCSum, tBytes, cudaMemcpyDefault);
	cudaMemcpy(pfStdSum, gfStdSum, tBytes, cudaMemcpyDefault);
	cudaFree(gfCC);
	cudaFree(gfStd);
	//--------------
	m_fCCSum = 0.0f;
	m_fStdSum = 0.0f;
	for(int i=0; i<iBlocks; i++)
	{	m_fCCSum += pfCCSum[i];
		m_fStdSum += pfStdSum[i];
	}
	if(m_fStdSum > 0) m_fCC = m_fCCSum / m_fStdSum;
	else m_fCC = 0.0f;
	//----------------
	delete[] pfCCSum;
	delete[] pfStdSum;
	//mTestOnCPU(gCmp1, gCmp2, iCmpSize);
	return m_fCC;
}

int GCC1D::mCalcWarps(int iSize, int iWarpSize)
{
        float fWarps = iSize / (float)iWarpSize;
        if(fWarps < 1.5) return 1;
        //------------------------
        int iExp = (int)(logf(fWarps) / logf(2.0f) + 0.5f);
        int iWarps = 1 << iExp;
        if(iWarps > 16) iWarps = 16;
        return iWarps;
}

void GCC1D::mTestOnCPU
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	int iCmpSize
) 
{	cufftComplex* pCmp1 = new cufftComplex[iCmpSize];
	cufftComplex* pCmp2 = new cufftComplex[iCmpSize];
	size_t tBytes = iCmpSize * sizeof(cufftComplex);
	cudaMemcpy(pCmp1, gCmp1, tBytes, cudaMemcpyDefault);
	cudaMemcpy(pCmp2, gCmp2, tBytes, cudaMemcpyDefault);
	//--------------------------------------------------
	double dCCSum = 0, dStdSum = 0, dCC = 0;
	for(int i=1; i<iCmpSize; i++)
	{	float fX = i * 0.5f / (iCmpSize - 1.0f);
		double dFilt = exp(-2.0 * m_fBFactor * fX * fX);
		double a1 = pCmp1[i].x * pCmp1[i].x + pCmp1[i].y * pCmp1[i].y;
		double a2 = pCmp2[i].x * pCmp2[i].x + pCmp2[i].y * pCmp2[i].y;
		a1 = sqrt(a1);
		a2 = sqrt(a2);
		double cc = pCmp2[i].x * pCmp1[i].x + pCmp2[i].y * pCmp1[i].y;
		cc = cc * dFilt / (a1 * a2 + 1e-20);
		if(cc > 0)
		{	dCCSum += cc;
			dStdSum += dFilt;
		}
	}
	if(dStdSum > 0) dCC = dCCSum / dStdSum;
	printf("GCC1D: CPU: %.3e  %.3e  %.3e\n", dCCSum, dStdSum, dCC);
	printf("GCC1D: GPU: %.3e  %.3e  %.3e\n", m_fCCSum, m_fStdSum, m_fCC);
	delete[] pCmp1;
	delete[] pCmp2;
}

