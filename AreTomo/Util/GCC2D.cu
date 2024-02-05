#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::Util;

extern __shared__ char s_acArray[];

static __global__ void mGConv
(	cufftComplex* gComp1, 
	cufftComplex* gComp2,
	int iCmpSizeY,
	float fBFactor,
	float* gfCC,
	float* gfStd
)
{	float* sfCC = (float*)&s_acArray[0];
	float* sfStd = (float*)&sfCC[blockDim.y];
	sfCC[threadIdx.y] = 0.0f;
	sfStd[threadIdx.y] = 0.0f;
	//------------------------
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i = y * gridDim.x + blockIdx.x;
	if(y < iCmpSizeY && i != 0)
	{	float fY = y * 1.0f / iCmpSizeY;
		if(fY > 0.5f) fY -= 1.0f;
		float fX = blockIdx.x * 0.5f /(gridDim.x - 1);
		float fFilt = expf(-2.0f * fBFactor * (fX * fX + fY * fY));
		//---------------------------------------------------------
		fX = gComp1[i].x * gComp1[i].x + gComp1[i].y * gComp1[i].y;
		fY = gComp2[i].x * gComp2[i].x + gComp2[i].y * gComp2[i].y;
		fX = sqrtf(fX); // Amplitude 1
		fY = sqrtf(fY); // Amplitude 2
		//----------------------------
		fX = (gComp2[i].x * gComp1[i].x + gComp2[i].y * gComp1[i].y)
			/ (fX * fY + (float)1e-20); // fCC
		if(fX > 0)
		{	sfCC[threadIdx.y] = fX * fFilt;
			sfStd[threadIdx.y] = fFilt;
		}
	}
	__syncthreads();
	//--------------
	i = blockDim.y >> 1;
	while(i > 0)
	{	if(threadIdx.y < i)
		{	sfCC[threadIdx.y] += sfCC[threadIdx.y + i];
			sfStd[threadIdx.y] += sfStd[threadIdx.y + i];
		}
		__syncthreads();
		i = i >> 1;
	}
	//-----------------
	if(threadIdx.y != 0) return;
	i = blockIdx.y * gridDim.x + blockIdx.x;
	gfCC[i] = sfCC[0];
	gfStd[i] = sfStd[0];
}

static __global__ void mGSum1D
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

GCC2D::GCC2D(void)
{
	m_fBFactor = 300.0f;
}

GCC2D::~GCC2D(void)
{
}

void GCC2D::SetBFactor(float fBFactor)
{
	m_fBFactor = fBFactor;
}

float GCC2D::DoIt
(	cufftComplex* gCmp1, 
	cufftComplex* gCmp2, 
	int* piCmpSize
)
{	int iWarps = mCalcWarps(piCmpSize[1], 32); // power of 2
	dim3 aBlockDim(1, iWarps * 32);
        dim3 aGridDim(piCmpSize[0], piCmpSize[1] / aBlockDim.y + 1);
	int iShmBytes = sizeof(float) * 2 * aBlockDim.y;
	//----------------------------------------------
	int iBlocks = aGridDim.x * aGridDim.y;
	float *gfCC = 0L, *gfStd = 0L;
	cudaMalloc(&gfCC, sizeof(float) * iBlocks * 2);
	cudaMalloc(&gfStd, sizeof(float) * iBlocks * 2);
	//----------------------------------------------
        mGConv<<<aGridDim, aBlockDim, iShmBytes>>>
	( gCmp1, gCmp2, piCmpSize[1],
	  m_fBFactor, gfCC, gfStd
	);
        //-----------------------
	float* gfCCRes = gfCC + iBlocks;
	float* gfStdRes = gfStd + iBlocks;
	bool bSwap = false;
	while(iBlocks > 0)
	{	if(bSwap)
		{	iBlocks = mDo1D
			( gfCCRes, gfStdRes, gfCC, gfStd, iBlocks
			);
		}
		else
		{	iBlocks = mDo1D
			( gfCC, gfStd, gfCCRes, gfStdRes, iBlocks
			);
		}
		if(iBlocks <= 1) break;
		bSwap = !bSwap;
	}
	cudaFree(gfCC);
	cudaFree(gfStd);
	//mTest(gCmp1, gCmp2, piCmpSize);
	return m_fCC;
}

int GCC2D::mDo1D
(	float* gfCC,
	float* gfStd,
	float* gfCCRes,
	float* gfStdRes,
	int iSize
)
{	int iWarps = mCalcWarps(iSize, 32);
	dim3 aBlockDim(iWarps * 32, 1);
	dim3 aGridDim(iSize / aBlockDim.x + 1, 1);
	int iShmBytes = sizeof(float) * aBlockDim.x * 2;
	//----------------------------------------------
	mGSum1D<<<aGridDim, aBlockDim, iShmBytes>>>
	( gfCC, gfStd, gfCCRes, gfStdRes, iSize
	);
	if(aGridDim.x >= 32) return aGridDim.x;
	//-------------------------------------
	float* pfCCRes = new float[aGridDim.x];
	float* pfStdRes = new float[aGridDim.x];
	size_t tBytes = sizeof(float) * aGridDim.x;
	cudaMemcpy(pfCCRes, gfCCRes, tBytes, cudaMemcpyDefault);
	cudaMemcpy(pfStdRes, gfStdRes, tBytes, cudaMemcpyDefault);
	double dCC = 0, dStd = 0;
	for(int i=0; i<aGridDim.x; i++)
	{	dCC += pfCCRes[i];
		dStd += pfStdRes[i];
	}
	m_fCCSum = (float)dCC;
	m_fStdSum = (float)dStd;
	if(m_fStdSum > 0) m_fCC = m_fCCSum / m_fStdSum;
	else m_fCC = 0.0f;
	//----------------
	delete[] pfCCRes;
	delete[] pfStdRes;
	return 0;
}

int GCC2D::mCalcWarps(int iSize, int iWarpSize)
{
	float fWarps = iSize / (float)iWarpSize;
	if(fWarps < 1.5) return 1;
	//------------------------
	int iExp = (int)(logf(fWarps) / logf(2.0f) + 0.5f);
	int iWarps = 1 << iExp;
	if(iWarps > 16) iWarps = 16;
	return iWarps;
}

void GCC2D::mTest
(	cufftComplex* gCmp1, 
	cufftComplex* gCmp2, 
	int* piCmpSize
)
{	int iCmpSize = piCmpSize[0] * piCmpSize[1];
	size_t tBytes = iCmpSize * sizeof(cufftComplex);
	cufftComplex* pCmp1 = new cufftComplex[iCmpSize];
	cufftComplex* pCmp2 = new cufftComplex[iCmpSize];
	cudaMemcpy(pCmp1, gCmp1, tBytes, cudaMemcpyDefault);
	cudaMemcpy(pCmp2, gCmp2, tBytes, cudaMemcpyDefault);
	//--------------------------------------------------
	double dCCSum = 0;
	double dStdSum = 0;
	for(int i=1; i<iCmpSize; i++)
	{	float fX = i % piCmpSize[0];
		float fY = i / piCmpSize[0];
		fX = fX * 0.5f / (piCmpSize[0] - 1);
		fY = fY / piCmpSize[1];
		if(fY > 0.5f) fY -= 1.0f;
		//-----------------------
		float fFilt = fX * fX + fY * fY;
		fFilt = (float)exp(-2.0f * m_fBFactor * fFilt);
		fX = pCmp1[i].x * pCmp1[i].x + pCmp1[i].y * pCmp1[i].y;
		fY = pCmp2[i].x * pCmp2[i].x + pCmp2[i].y * pCmp2[i].y;
		fX = (float)sqrt(fX); // amplitude 1
		fY = (float)sqrt(fY); // amplitude 2
		//----------------------------------
		fX = (pCmp1[i].x * pCmp2[i].x + pCmp1[i].y * pCmp2[i].y)
			/ (fX * fY + (float)1e-20); // CC
		if(fX > 0)
		{	dCCSum += (fX * fFilt);
			dStdSum += fFilt;
		}
	}
	delete[] pCmp1;
	delete[] pCmp2;
	//-------------
	float fCCSum = (float)dCCSum;
	float fStdSum = (float)dStdSum;
	float fCC = (fStdSum > 0) ? fCCSum / fStdSum : 0.0f;
	printf("GCC2Ds: GPU: %.3e  %.3e  %.3e\n",
		m_fCCSum, m_fStdSum, m_fCC);
	printf("GCC2Ds: CPU: %.3e  %.3e  %.3e\n",
		fCCSum, fStdSum, fCC);
		
}
