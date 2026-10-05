#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::FindCtf; 

static __constant__ float c_afResRange[2];  // res/pix_size
static __constant__ float c_afIceRange[2];  // res/pix_size

//-----------------------------------------------------------------------------
// 1. Fitting a 1D Gaussian distribution of f(x) = exp(-B * fx^2) to the
//    absolute 1D radio averaged spectrum, |gfSpectrum(x)|.
// 2. fx is in [0, 0.5]. x is the spatial frequency in pixel.
// 2. The absolute is needed because gfSpectrum contains negative after
//    background subtraction.
//-----------------------------------------------------------------------------
static __global__ void mGEstimate
(	float* gfSpectrum,
	int iSize,
	float fBStep,
	float* gfCCs
)
{	extern __shared__ float s_gfCC[];
	float* s_gfMeanS = &s_gfCC[blockDim.x];
	float* s_gfMeanB = &s_gfMeanS[blockDim.x];
	float* s_gfStdS = &s_gfMeanB[blockDim.x];
	float* s_gfStdB = &s_gfStdS[blockDim.x];
	float* s_gfCount = &s_gfStdB[blockDim.x];
	float fBFactor = blockIdx.x * fBStep;
	//---------------------------
	int x = 0;
	float fCC = 0.0f, fCount = 0.0f;
	float fMeanS = 0.0f, fMeanB = 0.0f;
	float fStdS = 0.0f, fStdB = 0.0f;
	float fN = (iSize - 1.0f) * 2;
	//---------------------------
	for(x=threadIdx.x; x<iSize; x+=blockDim.x)
	{	float fX = x / fN;
		if(fX < c_afResRange[0]) continue;
		//if(fX > c_afResRange[1]) continue;
		if(fX >= c_afIceRange[0] && 
		   fX <= c_afIceRange[1]) continue;
		//-------------------
		float fS = fabsf(gfSpectrum[x]);// / iSize;
		float fB = fX; // - c_afResRange[0];
		fB = expf(-fBFactor * fB * fB);
		//-------------------
		fCC += (fS * fB);
		fMeanS += fS;
		fMeanB += fB; 
		fStdS += (fS * fS);
		fStdB += (fB * fB);
		fCount += 1.0f;
	}	
	s_gfCC[threadIdx.x] = fCC;
	s_gfMeanS[threadIdx.x] = fMeanS;
	s_gfMeanB[threadIdx.x] = fMeanB;
	s_gfStdS[threadIdx.x] = fStdS;
	s_gfStdB[threadIdx.x] = fStdB;
	s_gfCount[threadIdx.x] = fCount;
	__syncthreads();
	//--------------
	x = blockDim.x / 2;
	while(x > 0)
	{	if(threadIdx.x < x)
		{	int j = x + threadIdx.x;
			s_gfCC[threadIdx.x] += s_gfCC[j];
			s_gfMeanB[threadIdx.x] += s_gfMeanB[j];
			s_gfMeanS[threadIdx.x] += s_gfMeanS[j];
			s_gfStdB[threadIdx.x] += s_gfStdB[j];
			s_gfStdS[threadIdx.x] += s_gfStdS[j];
			s_gfCount[threadIdx.x] += s_gfCount[j];
		}
		__syncthreads();
		x /= 2;
	}
	//-------------
	if(threadIdx.x != 0) return;
	gfCCs[blockIdx.x] = 0.0f;
	if(s_gfCount[0] == 0) return;
	//---------------------------
	fMeanS = s_gfMeanS[0] / s_gfCount[0];
	fMeanB = s_gfMeanB[0] / s_gfCount[0];
	fCC = s_gfCC[0] / s_gfCount[0] - fMeanS * fMeanB;
	fStdS = s_gfStdS[0] / s_gfCount[0] - fMeanS * fMeanS;
	fStdB = s_gfStdB[0] / s_gfCount[0] - fMeanB * fMeanB;
	if(fStdS == 0 || fStdB == 0) return;
	//---------------------------
	fStdS = sqrtf(fStdS);
	fStdB = sqrtf(fStdB);
	gfCCs[blockIdx.x] = fCC / (fStdS * fStdB);
}

GEstBFactor1D::GEstBFactor1D(void)
{
	m_gfBuf = 0L;
	m_fBStep = 10.0f;
	m_iNumSteps = 0;
}

GEstBFactor1D::~GEstBFactor1D(void)
{
	mClean();
}

void GEstBFactor1D::Setup
(	float* pfResRange, // [low, high], ex: [30.0f, 4.0f]
	float fPixSize,    // angstrom
	float fBStep,      // step size for searching opt. B-factor
	int iNumSteps      // number of searching steps
)
{	float afResRange[2] = {0.0f};
	afResRange[0] = fPixSize / pfResRange[0];
	afResRange[1] = fPixSize / pfResRange[1];
	cudaMemcpyToSymbol(c_afResRange, afResRange, sizeof(float) * 2);
	//---------------------------
	float afIceRange[2] = {1.0f, 2.0f};
	cudaMemcpyToSymbol(c_afIceRange, afIceRange, sizeof(float) * 2);
	//---------------------------
	if(iNumSteps > m_iNumSteps) mClean();
	m_iNumSteps = iNumSteps;
	m_fBStep = fBStep;
	//---------------------------
	if(m_gfBuf != 0L) return;
	cudaMalloc(&m_gfBuf, m_iNumSteps * sizeof(float));
}

float GEstBFactor1D::DoIt(float* gfSpectrum, int iSize)
{    	
	dim3 aBlockDim(256, 1);
	dim3 aGridDim(m_iNumSteps, 1);
	//---------------------------
	size_t tBytes = 6 * sizeof(float) * aBlockDim.x;
	mGEstimate<<<aGridDim, aBlockDim, tBytes>>>(
	   gfSpectrum, iSize, 
	   m_fBStep, m_gfBuf);
	//---------------------------
	float* pfRes = new float[m_iNumSteps];
	tBytes = m_iNumSteps * sizeof(float);
	cudaMemcpy(pfRes, m_gfBuf, tBytes, cudaMemcpyDefault);
	//---------------------------
	float fBestCC = (float)-1e30;
	float fBestB = 0.0f;
	//---------------------------
	for(int i=0; i<m_iNumSteps; i++)
	{	if(fBestCC >= pfRes[i]) continue;
		fBestCC = pfRes[i];
		fBestB = i * m_fBStep;
	}
	if(pfRes != 0L) delete[] pfRes;
	//---------------------------
	return fBestB;
}

void GEstBFactor1D::mClean(void)
{
	if(m_gfBuf == 0L) return;
	cudaFree(m_gfBuf);
	m_gfBuf = 0L;
	m_iNumSteps = 0;
}

