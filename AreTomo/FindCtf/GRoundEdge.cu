#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

static __global__ void mGRoundEdge(float* gfImg, int iSizeX,
	float* gfMaskCent, float* gfMaskSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= iSizeX) return;
	int i = blockIdx.y * iSizeX + x;
	//------------------------------
	float fX = 2 * fabsf(x - gfMaskCent[0]) / gfMaskSize[0];
	float fY = 2 * fabsf(blockIdx.y - gfMaskCent[1]) / gfMaskSize[1];
	float fR = sqrtf(fX * fX + fY * fY);
	if(fR >= 1.0f)
	{	gfImg[i] = 0.0f;
		return;
	}
	//-------------
	fR = 0.5f * (1 - cosf(3.1415926f * fR));
	fR = 1.0f - powf(fR, 3.0f);
	gfImg[i] = gfImg[i] * fR;
}

static __global__ void mGKeepCenter(float* gfImg, int iSizeX,
        float* gfMaskCent, float fMaskR)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= iSizeX) return;
	int i = blockIdx.y * iSizeX + x;
	//-----------------
	float fX = x - gfMaskCent[0]; 
	float fY = blockIdx.y - gfMaskCent[1];
	float fR = sqrtf(fX * fX + fY * fY) / fMaskR - 1.0f;
	if(fR <= 0.0f) return;
	//-----------------
	fR = 0.5f * (1 - cosf(3.1415926f * fR));
	fR = 1.0f - powf(fR, 2.0f);
	gfImg[i] = gfImg[i] * fR;
}

GRoundEdge::GRoundEdge(void)
{
	memset(m_afMaskCent, 0, sizeof(m_afMaskCent));
	memset(m_afMaskSize, 0, sizeof(m_afMaskSize));
}

GRoundEdge::~GRoundEdge(void)
{
}

void GRoundEdge::SetMask(float* pfCent, float* pfSize)
{
	m_afMaskCent[0] = pfCent[0];
	m_afMaskCent[1] = pfCent[1];
	m_afMaskSize[0] = pfSize[0];
	m_afMaskSize[1] = pfSize[1];
}

void GRoundEdge::DoIt(float* gfImg, int* piImgSize, bool bKeepCenter)
{
	if(m_afMaskCent[0] == 0 || m_afMaskCent[1] == 0)
	{	m_afMaskCent[0] = 0.5f * piImgSize[0];
		m_afMaskCent[1] = 0.5f * piImgSize[1];
	}
	if(m_afMaskSize[0] == 0 || m_afMaskSize[1] == 0)
	{	m_afMaskSize[0] = 1.0f * piImgSize[0];
		m_afMaskSize[1] = 1.0f * piImgSize[1];
	}
	//-----------------
	float* gfMaskCent = 0L;
	size_t tBytes = sizeof(float) * 2;
	cudaMalloc(&gfMaskCent, tBytes);
	cudaMemcpy(gfMaskCent, m_afMaskCent, tBytes, cudaMemcpyDefault);
	//-----------------
	float* gfMaskSize = 0L;
	cudaMalloc(&gfMaskSize, tBytes);
	cudaMemcpy(gfMaskSize, m_afMaskSize, tBytes, cudaMemcpyDefault);
	//-----------------
	dim3 aBlockDim(512, 1);
	int iGridX = piImgSize[0] / aBlockDim.x + 1;
	dim3 aGridDim(iGridX, piImgSize[1]);
	//-----------------
	if(!bKeepCenter)
	{	mGRoundEdge<<<aGridDim, aBlockDim>>>(gfImg, 
		   piImgSize[0], gfMaskCent, gfMaskSize);
	}
	else
	{	float fMaskR = (float)sqrtf(m_afMaskSize[0] *
		   m_afMaskSize[0] + m_afMaskSize[1] *
	   	   m_afMaskSize[1]) * 0.5f;
		mGKeepCenter<<<aGridDim, aBlockDim>>>(gfImg,
		   piImgSize[0], gfMaskCent, fMaskR);	
	}
	//-----------------
	if(gfMaskCent != 0L) cudaFree(gfMaskCent);
	if(gfMaskSize != 0L) cudaFree(gfMaskSize);
}
