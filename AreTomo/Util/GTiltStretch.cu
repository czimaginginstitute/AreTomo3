#include "CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Util;

static __device__ void mGCalcXY(float afXY[2], float* gfMatrix)
{
	float fX = afXY[0] * gfMatrix[0] + afXY[1] * gfMatrix[1];
        afXY[1] = afXY[0] * gfMatrix[1] + afXY[1] * gfMatrix[2];
	afXY[0] = fX;
}

static __device__ float mGBilinear
(	float afXY[2], 
	int iPadX,
	int iSizeY,
	float* gfImg
)
{	int iX = (int)afXY[0];
        int iY = (int)afXY[1];
        int i = iY * iPadX + iX;
	//----------------------
	afXY[0] -= iX;
	afXY[1] -= iY;
	float f2 = 1.0f - afXY[0];
	float f3 = 1.0f - afXY[1];
	f2 = gfImg[i] * f2 * f3 + gfImg[i+1] * afXY[0] * f3
		+ gfImg[i+iPadX] * f2 * afXY[1]
		+ gfImg[i+iPadX+1] * afXY[0] * afXY[1];
	return f2;
}

static __device__ float mGRandom
(	float afXY[2], 
	int iPadX, 
	int iSizeY, 
	float* gfImg
)
{	int x = (int)fabsf(afXY[0]);
	int y = (int)fabsf(afXY[1]);
	if(x >= gridDim.x) x = 2 * gridDim.x - x;
	if(y >= iSizeY) y = 2 * iSizeY - y;
	//---------------------------------
	int iWin = 31;
	int iWinPixels = iWin * iWin;
	unsigned int next = y * iPadX + x;
	for(int k=0; k<iWinPixels; k++)
	{	next = (next * 7) % iWinPixels;
		int iX = next % iWin - iWin / 2 + x;
		if(iX < 0 || iX >= gridDim.x) continue;
		int iY = next / iWin - iWin / 2 + y;
		if(iY < 0 || iY >= iSizeY) continue;
		return gfImg[iY * iPadX + iX];
	}
	//------------------------------------
	return gfImg[iSizeY / 2 * iPadX + gridDim.x / 2];		
}

static __global__ void mGTiltStretch
(	float* gfInFrm,
	int iPadX,  // if not padded, iPadX is equal to gridDim.x
	int iSizeY,
	float* gfMatrix,
	float* gfOutFrm
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * iPadX + blockIdx.x;
	gfOutFrm[i] = (float)-1e30;
	//-------------------------
	float afXY[4] = {0.0f};
	afXY[0] = blockIdx.x - 0.5f * gridDim.x + 0.5f;
	afXY[1] = y - 0.5 * iSizeY + 0.5f;
	mGCalcXY(afXY, gfMatrix);
	afXY[0] += (0.5f * gridDim.x - 0.5f);
	afXY[1] += (0.5f * iSizeY - 0.5f); 
	//--------------------------------
	if(afXY[0] < 0 || afXY[0] > (gridDim.x - 2) 
	|| afXY[1] < 0 || afXY[1] > (iSizeY - 2)) return;
	//-----------------------------------------------
	gfOutFrm[i] = mGBilinear(afXY, iPadX, iSizeY, gfInFrm);
}

static __global__ void mGTiltStretchRandom
(	float* gfInFrm,
	int iPadX,  // if not padded, iPadX is equal to gridDim.x
	int iSizeY,
	float* gfMatrix,
	float* gfOutFrm
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iSizeY) return;
	int i = y * iPadX + blockIdx.x;
	//-----------------------------
	float afXY[2] = {0.0f};
	afXY[0] = blockIdx.x - 0.5f * gridDim.x + 0.5f;
	afXY[1] = y - 0.5 * iSizeY + 0.5f;
	mGCalcXY(afXY, gfMatrix);
	afXY[0] += (0.5f * gridDim.x - 0.5f);
	afXY[1] += (0.5f * iSizeY - 0.5f);
	//--------------------------------
	if(afXY[0] < 0 || afXY[0] >= (gridDim.x - 1)
	|| afXY[1] < 0 || afXY[1] >= (iSizeY - 1))
	{	gfOutFrm[i] = mGRandom(afXY, iPadX, iSizeY, gfInFrm);
		return;
	}
	//-------------
	gfOutFrm[i] = mGBilinear(afXY, iPadX, iSizeY, gfInFrm);
}

GTiltStretch::GTiltStretch(void)
{
	m_gfMatrix = 0L;
}

GTiltStretch::~GTiltStretch(void)
{
	this->Clean();
}

void GTiltStretch::Clean(void)
{
	if(m_gfMatrix != 0L) cudaFree(m_gfMatrix);
	m_gfMatrix = 0L;
}

void GTiltStretch::CalcMatrix(float fStretch, float fTiltAxis)
{
	double d2T = 2 * fTiltAxis * 3.1416 / 180;
	double dSin2T = sin(d2T);
	double dCos2T = cos(d2T);
	double dP = 0.5 * (fStretch + 1);
	double dM = 0.5 * (fStretch - 1);
	//-------------------------------
	float a0 = (float)(dP + dM * dCos2T);
	float a1 = (float)(1 * dM * dSin2T);
	float a2 = (float)(dP - dM * dCos2T);
	float fDet = a0 * a2 - a1 * a1;
	//-----------------------------
	m_afMatrix[0] = a2 / fDet;
	m_afMatrix[1] = -a1 / fDet;
	m_afMatrix[2] = a0 / fDet;
}

void GTiltStretch::DoIt
(	float* gfInImg,   // input image
	int* piSize,
	bool bPadded,	
	float fStretch,
	float fTiltAxis,
	float* gfOutImg,   // output image
	bool bRandomFill,
	cudaStream_t stream 
)
{	this->CalcMatrix(fStretch, fTiltAxis);
	int iBytes = sizeof(float) * 3;
	if(m_gfMatrix == 0L) cudaMalloc(&m_gfMatrix, iBytes);
	//---------------------------------------------------
	this->CalcMatrix(fStretch, fTiltAxis);
	cudaMemcpy(m_gfMatrix, m_afMatrix, iBytes, cudaMemcpyDefault);
	//------------------------------------------------------------
	int iSizeX = bPadded ? (piSize[0] / 2 - 1) * 2 : piSize[0];
	int iSizeY = piSize[1];
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iSizeX, 1);
	aGridDim.y = (iSizeY + aBlockDim.y - 1) / aBlockDim.y;
	//----------------------------------------------------
	if(bRandomFill)
	{	mGTiltStretchRandom<<<aGridDim, aBlockDim, 0, stream>>>
		(gfInImg, piSize[0], iSizeY,  m_gfMatrix, gfOutImg);
	}
	else
	{	mGTiltStretch<<<aGridDim, aBlockDim, 0, stream>>>
		(gfInImg, piSize[0], iSizeY, m_gfMatrix, gfOutImg);
	}
}

void GTiltStretch::Unstretch(float* pfInShift, float* pfOutShift)
{
	float fX = pfInShift[0];
	float fY = pfInShift[1];
	pfOutShift[0] = fX * m_afMatrix[0] + fY * m_afMatrix[1]; 
	pfOutShift[1] = fX * m_afMatrix[1] + fY * m_afMatrix[2];
}

