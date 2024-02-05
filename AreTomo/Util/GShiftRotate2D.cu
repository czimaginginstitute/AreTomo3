#include "CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::Util;

static __device__ __constant__ int giInSize[2];
static __device__ __constant__ int giOutSize[2];
static __device__ __constant__ float gfCosSin[2];
static __device__ __constant__ float gfShift[2];

//-------------------------------------------------------------------
// 1. Shift is done before rotation.
// 2. Note that it is the image is shifted, not the coordinate.
//-------------------------------------------------------------------
static __device__ void mGCalcInXY(int y, int iInImgX, float afXY[2])
{
	afXY[0] = blockIdx.x - (gridDim.x - 1.0f) * 0.5f;
	afXY[1] = y - (giOutSize[1] - 1.0f) * 0.5f;
	//-----------------------------------------
        float fT = afXY[0] * gfCosSin[0] + afXY[1] * gfCosSin[1];
        afXY[1] = -afXY[0] * gfCosSin[1] + afXY[1] * gfCosSin[0];
	//-------------------------------------------------------
        afXY[0] = fT + (iInImgX - 1.0f) * 0.5f - gfShift[0];
        afXY[1] = afXY[1] + (giInSize[1] - 1.0f) * 0.5f - gfShift[1];
}

static __device__ float mGRandom(int x, int y, int iInImgX, float* gfInImg)
{
	if(x < 0) x = -x;
	else if(x >= iInImgX) x = 2 * iInImgX - x;
	if(y < 0) y = -y;
	else if(y >= giInSize[1]) y = 2 * giInSize[1] - y;
	//------------------------------------------------
	int iWin = 31;
	int iWinPixels = iWin * iWin;
	unsigned int next = y * giInSize[0] + x;
	for(int k=0; k<iWinPixels; k++)
	{	next = (next * 7) % iWinPixels;
		int ix = (next % iWin) - iWin / 2 + x;
		if(ix < 0 || ix >= iInImgX) continue;
		int iy = (next / iWin) - iWin / 2 + y;
		if(iy < 0 || iy >= giInSize[1]) continue;
		return gfInImg[iy * giInSize[0] + ix];
	}
	return gfInImg[y * giInSize[0] + x];
}

static __global__ void mGDoIt
(	float* gfInImg,
	int iInImgX,
	float* gfOutImg,
	bool bRandomFill
)
{	int y =  blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= giOutSize[1]) return;
	//---------------------------
	float afXY[2] = {0.0f};
	mGCalcInXY(y, iInImgX, afXY);
	int ix = (int)afXY[0];
	int iy = (int)afXY[1];
	//--------------------
	y = y * giOutSize[0] + blockIdx.x;
	if(ix < 0 || ix >= iInImgX || iy < 0 || iy >= giInSize[1])
	{	if(!bRandomFill) gfOutImg[y] = (float)-1e30;
		else gfOutImg[y] = mGRandom(ix, iy, iInImgX, gfInImg);
	}
	else
	{	gfOutImg[y] = gfInImg[iy * giInSize[0] + ix];
	}
}

GShiftRotate2D::GShiftRotate2D(void)
{
}

GShiftRotate2D::~GShiftRotate2D(void)
{
}

void GShiftRotate2D::SetSizes
(	int* piInSize,
	bool bInPadded, 
	int* piOutSize,
	bool bOutPadded
)
{	cudaMemcpyToSymbol(giInSize, piInSize, sizeof(giInSize));
	cudaMemcpyToSymbol(giOutSize, piOutSize, sizeof(giOutSize));
	//----------------------------------------------------------
	m_iInImgX = piInSize[0];
	if(bInPadded) m_iInImgX = (piInSize[0] / 2 - 1) * 2;
	//--------------------------------------------------
	m_iOutImgY = piOutSize[1];
	m_iOutImgX = piOutSize[0];
	if(bOutPadded) m_iOutImgX = (piOutSize[0] / 2 - 1) * 2;
	memcpy(m_aiOutSize, piOutSize, sizeof(float) * 2);
}

void GShiftRotate2D::DoIt
(	float* gfInImg,
	float* pfShift, 
	float fRotAngle, 
	float* gfOutImg,
	bool bRandomFill,
	cudaStream_t stream
) 
{	float afShift[] = {0.0f, 0.0f};
	if(pfShift != 0L) memcpy(afShift, pfShift, sizeof(afShift)); 
	cudaMemcpyToSymbol(gfShift, afShift, sizeof(gfShift));
	//----------------------------------------------------
	double dRad = atan(1.0) * 4 / 180.0;
	float afCosSin[2] = {0.0f};
        afCosSin[0] = (float)cos(fRotAngle * dRad);
        afCosSin[1] = (float)sin(fRotAngle * dRad);
        cudaMemcpyToSymbol(gfCosSin, afCosSin, sizeof(gfCosSin));
	//-------------------------------------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_iOutImgX, 1);
	aGridDim.y = (m_iOutImgY + aBlockDim.y - 1) / aBlockDim.y;
	mGDoIt<<<aGridDim, aBlockDim, 0, stream>>>
	(gfInImg, m_iInImgX, gfOutImg, bRandomFill);
}

