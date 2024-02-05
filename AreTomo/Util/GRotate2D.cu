#include "CUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::Util;

static __device__ __constant__ float gfCosSin[2];
static __device__ __constant__ int giImgSize[3]; // imageX, imageY, iPadX

//-------------------------------------------------------------------
// 1. The output image can be both unpadded or padded for FFT.
//    If unpadded, iPadSizeX store unpadded size.
// 2. gridDim.x always store unpadded size X because we don't
//    need to fill the padded region.
//-------------------------------------------------------------------
static __global__ void mGRotate
(	float* gfInImg,
 	float fFillVal,
	float* gfOutImg,
	int iPadSizeX
)
{	int x, y;
	y =  blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= giImgSize[1]) return;
	//---------------------------
	float fX, fY, fCentX, fCentY;
	fCentX = gridDim.x / 2 + 0.5f;
	fCentY = giImgSize[1] / 2 + 0.5f;
	fX = blockIdx.x - fCentX;
	fY = y - fCentY;
	fCentX = fX * gfCosSin[0] + fY * gfCosSin[1] + fCentX;
	fY = -fX * gfCosSin[1] + fY * gfCosSin[0] + fCentY;
	fX = fCentX;
	//-------------------------------------------------
	int i = y * iPadSizeX + blockIdx.x;
	if(fX < 0 || fY < 0 || fX > (gridDim.x - 1) || fY > (giImgSize[1] - 1)) 
	{	gfOutImg[y * iPadSizeX + blockIdx.x] = fFillVal;
		return;
	}
	//-----------------------------
	x = (int)fX;
	y = (int)fY;
	gfOutImg[i] = gfInImg[y * giImgSize[2] + x];
}

GRotate2D::GRotate2D(void)
{
	m_fFillVal = (float)-1e30;
}

GRotate2D::~GRotate2D(void)
{
}

void GRotate2D::SetFillValue(float fFillVal)
{
	m_fFillVal = fFillVal;
}

void GRotate2D::SetImage(float* gfInImg, int* piSize, bool bPadded)
{
	m_gfInImg = gfInImg;
	m_aiImgSize[0] = piSize[0];
	m_aiImgSize[1] = piSize[1];
	if(bPadded) m_aiImgSize[0] = (piSize[0] / 2 - 1) * 2;
	//---------------------------------------------------
	int aiSize[] = {m_aiImgSize[0], m_aiImgSize[1], piSize[0]};
	cudaMemcpyToSymbol(giImgSize, aiSize, sizeof(int) * 3);
}
//-----------------------------------------------------------------------------
// The images before and after rotaton must have the same image size but can
// be either padded or not.
//-----------------------------------------------------------------------------
void GRotate2D::DoIt(float fAngle, float* gfOutImg, bool bPadded) 
{
	mCalcTrig(fAngle);
	//----------------
	int iPadSizeX = m_aiImgSize[0];
	if(bPadded) iPadSizeX = (m_aiImgSize[0] / 2 + 1) * 2;
	//---------------------------------------------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_aiImgSize[0], 1);
	aGridDim.y = (m_aiImgSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//------------------------------------------------------------
	mGRotate<<<aGridDim, aBlockDim>>>(m_gfInImg, m_fFillVal,
	   gfOutImg, iPadSizeX);
}

void GRotate2D::mCalcTrig(float fAngle)
{
	float afCosSin[2] = {0.0f};
	float fAngRad = fAngle * 0.017453f;
	afCosSin[0] = (float)cos(fAngRad);
	afCosSin[1] = (float)sin(fAngRad);
	cudaMemcpyToSymbol(gfCosSin, afCosSin, sizeof(gfCosSin));
}
