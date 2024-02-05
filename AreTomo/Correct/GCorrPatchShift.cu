#include "CCorrectInc.h"
#include "../PatchAlign/CPatchAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::Correct;

// giInSize: padded sizeX, sizeY, numPatches
static __device__ __constant__ int giInSize[3];
static __device__ __constant__ int giOutSize[2];

static __device__ void mGCalcLocalShift
(	float* gfLocalAlnParams,
	int iInImgX,
     	float afXY[2],
	float afLS[2]
)
{	float* gfPatCentYs = gfLocalAlnParams + giInSize[2];
	float* gfLocalShiftXs = gfLocalAlnParams + giInSize[2] * 2;
	float* gfLocalShiftYs = gfLocalAlnParams + giInSize[2] * 3;
	float* gfGood = gfLocalAlnParams + giInSize[2] * 4;
	//-------------------------------------------------
	int iCount = 0;
	float fSx = 0.0f, fSy = 0.0f, fW = 0.0f, fSumW = 0.0f;
	for(int p=0; p<giInSize[2]; p++)
	{	if(gfGood[p] < 0.9f) continue;
		afLS[0] = (afXY[0] - gfLocalAlnParams[p]) / iInImgX;
		afLS[1] = (afXY[1] - gfPatCentYs[p]) / giInSize[1];
		fW = expf(-100.0f * (afLS[0] * afLS[0] + afLS[1] * afLS[1]));
		fSx += (gfLocalShiftXs[p] * fW);
		fSy += (gfLocalShiftYs[p] * fW);
		fSumW += fW; iCount += 1;
	}
	if(iCount > 0)
	{	afLS[0] = fSx / fSumW;
		afLS[1] = fSy / fSumW;
	}
	else
	{	afLS[0] = 0.0f;
		afLS[1] = 0.0f;
	}
}

static __device__ float mGRandom
(	int x, int y, 
	int iInImgX,
	float* gfInImg
)
{	if(x < 0) x = -x;
	if(y < 0) y = -y;
	if(x >= iInImgX) x = iInImgX - 1 - (x % iInImgX);
	if(y >= giInSize[1]) y = giInSize[1] - 1 - (y % giInSize[1]);
	//-----------------------------------------------------------
	int iWin = 51, ix = 0, iy = 0;
	int iSize = iWin * iWin;
	unsigned int next = y * giInSize[0] + x;
	for(int i=0; i<20; i++)
	{	next = (next * 19 + 57) % iSize;
		ix = (next % iSize) - iWin / 2 + x;
		if(ix < 0 || ix >= iInImgX) continue;
		//-----------------------------------
		iy = (next / iWin) - iWin / 2 + y;
		if(iy < 0 || iy >= giInSize[1]) continue;
		//---------------------------------------
		return gfInImg[iy * giInSize[0] + ix];
	}
	return gfInImg[y * giInSize[0] + x];
}
//-----------------------------------------------------------------------------
// Imod coordinate system: [0, Nx] where 0 is the left edge of the image and
//    Nx the right edge. The origin is at Nx * 0.5. The leftmost pixel is at
//    x = 0.5 and the rightmost is at Nx * 0.5 - 0.5.
// The conversioon from pixel coordinates to pixel index: we can simply
//    convert float to int. For example, when x in (0, 1), the pixel index
//    should be 0 since x is closest to 0.5, coordinate of pixel 0.
//-----------------------------------------------------------------------------
static __global__ void mGCorrect
(	float* gfInImg,
	int iInImgX,
	float fGlobalShiftX,
	float fGlobalShiftY,
	float fRotAngle, // tilt axis in radian
	float* gfLocalAlnParams,
	bool bRandomFill,
	float* gfOutImg
)
{	int x = 0, y = 0;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= giOutSize[1]) return;
	int i = y * giOutSize[0] + blockIdx.x;
	//------------------------------------
	float afXY[2] = {0.0f}, afTmp[2];
	afXY[0] = blockIdx.x + 0.5f - gridDim.x * 0.5f;
	afXY[1] = y + 0.5f - giOutSize[1] * 0.5f;
	//---------------------------------------
	afTmp[0] = cosf(fRotAngle);
	afTmp[1] = sinf(fRotAngle);
	float fT = afXY[0] * afTmp[0] - afXY[1] * afTmp[1]; 
	afXY[1] = afXY[0] * afTmp[1] + afXY[1] * afTmp[0];
	afXY[0] = fT;
	//-----------
	if(giInSize[2] > 0 && gfLocalAlnParams != 0L)
	{	mGCalcLocalShift(gfLocalAlnParams, iInImgX, afXY, afTmp);
		afXY[0] += afTmp[0];
		afXY[1] += afTmp[1];
	}
	afXY[0] += (fGlobalShiftX + iInImgX * 0.5f);
	afXY[1] += (fGlobalShiftY + giInSize[1] * 0.5f);
	//----------------------------------------------
	x = (int)(afXY[0] + 0.5f);
	y = (int)(afXY[1] + 0.5f);
	//------------------------
	if(x >= 0 && x < iInImgX && y >= 0 && y < giInSize[1]) 
	{	gfOutImg[i] = gfInImg[y * giInSize[0] + x];
		return;
	}
	//-------------
	if(bRandomFill) gfOutImg[i] = mGRandom(x, y, iInImgX, gfInImg);
	else gfOutImg[i] = (float)(-1e30);
}

GCorrPatchShift::GCorrPatchShift(void)
{
	m_fD2R = 3.141592654f / 180.0f;
}

GCorrPatchShift::~GCorrPatchShift(void)
{
}

void GCorrPatchShift::SetSizes
(	int* piInSize,
	bool bInPadded, 
	int* piOutSize,
	bool bOutPadded,
	int iNumPatches
)
{	int aiInSize[] = {piInSize[0], piInSize[1], iNumPatches};
	cudaMemcpyToSymbol(giInSize, aiInSize, sizeof(giInSize));
	cudaMemcpyToSymbol(giOutSize, piOutSize, sizeof(giOutSize));
	//----------------------------------------------------------
	m_iInImgX = piInSize[0];
	if(bInPadded) m_iInImgX = (piInSize[0] / 2 - 1) * 2;
	//--------------------------------------------------
	m_iOutImgY = piOutSize[1];
	m_iOutImgX = piOutSize[0];
	if(bOutPadded) m_iOutImgX = (piOutSize[0] / 2 - 1) * 2;
}

void GCorrPatchShift::DoIt
(	float* gfInImg,
	float* pfGlobalShift,
	float fRotAngle,
	float* gfLocalAlnParams,
	bool bRandomFill,
	float* gfOutImg
)
{	fRotAngle *= m_fD2R;
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_iOutImgX, 1);
	aGridDim.y = (m_iOutImgY + aBlockDim.y - 1) / aBlockDim.y;
	//--------------------------------------------------------
	mGCorrect<<<aGridDim, aBlockDim>>>(gfInImg, m_iInImgX,
	   pfGlobalShift[0], pfGlobalShift[1], fRotAngle,
	   gfLocalAlnParams, bRandomFill, gfOutImg);
}

