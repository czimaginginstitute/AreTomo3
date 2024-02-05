#include "CCommonLineInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::CommonLine;
namespace MAM = McAreTomo::AreTomo::MrcUtil;

static __device__ __constant__ float gfShift[2];

//-------------------------------------------------------------------
// 1. The common region is the subarea commonly seen in each tilted
//    image. When the tilt axis is rotated to the y axis (the 
//    vertical axis going through the image center), the common 
//    area can be described as a sets of line segments in x axis.
// 2. This class intends to determine the common region in the
//    untilt image. The common regions in tilted images will then
//    be determined based upon cosines of tilt angles.
// 3. giComRegion is a 2D array with the sizeof  2 * iImgSizeY and 
//    stores two numbers that describe the width at each y. The
//    first one is the width of the line segment to the left of y 
//    axis and the second and corresponds to the width of the line
//    segment to the right of y axis.
//-------------------------------------------------------------------
static __global__ void mGCalcCommonRegion
(	int iImgSizeX,
	int iImgSizeY,
	int iLineSize,
	float* gfRotAngles,
	int* giComRegion
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iLineSize) return;
	int i = blockIdx.x * iLineSize + y;
	//---------------------------------
	float fCos = gfRotAngles[blockIdx.x] * 3.1415926f / 180.0f;
	float fSin = sinf(fCos);
	fCos = cosf(fCos);
	//----------------
	int iHalf = iLineSize / 2;
	float fOffsetX = iImgSizeX * 0.5f + gfShift[0];
	float fOffsetY = iImgSizeY * 0.5f + gfShift[1];
	float fY = y - iLineSize * 0.5f;
	//------------------------------
	for(int x=1; x<iHalf; x++)
	{	float fOldX = -x * fCos - fY * fSin + fOffsetX;
		float fOldY = -x * fSin + fY * fCos + fOffsetY;
		if(fOldX < 0 || fOldX >= iImgSizeX) break;
		if(fOldY < 0 || fOldY >= iImgSizeY) break;
		giComRegion[2 * i] = -x;
	}
	for(int x=1; x<iHalf; x++)
	{	float fOldX = x * fCos - fY * fSin + fOffsetX;
		float fOldY = x * fSin + fY * fCos + fOffsetY;
		if(fOldX < 0 || fOldX >= iImgSizeX) break;
		if(fOldY < 0 || fOldY >= iImgSizeY) break;
		giComRegion[2 * i + 1] = x;
	}
}

GCalcCommonRegion::GCalcCommonRegion(void)
{
	m_giComRegion = 0L;
}

GCalcCommonRegion::~GCalcCommonRegion(void)
{
	this->Clean();
}

void GCalcCommonRegion::Clean(void)
{
	if(m_giComRegion != 0L) cudaFree(m_giComRegion);
	m_giComRegion = 0L;
}

void GCalcCommonRegion::DoIt(int iNthGpu)
{	
	this->Clean();
	//-----------------
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance(iNthGpu);
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	MAM::CAlignParam* pAlignParam = 
	   MAM::CAlignParam::GetInstance(iNthGpu);
	//-----------------
	int iNumLines = pClParam->m_iNumLines;
	int iLineSize = pClParam->m_iLineSize;
	int iBytes = sizeof(int) * iNumLines * iLineSize * 2;
	cudaMalloc(&m_giComRegion, iBytes);
	cudaMemset(m_giComRegion, 0, iBytes);	
	//-----------------
	float afShift[2] = {0.0f};
	int iZeroTilt = pAlignParam->GetFrameIdxFromTilt(0.0f);
	pAlignParam->GetShift(iZeroTilt, afShift);
	cudaMemcpyToSymbol(gfShift, afShift, sizeof(gfShift));
	//-----------------
	float* gfRotAngles = 0L;
	iBytes = sizeof(float) * iNumLines;
	cudaMalloc(&gfRotAngles, iBytes);
	cudaMemcpy(gfRotAngles, pClParam->m_pfRotAngles,
	   iBytes, cudaMemcpyDefault);
	//-----------------
	dim3 aBlockDim(1, 256);
	dim3 aGridDim(iNumLines, 1);
	aGridDim.y = (iLineSize + aBlockDim.y - 1) / aBlockDim.y;
	//-----------------
	mGCalcCommonRegion<<<aGridDim, aBlockDim>>>
	( pTiltSeries->m_aiStkSize[0], pTiltSeries->m_aiStkSize[1], 
	  iLineSize, gfRotAngles, m_giComRegion
	);
	cudaFree(gfRotAngles);
}

