#include "CCommonLineInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;

static __device__ __constant__ int giImgSize[2];
static __device__ __constant__ float gfOffset[2];

static __global__ void mGGenLine
(	float* gfImage,
 	float* gfRotAngles,
	float fCosTilt,   //cosine of tilt angle
	int* giComRegion, //common region
	int iLineSize,
	float* gfPadLines //multiple lines
)
{	int x, y, i;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iLineSize) return;
	//-------------------------
	float fCosRot = gfRotAngles[blockIdx.x] * 0.017453f;
	float fSinRot = sinf(fCosRot);
	fCosRot = cosf(fCosRot);
	//----------------------
	float fY = y - iLineSize * 0.5f;
	float fSum = 0.0f;
	int iCount = 0;
	//-------------
	i = blockIdx.x * iLineSize + y;
	int iLeftWidth = (int)(giComRegion[2 * i] * fCosTilt + 0.5f);
	for(x=iLeftWidth; x<0; x++)
	{	float fOldX = x * fCosRot - fY * fSinRot + gfOffset[0];
		float fOldY = x * fSinRot + fY * fCosRot + gfOffset[1];
		if(fOldX < 0 || fOldX >= giImgSize[0]) continue;
		if(fOldY < 0 || fOldY >= giImgSize[1]) continue;
		//----------------------------------------------
		fOldX = gfImage[((int)fOldY) * giImgSize[0] + (int)fOldX];
		if(fOldX < 0) continue;
		//---------------------
		fSum += fOldX;
		iCount++;
	}
	int iRightWidth = (int)(giComRegion[2*i+1] * fCosTilt + 0.5f);
	for(x=0; x<iRightWidth; x++)
	{	float fOldX = x * fCosRot - fY * fSinRot + gfOffset[0];
		float fOldY = x * fSinRot + fY * fCosRot + gfOffset[1];
		if(fOldX < 0 || fOldX >= giImgSize[0]) continue;
		if(fOldY < 0 || fOldY >= giImgSize[1]) continue;
		//----------------------------------------------
		fOldX = gfImage[((int)fOldY) * giImgSize[0] + (int)fOldX];
		if(fOldX < 0) continue;
		//---------------------
		fSum += fOldX;
		iCount++;
	}
	//------------
	y = blockIdx.x * (iLineSize / 2 + 1) * 2 + y;
	if(iCount == 0) gfPadLines[y] = (float)-1e30;
	else gfPadLines[y] = fSum / iCount;
}

GGenCommonLine::GGenCommonLine(void)
{
	m_gfRotAngles = 0L;
	m_gfImage = 0L;
}

GGenCommonLine::~GGenCommonLine(void)
{
	this->Clean();
}

void GGenCommonLine::Clean(void)
{
	if(m_gfRotAngles != 0L) cudaFree(m_gfRotAngles);
	if(m_gfImage != 0L) cudaFree(m_gfImage);
	m_gfRotAngles = 0L;
	m_gfImage = 0L;
}

void GGenCommonLine::Setup(int iNthGpu)
{	
	this->Clean();
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance(iNthGpu);
	//-----------------
	m_aiImgSize[0] = pTiltSeries->m_aiStkSize[0];
	m_aiImgSize[1] = pTiltSeries->m_aiStkSize[1];
	cudaMemcpyToSymbol(giImgSize, m_aiImgSize, sizeof(giImgSize));
	//-----------------
	m_iNumAngles = pClParam->m_iNumLines;
	int iBytes = sizeof(float) * m_iNumAngles;
	cudaMalloc(&m_gfRotAngles, iBytes);
	float* pfRotAngles = pClParam->m_pfRotAngles;
	cudaMemcpy(m_gfRotAngles, pfRotAngles, iBytes, cudaMemcpyDefault);
	//-----------------
	iBytes = sizeof(float) * m_aiImgSize[0] * m_aiImgSize[1];
	cudaMalloc(&m_gfImage, iBytes);
}

void GGenCommonLine::DoIt
(	int iProj,
	int* giComRegion,
	float* gfPadLines
)
{	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance(m_iNthGpu);
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	float* pfProj = (float*)pTiltSeries->GetFrame(iProj);
	cudaMemcpy(m_gfImage, pfProj, sizeof(float) * m_aiImgSize[0]
	   * m_aiImgSize[1], cudaMemcpyDefault);	
	//-----------------
	float afShift[2] = {0.0f};
	pAlnParam->GetShift(iProj, afShift);
	//-----------------
	float afOffset[2] = {0.0f};
	afOffset[0] = m_aiImgSize[0] * 0.5f + afShift[0];	
	afOffset[1] = m_aiImgSize[1] * 0.5f + afShift[1];
	cudaMemcpyToSymbol(gfOffset, afOffset, sizeof(gfOffset));
	//------------------
	float fTiltAng = pAlnParam->GetTilt(iProj);
	float fCosTilt = (float)cos(fTiltAng * 3.141593 / 180.0);
	//-----------------
	int iLineSize = pClParam->m_iLineSize;
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_iNumAngles, 1);
	aGridDim.y = (iLineSize + aBlockDim.y - 1) / aBlockDim.y;
	mGGenLine<<<aGridDim, aBlockDim>>>(m_gfImage, m_gfRotAngles, 
	   fCosTilt, giComRegion, iLineSize, gfPadLines);
}
