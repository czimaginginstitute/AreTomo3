#include "CMrcUtilInc.h"
#include "../Util/CUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CCropVolume::CCropVolume(void)
{
	m_pOutVol = 0L;
}

CCropVolume::~CCropVolume(void)
{
	this->Clean();
}

void CCropVolume::Clean(void)
{
	if(m_pOutVol != 0L) delete m_pOutVol;
	m_pOutVol = 0L;
}

MD::CTiltSeries* CCropVolume::DoIt
(	MD::CTiltSeries* pInVol,
	float fOutBin,
	CAlignParam* pFullParam,
	CLocalAlignParam* pLocalParam,
	int* piOutSizeXY
)
{	m_pInVol = pInVol;
	m_fOutBin = fOutBin;
	m_pFullParam = pFullParam;
	m_pLocalParam = pLocalParam;
	m_aiOutSize[0] = piOutSizeXY[0];
	m_aiOutSize[1] = piOutSizeXY[1];
	//------------------------------
	this->Clean();
	mCalcOutCenter();
	mCreateOutVol();
	mCalcOutVol();
	//------------
	MD::CTiltSeries* pOutVol = m_pOutVol;
	m_pOutVol = 0L;
	return pOutVol;
}

void CCropVolume::mCalcOutCenter(void)
{
	int iNumPatches = m_pLocalParam->m_iNumPatches;
	float* pfCentXs = new float[iNumPatches * 2];
	float* pfCentYs = pfCentXs + iNumPatches;
	//---------------------------------------
	int iZeroTilt = m_pFullParam->GetFrameIdxFromTilt(0.0f);
	m_pLocalParam->GetCoordXYs(iZeroTilt, pfCentXs, pfCentYs);
	//-----------------------
	float fMinX = pfCentXs[0], fMaxX = pfCentXs[0];
	float fMinY = pfCentYs[0], fMaxY = pfCentYs[0];
	for(int i=1; i<iNumPatches; i++)
	{	if(fMinX > pfCentXs[i]) fMinX = pfCentXs[i];
		else if(fMaxX < pfCentXs[i]) fMaxX = pfCentXs[i];
		//-----------------------------------------------
		if(fMinY > pfCentYs[i]) fMinY = pfCentYs[i];
		else if(fMaxY < pfCentYs[i]) fMaxY = pfCentYs[i];
	}
	m_aiOutCent[0] = (int)((fMinX + fMaxX) * 0.5f);
	m_aiOutCent[1] = (int)((fMinY + fMaxY) * 0.5f);
}

void CCropVolume::mCreateOutVol(void)
{
	int* piInStkSize = m_pInVol->m_aiStkSize;
	int aiStkSize[3] = {0};
	aiStkSize[0] = (int)(m_aiOutSize[0] / m_fOutBin + 0.5f) / 2 * 2;
	aiStkSize[1] = piInStkSize[1]; // z-axis, no cropping
	aiStkSize[2] = (int)(m_aiOutSize[1] / m_fOutBin + 0.5f) / 2 * 2;
	if(aiStkSize[0] > piInStkSize[0]) aiStkSize[0] = piInStkSize[0];
	if(aiStkSize[2] > piInStkSize[2]) aiStkSize[2] = piInStkSize[2];
	//--------------------------------------------------------------
	bool bAlloc = true;
	m_pOutVol = new MD::CTiltSeries;
	m_pOutVol->Create(aiStkSize, bAlloc);
}

void CCropVolume::mCalcOutVol(void)
{
	MAU::GShiftRotate2D aGshiftRotate2D;
	bool bPadded = true;
	int* piInStkSize = m_pInVol->m_aiStkSize;
	int* piOutStkSize = m_pOutVol->m_aiStkSize;
	aGshiftRotate2D.SetSizes(piInStkSize, !bPadded,
	   piOutStkSize, !bPadded);
	//-------------------------
	size_t tInBytes = piInStkSize[0] * piInStkSize[1] * sizeof(float);
	size_t tOutBytes = piOutStkSize[0] * piOutStkSize[1] * sizeof(float);
	float *gfInImg = 0L, *gfOutImg = 0L;
	cudaMalloc(&gfInImg, tInBytes);
	cudaMalloc(&gfOutImg, tOutBytes);
	//-------------------------------
	float afShift[2] = {0.0f};
	afShift[0] = -m_aiOutCent[0] / m_fOutBin;
	int iShiftY = (int)(m_aiOutCent[1] / m_fOutBin);
	int iOffsetY = (piInStkSize[2] -piOutStkSize[2]) / 2 + iShiftY;
	if(iOffsetY < 0) iOffsetY = 0;
	if((iOffsetY + piOutStkSize[2]) > piInStkSize[2])
	{	iOffsetY = piInStkSize[2] - piOutStkSize[2];
	}
	//--------------------------------------------------
	for(int y=0; y<m_pOutVol->m_aiStkSize[2]; y++)
	{	float* pfOutFrm = (float*)m_pOutVol->GetFrame(y);
		float* pfInFrm = (float*)m_pInVol->GetFrame(y + iOffsetY);
		cudaMemcpy(gfOutImg, pfOutFrm, tOutBytes, cudaMemcpyDefault);
		cudaMemcpy(gfInImg, pfInFrm, tInBytes, cudaMemcpyDefault);
		//--------------------------------------------------------
		bool bRandomFill = true;
		aGshiftRotate2D.DoIt(gfInImg, afShift, 0.0f,
		   gfOutImg, bRandomFill);
		//------------------------
		cudaMemcpy(pfOutFrm, gfOutImg, tOutBytes, cudaMemcpyDefault);
	}
	//-------------------------------------------------------------------
	cudaFree(gfInImg);
	cudaFree(gfOutImg);
}

