#include "CStreAlignInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::AreTomo::StreAlign;

CStreAlignMain::CStreAlignMain(void)
{
	m_pCorrTomoStack = 0L;
	m_pMeaParam = 0L;
}

CStreAlignMain::~CStreAlignMain(void)
{
	this->Clean();
}

void CStreAlignMain::Clean(void)
{
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	if(m_pMeaParam != 0L) delete m_pMeaParam;
	m_pCorrTomoStack = 0L;
	m_pMeaParam = 0L;
}

void CStreAlignMain::Setup(int iNthGpu)
{	
	this->Clean();
	//-----------------
	MD::CTsPackage* pPkg = MD::CTsPackage::GetInstance(iNthGpu);
	m_pTiltSeries = pPkg->GetSeries(0);
	m_pAlignParam = MAM::CAlignParam::GetInstance(iNthGpu);
	//-----------------
	int iBinX = (int)(m_pTiltSeries->m_aiStkSize[0] / 1024.0f + 0.1f);
	int iBinY = (int)(m_pTiltSeries->m_aiStkSize[1] / 1024.0f + 0.1f);
	int iBin = (iBinX < iBinY) ? iBinX : iBinY;
	if(iBin < 1) iBin = 1;
	//-----------------
	m_pCorrTomoStack = new MAC::CCorrTomoStack;
	bool bShiftOnly = true, bRandFill = true;
	bool bFourierCrop = true, bRWeight = true;
	m_pCorrTomoStack->Set0(iNthGpu);
	m_pCorrTomoStack->Set1(0, 0.0f);
	m_pCorrTomoStack->Set2((float)iBin, !bFourierCrop, bRandFill);
	m_pCorrTomoStack->Set3(bShiftOnly, false, !bRWeight);
	//-----------------
	m_pBinSeries = m_pCorrTomoStack->GetCorrectedStack(false);
	m_afBinning[0] = m_pTiltSeries->m_aiStkSize[0] * 1.0f
	   / m_pBinSeries->m_aiStkSize[0];
	m_afBinning[1] = m_pTiltSeries->m_aiStkSize[1] * 1.0f
	   / m_pBinSeries->m_aiStkSize[1];
}	

void CStreAlignMain::DoIt(void)
{	
	m_pMeaParam = m_pAlignParam->GetCopy();
	m_pMeaParam->ResetShift();
	//----------------
	printf("Pre-align tilt series\n");
	m_pCorrTomoStack->DoIt(0, m_pAlignParam);
	float fErr = mMeasure();
	mUpdateShift();
        printf("Error: %8.2f\n\n", fErr);
	//-----------------
	delete m_pMeaParam; 
	m_pMeaParam = 0L;
}

float CStreAlignMain::mMeasure(void)
{
	m_pMeaParam->ResetShift();
	//-----------------
	CStretchAlign stretchAlign;
	float fMaxErr = stretchAlign.DoIt(m_pBinSeries, m_pMeaParam, 
	   400.0, m_afBinning);
	//-----------------
	return fMaxErr;
}

void CStreAlignMain::mUpdateShift(void)
{
	int iZeroTilt = m_pMeaParam->GetFrameIdxFromTilt(0.0f);
	m_pMeaParam->MakeRelative(iZeroTilt);
	//-----------------------------------
	float afSumShift[2] = {0.0f};
	float afShift[2] = {0.0f};
	for(int i=iZeroTilt-1; i>=0; i--)
	{	m_pMeaParam->GetShift(i, afShift);
		afSumShift[0] += afShift[0];
		afSumShift[1] += afShift[1];
		mUnstretch(i+1, i, afSumShift);
		m_pAlignParam->AddShift(i, afSumShift);
	}
	//---------------------------------------------
	memset(afSumShift, 0, sizeof(afSumShift));
	for(int i=iZeroTilt+1; i<m_pMeaParam->m_iNumFrames; i++)
        {       m_pMeaParam->GetShift(i, afShift);
                afSumShift[0] += afShift[0];
                afSumShift[1] += afShift[1];
                mUnstretch(i-1, i, afSumShift);
                m_pAlignParam->AddShift(i, afSumShift);
        }
}

void CStreAlignMain::mUnstretch(int iLowTilt, int iHighTilt, float* pfShift)
{
        double dRad = 3.14159 / 180.0;
        float fLowTilt = m_pAlignParam->GetTilt(iLowTilt);
        float fHighTilt = m_pAlignParam->GetTilt(iHighTilt);
        float fTiltAxis = m_pAlignParam->GetTiltAxis(iHighTilt);
        double dStretch = cos(fLowTilt * dRad) / cos(fHighTilt * dRad);
	//----------------- 
	MAU::GTiltStretch tiltStretch;
        tiltStretch.CalcMatrix((float)dStretch, fTiltAxis);
        tiltStretch.Unstretch(pfShift, pfShift);
}
