#include "CTiltOffsetInc.h"
#include "../Util/CUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include "../StreAlign/CStreAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::TiltOffset;

CTiltOffsetMain::CTiltOffsetMain(void)
{
	m_pCorrTomoStack = 0L;
	m_pStretchCC2D = new MAS::CStretchCC2D;
}

CTiltOffsetMain::~CTiltOffsetMain(void)
{
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	m_pCorrTomoStack = 0L;
	if(m_pStretchCC2D != 0L) delete m_pStretchCC2D;
	m_pStretchCC2D = 0L;
}

void CTiltOffsetMain::Setup(int iXcfBin, int iNthGpu)
{
	m_pAlignParam = MAM::CAlignParam::GetInstance(iNthGpu);
	//-----------------
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	m_pCorrTomoStack = new MAC::CCorrTomoStack;
	//-----------------
	bool bShiftOnly = true, bRandomFill = true;
	bool bFourierCrop = true, bRWeight = true, bClean = true;
	//-----------------
	m_pCorrTomoStack->Set0(iNthGpu);
	m_pCorrTomoStack->Set1(0, 0.0f);
	m_pCorrTomoStack->Set2((float)iXcfBin, !bFourierCrop, !bRandomFill);
	m_pCorrTomoStack->Set3(bShiftOnly, false, !bRWeight);
	m_pTiltSeries = m_pCorrTomoStack->GetCorrectedStack(!bClean);
	//-----------------
	bool bPadded = true;
	m_pStretchCC2D->SetSize(m_pTiltSeries->m_aiStkSize, !bPadded);
}
	

float CTiltOffsetMain::DoIt(void)
{
	bool bClean = true;
	m_pCorrTomoStack->DoIt(0, m_pAlignParam);
	//-----------------
	printf("Determine tilt angle offset.\n");
	float fBestOffset = mSearch(31, 1.0f, 0.0f);
	//-----------------
	m_pStretchCC2D->Clean();
	return fBestOffset;
}

float CTiltOffsetMain::mSearch
(	int iNumSteps, 
	float fStep, 
	float fInitOffset
)
{	float fMaxCC = 0.0f;
        float fBestOffset = 0.0f;
        for(int i=0; i<iNumSteps; i++)
        {       float fOffset = fInitOffset + (i - iNumSteps / 2) * fStep;
                float fCC = mCalcAveragedCC(fOffset);
                if(fCC > fMaxCC)
                {       fMaxCC = fCC;
                        fBestOffset = fOffset;
                }
                printf("...... %8.2f  %.4e\n", fOffset, fCC);
        }
	printf("Tilt offset: %8.2f,  CC: %.4f\n\n", fBestOffset, fMaxCC);
	return fBestOffset;
}

float CTiltOffsetMain::mCalcAveragedCC(float fTiltOffset)
{
	m_pAlignParam->AddTiltOffset(fTiltOffset);
	//----------------------------------------
	int iCount = 0;
	float fCCSum = 0.0f;
	int iZeroTilt = m_pAlignParam->GetFrameIdxFromTilt(0.0f);
	for(int i=0; i<m_pAlignParam->m_iNumFrames; i++)
	{	if(i == iZeroTilt) continue;
		int iRefTilt = (i < iZeroTilt) ? i+1 : i-1;
		float fCC = mCorrelate(iRefTilt, i);
		fCCSum += fCC;
		iCount++;
	}
	float fMeanCC = fCCSum / iCount;
	//------------------------------
	m_pAlignParam->AddTiltOffset(-fTiltOffset);
	return fMeanCC;
}

float CTiltOffsetMain::mCorrelate(int iRefProj, int iProj)
{
	float* pfRefProj = (float*)m_pTiltSeries->GetFrame(iRefProj);
	float* pfProj = (float*)m_pTiltSeries->GetFrame(iProj);
	float fRefTilt = m_pAlignParam->GetTilt(iRefProj);
	float fTilt = m_pAlignParam->GetTilt(iProj);
	float fTiltAxis = m_pAlignParam->GetTiltAxis(iProj);
	//--------------------------------------------------
	float fCC = m_pStretchCC2D->DoIt(pfRefProj, pfProj, 
	   fRefTilt, fTilt, fTiltAxis);
	return fCC;
}

