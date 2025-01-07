#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;

CCommonLineMain::CCommonLineMain(void)
{
	m_pfFitAngles = 0L;
	m_iNumImgs = 0;
}

CCommonLineMain::~CCommonLineMain(void)
{
	this->Clean();
}

void CCommonLineMain::Clean(void)
{
	if(m_pfFitAngles != 0L) delete[] m_pfFitAngles;
	m_pfFitAngles = 0L;
}

float CCommonLineMain::DoInitial
(       int iNthGpu,
        float fAngRange,
        int iNumSteps
)
{	this->Clean();
	m_iNthGpu = iNthGpu;
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(iNthGpu);
	//-----------------
	CCommonLineParam* pCLParam = CCommonLineParam::GetInstance(iNthGpu);
	pCLParam->Setup(fAngRange, iNumSteps);
	//-----------------
	CGenLines genLines;
	CPossibleLines* pPossibleLines = genLines.DoIt(m_iNthGpu);
	//-----------------
	CLineSet* pLineSet = new CLineSet;
	pLineSet->Setup(m_iNthGpu);
	//-----------------
	CFindTiltAxis findTiltAxis;
	float fRotAngle = findTiltAxis.DoIt(pPossibleLines, pLineSet);
	//-----------------
	if(pPossibleLines != 0L) delete pPossibleLines;
	if(pLineSet != 0L) delete pLineSet;
	//-----------------
	printf("Initial estimate of tilt axes:\n");
	for(int i=0; i<pAlnParam->m_iNumFrames; i++)
	{	pAlnParam->SetTiltAxis(i, fRotAngle);
	}
	printf("New tilt axis: %.2f\n\n", fRotAngle);
	return findTiltAxis.m_fScore;
}

float CCommonLineMain::DoRefine(int iNthGpu)
{	
	this->Clean();
	m_iNthGpu = iNthGpu;
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(iNthGpu);
	//-----------------
	m_iNumImgs = pAlnParam->m_iNumFrames;
	//-----------------
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance(iNthGpu);
	pClParam->Setup(6.0f, 200);
	//-----------------
	CGenLines genLines;
	CPossibleLines* pPossibleLines = genLines.DoIt(m_iNthGpu);
	//-----------------
	CLineSet* pLineSet = new CLineSet;
	pLineSet->Setup(m_iNthGpu);
	//----------------
	CRefineTiltAxis refineTiltAxis;
	refineTiltAxis.Setup(3, 10, 0.0001f);
	float fScore = refineTiltAxis.Refine(pPossibleLines, pLineSet);
	//-------------------------------------------------------------
	float* pfRotAngles = new float[pPossibleLines->m_iNumProjs];
	refineTiltAxis.GetRotAngles(pfRotAngles);
	//--------------------------------------
	if(pPossibleLines != 0L) delete pPossibleLines;
	if(pLineSet != 0L) delete pLineSet;
	//---------------------------------
	for(int i=0; i<m_iNumImgs; i++)
	{	pAlnParam->SetTiltAxis(i, pfRotAngles[i]);
	}
	if(pfRotAngles != 0L) delete[] pfRotAngles;
	return fScore;
}
