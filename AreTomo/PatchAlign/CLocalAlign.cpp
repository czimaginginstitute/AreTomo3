#include "CPatchAlignInc.h"
#include "../CAreTomoInc.h"
#include "../Correct/CCorrectInc.h"
#include "../ProjAlign/CProjAlignInc.h"
#include "../Massnorm/CMassNormInc.h"
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::PatchAlign;
static float s_fD2R = 0.0174533f;

CLocalAlign::CLocalAlign(void)
{
	m_pProjAlignMain = 0L;
}

CLocalAlign::~CLocalAlign(void)
{
	if(m_pProjAlignMain != 0L) delete m_pProjAlignMain;
	m_pProjAlignMain = 0L;
}

void CLocalAlign::Setup(int iNthGpu)
{	
	if(m_pProjAlignMain != 0L) delete m_pProjAlignMain;
	m_pProjAlignMain = new ProjAlign::CProjAlignMain;
	m_iNthGpu = iNthGpu;
}

void CLocalAlign::DoIt(MAM::CAlignParam* pAlignParam, int* piRoi)
{	
	pAlignParam->ResetShift();
	MD::CTsPackage* pPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPkg->GetSeries(0);
	int iZeroTilt = pAlignParam->GetFrameIdxFromTilt(0.0f);
	//-----------------
	float afS1[2] = {0.0f}, afS2[2] = {0.0f}, afS3[2] = {0.0f};
	afS1[0] = piRoi[0] - pTiltSeries->m_aiStkSize[0] * 0.5f;
	afS1[1] = piRoi[1] - pTiltSeries->m_aiStkSize[1] * 0.5f;
	float fCosRef = (float)cos(pAlignParam->GetTilt(iZeroTilt) * s_fD2R);
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float fTilt = pAlignParam->GetTilt(i) * s_fD2R;
		float fTiltAxis = pAlignParam->GetTiltAxis(i);
		MAM::CAlignParam::RotShift(afS1, -fTiltAxis, afS2);
		afS2[0] *= (float)(cos(fTilt) / fCosRef);
		MAM::CAlignParam::RotShift(afS2, fTiltAxis, afS3);
		pAlignParam->SetShift(i, afS3);
	}
	//-------------------------------------
	CAtInput* pAtInput = CAtInput::GetInstance();
	ProjAlign::CParam* pParam = ProjAlign::CParam::GetInstance(m_iNthGpu);
	pParam->m_fXcfSize = 1024.0f * 1.0f;
	pParam->m_afMaskSize[0] = 0.8f;
	pParam->m_afMaskSize[1] = 0.8f;
	m_pProjAlignMain->Set0(400.0f, m_iNthGpu);
	m_pProjAlignMain->Set1(pParam);
	m_pProjAlignMain->Set2(true);
	//-----------------
	pAlignParam->SetRotationCenterZ(0.0f);
	float fLastErr = m_pProjAlignMain->DoIt(pAlignParam);
	MAM::CAlignParam* pLastParam = pAlignParam->GetCopy();
	//-----------------
	pParam->m_fXcfSize = 1024.0f * 2.0f;
	pParam->m_afMaskSize[0] = 0.25f;
	pParam->m_afMaskSize[1] = 0.25f;
	m_pProjAlignMain->Set1(pParam);
	//-----------------
	int iIterations = 2;
	int iLastIter = iIterations - 1;
	for(int i=0; i<iIterations; i++)
	{	float fErr = m_pProjAlignMain->DoIt(pAlignParam);
		//--------------------
		pParam->m_afMaskSize[0] = 0.125f;
		pParam->m_afMaskSize[1] = 0.125f;
		//-----------------------------	
		if(fErr < fLastErr)
		{	pLastParam->Set(pAlignParam);
			if((fLastErr - fErr) < 1) break;
			else fLastErr = fErr; 
		}
		else
		{	pAlignParam->Set(pLastParam);
			break;
		}
	}
	delete pLastParam;
}

