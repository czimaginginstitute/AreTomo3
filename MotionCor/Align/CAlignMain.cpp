#include "CAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Align;

CAlignMain::CAlignMain(void)
{
}

CAlignMain::~CAlignMain(void)
{
}

void CAlignMain::DoIt(int iNthGpu)
{
	nvtxRangePush ("CAlignMain::DoIt");
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	float fTilt = pMcPackage->m_fTilt;
	//-----------------
	CAlignBase* pAlignBase = 0L;
	CAlignParam* pAlignParam = CAlignParam::GetInstance();
	//------------------------------------------------------------
	// Temporarily diable local motion correction at high tilts.
	//------------------------------------------------------------
	nvtxRangePushA("align select");
	if(pAlignParam->bPatchAlign() && fabs(fTilt) < 5.0f) 
	{	printf("Patch based alignment\n");
		pAlignBase = new CPatchAlign;
	}
	else
	{	printf("Full frame alignment\n");
		pAlignBase = new CFullAlign;
	}
	nvtxRangePop();
	pAlignBase->DoIt(m_iNthGpu);
	//-----------------
	char* pcLogFile = mCreateLogFile();
	if(pcLogFile != 0L)
	{	pAlignBase->LogShift(pcLogFile);
		delete[] pcLogFile;
	}
	//-----------------
	if(pAlignBase != 0L) delete pAlignBase;
	nvtxRangePop();
}

char* CAlignMain::mCreateLogFile(void)
{	return 0L;	
}

