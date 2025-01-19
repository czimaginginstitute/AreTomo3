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
	//-----------------
	nvtxRangePushA("align select");
	if(pAlignParam->bPatchAlign() && fabs(fTilt) < 40.0f) 
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
{	/*
	CMcInput* pInput = CMcInput::GetInstance();
	if(0 == strlen(pInput->m_acLogDir)) return 0L;
	//-----------------
	MD::CMcPackage* pMcPkg = MD::CMcPackage::GetInstance(m_iNthGpu);
	char* pcInFile = strrchr(pMcPkg->m_pcInFileName, '/');
	if(pcInFile != 0L) pcInFile += 1;
	else pcInFile = pMcPkg->m_pcInFileName;
	//-----------------
	char* pcLogFile = new char[256];
	strcpy(pcLogFile, pInput->m_acLogDir);
	strcat(pcLogFile, pcInFile);
	//-----------------
	char* pcFileExt = strcasestr(pcLogFile, ".mrc");
	if(pcFileExt != 0L) 
	{	strcpy(pcFileExt, "");
		return pcLogFile;
	}
	pcFileExt = strcasestr(pcLogFile, ".tif");
	if(pcFileExt != 0L)
	{	strcpy(pcFileExt, "");
		return pcLogFile;
	}
	pcFileExt = strcasestr(pcLogFile, ".eer");
	if(pcFileExt != 0L)
	{	strcpy(pcFileExt, "");
		return pcLogFile;
	}
	//-----------------
	delete pcLogFile; return 0L;
	*/
	return 0L;
}

