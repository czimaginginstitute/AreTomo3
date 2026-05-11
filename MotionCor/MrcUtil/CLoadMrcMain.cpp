#include "CMrcUtilInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include <Util/Util_Time.h>
#include <Mrcfile/CMrcFileInc.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::MotionCor::MrcUtil;
namespace MMD = McAreTomo::MotionCor::DataUtil;

CLoadMrcMain::CLoadMrcMain(void)
{
}

CLoadMrcMain::~CLoadMrcMain(void)
{
}

bool CLoadMrcMain::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	m_bLoaded = true;
	//---------------------------
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	bool bOpen = mLoadHeader();
	if(!bOpen)
	{	fprintf(stderr, "CLoadMrcMain: cannot open MRC file, "
		   "skip,\n   %s\n\n", pPackage->m_acMoviePath);
		m_bLoaded = false;
		return false;
	}
	//---------------------------
	mLoadStack();
	return m_bLoaded;
}

bool CLoadMrcMain::mLoadHeader(void)
{
	Mrc::CLoadMrc aLoadMrc;
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	bool bOpen = aLoadMrc.OpenFile(pPackage->m_acMoviePath);
	if(!bOpen) return false;
	//---------------------------
	aLoadMrc.m_pLoadMain->GetSize(m_aiStkSize, 3);
	m_iMode = aLoadMrc.m_pLoadMain->GetMode();
	printf("MRC file size & mode: %d  %d  %d  %d\n",
	   m_aiStkSize[0], m_aiStkSize[1], m_aiStkSize[2], m_iMode);
	//---------------------------------------------------------
	// 1. If we have per-frame dose, we should use it since it
	//    is more accurate than the MDOC's result.
	// 2. If not, we use MDOC's result for better than nothing.
	//---------------------------------------------------------
	CInput* pInput = CInput::GetInstance();
	float fImgDose = pInput->m_fFmDose * m_aiStkSize[2];
	if(fImgDose > 0) pPackage->m_fTotalDose = fImgDose;
	//----------------------------------------------------
	// If s_pPackage does not have m_pFmIntParam, this is
	// a new package amd we create an object
	//----------------------------------------------------
	MMD::CFmIntParam* pFmIntParam = 
	   MMD::CFmIntParam::GetInstance(m_iNthGpu);
	pFmIntParam->Setup(m_aiStkSize[2], m_iMode, pPackage->m_fTotalDose);
	m_aiStkSize[2] = pFmIntParam->m_iNumIntFms;
	//------------------------------------------------------------
	// Calculate group parameters for global and local alignments
	//------------------------------------------------------------
	CMcInput* pMcInput = CMcInput::GetInstance();
	MMD::CFmGroupParam* pFmGroupParam = 0L; 
	pFmGroupParam = MMD::CFmGroupParam::GetInstance(m_iNthGpu, false);
	pFmGroupParam->Setup(pMcInput->m_aiGroup[0]);
	//---------------------------
	pFmGroupParam = MMD::CFmGroupParam::GetInstance(m_iNthGpu, true);
	pFmGroupParam->Setup(pMcInput->m_aiGroup[1]);
	//---------------------------------------------------------
	// Frame integration is not supported for MRC movies.
	//---------------------------------------------------------
	pPackage->m_pRawStack->Create(m_iMode, m_aiStkSize);
	printf("Rendered size & mode:  %d  %d  %d  %d\n", 
	   m_aiStkSize[0], m_aiStkSize[1], m_aiStkSize[2], m_iMode);
	return true;
}

void CLoadMrcMain::mLoadStack(void)
{
	Util_Time aTimer;
	aTimer.Measure();
	//---------------------------
	Mrc::CLoadMrc aLoadMrc;
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	bool bOpen = aLoadMrc.OpenFile(pPackage->m_acMoviePath);
	//---------------------------
	Mrc::CLoadImage* pLoadImg = aLoadMrc.m_pLoadImg;
	MD::CMrcStack* pRawStack = pPackage->m_pRawStack;
	//---------------------------
	for(int i=0; i<pRawStack->m_aiStkSize[2]; i++)
	{	void* pvFrame = pPackage->m_pRawStack->GetFrame(i);
		pLoadImg->DoIt(i, (char*)pvFrame);
	}
	m_fLoadTime = aTimer.GetElapsedSeconds();
}

