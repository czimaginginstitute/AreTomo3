#include "CMotionCorInc.h"
#include "Align/CAlignInc.h"
#include "BadPixel/CBadPixelInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include "TiffUtil/CTiffUtilInc.h"
#include "EerUtil/CEerUtilInc.h"
//------------------
#include <Util/Util_Time.h>
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>

using namespace McAreTomo;
using namespace McAreTomo::MotionCor;

CMotionCorMain::CMotionCorMain(void)
{
	m_iNthGpu = 0;
}

CMotionCorMain::~CMotionCorMain(void)
{
}

void CMotionCorMain::LoadRefs(void)
{
	CLoadRefs* pLoadRefs = CLoadRefs::GetInstance();
	CMcInput* pMcInput = CMcInput::GetInstance();
	//-----------------
	bool bGain = pLoadRefs->LoadGain(pMcInput->m_acGainFile);
	bool bDark = pLoadRefs->LoadDark(pMcInput->m_acDarkMrc);
	//-----------------
	pLoadRefs->PostProcess(pMcInput->m_iRotGain,
	   pMcInput->m_iFlipGain, pMcInput->m_iInvGain);
}

bool CMotionCorMain::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	if(!mLoadStack()) return false;
	if(!mCheckGain()) return false;
	mCreateBuffer();
	//-----------------
	mApplyRefs();
	mDetectBadPixels();
	mCorrectBadPixels();
	//-----------------
	mAlignStack();
	return true;
}

bool CMotionCorMain::mLoadStack(void)
{
	bool bStatus = false;
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	if(pMcPackage->bTiffFile())
	{	TiffUtil::CLoadTiffMain loadTiffMain;
		bStatus = loadTiffMain.DoIt(m_iNthGpu);
	}
	else if(pMcPackage->bEerFile())
	{	EerUtil::CLoadEerMain loadEerMain;
		bStatus = loadEerMain.DoIt(m_iNthGpu);
	}
	//-----------------
	MMD::CFmIntParam* pFmIntParam = 
	   MMD::CFmIntParam::GetInstance(m_iNthGpu);
	pMcPackage->m_fTotalDose = pFmIntParam->GetTotalDose();
	//-----------------
	return bStatus;
}

bool CMotionCorMain::mCheckGain(void)
{
	CLoadRefs* pLoadRefs = CLoadRefs::GetInstance();
	if(pLoadRefs->m_pfGain == 0L)
	{	printf("Warning: Gain ref not found.\n"
		   "......   Gain correction will be skipped.\n\n");
		return true;
	}
	//-----------------
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	int* piStkSize = pPackage->GetMovieSize();
	bool bAugRefs = pLoadRefs->AugmentRefs(piStkSize);
	return bAugRefs;
}

void CMotionCorMain::mCreateBuffer(void)
{
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CMcPackage* pMcPackage = 
	   MD::CMcPackage::GetInstance(m_iNthGpu);
	//-----------------
	int* piStkSize = pMcPackage->GetMovieSize();
	pBufferPool->Create(piStkSize);
}

void CMotionCorMain::mApplyRefs(void)
{
	CLoadRefs* pLoadRefs = CLoadRefs::GetInstance();
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	//-----------------
	MMM::CApplyRefs applyRefs;
	applyRefs.DoIt(pLoadRefs->m_pfGain, 
	   pLoadRefs->m_pfDark, m_iNthGpu);
}

void CMotionCorMain::mDetectBadPixels(void)
{
	MMB::CDetectMain* pDetectMain = 
	   MMB::CDetectMain::GetInstance(m_iNthGpu); 
	pDetectMain->DoIt(m_iNthGpu);
}

void CMotionCorMain::mCorrectBadPixels(void)
{
	bool bClean = true;
	int iDefectSize = 7;
	//-----------------
	MMB::CCorrectMain* pCorrectMain = 
	   MMB::CCorrectMain::GetInstance(m_iNthGpu);
	pCorrectMain->DoIt(iDefectSize, m_iNthGpu);
}

void CMotionCorMain::mAlignStack(void)
{
	Align::CAlignMain alignMain;
	alignMain.DoIt(m_iNthGpu);
}
