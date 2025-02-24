#include "CEerUtilInc.h"
#include <Util/Util_Time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <string.h>
#include <stdio.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::EerUtil;

CLoadEerMain::CLoadEerMain(void)
{
	m_iFile = -1;
	m_pLoadHeader = 0L;
	m_pLoadFrames = 0L;
}

CLoadEerMain::~CLoadEerMain(void)
{
	mClean();
}

bool CLoadEerMain::DoIt(int iNthGpu)
{
	mClean();
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	m_iFile = open(pPackage->m_acMoviePath, O_RDONLY);
	if(m_iFile == -1)
	{	fprintf(stderr, "Error: Unable to open EER file, skip.\n"
		   "   %s\n\n", pPackage->m_acMoviePath);
		return false;
	}
	//-----------------
	m_pLoadHeader = new CLoadEerHeader;
	m_pLoadFrames = new CLoadEerFrames;
	//-----------------
	mLoadHeader();
	mLoadStack();
	//-----------------
	mClean();
	return m_bLoaded;
}

void CLoadEerMain::mLoadHeader(void)
{
	CMcInput* pInput = CMcInput::GetInstance();
	m_bLoaded = m_pLoadHeader->DoIt(m_iFile, pInput->m_iEerSampling);
	if(!m_bLoaded) return;
	//-----------------
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	MMD::CFmIntParam* pFmIntParam = 
	   MMD::CFmIntParam::GetInstance(m_iNthGpu);
	pFmIntParam->Setup(m_pLoadHeader->m_iNumFrames, Mrc::eMrcUChar,
	   pPackage->m_fTotalDose);
	//----------------------------------------------------------------
	// Need two group parameters, one for global, one for local align.
	//----------------------------------------------------------------
	MMD::CFmGroupParam* pFmGroupParam = 0L;
	pFmGroupParam = MMD::CFmGroupParam::GetInstance(m_iNthGpu, false);
        pFmGroupParam->Setup(pInput->m_aiGroup[0]);
	//-----------------
	pFmGroupParam = MMD::CFmGroupParam::GetInstance(m_iNthGpu, true);
	pFmGroupParam->Setup(pInput->m_aiGroup[1]);
	//-------------------------------------------------
	// Create a MRC stack to store the rendered frames
	//-------------------------------------------------
	memcpy(m_aiStkSize, m_pLoadHeader->m_aiFrmSize, sizeof(int) * 2);
	m_aiStkSize[2] = pFmIntParam->m_iNumIntFms;
	pPackage->m_pRawStack->Create(Mrc::eMrcUChar, m_aiStkSize);
	//-----------------
	int* piCamSize = m_pLoadHeader->m_aiCamSize;
	printf("EER stack: %d  %d  %d\nRendered stack: %d  %d  %d\n\n", 
	   piCamSize[0], piCamSize[1], m_pLoadHeader->m_iNumFrames, 
	   m_aiStkSize[0], m_aiStkSize[1], m_aiStkSize[2]);
	m_bLoaded = true;
}

void CLoadEerMain::mLoadStack(void)
{
	if(!m_bLoaded) return;
	//-----------------
	m_bLoaded = m_pLoadFrames->DoIt(m_iFile, 
	   m_pLoadHeader->m_iNumFrames);
	if(!m_bLoaded) return;
	//-----------------
	CRenderMrcStack aRenderMrcStack;
	aRenderMrcStack.DoIt(m_pLoadHeader, m_pLoadFrames, m_iNthGpu);
	m_bLoaded = true;
}

void CLoadEerMain::mClean(void)
{
	if(m_pLoadHeader != 0L) delete m_pLoadHeader;
	if(m_pLoadFrames != 0L) delete m_pLoadFrames;
	m_pLoadHeader = 0L;
	m_pLoadFrames = 0L;
	//-----------------
	if(m_iFile == -1) return;
	close(m_iFile);
	m_iFile = -1;
}
