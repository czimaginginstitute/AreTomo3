#include "CAlignInc.h"
#include "../Correct/CCorrectInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;
namespace MMC = McAreTomo::MotionCor::Correct;

CPatchAlign::CPatchAlign(void)
{
	m_pPatchShifts = 0L;
}

CPatchAlign::~CPatchAlign(void)
{
	if(m_pPatchShifts != 0L) delete m_pPatchShifts;
}

void CPatchAlign::DoIt(int iNthGpu)
{
	if(m_pPatchShifts != 0L)
	{	delete m_pPatchShifts;
		m_pPatchShifts = 0L;
	}
	//-----------------
	CFullAlign::Align(iNthGpu);
	mCorrectFullShift();
	//-----------------
	Util_Time aTimer;
	aTimer.Measure();
	//-----------------
	mCalcPatchShifts();
	float fSeconds = aTimer.GetElapsedSeconds();
	printf("Patch alignment time: %.2f(sec)\n\n", fSeconds);
	//-----------------
	MMC::GCorrectPatchShift corrPatchShift;
	corrPatchShift.DoIt(m_pPatchShifts, m_iNthGpu); 
	//-----------------
	CSaveAlign* pSaveAlign = CSaveAlign::GetInstance(m_iNthGpu);
	pSaveAlign->DoLocal(m_pPatchShifts);
	//-----------------
	mLogShift();
}

void CPatchAlign::mCorrectFullShift(void)
{
	nvtxRangePushA("CPatchAlign::mCorrectFullShift");
	CAlignParam* pAlignParam = CAlignParam::GetInstance();
	int iRefFrame = pAlignParam->GetFrameRef(m_pFullShift->m_iNumFrames);
	m_pFullShift->MakeRelative(iRefFrame);
	//-----------------
	CMcInput* pInput = CMcInput::GetInstance();
	bool bCorrInterp = (pInput->m_iCorrInterp == 0) ? false : true;
	bool bMotionDecon = (pInput->m_iInFmMotion == 0) ? false : true;
	//-----------------
	Util_Time utilTime; utilTime.Measure();
	MMC::CGenRealStack genRealStack;
	genRealStack.Setup(MD::EBuffer::frm, 
	   bCorrInterp, bMotionDecon, m_iNthGpu);
	genRealStack.DoIt(m_pFullShift);
	//-----------------
	printf("Global shifts are corrected: %f sec\n",
	   utilTime.GetElapsedSeconds());
	nvtxRangePop();
}

void CPatchAlign::mCalcPatchShifts(void)
{
	nvtxRangePushA("CPatchAlign::mCalcPatchShifts");
	CAlignParam* pAlignParam = CAlignParam::GetInstance();
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	//-----------------
	m_pPatchShifts = new MMD::CPatchShifts;
	int* piStkSize = pBufferPool->m_aiStkSize;
	m_pPatchShifts->Setup(pAlignParam->GetNumPatches(), piStkSize);
	m_pPatchShifts->SetFullShift(m_pFullShift);
	//-----------------
	printf("Start to align patches.\n");
	CMeasurePatches meaPatches;
	meaPatches.DoIt(m_pPatchShifts, m_iNthGpu);
	m_pFullShift = 0L;
	//-----------------
	nvtxRangePop();
}

void CPatchAlign::LogShift(char* pcLogFile)
{
	m_pPatchShifts->LogFullShifts(pcLogFile);
	m_pPatchShifts->LogPatchShifts(pcLogFile);
	m_pPatchShifts->LogFrameShifts(pcLogFile);
}

void CPatchAlign::mLogShift(void)
{
	MD::CLogFiles* pLogFiles = MD::CLogFiles::GetInstance(m_iNthGpu);
	FILE* pFile = pLogFiles->m_pMcLocalLog;
	if(pFile == 0L) return;
	//-----------------
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	int iAcqIdx = pMcPackage->m_iAcqIdx;
	float fTilt = pMcPackage->m_fTilt;
	float fPixSize = pMcPackage->m_fPixSize;
	//-----------------
	for(int p=0; p<m_pPatchShifts->m_iNumPatches; p++)
	{	float afPatCent[2] = {0};
		m_pPatchShifts->GetPatchCenter(p, afPatCent);
		//----------------
		for(int f=0; f<m_pPatchShifts->m_aiFullSize[2]; f++)
		{	float afShift[2] = {0.0f};
			m_pPatchShifts->GetLocalShift(f, p, afShift);
			//---------------
			fprintf(pFile, "%3d %7.2f %6.2f %3d %3d "
			   "%7.1f %7.1f %8.2f %8.2f\n", iAcqIdx, fTilt, fPixSize,
			   p, f, afPatCent[0], afPatCent[1],
			   afShift[0], afShift[1]);
		}
	}
	fflush(pFile);
}

