#include "CAlignInc.h"
#include "../Correct/CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <Util/Util_Time.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;
namespace MMC = McAreTomo::MotionCor::Correct;

CFullAlign::CFullAlign(void)
{
}

CFullAlign::~CFullAlign(void)
{
}

void CFullAlign::Align(int iNthGpu)
{
	nvtxRangePushA ("AlignBase Clean+DoIt");
	CAlignBase::Clean();
	CAlignBase::DoIt(iNthGpu);
	nvtxRangePop();
	//-----------------
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(iNthGpu);
	MD::CStackBuffer* pFrmBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::frm);	
	//-----------------
	bool bForward = true;
	nvtxRangePushA ("mFourierTransform(bForward)");
	mFourierTransform(bForward);
	nvtxRangePop();
	//-----------------
	m_pFullShift = new MMD::CStackShift;
	m_pFullShift->Setup(pFrmBuffer->m_iNumFrames);
	//-----------------
	CGenXcfStack genXcfStack;
	genXcfStack.DoIt(m_pFullShift, iNthGpu);
	//-----------------
	mDoAlign();
	m_pFullShift->TruncateDecimal();
	m_pFullShift->DisplayShift("Full-frame alignment shift", 0);
}

void CFullAlign::DoIt(int iNthGpu)
{	
	this->Align(iNthGpu);
	mCorrect();
	//---------
	CSaveAlign* pSaveAlign = CSaveAlign::GetInstance(iNthGpu);
	pSaveAlign->DoGlobal(m_pFullShift);
}

void CFullAlign::mFourierTransform(bool bForward)
{
	const char* pcForward = "Forward FFT of stack";
	const char* pcInverse = "Inverse FFT of stack";
	if(bForward) printf("%s, please wait...\n", pcForward);
	else printf("%s, please wait...\n", pcInverse);
	//---------------------------------------------
	bool bNorm = true;
	CTransformStack transformStack;
	transformStack.Setup(MD::EBuffer::frm, bForward, bNorm, m_iNthGpu);
	transformStack.DoIt();
	printf("Fourier transform done.\n");
}

void CFullAlign::mDoAlign(void)
{
	printf("Full-frame alignment has been started.\n");
	CMcInput* pMcInput = CMcInput::GetInstance();
	CIterativeAlign iterAlign;
	//-----------------
	iterAlign.Setup(MD::EBuffer::xcf, m_iNthGpu);
	iterAlign.DoIt(m_pFullShift);
	//------------------
	char* pcErrLog = iterAlign.GetErrorLog();
	printf("%s\n", pcErrLog);
	if(pcErrLog != 0L) delete[] pcErrLog;
	//-----------------
	CAlignParam* pAlignParam = CAlignParam::GetInstance();
	int iFrmRef = pAlignParam->GetFrameRef(m_pFullShift->m_iNumFrames);
	//----------------------------------------------------------
	// For patch alignment, we need to detect patches that have
	// features for reliable alignment.
	//----------------------------------------------------------
	int* piNumPatches = pMcInput->m_aiNumPatches;
	if(piNumPatches[0] > 1 && piNumPatches[1] > 1)
	{	CDetectFeatures* pDetectFeatures = 
		   CDetectFeatures::GetInstance(m_iNthGpu);
		pDetectFeatures->DoIt(m_pFullShift, piNumPatches);
		//----------------
		CPatchCenters* pPatchCenters = 
		   CPatchCenters::GetInstance(m_iNthGpu);
		pPatchCenters->Calculate();
		//--------------------------------------------------
		// We need to transform back xcf stack back to real
		// space to extract patches.
		//--------------------------------------------------
		bool bGenReal = true;
		MMC::CGenRealStack genRealStack;
		genRealStack.Setup(MD::EBuffer::xcf, bGenReal, m_iNthGpu); 
		genRealStack.DoIt(m_pFullShift);
	}
	//-----------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	float* pfXcfBin = pBufferPool->m_afXcfBin;	
	m_pFullShift->Multiply(pfXcfBin[0], pfXcfBin[1]);
	//-----------------
	mLogShift();
}

void CFullAlign::mCorrect(void)
{
	printf("Create aligned sum based upon full frame alignment.\n");
	MMC::CCorrectFullShift corrFullShift;
	corrFullShift.Setup(m_pFullShift, m_iNthGpu);
	corrFullShift.DoIt();
}

void CFullAlign::mLogShift(void)
{
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	int iAcqIdx = pMcPackage->m_iAcqIdx;
	float fTilt = pMcPackage->m_fTilt;
	float fPixSize = pMcPackage->m_fPixSize;
	//-----------------
	MD::CLogFiles* pLogFiles = MD::CLogFiles::GetInstance(m_iNthGpu);
	FILE* pFile = pLogFiles->m_pMcGlobalLog;
	if(pFile == 0L) return;
	//-----------------
	float afShift0[2] = {0.0f}, afShift[2] = {0.0f};
	m_pFullShift->GetShift(0, afShift0);
	//-----------------
	for(int i=0; i<m_pFullShift->m_iNumFrames; i++)
	{	m_pFullShift->GetShift(i, afShift);
		float fSx = afShift[0] - afShift0[0];
		float fSy = afShift[1] - afShift0[1];
		//----------------
		fprintf(pFile, "%3d %3d %7.2f %6.2f %8.2f %8.2f\n",
		   i, iAcqIdx, fTilt, fPixSize, fSx, fSy); 
	}
	fflush(pFile);
}

