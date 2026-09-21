#include "CAlignInc.h"
#include "../Correct/CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::MotionCor::Align;
namespace MMC = McAreTomo::MotionCor::Correct;

CAlignMain::CAlignMain(void)
{
}

CAlignMain::~CAlignMain(void)
{
}

void CAlignMain::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------------------------------------------------
	// -InMotion 1: replay a .mcaln motion alignment through the
	// correction stages only, skipping all measurement. Falls
	// back to measurement when the file cannot be loaded.
	//-----------------------------------------------------------
	CMcInput* pMcInputMotion = CMcInput::GetInstance();
	if(pMcInputMotion->m_iInMotion != 0)
	{	if(mCorrectFromFile()) return;
		fprintf(stderr, "Warning: -InMotion failed, falling back "
		   "to motion measurement.\n\n");
	}
	//-----------------
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	float fTilt = pMcPackage->m_fTilt;
	//-----------------
	CAlignBase* pAlignBase = 0L;
	CAlignParam* pAlignParam = CAlignParam::GetInstance();
	//------------------------------------------------------------
	// Temporarily diable local motion correction at high tilts.
	//------------------------------------------------------------
	if(pAlignParam->bPatchAlign()) 
	{	printf("Patch based alignment\n");
		pAlignBase = new CPatchAlign;
	}
	else
	{	printf("Full frame alignment\n");
		pAlignBase = new CFullAlign;
	}
	pAlignBase->DoIt(m_iNthGpu);
	//-----------------
	char* pcLogFile = mCreateLogFile();
	if(pcLogFile != 0L)
	{	pAlignBase->LogShift(pcLogFile);
		delete[] pcLogFile;
	}
	//-----------------
	if(pAlignBase != 0L) delete pAlignBase;
}

char* CAlignMain::mCreateLogFile(void)
{	return 0L;	
}

bool CAlignMain::mCorrectFromFile(void)
{
	CLoadAlign aLoadAlign;
	if(!aLoadAlign.DoIt(m_iNthGpu)) return false;
	//-----------------------------------------------------------
	// Buffer prep + forward FFT of the frame stack, exactly as
	// CFullAlign::Align does before any measurement.
	//-----------------------------------------------------------
	CAlignBase aAlignBase;
	aAlignBase.Clean();
	aAlignBase.DoIt(m_iNthGpu);
	//-----------------
	bool bForward = true, bNorm = true;
	CTransformStack aTransformStack;
	aTransformStack.Setup(MD::EBuffer::frm, bForward, bNorm, m_iNthGpu);
	aTransformStack.DoIt();
	//-----------------
	MMD::CPatchShifts* pPatchShifts = aLoadAlign.TakePatchShifts();
	if(pPatchShifts == 0L)
	{	MMD::CStackShift* pFullShift = aLoadAlign.TakeFullShift();
		MMC::CCorrectFullShift aCorrFullShift;
		aCorrFullShift.Setup(pFullShift, m_iNthGpu);
		aCorrFullShift.DoIt();
		delete pFullShift;
	}
	else
	{	// Global correction in place (Fourier phase shift), then
		// the residual local field - the same sequence as the
		// patch path. NO MakeRelative: file shifts are final.
		MMC::CGenRealStack aGenRealStack;
		bool bGenReal = true;
		aGenRealStack.Setup(MD::EBuffer::frm, !bGenReal, m_iNthGpu);
		aGenRealStack.DoIt(pPatchShifts->m_pFullShift);
		//-----------------
		MMC::GCorrectPatchShift aCorrPatchShift;
		aCorrPatchShift.DoIt(pPatchShifts, m_iNthGpu);
		delete pPatchShifts;
	}
	return true;
}

