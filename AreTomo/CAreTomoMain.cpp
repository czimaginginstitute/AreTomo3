#include "CAreTomoInc.h"
#include "Massnorm/CMassNormInc.h"
#include "StreAlign/CStreAlignInc.h"
#include "ProjAlign/CProjAlignInc.h"
#include "CommonLine/CCommonLineInc.h"
#include "TiltOffset/CTiltOffsetInc.h"
#include "PatchAlign/CPatchAlignInc.h"
#include "Correct/CCorrectInc.h"
#include "Recon/CReconInc.h"
#include "DoseWeight/CDoseWeightInc.h"
#include "FindCtf/CFindCtfInc.h"
#include "ImodUtil/CImodUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <Util/Util_Time.h>

using namespace McAreTomo::AreTomo;

static MAM::CAlignParam* sGetAlignParam(int iNthGpu)
{
	MAM::CAlignParam* pParam = MAM::CAlignParam::GetInstance(iNthGpu);
	return pParam;
}

static MAM::CLocalAlignParam* sGetLocalParam(int iNthGpu)
{
	MAM::CLocalAlignParam* pParam = 
	   MAM::CLocalAlignParam::GetInstance(iNthGpu);
	return pParam;
}

static MD::CTiltSeries* sGetTiltSeries(int iNthGpu, int iSeries)
{
	MD::CTsPackage* pPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pSeries = pPkg->GetSeries(iSeries);
	return pSeries;
}

CAreTomoMain::CAreTomoMain(void)
{
	m_pCorrTomoStack = 0L;
}

CAreTomoMain::~CAreTomoMain(void)
{
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
}

//--------------------------------------------------------------------
// m_iCmd = 0: full processing that starts from motion correction.
// m_iCmd = 1: skip motion correction, starts from tomo alignment.
// m_iCmd = 2: do tomo reconstruction only.
// m_iCmd = 3: do CTF estimation only.
//--------------------------------------------------------------------
bool CAreTomoMain::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iCmd == 0 || pInput->m_iCmd == 1) mDoFull();
	else if(pInput->m_iCmd == 2) mSkipAlign();
	else if(pInput->m_iCmd == 3) mEstimateCtf();
	//-----------------
	printf("Process thread exits.\n\n");
	return true;
}

void CAreTomoMain::mDoFull(void)
{
	//-----------------------------------------------
	// 1) This runs on the full tilt series. 2) In 
	// the future, refinement will be performed on 
	// dark removed tilt series to determine alpha 
	// and beta tilt offset.
	//-----------------------------------------------
	mFindCtf();
	mRemoveDarkFrames();
	mRemoveDarkCtfs();
	//-----------------
	mCreateAlnParams();
	mRemoveSpikes();	
	mMassNorm();
	//-----------------
	mAlign();
	mSaveAlignment();
	//-----------------------------------------------
	// 1) -OutImod 3 saves the aligned tilt series.
	// 2) Hence we need to configure Tilt series
	// correction before saving files to Imod
	// sub-directory.
	//-----------------------------------------------
	mSetupTsCorrection();
	mSaveForImod();
	//-----------------	
	mCorrectCTF();
	mRecon();
	//-----------------------------------------------
	// 1) If -OutImod 3 is specified, we save the
	// aligned tilt series and aligned CTF results.
	// 2) mAlignCTF aligns CTF results and saves
	// to IMOD sub-directory.
	//-----------------------------------------------
	mAlignCTF();
}	

//--------------------------------------------------------------------
// 1) This implements CInput::m_iCmd = 2 that performs tomographic
//    reconstruction only.
// 2) This workflow includes local CTF correction.
// 3) This workflow needs to load aln file and CTF estimation file. 
//--------------------------------------------------------------------
void CAreTomoMain::mSkipAlign(void)
{
	MAM::CRemoveDarkFrames remDarkFrames;
        remDarkFrames.Setup(m_iNthGpu);
	//---------------------------------------------------------
	// 1) When loading alignment file is successful, the dark
	// frames are saved in CDarkFrames. 2) CAlignParam and
	// CLocalAlignParam objects are created. 3) We need to
	// remove dark frames from tilt series, and corresponding
	// entries in CCtfResults.
	//---------------------------------------------------------
	MAM::CLoadAlignFile loadAlnFile;
	bool bLoaded = loadAlnFile.DoIt(m_iNthGpu);
	if(!bLoaded) return;
	//-----------------
	FindCtf::CLoadCtfResults loadCtfResults;
	loadCtfResults.DoIt(m_iNthGpu);
	//-----------------
	remDarkFrames.Remove();
	mRemoveDarkCtfs();
	//-----------------
	mRemoveSpikes();
	mMassNorm();
	//-----------------
	mCorrectCTF();
	mSetupTsCorrection();
	mRecon();
}

//--------------------------------------------------------------------
// 1) This implements CInput::m_iCmd = 3 that repeats CTF estimation.
// 2) This workflow needs to load aln file for removing dark entries
//    in the generated CTF files and aligning CTF entries if users
//    previously choose -OutImod 3.
//--------------------------------------------------------------------
void CAreTomoMain::mEstimateCtf(void)
{
	MAM::CRemoveDarkFrames remDarkFrames;
        remDarkFrames.Setup(m_iNthGpu);
	//-----------------
	MAM::CLoadAlignFile loadAlnFile;
        bool bLoaded = loadAlnFile.DoIt(m_iNthGpu);
        if(!bLoaded) return;
	//-----------------
	mFindCtf();
	mRemoveDarkCtfs();
	//-----------------
	ImodUtil::CImodUtil* pImodUtil = 0L;
	pImodUtil = ImodUtil::CImodUtil::GetInstance(m_iNthGpu);
	int iOutImod = pImodUtil->FindOutImodVal();
	//-----------------
	CAtInput* pAtInput = CAtInput::GetInstance();
	pAtInput->m_iOutImod = iOutImod;
	mSaveForImod();
	//-----------------
	mAlignCTF();	
}

void CAreTomoMain::mRemoveDarkFrames(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	MAM::CRemoveDarkFrames remDarkFrames;
	remDarkFrames.Setup(m_iNthGpu);
	remDarkFrames.Detect(pAtInput->m_fDarkTol);
	remDarkFrames.Remove();
}

void CAreTomoMain::mRemoveDarkCtfs(void)
{
	MD::CCtfResults* pCtfResults = 
	   MD::CCtfResults::GetInstance(m_iNthGpu);
	pCtfResults->RemoveDarkCTFs();
}

void CAreTomoMain::mCreateAlnParams(void)
{
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pRawSeries = pPackage->GetSeries(0);
	//-----------------
	MAM::CAlignParam* pAlignParam = sGetAlignParam(m_iNthGpu); 
	pAlignParam->Create(pRawSeries->m_aiStkSize[2]);
	for(int i=0; i<pRawSeries->m_aiStkSize[2]; i++)
	{	pAlignParam->SetTilt(i, pRawSeries->m_pfTilts[i]);
	}
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
	if(!pInput->bLocalAlign()) return;
	int iNumPatches = pInput->GetNumPatches();
	//-----------------
	MAM::CLocalAlignParam* pLocalParam = sGetLocalParam(m_iNthGpu);
	pLocalParam->Setup(pRawSeries->m_aiStkSize[2], iNumPatches);
}

void CAreTomoMain::mRemoveSpikes(void)
{
	MD::CTsPackage* pPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pRawSeries = pPkg->GetSeries(0);
	MAJ::CRemoveSpikes remSpikes;
	remSpikes.DoIt(pRawSeries);	
}

//--------------------------------------------------------------------
// 1. CFindCtfMain estimates CTFs and saves them into the output
//    directory, but does not save in the Imod directory.
//--------------------------------------------------------------------
void CAreTomoMain::mFindCtf(void)
{
	if(!FindCtf::CFindCtfMain::bCheckInput()) return;
	FindCtf::CFindCtfMain findCtfMain;
	findCtfMain.DoIt(m_iNthGpu);
}

void CAreTomoMain::mMassNorm(void)
{
	MassNorm::CLinearNorm linearNorm;
	linearNorm.DoIt(m_iNthGpu);
}

void CAreTomoMain::mAlign(void)
{
	m_fRotScore = 0.0f;
	mCoarseAlign();
	//-----------------
	MAM::CAlignParam* pAlignParam = sGetAlignParam(m_iNthGpu);
	pAlignParam->ResetShift();
	//-----------------
	ProjAlign::CParam* pParam = ProjAlign::CParam::GetInstance(m_iNthGpu);
	pParam->m_fXcfSize = 2048.0f;
	mProjAlign();
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_afTiltAxis[1] >= 0)
	{	float fRange = (pInput->m_afTiltAxis[0] == 0) ? 20.0f : 6.0f;
		int iIters = (pInput->m_afTiltAxis[0] == 0) ? 4 : 2;
		for(int i=1; i<=iIters; i++) 
		{	mRotAlign(fRange/i, 100);
			if(i == 1) mProjAlign();
		}
		mProjAlign();
	}
	//-----------------
	mPatchAlign();
	//-----------------
	if(pInput->m_afTiltCor[0] == 0) 
	{	pAlignParam->AddAlphaOffset(-m_fTiltOffset);
	}
	else
	{	MAM::CDarkFrames* pDarkFrames = 
		   MAM::CDarkFrames::GetInstance(m_iNthGpu);
		pDarkFrames->AddTiltOffset(m_fTiltOffset);
	}
	//-----------------
	mLogGlobalShift();
	mLogLocalShift();
}

void CAreTomoMain::mCoarseAlign(void)
{
	MAS::CStreAlignMain streAlignMain;
	streAlignMain.Setup(m_iNthGpu);
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_afTiltAxis[0] == 0)
	{	for(int i=1; i<=3; i++)
		{	streAlignMain.DoIt();
			mRotAlign(180.0f / i, 100);
		}
		mFindTiltOffset();
		return;
	}
	//-----------------
	MAM::CAlignParam* pAlignParam = sGetAlignParam(m_iNthGpu);
	pAlignParam->SetTiltAxisAll(pInput->m_afTiltAxis[0]);
	//-----------------
	if(pInput->m_afTiltAxis[1] < 0)
	{	streAlignMain.DoIt();
		streAlignMain.DoIt();
	}
	else
	{	for(int i=1; i<=2; i++)
		{	streAlignMain.DoIt();
			mRotAlign(10.0f / i, 100);
		}
	}	
        mFindTiltOffset();
}

void CAreTomoMain::mProjAlign(void)
{
	CAtInput* pInput = CAtInput::GetInstance();
	ProjAlign::CParam* pParam = ProjAlign::CParam::GetInstance(m_iNthGpu);
	pParam->m_iVolZ = pInput->m_iAlignZ;
        pParam->m_afMaskSize[0] = 0.7f;
        pParam->m_afMaskSize[1] = 0.7f;
	//-----------------
	MAM::CAlignParam* pAlignParam = sGetAlignParam(m_iNthGpu);	
	ProjAlign::CProjAlignMain aProjAlign;
	aProjAlign.Set0(500, m_iNthGpu);
	aProjAlign.Set1(pParam);
	float fLastErr = aProjAlign.DoIt(pAlignParam);
	MrcUtil::CAlignParam* pLastParam = pAlignParam->GetCopy();
	//-----------------
	int iIterations = 1; //10;
	int iLastIter = iIterations - 1;
	pParam->m_afMaskSize[0] = 0.55f;
	pParam->m_afMaskSize[1] = 0.55f;
	//-----------------
	for(int i=1; i<iIterations; i++)
	{	float fErr = aProjAlign.DoIt(pAlignParam);
		if(fErr < 2.0f) break;
		//--------------------
		if(fErr <= fLastErr)
		{	fLastErr = fErr;
			pLastParam->Set(pAlignParam);
		}
		else
		{	pAlignParam->Set(pLastParam);
			break;
		}
	}
	delete pLastParam;
}

void CAreTomoMain::mRotAlign(float fAngRange, int iNumSteps)
{
	CommonLine::CCommonLineMain clMain;
	clMain.DoInitial(m_iNthGpu, fAngRange, iNumSteps);
}

void CAreTomoMain::mRotAlign(void)
{
        CommonLine::CCommonLineMain clMain;
        m_fRotScore = clMain.DoRefine(m_iNthGpu);
	printf("Rotation align score: %f\n\n", m_fRotScore);
}

void CAreTomoMain::mFindTiltOffset(void)
{
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_afTiltCor[0] < 0) return;
	//-----------------
	MAM::CAlignParam* pAlignParam = sGetAlignParam(m_iNthGpu);
	if(fabs(pInput->m_afTiltCor[1]) > 0.1)
        {       m_fTiltOffset = pInput->m_afTiltCor[1];
                pAlignParam->AddAlphaOffset(m_fTiltOffset);
		return;
        }
	//-----------------
	TiltOffset::CTiltOffsetMain aTiltOffsetMain;
	aTiltOffsetMain.Setup(4, m_iNthGpu);
	float fTiltOffset = aTiltOffsetMain.DoIt();
	m_fTiltOffset += fTiltOffset;
	//-----------------
	pAlignParam->AddAlphaOffset(fTiltOffset);
}

void CAreTomoMain::mPatchAlign(void)
{
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->GetNumPatches() == 0) return;
	//-----------------
	MAP::CPatchTargets* pPatchTgts = 
     	   MAP::CPatchTargets::GetInstance(m_iNthGpu);
	pPatchTgts->Detect();
	if(pPatchTgts->m_iNumTgts < 4) return;
	//-----------------
	MAP::CPatchAlignMain* pPatchAlignMain = 
	   MAP::CPatchAlignMain::GetInstance(m_iNthGpu);
	pPatchAlignMain->DoIt(m_fTiltOffset); 
}

void CAreTomoMain::mCorrectCTF(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	if(pAtInput->m_aiCorrCTF[0] == 0) return;
	//-----------------
	MD::CCtfResults* pCtfResults = 
	   MD::CCtfResults::GetInstance(m_iNthGpu);
	if(!pCtfResults->bHasCTF()) return;
	//-----------------
	printf("GPU %d: Tilt series CTF deconvolution\n\n", m_iNthGpu);
	bool bPhaseFlip = false;
	if(pAtInput->m_aiCorrCTF[0] == 2) bPhaseFlip = true;
	//-----------------
	MAF::CCorrCtfMain corrCtfMain;
	corrCtfMain.DoIt(m_iNthGpu, bPhaseFlip, pAtInput->m_aiCorrCTF[1]);
}

void CAreTomoMain::mSetupTsCorrection(void)
{
	//---------------------------------------------------------
	// The aligned tilt series is buffered in m_pCorrTomoStack
	// and can be retrieved by calling
	// m_pCorrTomoStack->GetCorrectedStack(bool bClean)
	//---------------------------------------------------------
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	m_pCorrTomoStack = new Correct::CCorrTomoStack;
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
        bool bIntpCor = pInput->m_bIntpCor;
        bool bShiftOnly = true, bRandFill = true;
        bool bFFTCrop = true, bRWeight = true;
	//-----------------
	MAM::CAlignParam* pAlignParam = sGetAlignParam(m_iNthGpu);
	MAM::CLocalAlignParam* pLocalParam = sGetLocalParam(m_iNthGpu);
        float fTiltAxis = pAlignParam->GetTiltAxis(0);
	//-----------------
	m_pCorrTomoStack->Set0(m_iNthGpu);
	m_pCorrTomoStack->Set1(pLocalParam->m_iNumPatches, fTiltAxis);
	m_pCorrTomoStack->Set2(1.0f, bFFTCrop, bRandFill);
	m_pCorrTomoStack->Set3(!bShiftOnly, bIntpCor, !bRWeight);
	m_pCorrTomoStack->Set4(true);
}

//--------------------------------------------------------------------
// 1. Save tilt series in the Imod sub-directory.
//--------------------------------------------------------------------
void CAreTomoMain::mSaveForImod(void)
{
	CInput* pInput = CInput::GetInstance();
	CAtInput* pAtInput = CAtInput::GetInstance();
	ImodUtil::CImodUtil* pImodUtil = 0L;
        pImodUtil = ImodUtil::CImodUtil::GetInstance(m_iNthGpu);
	if(pAtInput->m_iOutImod <= 0) return;
	//--------------------------------------------------
	// CreateFolder is skipped if the folder exists.
	//--------------------------------------------------
	pImodUtil->CreateFolder();
	MD::CTiltSeries* pSeries = sGetTiltSeries(m_iNthGpu, 0);
	//--------------------------------------------------
	// CTF estimation is performed for m_iCmd = 0 and 1
	//--------------------------------------------------
	if(pInput->m_iCmd == 0 || pInput->m_iCmd == 1)
	{	if(pAtInput->m_iOutImod == 1) // for Relion 4
		{	pImodUtil->SaveTiltSeries(0L);
		}
		else if(pAtInput->m_iOutImod == 2) // for warp
		{	pImodUtil->SaveTiltSeries(pSeries);
			pImodUtil->SaveCtfFile();
		}
		//-----------------------------------------------
		// 1) This is for aligned tilt series not CTF
		// corrected. 2) We do not save the CTF results
		// into the IMOD sub-folder here since they are
		// not aligned yet to match the aligned tilt
		// series. 3) The aligned CTF results will be
		// created after CTF correction is done on raw
		// and dark-removed tilt series.
		//-----------------------------------------------
		else if(pAtInput->m_iOutImod == 3)
		{	m_pCorrTomoStack->DoIt(0, 0L);
			MD::CTiltSeries* pAlnSeries = 
		   	   m_pCorrTomoStack->GetCorrectedStack(false);
			pImodUtil->SaveTiltSeries(pAlnSeries);
		}
	}
	//--------------------------------------------------
	// 1) Perform tomographic reconstruction only.
	// 2) Nothing else needs to update in Imod folder.
	//--------------------------------------------------
	else if(pInput->m_iCmd == 2)
	{	// do not update Imod subfolder	
	}
	//--------------------------------------------------
	// 1) Since CTF estimation is repeated, needs to
	// update the corresponding files in Imod sub
	// when the last processing uses -OutImod 2.
	// 2) When last processing uses -OutImod 3, we nned
	// to re-align CTF by calling mAlignCTF.
	//--------------------------------------------------
	else if(pInput->m_iCmd == 3)
	{	if(pAtInput->m_iOutImod == 2)
		{	pImodUtil->SaveCtfFile();
		}
	}
}

void CAreTomoMain::mAlignCTF(void)
{
	CInput* pInput = CInput::GetInstance();
	CAtInput* pAtInput = CAtInput::GetInstance();
	if(pAtInput->m_iOutImod != 3) return;
	//-----------------
	FindCtf::CAlignCtfResults alignCtfResults;
	alignCtfResults.DoIt(m_iNthGpu);
	//-----------------
	ImodUtil::CImodUtil* pImodUtil = 0L;
	pImodUtil = ImodUtil::CImodUtil::GetInstance(m_iNthGpu);
	pImodUtil->SaveCtfFile();
}

void CAreTomoMain::mRecon(void)
{
	int iNumSeries = MD::CAlnSums::m_iNumSums;
	for(int i=0; i<iNumSeries; i++)
	{	m_pCorrTomoStack->DoIt(i, 0L);
		mReconSeries(i);
	}
	//-----------------
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	m_pCorrTomoStack = 0L;
}

void CAreTomoMain::mReconSeries(int iSeries)
{
	CAtInput* pInput = CAtInput::GetInstance();
	int iVolZ = (int)(pInput->m_iVolZ / pInput->m_fAtBin) / 2 * 2;
	if(iVolZ <= 16) return;
	//-----------------
	MD::CTiltSeries* pAlnSeries = 
	   m_pCorrTomoStack->GetCorrectedStack(false);
	if(pAlnSeries == 0L || pAlnSeries->bEmpty()) return;
	//-----------------
	MAC::CBinStack binStack;
	MD::CTiltSeries* pBinSeries = binStack.DoFFT(pAlnSeries,
	   pInput->m_fAtBin, m_iNthGpu);
	//-----------------
	if(pInput->m_iWbp != 0) mWbpRecon(iVolZ, iSeries, pBinSeries);
	else mSartRecon(iVolZ, iSeries, pBinSeries);
	//-----------------
	if(pBinSeries != 0L) delete pBinSeries;
}

//--------------------------------------------------------------------
// 1. Missing setting positivity.
//--------------------------------------------------------------------
void CAreTomoMain::mSartRecon
(	int iVolZ, int iSeries, 
	MD::CTiltSeries* pSeries
)
{	CAtInput* pInput = CAtInput::GetInstance();
	MAM::CAlignParam* pAlnParam = sGetAlignParam(m_iNthGpu);
	//-----------------
	int iStartTilt = pAlnParam->GetFrameIdxFromTilt(
	   pInput->m_afReconRange[0]);
	int iEndTilt = pAlnParam->GetFrameIdxFromTilt(
	   pInput->m_afReconRange[1]);
	if(iStartTilt == iEndTilt) return;
	//-----------------
	int iIters = pInput->m_aiSartParam[0];
	int iNumTilts = iEndTilt - iStartTilt + 1;
	int iNumSubsets = iNumTilts / pInput->m_aiSartParam[1];
	if(iNumSubsets < 1) iNumSubsets = 1;
	//-----------------
	printf("GPU %d: start SART reconstruction...\n", m_iNthGpu);
	Util_Time aTimer;
	aTimer.Measure();
	//-----------------
	Recon::CDoSartRecon doSartRecon;
	MD::CTiltSeries* pVolStack = doSartRecon.DoIt(pSeries, 
	   pAlnParam, iStartTilt, iNumTilts, iVolZ, iIters, iNumSubsets);
	pVolStack->m_fPixSize = pSeries->m_fPixSize;
	//-----------------
	printf("GPU %d: SART Recon: %.2f sec\n\n", m_iNthGpu, 
	   aTimer.GetElapsedSeconds());
	//-----------------
	pVolStack = mFlipVol(pVolStack);
	//-----------------
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	pTsPackage->SaveVol(pVolStack, iSeries);
	//-----------------
	if(pVolStack != 0L) delete pVolStack;
}


void CAreTomoMain::mWbpRecon
(	int iVolZ, int iSeries,
	MD::CTiltSeries* pSeries
)
{	printf("GPU %d: start WBP reconstruction...\n", m_iNthGpu);
	CAtInput* pInput = CAtInput::GetInstance();
	//-----------------
	Util_Time aTimer;
	aTimer.Measure();
	//-----------------
	MAM::CAlignParam* pAlnParam = sGetAlignParam(m_iNthGpu);
	//-----------------
	Recon::CDoWbpRecon doWbpRecon;
	MD::CTiltSeries* pVolStack = doWbpRecon.DoIt(pSeries, 
	   pAlnParam, iVolZ);
	pVolStack->m_fPixSize = pSeries->m_fPixSize;
	printf("GPU %d: WBP Recon: %.2f sec\n\n", m_iNthGpu,
	   aTimer.GetElapsedSeconds());
	//-----------------
	pVolStack = mFlipVol(pVolStack);
	//-----------------
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
        pTsPackage->SaveVol(pVolStack, iSeries);
	//-----------------
	if(pVolStack != 0L) delete pVolStack;
}

MD::CTiltSeries* CAreTomoMain::mFlipVol(MD::CTiltSeries* pVolSeries)
{
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_iFlipVol == 0) return pVolSeries;
	if(pVolSeries == 0L) return 0L;
	//-----------------
	printf("GPU %d: Flip volume from xzy view to xyz view.\n", m_iNthGpu);
	bool bFlip = (pInput->m_iFlipVol == 1) ? true : false;
	MD::CTiltSeries* pVolXZY = pVolSeries->FlipVol(bFlip);
	printf("GPU %d: Flip volume completed.\n\n", m_iNthGpu);
	//-----------------
	delete pVolSeries;
	return pVolXZY;
}

void CAreTomoMain::mSaveAlignment(void)
{
	MrcUtil::CSaveAlignFile saveAlignFile;
	saveAlignFile.DoIt(m_iNthGpu); 
}

void CAreTomoMain::mLogGlobalShift(void)
{
	MD::CLogFiles* pLogFiles = MD::CLogFiles::GetInstance(m_iNthGpu);
	FILE* pFile = pLogFiles->m_pAtGlobalLog;
	if(pFile == 0L) return;
	//-----------------
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pRawSeries = pTsPackage->GetSeries(0);
	MAM::CAlignParam* pAlnParam = sGetAlignParam(m_iNthGpu);
	//-----------------
	float fPixSize = pRawSeries->m_fPixSize;
	float fImgDose = pRawSeries->m_fImgDose;
	float afShift[2] = {0.0f}, fTiltAxis = 0.0f;
	//-----------------
	for(int i=0; i<pRawSeries->m_aiStkSize[2]; i++)
	{	pAlnParam->GetShift(i, afShift);
		fTiltAxis = pAlnParam->GetTiltAxis(i);
		float fTilt = pRawSeries->m_pfTilts[i];
		int iAcqIdx = pRawSeries->m_piAcqIndices[i];
		float fDose = iAcqIdx * fImgDose;
		//----------------
		fprintf(pFile, "%3d %3d %7.2f %6.2f "
		   "%7.2f %7.2f %8.2f %8.2f\n",
		   i, iAcqIdx, fTilt, fPixSize, 
		   fDose, fTiltAxis, afShift[0], afShift[1]);
	}
	fflush(pFile);
}

void CAreTomoMain::mLogLocalShift(void)
{
	MD::CLogFiles* pLogFiles = MD::CLogFiles::GetInstance(m_iNthGpu);
	FILE* pFile = pLogFiles->m_pAtLocalLog;
	if(pFile == 0L) return;
	//-----------------
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pRawSeries = pTsPackage->GetSeries(0);
	float fPixSize = pRawSeries->m_fPixSize;
	//-----------------
	MAM::CLocalAlignParam* pLocalParam = 
	   MAM::CLocalAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	float afCoord[2] = {0.0f}, afShift[2] = {0.0f};
	float fGoodShift = 0.0f, fTilt = 0.0f, fTiltAxis = 0.0f;
	//-----------------
	for(int t=0; t<pLocalParam->m_iNumTilts; t++)
	{	int iAcqIdx = pRawSeries->m_piAcqIndices[t];
		fTilt = pRawSeries->m_pfTilts[t];
		//----------------
		for(int p=0; p<pLocalParam->m_iNumPatches; p++)
		{	pLocalParam->GetCoordXY(t, p, afCoord);
			pLocalParam->GetShift(t, p, afShift);
			fGoodShift = pLocalParam->GetGood(t, p);
			//---------------
			fprintf(pFile, "%3d %3d %3d ", t, p, iAcqIdx);
			fprintf(pFile, "%7.2f %6.2f ", fTilt, fPixSize);
			fprintf(pFile, "%8.2f %8.2f ", afCoord[0], afCoord[1]);
			fprintf(pFile, "%7.2f %8.2f ", afShift[0], afShift[1]);
			fprintf(pFile, "%4.1f\n", fGoodShift);
		}
	}
	fflush(pFile);	
}
