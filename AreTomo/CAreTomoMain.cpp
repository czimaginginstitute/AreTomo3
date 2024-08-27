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
	mGenCtfTiles();
	if(pInput->m_iCmd == 0 || pInput->m_iCmd == 1) mDoFull();
	else if(pInput->m_iCmd == 2) mSkipAlign();
	else if(pInput->m_iCmd == 3) mEstimateCtf();
	//-----------------
	FindCtf::CTsTiles::DeleteInstance(m_iNthGpu);
	//-----------------
	MD::CAsyncSaveVol* pSaveVol = 
	   MD::CAsyncSaveVol::GetInstance(m_iNthGpu);
	pSaveVol->WaitForExit(10.0f);
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
	mFindCtf(false);
	mRemoveDarkFrames();
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
	mRemoveDarkCtfs();
	mSetupTsCorrection();
	mSaveForImod();
	//-----------------
	mRecon2nd();	
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
	mSetupTsCorrection();
	mRecon2nd();
	//-----------------
	mCorrectCTF();
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
	mFindCtf(false);
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

void CAreTomoMain::mGenCtfTiles(void)
{
	if(!FindCtf::CFindCtfMain::bCheckInput()) return;
	FindCtf::CTsTiles *pTsTiles = 
	   FindCtf::CTsTiles::GetInstance(m_iNthGpu);
	CAtInput* pAtInput = CAtInput::GetInstance();
	pTsTiles->Generate(pAtInput->m_iCtfTileSize);
}

//--------------------------------------------------------------------
// 1. CFindCtfMain estimates CTFs and saves them into the output
//    directory, but does not save in the Imod directory.
//--------------------------------------------------------------------
void CAreTomoMain::mFindCtf(bool bRefine)
{
	if(!FindCtf::CFindCtfMain::bCheckInput()) return;
	if(!bRefine)
	{	FindCtf::CFindCtfMain findCtfMain;
		findCtfMain.DoIt(m_iNthGpu);
	}
	else 
	{	FindCtf::CRefineCtfMain refineCtfMain;
		refineCtfMain.DoIt(m_iNthGpu);
	}
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
	mFindCtf(true);
	mCalcThickness();
	//-----------------
	Recon::CAlignMetric alignMetric;
	//alignMetric.Calculate(m_iNthGpu, m_iThickness);
	//printf("Coarse align: %e\n", alignMetric.m_fRms);
	
	
	MAM::CAlignParam* pAlignParam = sGetAlignParam(m_iNthGpu);
	pAlignParam->ResetShift();
	//-----------------
	ProjAlign::CParam* pParam = ProjAlign::CParam::GetInstance(m_iNthGpu);
	pParam->m_fXcfSize = 2048.0f;
	mProjAlign();

	//alignMetric.Calculate(m_iNthGpu, m_iThickness);
        //printf("Global align 1: %e\n", alignMetric.m_fRms);
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
	
	//alignMetric.Calculate(m_iNthGpu, m_iThickness);
        //printf("Global align 2: %e\n", alignMetric.m_fRms);
	

	pAlignParam->FitRotCenterZ();
        pAlignParam->RemoveOffsetZ(1.0f);
	//-----------------
	mPatchAlign();

	//alignMetric.Calculate(m_iNthGpu, m_iThickness);
        //printf("Patch align: %e\n", alignMetric.m_fRms);
	//-----------------
	CTsMetrics* pTsMetrics = CTsMetrics::GetInstance();
	pTsMetrics->Save(m_iNthGpu);
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
	//-----------------------------------------------
	// 1) This function is deprecated and replaced
	// by FindCtf/CRefineCtfMain.cpp to estimate
	// the tilt angle offset. 2) Do not delete util
	// CRefineCtfMain is tested with more cases.
	//-----------------------------------------------
	/*
	float fTiltOffset = 0.0f;
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_afTiltCor[0] < 0) return;
	//-----------------
	MAM::CAlignParam* pAlignParam = sGetAlignParam(m_iNthGpu);
	if(fabs(pInput->m_afTiltCor[1]) > 0.1)
        {       fTiltOffset = pInput->m_afTiltCor[1];
                pAlignParam->AddAlphaOffset(fTiltOffset);
		return;
        }
	//-----------------
	TiltOffset::CTiltOffsetMain aTiltOffsetMain;
	aTiltOffsetMain.Setup(4, m_iNthGpu);
	fTiltOffset = aTiltOffsetMain.DoIt();
	//-----------------
	printf("Stretching based tilt offset: %f\n\n", fTiltOffset);
	*/
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
	pPatchAlignMain->DoIt(); 
}

void CAreTomoMain::mCalcThickness(void)
{
	Recon::CCalcVolThick calcVolThick;
        calcVolThick.DoIt(m_iNthGpu);
	float fThickness = calcVolThick.GetThickness(false);
	m_iThickness = (int)fThickness / 2 * 2;
	//-----------------
	CAtInput* pAtInput = CAtInput::GetInstance();
	ProjAlign::CParam* pParam = ProjAlign::CParam::GetInstance(m_iNthGpu);
	int iThickness = m_iThickness * 8 / 20 * 2;
	if(iThickness < 100) iThickness = 100;
	else if(iThickness > 2000) iThickness = 2000;
	//-----------------
	if(pAtInput->m_iAlignZ <= 0) pParam->m_iAlignZ = iThickness;
	else pParam->m_iAlignZ = pAtInput->m_iAlignZ;
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

MD::CTiltSeries* CAreTomoMain::mBinAlnSeries(float fBin)
{
	MD::CTiltSeries* pAlnSeries =
           m_pCorrTomoStack->GetCorrectedStack(false);
	if(pAlnSeries == 0L || pAlnSeries->bEmpty()) return 0L;
	if(fBin < 0.01f) return 0L;
        //-----------------
        /*
        if(iSeries == 0)
        {       MU::CSaveTempMrc saveMrc;
                saveMrc.SetFile("/home/shawn.zheng/szheng/Temp/TestAlnCTF",
                   ".mrc");
                void** ppvImgs = pAlnSeries->GetFrames();
                saveMrc.DoMany(ppvImgs, 2, pAlnSeries->m_aiStkSize);
                printf("GPU %d: Save aln tilt series done.\n\n",
                   m_iNthGpu);
        }
        */
        //-----------------
        MAC::CBinStack binStack;
        MD::CTiltSeries* pBinSeries = 0L;
        pBinSeries = binStack.DoFFT(pAlnSeries, fBin, m_iNthGpu);
	return pBinSeries;
}

void CAreTomoMain::mRecon(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	int iVolZ = pAtInput->m_iVolZ;
	if(iVolZ < 0) iVolZ = m_iThickness + 64;
	iVolZ = (int)(iVolZ / pAtInput->m_afAtBin[0]) / 2 * 2;
	if(iVolZ < 16) iVolZ = 16;
	//-----------------
	bool bWbp = (pAtInput->m_iWbp == 1);
	//-----------------
	int iNumSeries = MD::CAlnSums::m_iNumSums;
	MD::CTiltSeries* pBinnedSeries = 0L;
	//-----------------
	for(int i=0; i<iNumSeries; i++)
	{	m_pCorrTomoStack->DoIt(i, 0L);
		pBinnedSeries = mBinAlnSeries(pAtInput->m_afAtBin[0]);
		mReconVol(pBinnedSeries, iVolZ, i, bWbp);
		if(pBinnedSeries != 0L) delete pBinnedSeries;
	}
	//-----------------
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	m_pCorrTomoStack = 0L;
}

void CAreTomoMain::mRecon2nd(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	float fBin1 = pAtInput->m_afAtBin[1];
	float fBin2 = pAtInput->m_afAtBin[2];
	if(fBin1 < 1 && fBin2 < 1) return;
	//-----------------
	m_pCorrTomoStack->DoIt(0, 0L);
	//-----------------
	int iVolZ = pAtInput->m_iVolZ;
	if(iVolZ < 0) iVolZ = m_iThickness + 64;
	//-----------------
	MD::CTiltSeries* pBinnedSeries = 0L;
	if(fBin1 >= 1)
	{	int iVolZ1 = (int)(iVolZ / fBin1) / 2 * 2;
		pBinnedSeries = mBinAlnSeries(fBin1);
		mReconVol(pBinnedSeries, iVolZ1, 3, true);
		if(pBinnedSeries != 0L) delete pBinnedSeries;
	}
	//-----------------
	if(fBin2 < 1) return;
	int iVolZ2 = (int)(iVolZ / fBin2) / 2 * 2;
	pBinnedSeries = mBinAlnSeries(fBin2);
        mReconVol(pBinnedSeries, iVolZ2, 4, false);
        if(pBinnedSeries != 0L) delete pBinnedSeries;
}

void CAreTomoMain::mReconVol
(	MD::CTiltSeries* pTiltSeries, 
	int iVolZ,
	int iSeries, 
	bool bWbp
)
{	if(iVolZ <= 16) return;
	if(pTiltSeries == 0L) return;
	if(bWbp) mWbpRecon(iVolZ, iSeries, pTiltSeries);
        else mSartRecon(iVolZ, iSeries, pTiltSeries);
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
	MD::CTiltSeries* pNewVol = mFlipVol(pVolStack);
        if(pNewVol != 0L) pVolStack = pNewVol;
	//-----------------
	bool bClean = true;
	mSaveVol(pVolStack, iSeries, bClean);
	if(!bClean && pVolStack != 0L) delete pVolStack;
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
	MD::CTiltSeries* pNewVol = mFlipVol(pVolStack);
	if(pNewVol != 0L) pVolStack = pNewVol;
	//-----------------
	bool bClean = true;
	mSaveVol(pVolStack, iSeries, bClean);
	if(!bClean && pVolStack != 0L) delete pVolStack;
}

void CAreTomoMain::mSaveVol
(	MD::CTiltSeries* pVolSeries, 
	int iNthVol,
	bool bClean
)
{	bool bAsync = true;
	MD::CAsyncSaveVol* pSaveVol = 
	   MD::CAsyncSaveVol::GetInstance(m_iNthGpu);
	pSaveVol->WaitForExit(-1.0f);
	pSaveVol->DoIt(pVolSeries, iNthVol, bAsync, bClean);
}

MD::CTiltSeries* CAreTomoMain::mFlipVol(MD::CTiltSeries* pVolSeries)
{
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_iFlipVol == 0) return 0L;
	//-----------------
	printf("GPU %d: Flip volume from xzy view to xyz view.\n", m_iNthGpu);
	bool bFlip = (pInput->m_iFlipVol == 1) ? true : false;
	MD::CTiltSeries* pVolXYZ = pVolSeries->FlipVol(bFlip);
	printf("GPU %d: Flip volume completed.\n\n", m_iNthGpu);
	//-----------------
	delete pVolSeries;
	return pVolXYZ;
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
