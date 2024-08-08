#include "CProjAlignInc.h"
#include "../CAreTomoInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::ProjAlign;

CProjAlignMain::CProjAlignMain(void)
{
	m_pfReproj = 0L;
	m_pbSkipProjs = 0L;
	m_pCorrTomoStack = 0L;
	m_pCorrProj = 0L;
	m_iBin = 0;
	m_iNthGpu = -1;
	m_pcLog = 0L;
	m_bLocal = false;
}

CProjAlignMain::~CProjAlignMain(void)
{
	this->Clean();
}

void CProjAlignMain::Clean(void)
{
	if(m_pfReproj != 0L) cudaFreeHost(m_pfReproj);
	if(m_pbSkipProjs != 0L) delete[] m_pbSkipProjs;
	if(m_pCorrTomoStack != 0L) delete m_pCorrTomoStack;
	if(m_pCorrProj != 0L) delete m_pCorrProj;
	if(m_pcLog != 0L) delete[] m_pcLog;
	m_pfReproj = 0L;
	m_pbSkipProjs = 0L;
	m_pCorrTomoStack = 0L;
	m_pCorrProj = 0L;
	m_pcLog = 0L;
	//-----------------
	m_centralXcf.Clean();
	m_iBin = 0;
	m_iNthGpu = -1;
}

void CProjAlignMain::Set0(float fBFactor, int iNthGpu)
{	
	this->Clean();
	m_fBFactor = fBFactor;
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	m_pTiltSeries = pPackage->GetSeries(0);
	m_iNumProjs = m_pTiltSeries->m_aiStkSize[2];
	//-----------------
	m_pCorrTomoStack = new Correct::CCorrTomoStack;
        m_pCorrTomoStack->Set0(m_iNthGpu);
	//-----------------
	m_pbSkipProjs = new bool[m_iNumProjs];
	//-----------------
	m_pCorrProj = new MAC::CCorrProj;
}

void CProjAlignMain::Set1(CParam* pParam)
{
	int iBinX = (int)(m_pTiltSeries->m_aiStkSize[0] 
	   / pParam->m_fXcfSize + 0.5f);
	int iBinY = (int)(m_pTiltSeries->m_aiStkSize[1] 
	   / pParam->m_fXcfSize + 0.5f);
	m_iBin = (iBinX > iBinY) ? iBinX : iBinY;
	if(m_iBin < 1) m_iBin = 1;
	//-----------------
	bool bShiftOnly = true, bRandomFill = true;
	bool bFourierCrop = true, bRWeight = true;
	//-----------------
	MAM::CAlignParam* pAlignParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	float fTiltAxis = pAlignParam->GetTiltAxis(m_iNumProjs / 2);
	//-----------------
	m_pCorrTomoStack->Set1(0, fTiltAxis);
	m_pCorrTomoStack->Set2((float)m_iBin, !bFourierCrop, bRandomFill);
	m_pCorrTomoStack->Set3(!bShiftOnly, false, !bRWeight);
	m_pBinSeries = m_pCorrTomoStack->GetCorrectedStack(false);
	//-----------------
	m_iVolZ = pParam->m_iAlignZ / m_iBin / 2 * 2;
	//-----------------
	int iPixels = m_pBinSeries->GetPixels();
	if(m_pfReproj != 0L) cudaFreeHost(m_pfReproj);
	cudaMallocHost(&m_pfReproj, sizeof(float) * iPixels);
	//-----------------
	m_centralXcf.Setup(m_pBinSeries->m_aiStkSize, m_iVolZ, m_iNthGpu);
	//-----------------
	bool bPadded = true;
	m_pCorrProj->Setup(m_pTiltSeries->m_aiStkSize, !bPadded,
	   bRandomFill, !bFourierCrop, fTiltAxis, (float)m_iBin, m_iNthGpu);
	//-----------------
	m_aCalcReproj.Setup(m_pBinSeries->m_aiStkSize, m_iVolZ, m_iNthGpu);
}

float CProjAlignMain::DoIt(MAM::CAlignParam* pAlignParam)
{	
	m_pAlignParam = pAlignParam;
	m_iZeroTilt = m_pAlignParam->GetFrameIdxFromTilt(0.0f);
	//-----------------
	if(m_pcLog != 0L) delete[] m_pcLog;
	int iSize = (m_iNumProjs + 16) * 256;
	m_pcLog = new char[iSize];
	memset(m_pcLog, 0, sizeof(char) * iSize); 
	//-----------------
	mBinStack();
	float fError = mDoAll();
	//-----------------
	printf("%s\n", m_pcLog);
	if(m_pcLog != 0L) delete[] m_pcLog;
	m_pcLog = 0L;
	return fError;
}

float CProjAlignMain::mDoAll(void)
{
	m_pAlignParam->FitRotCenterZ();
	m_pAlignParam->RemoveOffsetZ(-1.0f);
	//-----------------
	for(int i=0; i<m_iNumProjs; i++)
	{	m_pbSkipProjs[i] = true;
	}
	m_pbSkipProjs[m_iZeroTilt] = false;
	//-----------------
	strcpy(m_pcLog, "# Projection matching measurements\n");
	strcat(m_pcLog, "# tilt angle   x shift   y shift\n");
	//-----------------
	float fMaxErr = (float)-1e20;
	for(int i=1; i<m_iNumProjs; i++)
	{	int iProj = m_iZeroTilt + i;
		if(iProj < m_iNumProjs && iProj >= 0)
		{	float fErr = mAlignProj(iProj);
			m_pbSkipProjs[iProj] = false;
			if(fErr > fMaxErr) fMaxErr = fErr;
		}
		//----------------
		iProj = m_iZeroTilt - i;
		if(iProj >= 0 && iProj < m_iNumProjs)
		{	float fErr = mAlignProj(iProj);
			m_pbSkipProjs[iProj] = false;
			if(fErr > fMaxErr) fMaxErr = fErr;
		}
	}
	strcat(m_pcLog, "\n");
	return fMaxErr;
}

float CProjAlignMain::mAlignProj(int iProj)
{
	float* pfRawProj = (float*)m_pTiltSeries->GetFrame(iProj);
	m_pCorrProj->SetProj(pfRawProj);
	mCalcReproj(iProj);
	//-----------------
	float fShift = mMeasure(iProj);
	mCorrectProj(iProj);
	return fShift;
}

float CProjAlignMain::mMeasure(int iProj)
{
	float fTilt = m_pAlignParam->GetTilt(iProj);
	//-----------------
	float afShift[2] = {0.0f};
	float* pfProj = (float*)m_pBinSeries->GetFrame(iProj);
	m_centralXcf.SetupXcf(0.5f, m_fBFactor);
	m_centralXcf.DoIt(m_pfReproj, pfProj, fTilt);
        m_centralXcf.GetShift(afShift);
        afShift[0] *= m_afBinning[0];
        afShift[1] *= m_afBinning[1];
	//-----------------
	char acLog[256] = {'\0'};
	sprintf(acLog, "  %6.2f  %8.2f  %8.2f\n", fTilt, 
	   afShift[0], afShift[1]);
	strcat(m_pcLog, acLog);
	//-----------------
	float fShift = afShift[0] * afShift[0] + afShift[1] * afShift[1]; 
	fShift = (float)sqrt(fShift);
	//-----------------
	float fTiltAxis = m_pAlignParam->GetTiltAxis(iProj);
	MrcUtil::CAlignParam::RotShift(afShift, fTiltAxis, afShift);
	float afInducedS[2] = {0.0f};
	m_pAlignParam->CalcZInducedShift(iProj, afInducedS);
	afShift[0] += afInducedS[0];
	afShift[1] += afInducedS[1];
	m_pAlignParam->AddShift(iProj, afShift);
	//-----------------
	return fShift;
}

void CProjAlignMain::mCalcBinning(void)
{	
	int* piProjSize = m_pTiltSeries->m_aiStkSize;
	CParam* pParam = CParam::GetInstance(m_iNthGpu);
	int iBinX = (int)(piProjSize[0] / pParam->m_fXcfSize + 0.5f);
	int iBinY = (int)(piProjSize[1] / pParam->m_fXcfSize + 0.5f);
	m_iBin = (iBinX > iBinY) ? iBinX : iBinY;
	if(m_iBin < 1) m_iBin = 1;
}

void CProjAlignMain::mBinStack(void)
{
	m_pCorrTomoStack->DoIt(0, m_pAlignParam);
	m_pCorrTomoStack->GetBinning(m_afBinning);
}
	
void CProjAlignMain::mRemoveSpikes(MD::CTiltSeries* pTiltSeries)
{
	CRemoveSpikes remSpikes;
	remSpikes.DoIt(pTiltSeries);
}

void CProjAlignMain::mCalcReproj(int iProj)
{
	m_pbSkipProjs[iProj] = true;
	float** ppfImgs = m_pBinSeries->GetImages();
	float* pfTilts = m_pAlignParam->GetTilts(false);
	m_aCalcReproj.DoIt(ppfImgs, pfTilts, 
	  m_pbSkipProjs, iProj, m_pfReproj);
	m_pbSkipProjs[iProj] = false;
}

void CProjAlignMain::mCorrectProj(int iProj)
{
	float afShift[2] = {0.0f};
	m_pAlignParam->GetShift(iProj, afShift);
	float fTiltAxis = m_pAlignParam->GetTiltAxis(iProj);
	m_pCorrProj->DoIt(afShift, fTiltAxis);
	//-----------------
	bool bPadded = true;
	float* pfBinProj = (float*)m_pBinSeries->GetFrame(iProj);
	m_pCorrProj->GetProj(pfBinProj, m_pBinSeries->m_aiStkSize, !bPadded);
}
