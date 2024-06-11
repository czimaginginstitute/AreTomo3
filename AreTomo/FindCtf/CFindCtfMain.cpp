#include "CFindCtfInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

int CFindCtfMain::m_aiSpectSize[] = {512, 512};

CFindCtfMain::CFindCtfMain(void)
{
	m_ppfHalfSpects = 0L;
	m_pFindCtf2D = 0L;
	m_iNumTilts = 0;
	m_iRefTilt = 0;
}

CFindCtfMain::~CFindCtfMain(void)
{
	this->Clean();
}

void CFindCtfMain::Clean(void)
{
	if(m_ppfHalfSpects != 0L)
	{	for(int i=0; i<m_iNumTilts; i++)
		{	if(m_ppfHalfSpects[i] == 0L) continue;
			else delete[] m_ppfHalfSpects[i];
		}
		delete[] m_ppfHalfSpects;
		m_ppfHalfSpects = 0L;
	}
	if(m_pFindCtf2D != 0L)
	{	delete m_pFindCtf2D;
		m_pFindCtf2D = 0L;
	}
	m_iNumTilts = 0;
}

bool CFindCtfMain::bCheckInput(void)
{
	CInput* pInput = CInput::GetInstance();
	bool bEstimate = true;
	if(pInput->m_fCs == 0.0) bEstimate = false;
	else if(pInput->m_iKv == 0) bEstimate = false;
	//--------------------------------------------------
	// 1) Pixel size is needed. 2) We will check each
	// CTiltSeries object since CTF estimation is 
	// performed on it. 3) CInput is not checked.
	//--------------------------------------------------
	if(bEstimate) return true;
	//------------------------
	printf("Skip CTF estimation. Need the following parameters.\n");
	printf("High tension: %d\n", pInput->m_iKv);
	printf("Cs value:     %.4f\n", pInput->m_fCs);
	printf("Pixel size:   %.4f\n\n", pInput->m_fPixSize);
	return false;	
}

void CFindCtfMain::DoIt(int iNthGpu)
{	
	this->Clean();
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	m_pTiltSeries = pTsPkg->GetSeries(0);
	m_iNumTilts = m_pTiltSeries->m_aiStkSize[2];
	//---------------------------------------------------------
	// 1) Check whether pixel size is given in CTiltSeries.
	//---------------------------------------------------------
	if(m_pTiltSeries->m_fPixSize < 0.001f) 
	{	printf("(Warning (GPU %d): pixel size is not given, "
		   "CTF estimation is skipped.\n\n", m_iNthGpu);
		return;
	}
	//-----------------
	CInput* pInput = CInput::GetInstance();
	CAtInput* pAtInput = CAtInput::GetInstance();
	//---------------------------------------------------------
	// When pixel size is small, Fourier cropping to cut
	// high-frequency components to reduce noise.
	//---------------------------------------------------------
	float fBinning = 1.0f / m_pTiltSeries->m_fPixSize;
	if(fBinning <= 1.001f) fBinning = 1.0f;
	else if(fBinning > 2) fBinning = 2.0f;
	//-----------------
	float fPixSize = m_pTiltSeries->m_fPixSize;
	m_aiBinSize[0] = m_pTiltSeries->m_aiStkSize[0];
	m_aiBinSize[1] = m_pTiltSeries->m_aiStkSize[1];
	if(fBinning > 1.001)
	{	m_aiBinSize[0] = (int)(m_aiBinSize[0] / fBinning) / 2 * 2;
		m_aiBinSize[1] = (int)(m_aiBinSize[1] / fBinning) / 2 * 2;
		fPixSize *= fBinning;
	}
	//-----------------
	float fExtPhase = pAtInput->m_afExtPhase[0] * 0.017453f;
	CCtfTheory aInputCTF;
	aInputCTF.Setup(pInput->m_iKv, pInput->m_fCs,
	   pAtInput->m_fAmpContrast, fPixSize,
	   100.0f, fExtPhase);
	//-----------------
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
        pCtfResults->Setup(m_iNumTilts, CFindCtfMain::m_aiSpectSize,
	   aInputCTF.GetParam(false));
	//-----------------
	m_pFindCtf2D = new CFindCtf2D;
	m_pFindCtf2D->Setup1(&aInputCTF, CFindCtfMain::m_aiSpectSize[0]);
	m_pFindCtf2D->Setup2(m_aiBinSize);
	m_pFindCtf2D->SetPhase(pAtInput->m_afExtPhase[0], 
	   pAtInput->m_afExtPhase[1]);
	//-----------------
	mGenSpectrums();
	printf("GPU %d: estimate tilt series CTF, "
	   "please wait ......\n", m_iNthGpu);
	mDoZeroTilt();
	mDo2D();
	CSaveCtfResults saveCtfResults;
	saveCtfResults.DoIt(m_iNthGpu);
	printf("GPU %d: estimate tilt series CTF, done\n\n", m_iNthGpu);
}

void CFindCtfMain::mGenSpectrums(void)
{
	MU::GFourierResize2D* gFtResize = 0L;
	float* pfBinImg = 0L;
	if(m_aiBinSize[0] < m_pTiltSeries->m_aiStkSize[0] &&
	   m_aiBinSize[1] < m_pTiltSeries->m_aiStkSize[1])
	{	gFtResize = new MU::GFourierResize2D;
		gFtResize->Setup(m_pTiltSeries->m_aiStkSize, m_aiBinSize);
		pfBinImg = new float[m_aiBinSize[0] * m_aiBinSize[1]];
	}
	//-----------------
	bool bRaw = true, bToHost = true;
	if(m_ppfHalfSpects == 0L) m_ppfHalfSpects = new float*[m_iNumTilts];
	//-----------------
	int iSize = (m_iNumTilts + 16) * 64;
	char* pcLog = new char[iSize];
	memset(pcLog, 0, sizeof(char) * iSize);
	strcpy(pcLog, "Generate tile spectrum\n");
	//-----------------
	char acBuf[64] = {'\0'};
	for(int i=0; i<m_iNumTilts; i++)
	{	sprintf(acBuf, "...... spectrum of tilt %4d created, "
		   "%4d left\n", i+1, m_iNumTilts - 1 - i);
		strcat(pcLog, acBuf);	
		float* pfImage = (float*)m_pTiltSeries->GetFrame(i);
		if(gFtResize == 0L) 
		{	m_pFindCtf2D->GenHalfSpectrum(pfImage);
		}
		else
		{	gFtResize->DoIt(pfImage, pfBinImg);
			m_pFindCtf2D->GenHalfSpectrum(pfBinImg);
		}
		m_ppfHalfSpects[i] = m_pFindCtf2D->GetHalfSpect(!bRaw, bToHost);
	}
	printf("%s\n", pcLog);
	if(pcLog != 0L) delete[] pcLog;
	if(pfBinImg != 0L) delete[] pfBinImg;
	if(gFtResize != 0L) delete gFtResize;
}

void CFindCtfMain::mDoZeroTilt(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
        float fPhaseRange = fmaxf(pAtInput->m_afExtPhase[1], 0.0f);
	m_pFindCtf2D->SetPhase(pAtInput->m_afExtPhase[0], fPhaseRange);
	//-----------------
	int iZeroTilt = m_pTiltSeries->GetTiltIdx(0.0f);
	m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[iZeroTilt]);
	m_pFindCtf2D->Do2D();
	mGetResults(iZeroTilt);
	//-----------------------------------------------
	// Sometimes image at zero tilt is collected at
	// much higher defocus, we should use other
	// tilt images;
	//-----------------------------------------------
	m_fLowTilt = 12.5f;
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	//-----------------
	for(int i=0; i<m_iNumTilts; i++)
	{	if(fabs(m_pTiltSeries->m_pfTilts[i]) > m_fLowTilt) continue;
		else if(i == iZeroTilt) continue;
		//----------------
		m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[i]);
		m_pFindCtf2D->Do2D();
		mGetResults(i);
	}
	//------------------
	int iCount = 0.0f;
	float fSum1 = 0.0f, fSum2 = 0.0f;
	for(int i=0; i<m_iNumTilts; i++)
	{	if(fabs(m_pTiltSeries->m_pfTilts[i]) > m_fLowTilt) continue;
		else if(i == iZeroTilt) continue;
		//----------------
		float fDf = (pCtfResults->GetDfMin(i) + 
		   pCtfResults->GetDfMax(i)) * 0.5f;
		fSum1 += fDf;
		fSum2 += (fDf * fDf);
		iCount += 1;
	}
	m_fDfMean = fSum1 / (iCount + 0.001f);
	m_fDfStd = fSum2 / (iCount + 0.001f) - m_fDfMean * m_fDfMean;
	if(m_fDfStd <= 0) m_fDfStd = 0.0f;
	else m_fDfStd = sqrtf(m_fDfStd);
	//-----------------
	if(iCount == 0)
	{	m_fDfMean = (pCtfResults->GetDfMax(iZeroTilt) +
		   pCtfResults->GetDfMin(iZeroTilt)) * 0.5f;
		m_fDfStd = m_fDfMean * 0.5f;
	}
}
	   
void CFindCtfMain::mDo2D(void)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* m_pTiltSeries = pTsPkg->GetSeries(0);
	int iZeroTilt = m_pTiltSeries->GetTiltIdx(0.0f);
	//-----------------	
	float afDfRange[2], afAstRatio[2];
	float afAstAngle[2], afExtPhase[2];
	//-----------------
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	afDfRange[0] = m_fDfMean; 
	afDfRange[1] = fmaxf(m_fDfStd * 2.0f, 30000.0f);
	//-----------------
	MD::CCtfParam* pCtfParam = pCtfResults->GetCtfParam(iZeroTilt);
	afAstRatio[0] = pCtfParam->GetDfSigma(false) / 
	   (pCtfParam->GetDfMean(false) + 0.001f);
	afAstRatio[1] = fmaxf(afAstRatio[0] * 0.10f, 0.01f);
	//-----------------
	afAstAngle[0] = pCtfResults->GetAzimuth(iZeroTilt);
	afAstAngle[1] = 0.0f;
	//-----------------
	afExtPhase[0] = pCtfResults->GetExtPhase(iZeroTilt);
	afExtPhase[1] = 0.0f;
	//-----------------
	for(int i=0; i<m_pTiltSeries->m_aiStkSize[2]; i++)
	{	if(fabs(m_pTiltSeries->m_pfTilts[i]) <= m_fLowTilt) continue;
		else if(i == iZeroTilt) continue;
		//----------------
		m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[i]);
		m_pFindCtf2D->Refine(afDfRange, afAstRatio, 
		   afAstAngle, afExtPhase);
		float fScore = mGetResults(i);
	}
	//-----------------
	pCtfResults->DisplayAll();
}

float CFindCtfMain::mGetResults(int iTilt)
{
	float fTilt = m_pTiltSeries->m_pfTilts[iTilt];
	//-----------------
	MD::CCtfResults* pCtfResults = 
	   MD::CCtfResults::GetInstance(m_iNthGpu);
	pCtfResults->SetTilt(iTilt, fTilt);
	pCtfResults->SetDfMin(iTilt, m_pFindCtf2D->m_fDfMin);
	pCtfResults->SetDfMax(iTilt, m_pFindCtf2D->m_fDfMax);
	pCtfResults->SetAzimuth(iTilt, m_pFindCtf2D->m_fAstAng);
	pCtfResults->SetExtPhase(iTilt, m_pFindCtf2D->m_fExtPhase);
	pCtfResults->SetScore(iTilt, m_pFindCtf2D->m_fScore);
	pCtfResults->SetCtfRes(iTilt, m_pFindCtf2D->m_fCtfRes);
	//-----------------
	float* pfSpect = m_pFindCtf2D->GenFullSpectrum();
	pCtfResults->SetSpect(iTilt, pfSpect);
	//-----------------
	return m_pFindCtf2D->m_fScore;
}

char* CFindCtfMain::mGenSpectFileName(void)
{
	CInput* pInput = CInput::GetInstance();
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	//----------------
	char* pcSpectName = new char[256];
	memset(pcSpectName, 0, sizeof(char) * 256);
	//-----------------
	strcpy(pcSpectName, pInput->m_acOutDir);
	strcat(pcSpectName, pTsPackage->m_acMrcMain);
	strcat(pcSpectName, "_CTF.mrc");
	return pcSpectName;
}	

