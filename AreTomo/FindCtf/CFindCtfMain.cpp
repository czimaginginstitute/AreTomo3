#include "CFindCtfInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

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

bool CFindCtfMain::CheckInput(void)
{
	CInput* pInput = CInput::GetInstance();
	bool bEstimate = true;
	if(pInput->m_fCs == 0.0) bEstimate = false;
	else if(pInput->m_iKv == 0) bEstimate = false;
	else if(pInput->m_fPixSize == 0) bEstimate = false;
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
	//-----------------
	m_iNumTilts = m_pTiltSeries->m_aiStkSize[2];
	//-----------------
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	int aiSpectSize[] = {512, 512};
	pCtfResults->Setup(m_iNumTilts, aiSpectSize);
	//-----------------
	CInput* pInput = CInput::GetInstance();
	CAtInput* pAtInput = CAtInput::GetInstance();
	//-----------------
	float fExtPhase = pAtInput->m_afExtPhase[0] * 0.017453f;
	CCtfTheory aInputCTF;
	aInputCTF.Setup(pInput->m_iKv, pInput->m_fCs,
	   pAtInput->m_fAmpContrast, m_pTiltSeries->m_fPixSize,
	   100.0f, fExtPhase);
	//-----------------
	m_pFindCtf2D = new CFindCtf2D;
	m_pFindCtf2D->Setup1(&aInputCTF);
	m_pFindCtf2D->Setup2(m_pTiltSeries->m_aiStkSize);
	m_pFindCtf2D->SetPhase(pAtInput->m_afExtPhase[0], 
	   pAtInput->m_afExtPhase[1]);
	//-----------------
	mGenSpectrums();
	mDoZeroTilt();
	mDo2D();
	//-----------------
	CSaveCtfResults saveCtfResults;
	saveCtfResults.DoIt(m_iNthGpu);
}

void CFindCtfMain::mGenSpectrums(void)
{
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
		m_pFindCtf2D->GenHalfSpectrum(pfImage);
		m_ppfHalfSpects[i] = m_pFindCtf2D->GetHalfSpect(!bRaw, bToHost);
	}
	printf("%s\n", pcLog);
	if(pcLog != 0L) delete[] pcLog;
}

void CFindCtfMain::mDoZeroTilt(void)
{
	MAM::CAlignParam* pAlignParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	m_iRefTilt = pAlignParam->GetFrameIdxFromTilt(0.0f);
	//-----------------
	CAtInput* pAtInput = CAtInput::GetInstance();
        float fPhaseRange = fmaxf(pAtInput->m_afExtPhase[1], 0.0f);
	m_pFindCtf2D->SetPhase(pAtInput->m_afExtPhase[0], fPhaseRange);
	//-----------------
	m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[m_iRefTilt]);
	m_pFindCtf2D->Do2D();
	//-------------------
	mGetResults(m_iRefTilt);
}
	   
void CFindCtfMain::mDo2D(void)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* m_pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------	
	float afDfRange[2], afAstRatio[2];
	float afAstAngle[2], afExtPhase[2];
	//-----------------
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	float fDfMin = pCtfResults->GetDfMin(m_iRefTilt);
	float fDfMax = pCtfResults->GetDfMax(m_iRefTilt);
	afDfRange[0] = 0.5f * (fDfMin + fDfMax); 
	afDfRange[1] = fmaxf(afDfRange[0] * 0.25f, 10000.0f);
	//-----------------
	afAstRatio[0] = CFindCtfHelp::CalcAstRatio(fDfMin, fDfMax);
	afAstRatio[1] = fmaxf(afAstRatio[0] * 0.10f, 0.01f);
	//-----------------
	afAstAngle[0] = pCtfResults->GetAzimuth(m_iRefTilt);
	afAstAngle[1] = 0.0f;
	//-----------------
	afExtPhase[0] = pCtfResults->GetExtPhase(m_iRefTilt);
	afExtPhase[1] = 0.0f;
	//-----------------
	printf("GPU %d: estimate tilt series CTF, "
	   "please wait......\n", m_iNthGpu);
	//-----------------
	for(int i=0; i<m_pTiltSeries->m_aiStkSize[2]; i++)
	{	if(i == m_iRefTilt) continue;
		m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[i]);
		//----------------
		m_pFindCtf2D->Refine(afDfRange, afAstRatio, 
		   afAstAngle, afExtPhase);
		float fScore = mGetResults(i);
	}
	//-----------------
	pCtfResults->DisplayAll();
}

float CFindCtfMain::mGetResults(int iTilt)
{
	MAM::CAlignParam* pAlignParam =
           MAM::CAlignParam::GetInstance(m_iNthGpu);
	float fTilt = pAlignParam->GetTilt(iTilt);
	//-----------------
	MD::CCtfResults* pCtfResults = 
	   MD::CCtfResults::GetInstance(m_iNthGpu);
	pCtfResults->SetTilt(iTilt, fTilt);
	pCtfResults->SetDfMin(iTilt, m_pFindCtf2D->m_fDfMin);
	pCtfResults->SetDfMax(iTilt, m_pFindCtf2D->m_fDfMax);
	pCtfResults->SetAzimuth(iTilt, m_pFindCtf2D->m_fAstAng);
	pCtfResults->SetExtPhase(iTilt, m_pFindCtf2D->m_fExtPhase);
	pCtfResults->SetScore(iTilt, m_pFindCtf2D->m_fScore);
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

