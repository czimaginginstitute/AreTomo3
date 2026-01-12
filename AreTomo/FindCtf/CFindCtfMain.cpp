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
}

CFindCtfMain::~CFindCtfMain(void)
{
	this->Clean();
}

void CFindCtfMain::Clean(void)
{
	mCleanSpects();
	//-----------------
	if(m_pFindCtf2D != 0L)
	{	delete m_pFindCtf2D;
		m_pFindCtf2D = 0L;
	}
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
	mInit(false);
	//-----------------
	mGenAvgSpects(0.0f, 0.0f, 100.0f);
	printf("GPU %d: initial estimation of tilt series CTF, "
	   "please wait ......\n\n", m_iNthGpu);
	//-----------------
	mDoTilts();
	mRefineTilts();
	//-----------------
	MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
	pCtfRes->DisplayAll();
	printf("GPU %d: initial estimation of tilt series CTF, "
	   "done.\n\n", m_iNthGpu);
	
}

void CFindCtfMain::mInit(bool bRefine)
{
	m_pFindCtf2D = new CFindCtf2D;
	m_pFindCtf2D->SetGpu(m_iNthGpu);
	//-----------------
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	m_iNumTilts = pTsTiles->GetNumTilts();
	CTile* pTile = pTsTiles->GetTile(0);
	//-----------------
	CInput* pInput = CInput::GetInstance();
	CAtInput* pAtInput = CAtInput::GetInstance();
	//-----------------
	CCtfTheory aInitCTF;
	aInitCTF.Setup(pInput->m_iKv, pInput->m_fCs,
	   pAtInput->m_fAmpContrast, pTile->GetPixSize(),
	   100.0f, 0.0f);
	//-----------------
	m_pFindCtf2D->Setup1(&aInitCTF);
	if(bRefine) return;
	//-----------------
	m_pFindCtf2D->SetPhase(pAtInput->m_afExtPhase[0],
	   pAtInput->m_afExtPhase[1]);
	//-----------------
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
        int aiTileSize[] = {pAtInput->m_iCtfTileSize, pAtInput->m_iCtfTileSize};
        pCtfResults->Setup(m_iNumTilts, aiTileSize,
           aInitCTF.GetParam(false));
}

void CFindCtfMain::mGenAvgSpects
(	float fTiltOffset, 
	float fBetaOffset,
	float fMaxTilt
)
{	mCleanSpects();
	m_ppfHalfSpects = new float*[m_iNumTilts];
	memset(m_ppfHalfSpects, 0, sizeof(float*) * m_iNumTilts);
	//-----------------
	MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
	bool bRaw = true, bToHost = true;
	//-----------------
	for(int i=0; i<m_iNumTilts; i++)
	{	float fTilt = fabs(pCtfRes->GetTilt(i));
		if(fTilt > fMaxTilt) continue;
		//----------------	
		m_pFindCtf2D->GenHalfSpectrum(i, fTiltOffset, fBetaOffset);
		m_ppfHalfSpects[i] = m_pFindCtf2D->GetHalfSpect(!bRaw, bToHost);
	}
}

void CFindCtfMain::mDoTilts(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
        float fPhaseRange = fmaxf(pAtInput->m_afExtPhase[1], 0.0f);
	m_pFindCtf2D->SetPhase(pAtInput->m_afExtPhase[0], fPhaseRange);
	//---------------------------
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	int iZeroTilt = pTsTiles->GetTiltIdx(0.0f);
	m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[iZeroTilt]);
	m_pFindCtf2D->Do2D();
	mGetResults(iZeroTilt);
	//---------------------------
	float fInitPhase = m_pFindCtf2D->m_fExtPhase;
	if(fPhaseRange > 0) fPhaseRange = fminf(fPhaseRange, 5.0f);
	//---------------------------
	for(int i=0; i<m_iNumTilts; i++)
	{	m_pFindCtf2D->SetPhase(fInitPhase, fPhaseRange);
		m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[i]);
		m_pFindCtf2D->Do2D();
		mGetResults(i);
	}
}

void CFindCtfMain::mRefineTilts(void)
{
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	int iWinSize = 5;
	for(int i=0; i<m_iNumTilts; i++)
	{	int iStart = i - iWinSize / 2;
		if(iStart < 0) iStart = 0;
		int iEnd = iStart + iWinSize;
		if(iEnd > m_iNumTilts) iEnd = m_iNumTilts;
		iStart = iEnd - iWinSize;
		//--------------------------
		int iBestTilt = -1;
		float fBestScore = -99.0f;
		for(int j=iStart; j<iEnd; j++)
		{	float fScore = pCtfResults->GetScore(j);
			if(fScore > fBestScore)
			{	fBestScore = fScore;
				iBestTilt = j;
			}
		}
		float fTiltScore = pCtfResults->GetScore(i);
		float fDiff = (fBestScore - fTiltScore) / 
		   (fBestScore + (float)1e-30);
		if(fDiff < 0.4f) continue;
		else mRefineTilt(i, iBestTilt);
	}
}

void CFindCtfMain::mRefineTilt(int iTilt, int iRefTilt)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
        CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
        MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	//---------------------------
	float fPixSize = pTsTiles->GetPixSize();
	float afExtPhase[2] = {0.0f};
	afExtPhase[0] = pCtfResults->GetExtPhase(iRefTilt);
	afExtPhase[1] = 0.0f;
	//---------------------------
	float afDfRange[2] = {0.0f};
	float fPixSize2 = fPixSize * fPixSize;
	float fDfMin = pCtfResults->GetDfMean(iRefTilt) * 0.60f;
	float fDfMax = pCtfResults->GetDfMean(iRefTilt) * 1.40f;
	fDfMin = fmaxf(fDfMin,  3000.0f * fPixSize2);
	fDfMax = fminf(fDfMax, 40000.0f * fPixSize2);
	afDfRange[0] = (fDfMin + fDfMax) * 0.5f;
	afDfRange[1] = fDfMax - fDfMin;
	//---------------------------
	float afAstRatio[2] = {0.0f};
        MD::CCtfParam* pCtfParam = pCtfResults->GetCtfParam(iRefTilt);
        afAstRatio[0] = pCtfParam->GetDfSigma(false) /
           (pCtfParam->GetDfMean(false) + 0.001f);
        afAstRatio[1] = 0.0f;
	//---------------------------
	float afAstAngle[2] = {0.0f};
        afAstAngle[0] = pCtfResults->GetAzimuth(iRefTilt);
        afAstAngle[1] = 0.0f;
	//---------------------------
	m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[iTilt]);
	m_pFindCtf2D->Refine(afDfRange, afAstRatio, afAstAngle, afExtPhase);
	mGetResults(iTilt);
}
	   
float CFindCtfMain::mGetResults(int iTilt)
{
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	float fTilt = pTsTiles->GetTilt(iTilt);
	//-----------------
	MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
	pCtfRes->SetTilt(iTilt, fTilt);
	pCtfRes->SetDfMin(iTilt, m_pFindCtf2D->m_fDfMin);
	pCtfRes->SetDfMax(iTilt, m_pFindCtf2D->m_fDfMax);
	pCtfRes->SetAzimuth(iTilt, m_pFindCtf2D->m_fAstAng);
	pCtfRes->SetExtPhase(iTilt, m_pFindCtf2D->m_fExtPhase);
	pCtfRes->SetScore(iTilt, m_pFindCtf2D->m_fScore);
	pCtfRes->SetCtfRes(iTilt, m_pFindCtf2D->m_fCtfRes);
	//-----------------
	float* pfSpect = pCtfRes->GetSpect(iTilt, false);
	m_pFindCtf2D->GenFullSpectrum(pfSpect);
	//-----------------
	return m_pFindCtf2D->m_fScore;
}

void CFindCtfMain::mCleanSpects(void)
{
	if(m_ppfHalfSpects == 0L) return;
	for(int i=0; i<m_iNumTilts; i++)
	{	if(m_ppfHalfSpects[i] == 0L) continue;
		else delete[] m_ppfHalfSpects[i];
	}
	delete[] m_ppfHalfSpects;
	m_ppfHalfSpects = 0L;
}
