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
	mDoLowTilts();
	mDoHighTilts();
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

void CFindCtfMain::mDoLowTilts(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
        float fPhaseRange = fmaxf(pAtInput->m_afExtPhase[1], 0.0f);
	m_pFindCtf2D->SetPhase(pAtInput->m_afExtPhase[0], fPhaseRange);
	//-----------------
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	int iZeroTilt = pTsTiles->GetTiltIdx(0.0f);
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
	float fInitPhase = m_pFindCtf2D->m_fExtPhase;
	if(fPhaseRange > 0) fPhaseRange = fminf(fPhaseRange, 5.0f);
	//-----------------
	for(int i=0; i<m_iNumTilts; i++)
	{	float fTilt = pTsTiles->GetTilt(i);
		if(fabs(fTilt) > m_fLowTilt) continue;
		else if(i == iZeroTilt) continue;
		//----------------
		m_pFindCtf2D->SetPhase(fInitPhase, fPhaseRange);
		m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[i]);
		m_pFindCtf2D->Do2D();
		mGetResults(i);
	}
	//------------------
	int iCount = 0.0f;
	float fSum1 = 0.0f, fSum2 = 0.0f;
	for(int i=0; i<m_iNumTilts; i++)
	{	float fTilt = pTsTiles->GetTilt(i);
		if(fabs(fTilt) > m_fLowTilt) continue;
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
	   
void CFindCtfMain::mDoHighTilts(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	//---------------------------
	int iZeroTilt = pTsTiles->GetTiltIdx(0.0f);
	float fPixSize = pTsTiles->GetPixSize();
	//---------------------------
	float afExtPhase[2] = {0.0f};
	afExtPhase[0] = pCtfResults->GetExtPhase(iZeroTilt);
        afExtPhase[1] = pAtInput->m_afExtPhase[1];
	if(afExtPhase[1] > 0)
	{	afExtPhase[1] *= 0.25f;
		if(afExtPhase[1] > 5) afExtPhase[1] = 5.0f;
	}
	//---------------------------
	float afDfRange[2] = {0.0f};
	float fPixSize2 = fPixSize * fPixSize;
	float fDfRange = fmaxf(m_fDfStd * 2.0f, 20000 * fPixSize2);
	float fMin = m_fDfMean - fDfRange * 0.5f;
	float fMax = m_fDfMean + fDfRange * 0.5f;
	afDfRange[0] = fmaxf(fMin, 3000 * fPixSize2);
	afDfRange[1] = fminf(fMax, 30000 * fPixSize2);
	//---------------------------
	float afAstRatio[2] = {0.0f};
	MD::CCtfParam* pCtfParam = pCtfResults->GetCtfParam(iZeroTilt);
	afAstRatio[0] = pCtfParam->GetDfSigma(false) / 
	   (pCtfParam->GetDfMean(false) + 0.001f);
	afAstRatio[1] = 0.0f;
	//---------------------------
	float afAstAngle[2] = {0.0f};
	afAstAngle[0] = pCtfResults->GetAzimuth(iZeroTilt);
	afAstAngle[1] = 10.0f;
	//---------------------------
	int iNumTilts = pTsTiles->GetNumTilts();
	float fPhaseRange = afExtPhase[1];
	//---------------------------
	for(int i=0; i<iNumTilts; i++)
	{	float fTilt = pTsTiles->GetTilt(i);
		afExtPhase[1] =  fPhaseRange * (float)cos(fTilt * 0.01744);
		//--------------------------
		m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[i]);
		m_pFindCtf2D->Refine(afDfRange, afAstRatio, 
		   afAstAngle, afExtPhase);
		float fScore = mGetResults(i);
	}
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
