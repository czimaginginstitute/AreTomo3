#include "CFindCtfInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

CRefineCtfMain::CRefineCtfMain(void)
{
	m_fTiltOffset = 0.0f;
	m_fBetaOffset = 0.0f;
	m_fBestScore = 0.0f;
	m_pBestCtfRes = 0L;
	m_fLowTilt = 35.9f;
}

CRefineCtfMain::~CRefineCtfMain(void)
{
	this->Clean();
}

void CRefineCtfMain::Clean(void)
{
	if(m_pBestCtfRes != 0L) delete m_pBestCtfRes;
	//-----------------
	m_pBestCtfRes = 0L;
	m_fTiltOffset = 0.0f;
	m_fBetaOffset = 0.0f;
	//-----------------
	CFindCtfMain::Clean();
}

void CRefineCtfMain::DoIt(int iNthGpu)
{
	const char* pcMsg1 = "Find defocus handedness, please wait ......";
	const char* pcMsg2 = "Local CTF estimation, please wait ......";
	//-----------------
	this->Clean();
	m_iNthGpu = iNthGpu;
	//-----------------
	mInit(true);
	bool bBeta = true;
	printf("GPU %d: %s\n\n", iNthGpu, pcMsg1);
	mFindHandedness();
	//-----------------
	mRefineOffset(3.1f, 22, !bBeta);
	printf("GPU %d: %s\n\n", iNthGpu, pcMsg2);
	mRefineOffset(1.1f, 9, !bBeta);
	//-----------------
	printf("GPU %d: %s\n\n", iNthGpu, pcMsg2);
	mRefineOffset(3.0f, 7, bBeta);
	printf("GPU %d: %s\n\n", iNthGpu, pcMsg2);
	mRefineOffset(1.1f, 7, bBeta);
	//-----------------
	printf("GPU %d: %s\n\n", iNthGpu, pcMsg2);
	mGenAvgSpects(m_fTiltOffset, m_fBetaOffset, 100.0f);
	mRefineCTF(3);
	//-----------------
	CSaveCtfResults saveCtfResults;
	saveCtfResults.DoIt(m_iNthGpu);
	//-----------------
	MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
        pCtfRes->DisplayAll();
	//-----------------
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	printf("GPU %d: CTF refined tilt offsets\n"
	   "     tilt and beta offsets: %8.2f  %8.2f\n\n",
	   m_iNthGpu, m_fTiltOffset, m_fBetaOffset);
	//-----------------
	pCtfRes->m_fAlphaOffset = m_fTiltOffset;
	pCtfRes->m_fBetaOffset = m_fBetaOffset;
}

void CRefineCtfMain::mFindHandedness(void)
{
	MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
	//-------------------------------------
	// Try positive handedness.
	//-------------------------------------
	pCtfRes->m_iDfHand = 1;
	mGenAvgSpects(0.0f, 0.0f, 100.0f);
	float fScore1 = mRefineCTF(3);
	//pCtfRes->DisplayAll();
	//-------------------------------------
	// Try negative handedness.
	//-------------------------------------
	pCtfRes->m_iDfHand = -1;
	mGenAvgSpects(0.0f, 0.0f, 100.0f);
	float fScore2 = mRefineCTF(3);
	//pCtfRes->DisplayAll();
	//-----------------
	if(fScore1 > fScore2) pCtfRes->m_iDfHand = 1;
	else pCtfRes->m_iDfHand = -1;
	printf("Defocus handedness (GPU %d): "
	   "1 score: %.5f; -1 score: %.5f\n\n",
	   m_iNthGpu, fScore1, fScore2);
}

void CRefineCtfMain::mRefineOffset(float fStep, int iNumSteps, bool bBeta)
{
	MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
	m_fBestScore = pCtfRes->GetLowTiltScore(m_fLowTilt);
        m_pBestCtfRes = pCtfRes->GetCopy();
	//---------------------------
	float fInitOffset = m_fTiltOffset;
	if(bBeta) fInitOffset = m_fBetaOffset;
	int iKind = bBeta ? 2 : 1;
	float fMaxTilt = m_fLowTilt + 1.0f;
	//---------------------------
	for(int s=0; s<iNumSteps; s++)
	{	int i = s - iNumSteps / 2;
		float fOffset = fInitOffset + i * fStep;
		if(fabs(fOffset) > 30) continue;
		//----------------
		if(bBeta) mGenAvgSpects(m_fTiltOffset, fOffset, fMaxTilt);
		else mGenAvgSpects(fOffset, m_fBetaOffset, fMaxTilt);
		//----------------
		float fScore = mRefineCTF(iKind);
		if(fScore <= m_fBestScore) continue;
		//----------------
		if(bBeta) m_fBetaOffset = fOffset;
		else m_fTiltOffset = fOffset;
		//----------------
		m_fBestScore = fScore;
		if(m_pBestCtfRes != 0L) delete m_pBestCtfRes;
		m_pBestCtfRes = pCtfRes->GetCopy();
	}
	//-----------------
	MD::CCtfResults::Replace(m_iNthGpu, m_pBestCtfRes);
	m_pBestCtfRes = 0L;
}

//--------------------------------------------------------------------
// iKind = 1: refine for better alpha offset
// iKind = 2: refine for better beta offset
// iKind = 3: refine for all better CTFs at all tilts
//--------------------------------------------------------------------
float CRefineCtfMain::mRefineCTF(int iKind)
{
        CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
        //-----------------
        float afDfRange[2], afAstRatio[2];
        float afAstAngle[2], afExtPhase[2];
        //-----------------
        MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
        afDfRange[1] = 5000.0f;
        afAstRatio[1] = 0.0f;
        afAstAngle[1] = 0.0f;
        afExtPhase[1] = 0.0f;
        int iNumTilts = pTsTiles->GetNumTilts();
        //-----------------
        for(int i=0; i<iNumTilts; i++)
        {       float fTilt = fabs(pCtfRes->GetTilt(i));
		if(iKind == 1 && fTilt > m_fLowTilt) continue;
		if(iKind == 2 && fTilt > m_fLowTilt) continue;
		//----------------
		afDfRange[0] = pCtfRes->GetDfMean(i);
		afAstRatio[0] = pCtfRes->GetAstMag(i);
		afAstAngle[0] = pCtfRes->GetAzimuth(i);
		afExtPhase[0] = pCtfRes->GetExtPhase(i);
		//----------------
		m_pFindCtf2D->SetHalfSpect(m_ppfHalfSpects[i]);
		m_pFindCtf2D->Refine(afDfRange, afAstRatio,
		   afAstAngle, afExtPhase);
		mGetResults(i);
        }
	return pCtfRes->GetLowTiltScore(m_fLowTilt);
}

