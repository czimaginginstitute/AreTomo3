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
	printf("GPU %d: local CTF estimation, "
	   "please wait ......\n\n", iNthGpu);
	//-----------------
	this->Clean();
	m_iNthGpu = iNthGpu;
	//-----------------
	mInit(true);
	bool bBeta = true;
	mRefineOffset(3.0f, !bBeta);
	mRefineOffset(1.0f, !bBeta);
	mRefineOffset(3.0f, bBeta);
	mRefineOffset(1.0f, bBeta);
	//-----------------
	CSaveCtfResults saveCtfResults;
	saveCtfResults.DoIt(m_iNthGpu);
	//-----------------
	MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
        pCtfRes->DisplayAll();
	//-----------------
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	printf("GPU %d: CTF refined tilt offsets\n"
	   "     tilt and beta offsets: %8.2f  %8.2f\n"
	   "     reference offsets:     %8.2f  %8.2f\n\n",
	   m_iNthGpu, m_fTiltOffset, m_fBetaOffset,
	   pAlnParam->m_fAlphaOffset, pAlnParam->m_fBetaOffset);
	//-----------------
	CAtInput* pAtInput = CAtInput::GetInstance();
	if(pAtInput->m_afTiltCor[0] == 0) return;
	pAlnParam->AddAlphaOffset(m_fTiltOffset);

}

void CRefineCtfMain::mRefineOffset(float fStep, bool bBeta)
{
	MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
        m_fBestScore = pCtfRes->GetTsScore();
        m_pBestCtfRes = pCtfRes->GetCopy();
	//-----------------
	float fInitOffset = m_fTiltOffset;
	if(bBeta) fInitOffset = m_fBetaOffset;
	//-----------------
	for(int i=-3; i<=3; i++)
	{	float fOffset = fInitOffset + i * fStep;
		if(fabs(fOffset) > 15) continue;
		//----------------
		if(bBeta) mGenAvgSpects(m_fTiltOffset, fOffset);
		else mGenAvgSpects(fOffset, m_fBetaOffset);
		//----------------
		float fScore = mRefineCTF(bBeta);
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

float CRefineCtfMain::mRefineCTF(bool bBeta)
{
        CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
        //-----------------
        float afDfRange[2], afAstRatio[2];
        float afAstAngle[2], afExtPhase[2];
        //-----------------
        MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
        afDfRange[1] = 4000.0f;
        afAstRatio[1] = 0.0f;
        afAstAngle[1] = 0.0f;
        afExtPhase[1] = 0.0f;
        int iNumTilts = pTsTiles->GetNumTilts();
        //-----------------
	float fLowTilt = 20.9f;
	//-----------------
        for(int i=0; i<iNumTilts; i++)
        {       float fTilt = fabs(pCtfRes->GetTilt(i));
		if(bBeta && fTilt > fLowTilt) continue;
		if(!bBeta && fTilt <= fLowTilt) continue;
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
	return pCtfRes->GetTsScore();
}

