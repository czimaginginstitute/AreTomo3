#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

CFindCtf2D::CFindCtf2D(void)
{
	m_pFindDefocus2D = 0L;
}

CFindCtf2D::~CFindCtf2D(void)
{
	this->Clean();
}

void CFindCtf2D::Clean(void)
{
	if(m_pFindDefocus2D != 0L) 
	{	delete m_pFindDefocus2D;
		m_pFindDefocus2D = 0L;
	}
	CFindCtf1D::Clean();
}

void CFindCtf2D::Setup1(CCtfTheory* pCtfTheory)
{
	this->Clean();
	CFindCtf1D::Setup1(pCtfTheory);
	//---------------------------
	m_pFindDefocus2D = new CFindDefocus2D;
	MD::CCtfParam* pCtfParam = m_pCtfTheory->GetParam(false);
	m_pFindDefocus2D->Setup1(pCtfParam, m_aiCmpSize);
	m_pFindDefocus2D->Setup2(m_afResRange);
}

void CFindCtf2D::Do2D(void)
{
	CFindCtf1D::Do1D();
	float fDfMean = (m_fDfMin + m_fDfMax) * 0.5f;
	//---------------------------
	float afDfRange[] = {fDfMean - 3000.0f, fDfMean + 3000.0f};
	afDfRange[0] = fmax(afDfRange[0], 2000.0f);
	//---------------------------
	float afPhaseRange[] = {m_afPhaseRange[0], m_afPhaseRange[1]};
	m_pFindDefocus2D->SetInitVals(fDfMean, 0.01f, 0.0f, m_fExtPhase);
	m_pFindDefocus2D->DoIt(m_gfCtfSpect, afDfRange, afPhaseRange);
	mGetResults();
	//---------------------------
	float fDfRange = 6000.0f;
	float fAstMagRange = 0.1f;
	float fAstAngRange = 180.0f;
	float fPhaseRange = afPhaseRange[1] - afPhaseRange[0];
	for(int i=0; i<2; i++)
	{	float fFact = 1.0f / (1.0f + i);
		mRefine(fDfRange * fFact, fAstMagRange * fFact, 
		   fAstAngRange * fFact,  fPhaseRange * fFact);
	}
	//---------------------------
	for(int i=0; i<5; i++)
	{       fPhaseRange = fminf(fPhaseRange, 20.0f); 
		mRefine(2000.0f, 0.05f, 20.0f, fPhaseRange);
	}
}

void CFindCtf2D::Refine
(	float afDfMean[2],
	float afAstRatio[2],
	float afAstAngle[2],
	float afExtPhase[2]
)
{	m_pFindDefocus2D->SetInitVals(afDfMean[0], afAstRatio[0],
	   afAstAngle[0], afExtPhase[0]);
	//---------------------------
	m_pFindDefocus2D->Refine(m_gfCtfSpect, afDfMean[1], afExtPhase[1]);
	mGetResults();
}

void CFindCtf2D::mRefine
(       float fDfRange,
	float fAstMagRange,
	float fAstAngRange,
	float fPhaseRange
)
{	float fDfMean = (m_fDfMin + m_fDfMax) * 0.5f;
	float fMinDf = fDfMean - 0.5f * fDfRange;
	float fMaxDf = fDfMean + 0.5f * fDfRange;
	fMinDf = fmaxf(fMinDf, 2000.0f);
	float fStepDf = (fMaxDf - fMinDf) / 100.0f;
	//---------------------------
	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
	   fMinDf, fMaxDf, fStepDf, 0);
	//---------------------------
	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
	   m_fAstAng - 0.5f * fAstAngRange,
	   m_fAstAng + 0.5f * fAstAngRange,
	   2.0f, 2);
	//---------------------------
	float fAstRatio = (m_fDfMax - m_fDfMin) / (m_fDfMax + m_fDfMin);	
	float fMinRatio = fAstRatio - 0.5f * fAstMagRange;
	float fMaxRatio = fAstRatio + 0.5f * fAstMagRange;
	fMinRatio = fmax(fMinRatio, 0.0f);
	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
	   fMinRatio, fMaxRatio, 0.001, 1);
	//---------------------------
	if(fPhaseRange <= 0.5f)
	{	mGetResults();
		return;
	}
	//---------------------------
	float fMinPhase = m_fExtPhase - fPhaseRange;
	float fMaxPhase = m_fExtPhase + fPhaseRange;
	fMinPhase = fmax(fMinPhase, m_afPhaseRange[0]);
	fMaxPhase = fmin(fMaxPhase, m_afPhaseRange[1]);
	m_pFindDefocus2D->RefineParam(m_gfCtfSpect,
	   fMinPhase, fMaxPhase, 1.0f, 3);
	//---------------------------
	mGetResults();
}

void CFindCtf2D::mGetResults(void)
{
	m_fDfMin = m_pFindDefocus2D->GetDfMin();
	m_fDfMax = m_pFindDefocus2D->GetDfMax();
	m_fAstAng = m_pFindDefocus2D->GetAngle();
	m_fExtPhase = m_pFindDefocus2D->GetExtPhase();
	m_fScore = m_pFindDefocus2D->GetScore();
	m_fCtfRes = m_pFindDefocus2D->GetCtfRes();	
}
