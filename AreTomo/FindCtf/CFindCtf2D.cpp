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
	MD::CCtfInput* pCtfInput = MD::CCtfInput::GetInstance();
	float fRangeDF = pCtfInput->m_afDfRange[1] - 
	   pCtfInput->m_afDfRange[0];
	float fRangeAM = pCtfInput->m_afAstMagRange[1] -
	   pCtfInput->m_afAstMagRange[0];
	float fRangeAA = pCtfInput->m_afAstAngRange[1] - 
	   pCtfInput->m_afAstAngRange[0];
	float fRangePP = pCtfInput->m_afPhaseRange[1] -
	   pCtfInput->m_afPhaseRange[0];
	//---------------------------
	m_pFindDefocus2D->SetInitVals(fDfMean, 0.0f, 0.0f, m_fExtPhase);
	m_pFindDefocus2D->DoIt(
	   m_gfCtfSpect, fRangeDF,
	   fRangeAM, fRangeAA, fRangePP);
	mGetResults();
	//---------------------------
	mCGRefine();
}

void CFindCtf2D::Refine
(	float* pfRangeDF,
	float* pfRangeAM,
	float* pfRangeAA,
	float* pfRangePP
)
{	m_pFindDefocus2D->SetInitVals(
	   pfRangeDF[0], pfRangeAM[0],
	   pfRangeAA[0], pfRangePP[0]);
	//---------------------------
	m_pFindDefocus2D->DoIt(
	   m_gfCtfSpect, 
	   pfRangeDF[1], pfRangeAM[1],
	   pfRangeAA[1], pfRangePP[1]);
	mGetResults();
	//---------------------------
	//mCGRefine();
	//printf("CFindCtf2D: CG: %f %f\n", m_fDfMin, m_fDfMax);
}

void CFindCtf2D::mCGRefine(void)
{
        float fDfMean = (m_fDfMin + m_fDfMax) / 2.0f;
	float fAstMag = CFindCtfHelp::CalcAstRatio(m_fDfMin, m_fDfMax);
        float afInitPoint[] = {fDfMean, fAstMag, m_fAstAng, m_fExtPhase};
        float afSeaRange[] = {8000.0f, 0.00f, 0.0f, 3.0f};
        //---------------------------
	MD::CCtfInput* pCtfInput = MD::CCtfInput::GetInstance();
	int iDim = pCtfInput->bExtPhase() ? 4 : 3;
        //---------------------------
        CCGradient* pCGradient = new CCGradient;
        pCGradient->Setup(iDim, 10, 0.001f);
	MD::CCtfParam* pCtfParam = m_pCtfTheory->GetParam(false);
        pCGradient->SetCtfParam(pCtfParam);
        pCGradient->SetSpect(m_gfCtfSpect, m_aiCmpSize);
        float fScore = pCGradient->DoIt(afInitPoint, afSeaRange, 30);
        //---------------------------
	if(fScore > m_fScore)
	{	m_fDfMin = pCGradient->GetDfMin();
        	m_fDfMax = pCGradient->GetDfMax();
        	m_fAstAng = pCGradient->GetAstAngle();
        	m_fExtPhase = pCGradient->GetExtPhase();
		m_fScore = fScore;
	}
        //---------------------------
        if(pCGradient != 0L) delete pCGradient;
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
