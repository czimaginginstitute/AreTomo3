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
	//-----------------------------
	m_pFindDefocus2D = new CFindDefocus2D;
	CCtfParam* pCtfParam = m_pCtfTheory->GetParam(false);
	m_pFindDefocus2D->Setup1(pCtfParam, m_aiCmpSize);
	m_pFindDefocus2D->Setup2(m_afResRange);
}

void CFindCtf2D::Do2D(void)
{	
	CFindCtf1D::Do1D();
	float fDfMean = (m_fDfMin + m_fDfMax) * 0.5f;
	m_pFindDefocus2D->Setup3(fDfMean, 0.0f, 0.0f, m_fExtPhase);
	m_pFindDefocus2D->DoIt(m_gfCtfSpect, m_afPhaseRange);
	mGetResults();
}

void CFindCtf2D::Refine
(	float afDfMean[2],
	float afAstRatio[2],
	float afAstAngle[2],
	float afExtPhase[2]
)
{	m_pFindDefocus2D->Setup3(afDfMean[0], afAstRatio[0],
	   afAstAngle[0], afExtPhase[0]);
	m_pFindDefocus2D->Refine(m_gfCtfSpect, afDfMean[1],
	   afAstRatio[1], afAstAngle[1], afExtPhase[1]);
	mGetResults();
}

void CFindCtf2D::mGetResults(void)
{
	m_fDfMin = m_pFindDefocus2D->GetDfMin();
	m_fDfMax = m_pFindDefocus2D->GetDfMax();
	m_fAstAng = m_pFindDefocus2D->GetAngle();
	m_fExtPhase = m_pFindDefocus2D->GetExtPhase();
	m_fScore = m_pFindDefocus2D->GetScore();	
}
