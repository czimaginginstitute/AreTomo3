#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

CFindCtf1D::CFindCtf1D(void)
{
	m_pFindDefocus1D = 0L;
	m_gfRadialAvg = 0L;
}

CFindCtf1D::~CFindCtf1D(void)
{
	this->Clean();
}

void CFindCtf1D::Clean(void)
{
	if(m_pFindDefocus1D != 0L) 
	{	delete m_pFindDefocus1D;
		m_pFindDefocus1D = 0L;
	}
	if(m_gfRadialAvg != 0L)
	{	cudaFree(m_gfRadialAvg);
		m_gfRadialAvg = 0L;
	}
	CFindCtfBase::Clean();
}

void CFindCtf1D::Setup1(CCtfTheory* pCtfTheory)
{
	this->Clean();
	CFindCtfBase::Setup1(pCtfTheory);
	cudaMalloc(&m_gfRadialAvg, sizeof(float) * m_aiCmpSize[0]);
	//-----------------
	m_pFindDefocus1D = new CFindDefocus1D;
	MD::CCtfParam* pCtfParam = m_pCtfTheory->GetParam(false);
	m_pFindDefocus1D->Setup(pCtfParam, m_aiCmpSize[0]);
	//-----------------
	m_pFindDefocus1D->SetResRange(m_afResRange);
}

void CFindCtf1D::Do1D(void)
{	
	mCalcRadialAverage();
	mFindDefocus();
	//-----------------
	float fDfRange = fmaxf(0.3f * m_fDfMin, 2000.0f); 
	mRefineDefocus(fDfRange);
}

void CFindCtf1D::Refine1D(float fInitDf, float fDfRange)
{
	m_fDfMin = fInitDf;
	m_fDfMax = fInitDf;
	m_fScore = (float)-1e20;
	//----------------------
	mCalcRadialAverage();
	mRefineDefocus(fDfRange);
}

void CFindCtf1D::mFindDefocus(void)
{
	float fPixSize = m_pCtfTheory->GetPixelSize();
	float fPixSize2 = fPixSize * fPixSize;
	//-----------------
	m_pFindDefocus1D->DoIt(m_afDfRange, m_afPhaseRange, m_gfRadialAvg);
	m_fExtPhase = m_pFindDefocus1D->m_fBestPhase;
	m_fDfMin = m_pFindDefocus1D->m_fBestDf;
	m_fDfMax = m_fDfMin;
	m_fScore = m_pFindDefocus1D->m_fMaxCC;
}

void CFindCtf1D::mRefineDefocus(float fDfRange)
{
	float fPixSize = m_pCtfTheory->GetPixelSize();
	float fPixSize2 = fPixSize * fPixSize;
	float fMinDf = 1000.0f * fPixSize2;
	//---------------------------
	float afDfRange[2] = {0.0f};
	afDfRange[0] = fmaxf(m_fDfMin - 0.5f * fDfRange, fMinDf);
	afDfRange[1] = afDfRange[0] + fDfRange;
	//---------------------------
	float afPhaseRange[2] = {0.0f};
	float fPhaseRange = 0.2f * (m_afPhaseRange[1] - m_afPhaseRange[0]);
	afPhaseRange[0] = m_fExtPhase - fPhaseRange / 2; 
	afPhaseRange[1] = m_fExtPhase + fPhaseRange / 2;
	afPhaseRange[0] = fmaxf(afPhaseRange[0], m_afPhaseRange[0]);
	afPhaseRange[1] = fminf(afPhaseRange[1], m_afPhaseRange[1]);
	//---------------------------
	m_pFindDefocus1D->DoIt(afDfRange, afPhaseRange, m_gfRadialAvg);
	m_fExtPhase = m_pFindDefocus1D->m_fBestPhase;
	m_fDfMin = m_pFindDefocus1D->m_fBestDf;
	m_fDfMax = m_fDfMin;
	m_fScore = m_pFindDefocus1D->m_fMaxCC;
}

void CFindCtf1D::mCalcRadialAverage(void)
{
	GRadialAvg aGRadialAvg;
	aGRadialAvg.DoIt(m_gfCtfSpect, m_gfRadialAvg, m_aiCmpSize);
}

