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
	float fDfRange = fmaxf(0.3f * m_fDfMin, 3000.0f); 
	mRefineDefocus(fDfRange);
	//-----------------
	float fPsRange = (m_afPhaseRange[1] - m_afPhaseRange[1]) * 0.25f;
	mRefinePhase(fPsRange);
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
	float fDfRange1 = fDfRange * fPixSize * fPixSize; 
	//-----------------
	float afDfRange[2] = {0.0f};
	afDfRange[0] = m_fDfMin - 0.5f * fDfRange1;
	afDfRange[1] = m_fDfMin + 0.5f * fDfRange1;
	if(afDfRange[0] < 50) afDfRange[0] = 50.0f;
	//-----------------
	float afPhaseRange[] = {m_fExtPhase, m_fExtPhase};
	//-----------------
	m_pFindDefocus1D->DoIt(afDfRange, afPhaseRange, m_gfRadialAvg);
	m_fExtPhase = m_pFindDefocus1D->m_fBestPhase;
	m_fDfMin = m_pFindDefocus1D->m_fBestDf;
	m_fDfMax = m_fDfMin;
	m_fScore = m_pFindDefocus1D->m_fMaxCC;
}

void CFindCtf1D::mRefinePhase(float fPhaseRange)
{
	if(fPhaseRange <= 0.0001f) return;
	float afPsRange[2] = {0.0f};
	afPsRange[0] = m_fExtPhase - fPhaseRange * 0.5f;
	afPsRange[1] = m_fExtPhase + fPhaseRange * 0.5f;
	if(afPsRange[0] < 0) afPsRange[0] = 0.0f;
	if(afPsRange[1] > 180) afPsRange[1] = 180.0f;
	//-----------------
	float afDfRange[] = {m_fDfMin, m_fDfMin};
	//-----------------
	m_pFindDefocus1D->DoIt(afDfRange, afPsRange, m_gfRadialAvg);
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

