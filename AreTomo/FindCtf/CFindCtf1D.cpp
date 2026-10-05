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
	//mEstimateBFactor();
	mFindDefocus();
	//-----------------
	float fDfRange = fmaxf(0.3f * m_fDfMin, 3000.0f); 
	mRefineDefocus(fDfRange);
}

void CFindCtf1D::Refine1D(float fInitDf, float fDfRange)
{
	m_fDfMin = fInitDf;
	m_fDfMax = fInitDf;
	m_fScore = (float)-1e20;
	//----------------------
	mCalcRadialAverage();
	//mEstimateBFactor();
	mRefineDefocus(fDfRange);
}

//--------------------------------------------------------------------
// Note: Using the fitted B-factor in GCC1D/GCC2D has negative
// effect on CTF fitting accuracy. Retire this
//--------------------------------------------------------------------
void CFindCtf1D::mEstimateBFactor(void)
{
	MD::CCtfParam* pCtfParam = m_pCtfTheory->GetParam(false);
        GEstBFactor1D estBFactor;
        //---------------------------
        float fBStep = 20.0f;
        int iNumSteps = 100;
        //---------------------------
        estBFactor.Setup(m_afResRange,
           pCtfParam->m_fPixelSize,
           fBStep, iNumSteps);
        float fBestB = estBFactor.DoIt(
           m_gfRadialAvg,
           m_aiCmpSize[0]);
        //---------------------------
        pCtfParam->m_fBFactor = fBestB;
}

void CFindCtf1D::mFindDefocus(void)
{
	MD::CCtfInput* pCtfInput = MD::CCtfInput::GetInstance();
	m_pFindDefocus1D->DoIt(
	   pCtfInput->m_afDfRange, 
	   pCtfInput->m_afPhaseRange, 
	   m_gfRadialAvg);
	//---------------------------
	m_fExtPhase = m_pFindDefocus1D->m_fBestPhase;
	m_fDfMin = m_pFindDefocus1D->m_fBestDf;
	m_fDfMax = m_fDfMin;
	m_fScore = m_pFindDefocus1D->m_fMaxCC;
}

void CFindCtf1D::mRefineDefocus(float fDfRange)
{
	MD::CCtfInput* pCtfInput = MD::CCtfInput::GetInstance();
	float afDfRange[2] = {0.0f};
	pCtfInput->GetDfRange(m_fDfMin, fDfRange, afDfRange);
	//---------------------------
	float afPhaseRange[2] = {0.0f};
	pCtfInput->GetPhaseRange(m_fExtPhase, 10.0f, afPhaseRange);
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

