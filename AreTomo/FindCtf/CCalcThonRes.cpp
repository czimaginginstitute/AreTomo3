#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

CCalcThonRes::CCalcThonRes(void)
{
	m_gFFT1D = 0L;
	m_gfRadialAvg = 0L;
}

CCalcThonRes::~CCalcThonRes(void)
{
	this->Clean();
}

void CCalcThonRes::Clean(void)
{
	if(m_gfRadialAvg != 0L) cudaFree(m_gfRadialAvg);
	if(m_gFFT1D != 0L) delete m_gFFT1D;
	m_gfRadialAvg = 0L;
	m_gFFT1D = 0L;
}

void CCalcThonRes::Setup(CCtfTheory* pCtfTheory, int* piSpectSize)
{
	this->Clean();
	//-----------------
	m_aiSpectSize[0] = piSpectSize[0];
	m_aiSpectSize[1] = piSpectSize[1];
	//-----------------
	cudaMalloc(&m_gfRadialAvg, sizeof(float) * m_aiSpectSize[0]);
	//-----------------
	m_pFindDefocus1D = new CFindDefocus1D;
	CCtfParam* pCtfParam = m_pCtfTheory->GetParam(false);
	m_pFindDefocus1D->Setup(pCtfParam, m_aiCmpSize[0]);

}

void CCalcThonRes::DoIt(void)
{	
	mCalcRadialAverage();
	mFindDefocus();
}

void CCalcThonRes::mFindDefocus(void)
{
	float fPixSize = m_pCtfTheory->GetPixelSize();
	float fPixSize2 = fPixSize * fPixSize;
	//-----------------
	float afDfRange[2] = {0.0f};
	afDfRange[0] = 3000.0f * fPixSize2;
	afDfRange[1] = 30000.0f * fPixSize2;
	//------------------
	m_pFindDefocus1D->DoIt(afDfRange, m_afPhaseRange, m_gfRadialAvg);
	m_fExtPhase = m_pFindDefocus1D->m_fBestPhase;
	m_fDfMin = m_pFindDefocus1D->m_fBestDf;
	m_fDfMax = m_fDfMin;
	m_fScore = m_pFindDefocus1D->m_fMaxCC;
}

void CCalcThonRes::mCalcRadialAverage(void)
{
	GRadialAvg aGRadialAvg;
	aGRadialAvg.DoIt(m_gfCtfSpect, m_gfRadialAvg, m_aiCmpSize);
}

