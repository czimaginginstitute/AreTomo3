#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.01745329f;

CFindDefocus2D::CFindDefocus2D(void)
{
	m_gfCtf2D = 0L;
	m_pGCC2D = 0L;
	m_fAstRatio = 0.0f; // (m_fDfMean - fMinDf) / m_fDfMean;
	m_fAstAngle = 0.0f; // degree
}

CFindDefocus2D::~CFindDefocus2D(void)
{
	this->Clean();
}

void CFindDefocus2D::Clean(void)
{
	if(m_gfCtf2D != 0L) 
	{	cudaFree(m_gfCtf2D);
		m_gfCtf2D = 0L;
	}
	if(m_pGCC2D != 0L)
	{	delete m_pGCC2D;
		m_pGCC2D = 0L;
	}
}

float CFindDefocus2D::GetDfMin(void)
{
	float fDfMin = m_fDfMean * (1.0f - m_fAstRatio);
	return fDfMin;
}

float CFindDefocus2D::GetDfMax(void)
{
	float fDfMax = m_fDfMean * (1.0f + m_fAstRatio);
	return fDfMax;
}

float CFindDefocus2D::GetAngle(void)
{
	return m_fAstAngle;
}

float CFindDefocus2D::GetExtPhase(void)
{
	return m_fExtPhase;
}

float CFindDefocus2D::GetScore(void)
{
	return m_fCCMax;
}

void CFindDefocus2D::Setup1(CCtfParam* pCtfParam, int* piCmpSize)
{
	this->Clean();
	//------------
	m_pCtfParam = pCtfParam;
	memcpy(m_aiCmpSize, piCmpSize, sizeof(int) * 2);
	//----------------------------------------------
	m_aGCalcCtf2D.SetParam(m_pCtfParam);
	//----------------------------------
	cudaMalloc(&m_gfCtf2D, sizeof(float) 
	   * m_aiCmpSize[0] * m_aiCmpSize[1]);
	//------------------------------------
	m_pGCC2D = new GCC2D;
	m_pGCC2D->SetSize(m_aiCmpSize);	
}

void CFindDefocus2D::Setup2(float afResRange[2])
{
	float fRes1 = m_aiCmpSize[1] * m_pCtfParam->m_fPixelSize;
	float fMinFreq = fRes1 / afResRange[0];
	float fMaxFreq = fRes1 / afResRange[1];
	m_pGCC2D->Setup(fMinFreq, fMaxFreq, 81.0f);
}

void CFindDefocus2D::Setup3
(	float fDfMean,
	float fAstRatio,
	float fAstAngle,
	float fExtPhase
)
{	m_fDfMean = fDfMean;
	m_fAstRatio = fAstRatio;
	m_fAstAngle = fAstAngle;
	m_fExtPhase = fExtPhase;
}

void CFindDefocus2D::DoIt(float* gfSpect, float* pfPhaseRange)
{
	m_gfSpect = gfSpect;
        //-----------------
	m_afDfRange[0] = m_fDfMean * 0.9f;
	m_afDfRange[1] = m_fDfMean * 1.1f;
	m_afAstRange[0] = 0.0f;
	m_afAstRange[1] = 0.1f;
	m_afAngRange[0] = -90.0f;
	m_afAngRange[1] = 90.0f;
	memcpy(m_afPhaseRange, pfPhaseRange, sizeof(float) * 2);
	//-----------------
	mIterate();	
}

void CFindDefocus2D::Refine
(	float* gfSpect,
	float fDfRange,
	float fAstRange,
	float fAngRange,
	float fPhaseRange
)
{	m_gfSpect = gfSpect;
	//-----------------
	float fHalfR = 0.5f * fDfRange;
	m_afDfRange[0] = fmaxf(m_fDfMean - fHalfR, 3000.0f);
	m_afDfRange[1] = m_fDfMean + fHalfR;
	//-----------------
	fHalfR = 0.5f * fAstRange;
	m_afAstRange[0] = fmaxf(m_fAstRatio - fHalfR, 0.0f);
	m_afAstRange[1] = fminf(m_fAstRatio + fHalfR, 0.1f);
	//-----------------
	fHalfR = 0.5f * fAngRange;
	m_afAngRange[0] = fmaxf(m_fAstAngle - fHalfR, -90.0f);
	m_afAngRange[1] = fminf(m_fAstAngle + fHalfR, 90.0f);
	//-----------------
	fHalfR = 0.5f * fPhaseRange;
	m_afPhaseRange[0] = fmaxf(m_fExtPhase - fHalfR, 0.0f);
	m_afPhaseRange[1] = fminf(m_fExtPhase + fHalfR, 179.0f);
	//-----------------
	mIterate();
}

void CFindDefocus2D::mIterate(void)
{
	m_fCCMax = mCorrelate(m_fAstRatio, m_fAstAngle, m_fExtPhase);
	float fOldCC = m_fCCMax;
	//-----------------
	float fDfRange = m_afDfRange[1] - m_afDfRange[0];
	float fAstRange = m_afAstRange[1] - m_afAstRange[0];
	float fAngRange = m_afAngRange[1] - m_afAngRange[0];
	float fPhaseRange = m_afPhaseRange[1] - m_afPhaseRange[0];
	//-----------------
	int iIterations = 30;
	float afDfRange[2], afAstRange[2], afAngRange[2], afPhaseRange[2];
        //-----------------
	for(int i=0; i<iIterations; i++)
	{	float fScale = 1.0f - i * 0.5f / iIterations;
		float fRange = fScale * fDfRange;
		mGetRange(m_fDfMean, fRange, m_afDfRange, afDfRange);
		//----------------
		fRange = fScale * fAstRange;
		mGetRange(m_fAstRatio, fRange, m_afAstRange, afAstRange);
		//----------------
		fRange = fScale * fAngRange;
		mGetRange(m_fAstAngle, fRange, m_afAngRange, afAngRange);
		//----------------
		fRange = fScale * fPhaseRange;
		mGetRange(m_fExtPhase, fRange, m_afPhaseRange, afPhaseRange);
		//----------------
		mDoIt(afDfRange, afAstRange, afAngRange, afPhaseRange);
		if(fOldCC > m_fCCMax) break;
	}
}

void CFindDefocus2D::mDoIt
(       float* pfDfRange,
        float* pfAstRange,
        float* pfAngRange,
        float* pfPhaseRange
)
{	float fCCMaxOld = m_fCCMax;
	float fAstErr = mGridSearch(pfAstRange, pfAngRange);
	float fDfMeanErr = mRefineDfMean(pfDfRange);
	float fPhaseErr = mRefinePhase(pfPhaseRange);
	//-----------------
	float fSumErr = fAstErr + fDfMeanErr + fPhaseErr;
	//printf("Iter: %3d  %9.5e  %9.5e\n", i+1, m_fCCMax, fSumErr);
}

float CFindDefocus2D::mGridSearch(float* pfAstRange, float* pfAngRange)
{
	float fTiny = (float)1e-20;
	float fAstRange = pfAstRange[1] - pfAstRange[0];
	float fAngRange = pfAngRange[1] - pfAngRange[0];
	if(fAstRange < fTiny  && fAngRange < fTiny) return 0.0f;
	//-----------------
	int iAstSteps = 51, iAngSteps = 51;
	float fAstStep = fAstRange / iAstSteps;
	float fAngStep = fAngRange / iAngSteps;
	//-----------------
	iAstSteps = (fAstStep < 1e-5) ? 1 : 51;
	iAngSteps = (fAngStep < 1e-5) ? 1 : 51;
	if(iAstSteps == 1 && iAngSteps == 1) return 0.0f;
	//-----------------
	float fAngMax, fAstMax, fCCMax = (float)-1e20;
	for(int j=0; j<iAngSteps; j++)
	{	float fAng = pfAngRange[0] + j * fAngStep;
   		for(int i=0; i<iAstSteps; i++)
		{	float fAst = pfAstRange[0] + i * fAstStep;
			float fCC = mCorrelate(fAst, fAng, m_fExtPhase);
			if(fCC <= fCCMax) continue;
			//---------------
			fCCMax = fCC;
			fAngMax = fAng;
			fAstMax = fAst;
		}
	}
	//-----------------
	if(fCCMax <= m_fCCMax) return 0.0f;
	float fErr = fabsf((m_fAstRatio - fAstMax) / (m_fAstRatio + fTiny))
	   + fabs((m_fAstAngle - fAngMax) / (m_fAstAngle + fTiny));
	//-----------------
	m_fAstRatio = fAstMax;
	m_fAstAngle = fAngMax;
	m_fCCMax = fCCMax;
	return fErr;
}

float CFindDefocus2D::mRefineDfMean(float* pfDfRange)
{
	float fTiny = (float)1e-30;
	float fDfRange = pfDfRange[1] - pfDfRange[0];
	if(fDfRange < fTiny) return 0.0f;
	//-----------------
	float fDfMeanOld = m_fDfMean;
	int iNumSteps = 101;
	float fStep = fDfRange / iNumSteps;
	//-----------------
	float fDfMeanMax = 0.0f, fCCMax = (float)-1e20;
	for(int i=0; i<iNumSteps; i++)
	{	m_fDfMean = pfDfRange[0] + i * fStep;
		float fCC = mCorrelate(m_fAstRatio, m_fAstAngle, m_fExtPhase);
		if(fCC <= fCCMax) continue;
		//----------------
		fDfMeanMax = m_fDfMean;
		fCCMax = fCC;
	}
	//-----------------
	if(fCCMax <= m_fCCMax)
	{	m_fDfMean = fDfMeanOld;
		return 0.0f;
	}
	//-----------------
	float fErr = fabsf((m_fDfMean - fDfMeanMax) / (m_fDfMean + fTiny));
	m_fDfMean = fDfMeanMax;
	m_fCCMax = fCCMax;
	return fErr;
}

float CFindDefocus2D::mRefinePhase(float* pfPhaseRange)
{
	float fTiny = (float)1e-30;
	float fPhaseRange = pfPhaseRange[1] - pfPhaseRange[0];
	if(fPhaseRange < fTiny) return 0.0f;
	//-----------------
	int iNumSteps = 61;
	float fStep = fPhaseRange / iNumSteps;
	//-----------------
	float fCCMax = (float)-1e20, fPhaseMax = 0.0f;
	for(int i=0; i<iNumSteps; i++)
	{	float fExtPhase = pfPhaseRange[0] + i * fStep;
		float fCC = mCorrelate(m_fAstRatio, m_fAstAngle, fExtPhase);
		if(fCC <= fCCMax) continue;
		//----------------
		fPhaseMax = fExtPhase;
		fCCMax = fCC;
	}
	if(fCCMax <= m_fCCMax) return 0.0f;
	//-----------------
	float fErr = fabsf((m_fExtPhase - fPhaseMax) / (m_fExtPhase + fTiny));
	m_fExtPhase = fPhaseMax;
	m_fCCMax = fCCMax;
	return fErr;
}

float CFindDefocus2D::mCorrelate
(	float fAstRatio, 
	float fAstAngle, 
	float fExtPhase
)
{	float fExtPhaseRad = fExtPhase * s_fD2R;
	float fAstRad = fAstAngle * s_fD2R;
	//---------------------------------
	float fDfMin = CFindCtfHelp::CalcDfMin(m_fDfMean, fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	float fDfMax = CFindCtfHelp::CalcDfMax(m_fDfMean, fAstRatio)
	   / m_pCtfParam->m_fPixelSize;
	//-----------------------------
	m_aGCalcCtf2D.DoIt(fDfMin, fDfMax, fAstRad, fExtPhaseRad, 
	   m_gfCtf2D, m_aiCmpSize);
	float fCC = m_pGCC2D->DoIt(m_gfCtf2D, m_gfSpect);
	return fCC;
}

void CFindDefocus2D::mGetRange
(	float fCentVal,
	float fRange,
	float* pfMinMax,
	float* pfRange
)
{	pfRange[0] = fCentVal - fRange * 0.5f;
	pfRange[1] = fCentVal + fRange * 0.5f;
	if(pfRange[0] < pfMinMax[0]) pfRange[0] = pfMinMax[0];
	if(pfRange[1] > pfMinMax[1]) pfRange[1] = pfMinMax[1];
}

