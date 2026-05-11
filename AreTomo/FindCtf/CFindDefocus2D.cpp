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
}

CFindDefocus2D::~CFindDefocus2D(void)
{
	this->Clean();
}

void CFindDefocus2D::Clean(void)
{
	if(m_gfCtf2D != 0L) cudaFree(m_gfCtf2D);
	if(m_pGCC2D != 0L) delete m_pGCC2D;
	m_gfCtf2D = 0L;
	m_pGCC2D = 0L;
}

float CFindDefocus2D::GetDfMin(void)
{
	float fDfMean = m_afNewParam[0];
	float fAstRatio = m_afNewParam[1];
	float fDfMin = fDfMean * (1.0f - fAstRatio);
	return fDfMin;
}

float CFindDefocus2D::GetDfMax(void)
{
	float fDfMean = m_afNewParam[0];
	float fAstRatio = m_afNewParam[1];
	float fDfMax = fDfMean * (1.0f + fAstRatio);
	return fDfMax;
}

float CFindDefocus2D::GetAngle(void)
{
	return m_afNewParam[2];
}

float CFindDefocus2D::GetExtPhase(void)
{
	return m_afNewParam[3];
}

float CFindDefocus2D::GetScore(void)
{
	return m_afNewParam[4];
}

float CFindDefocus2D::GetCtfRes(void)
{
	return m_afNewParam[5];
}

void CFindDefocus2D::Setup1(MD::CCtfParam* pCtfParam, int* piCmpSize)
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
	m_pGCC2D->Setup(fMinFreq, fMaxFreq, 1.0f);
}

//--------------------------------------------------------------------
// 1. DoIt() should be called after CFindDefocus1D::DoIt(), which
//    generates an estimate of m_fDfMean.
//--------------------------------------------------------------------
void CFindDefocus2D::SetInitVals
(	float fDfMean,
	float fAstRatio,
	float fAstAngle,
	float fExtPhase
)
{	m_afNewParam[0] = fDfMean;
	m_afNewParam[1] = fAstRatio;
	m_afNewParam[2] = fAstAngle;
	m_afNewParam[3] = fExtPhase;
	m_afNewParam[4] = (float)-1e20;
	m_afNewParam[5] = (float)1e20;
}

void CFindDefocus2D::DoIt
(	float* gfSpect,
	float* pfDfRange,
	float* pfPhaseRange
)
{	m_gfSpect = gfSpect;
	this->RefineParam(m_gfSpect, 0.0, 180.0f, 1.0f, 2);
	this->RefineParam(m_gfSpect, 0.0, 0.5f, 0.005f, 1);
	mCalcCtfRes();
	//---------------------------
	mGridSearch(pfDfRange, pfPhaseRange);
	this->RefineParam(m_gfSpect, 0.0, 0.5f, 0.005f, 1);
	this->RefineParam(m_gfSpect, 0.0f, 180.0f, 1.0f, 2);
	//---------------------------
	float fPhaseRange = pfPhaseRange[1] - pfPhaseRange[0];
	if(fPhaseRange > 0.5f)
	{	this->RefineParam(m_gfSpect, 0.0f, 180.0f, 1.0f, 3);
	}
	mCalcCtfRes();
}

void CFindDefocus2D::Refine
(	float* gfSpect,
	float fDfRange,
	float fPhaseRange
)
{	m_gfSpect = gfSpect;
	//---------------------------
	float fDfMean = m_afNewParam[0];
	float fMinDf = fDfMean - 0.5f * fDfRange;
	float fMaxDf = fDfMean + 0.5f * fDfRange;
	fMinDf = fmax(fMinDf, 1000.0f);
	this->RefineParam(gfSpect, fMinDf, fMaxDf, 100.0f, 0);
	//---------------------------
	if(fPhaseRange >= 0.5f)
	{	float fExtPhase = m_afNewParam[3];
		float fMinPhase = fExtPhase - 0.5f * fPhaseRange;
		float fMaxPhase = fExtPhase + 0.5f * fPhaseRange;
		fMinPhase = fmaxf(fMinPhase, 0.0f);
		fMaxPhase = fminf(fMaxPhase, 180.0f);
		this->RefineParam(gfSpect, fMinPhase, fMaxPhase, 1.0f, 3);
	}
	//---------------------------
	mCalcCtfRes();
}

void CFindDefocus2D::RefineParam
(	float* gfSpect,
	float fMinVal,
	float fMaxVal,
	float fStep,
	int iParam
)
{	m_gfSpect = gfSpect;
	float fRange = fMaxVal - fMinVal;
	if(fRange == 0.0f) return;
	else if(fStep <= 0) return;
	else memcpy(m_afOldParam, m_afNewParam, sizeof(m_afNewParam));
	//---------------------------
	int iNumSteps = (int)(fRange / fStep + 0.5f);
	iNumSteps = iNumSteps / 2 * 2 + 1;
	int iCent = iNumSteps / 2;
	//---------------------------
	float fMaxCC = mCorrelate();
	float fBestVal = m_afNewParam[iParam];
	float fInitVal = fBestVal;
	//---------------------------
	for(int i=0; i<iNumSteps; i++)
	{	m_afNewParam[iParam] = fInitVal + fStep * (i - iCent);
		if(m_afNewParam[iParam] < fMinVal) continue;
		else if(m_afNewParam[iParam] > fMaxVal) continue;
		//-------------------
		float fCC = mCorrelate();
		if(fCC > fMaxCC)
		{	fMaxCC = fCC;
			fBestVal = m_afNewParam[iParam];
		}
	}
        //---------------------------
	if(fMaxCC > m_afNewParam[4])
	{	m_afNewParam[iParam] = fBestVal;
		m_afNewParam[4] = fMaxCC;
		if(m_afNewParam[3] > 180) m_afNewParam[3] -= 180.0f;
	}
	else memcpy(m_afNewParam, m_afOldParam, sizeof(m_afOldParam));
}

void CFindDefocus2D::mGridSearch
(	float* pfDfRange,
	float* pfPhaseRange
)
{       memcpy(m_afOldParam, m_afNewParam, sizeof(m_afOldParam));
	//---------------------------
	float fDfStep = 100.0f;
	float fPhStep = 1.0f;
	//---------------------------
        float fBestDF = 0.0f;
        float fBestPH = 0.0f;
        float fBestCC = (float)-1e20;
	//---------------------------
	for(float p=pfPhaseRange[0]; p<=pfPhaseRange[1]; p+=fPhStep)
	{	m_afNewParam[3] = p;
		for(float f=pfDfRange[0]; f<=pfDfRange[1]; f+=fDfStep)
		{	m_afNewParam[0] = f;
			float fCC = mCorrelate();
			if(fCC > fBestCC)
			{	fBestDF = f;
				fBestPH = p;
				fBestCC = fCC;
			}
		}
	}
	m_afNewParam[0] = fBestDF;
	m_afNewParam[3] = fBestPH;
	m_afNewParam[4] = fBestCC;
	//---------------------------
	if(m_afNewParam[4] < m_afOldParam[4])
	{	memcpy(m_afNewParam, m_afOldParam, sizeof(m_afNewParam));
	}
}

float CFindDefocus2D::mCorrelate(void)
{	
	float fDfMean = m_afNewParam[0];
        float fAstRatio = m_afNewParam[1];
	float fAstRad = m_afNewParam[2] * s_fD2R;
	float fExtPhaseRad = m_afNewParam[3] * s_fD2R;
	//---------------------------	
	float fDfMin = CFindCtfHelp::CalcDfMin(fDfMean, fAstRatio);
	float fDfMax = CFindCtfHelp::CalcDfMax(fDfMean, fAstRatio);
	fDfMin /= m_pCtfParam->m_fPixelSize;
	fDfMax /= m_pCtfParam->m_fPixelSize;
	//---------------------------
	m_aGCalcCtf2D.DoIt(fDfMin, fDfMax, fAstRad, fExtPhaseRad, 
	   m_gfCtf2D, m_aiCmpSize);
	float fCC = m_pGCC2D->DoIt(m_gfCtf2D, m_gfSpect);
	return fCC;
}

void CFindDefocus2D::mCalcCtfRes(void)
{
	float fDfMean = m_afNewParam[0];
	float fAstRatio = m_afNewParam[1];
	float fExtPhaseRad = m_afNewParam[3] * s_fD2R;
	float fAstRad = m_afNewParam[2] * s_fD2R;
	//---------------------------	
	float fDfMin = CFindCtfHelp::CalcDfMin(fDfMean, fAstRatio);
	float fDfMax = CFindCtfHelp::CalcDfMax(fDfMean, fAstRatio);
	fDfMin /= m_pCtfParam->m_fPixelSize;
	fDfMax /= m_pCtfParam->m_fPixelSize;
	//---------------------------
	m_aGCalcCtf2D.DoIt(fDfMin, fDfMax, fAstRad, fExtPhaseRad,
	   m_gfCtf2D, m_aiCmpSize);
	//---------------------------
	GSpectralCC2D gSpectCC;
	gSpectCC.SetSize(m_aiCmpSize);
	int iShell = gSpectCC.DoIt(m_gfCtf2D, m_gfSpect);
	m_afNewParam[5] = m_aiCmpSize[1] * m_pCtfParam->m_fPixelSize / iShell;
}

