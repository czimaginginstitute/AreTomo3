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
	m_pGCC2D->SetResRange(afResRange, m_pCtfParam->m_fPixelSize);
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
	float fRangeDF,
	float fRangeAM,
	float fRangeAA,
	float fRangePP
)
{	m_gfSpect = gfSpect;
	MD::CCtfInput* pCtfInput = MD::CCtfInput::GetInstance();
	//---------------------------
	float afRangeDF[2] = {0.0f};
	float afRangeAM[2] = {0.0f};
	float afRangeAA[2] = {0.0f};
	float afRangePP[2] = {0.0f};  // phase plate
	for(int i=1; i<3; i++)
	{	pCtfInput->GetDfRange(
		   m_afNewParam[0], fRangeDF / i,
	   	   afRangeDF);	   
		pCtfInput->GetAstMagRange(
		   m_afNewParam[1], fRangeAM / i,
		   afRangeAM);
		pCtfInput->GetAstAngRange(
		   m_afNewParam[2], fRangeAA / i,
		   afRangeAA);
		pCtfInput->GetPhaseRange(
		   m_afNewParam[3], fRangePP / i,
		   afRangePP);
		//-------------------
		mGridSearchAA(afRangeAM, afRangeAA);
		mGridSearchFP(afRangeDF, afRangePP);
		mCalcCtfRes();
	}
}

void CFindDefocus2D::Refine
(	float* gfSpect,
	float fRangeDF,
	float fRangePP
)
{	m_gfSpect = gfSpect;
	MD::CCtfInput* pCtfInput = MD::CCtfInput::GetInstance();
	//---------------------------
	float afRangeDF[2] = {0.0f};
	float afRangePP[2] = {0.0f};
	//---------------------------
	for(int i=1; i<3; i++)
	{	pCtfInput->GetDfRange(
                   m_afNewParam[0], fRangeDF / i,
                   afRangeDF);
		pCtfInput->GetPhaseRange(
                   m_afNewParam[3], fRangePP / i,
                   afRangePP);
		mGridSearchFP(afRangeDF, afRangePP);
		mCalcCtfRes();
	}
}

void CFindDefocus2D::RefineParam
(	float* gfSpect,
	float* pfRange, 
	float fStep,
	int iParam
)
{	m_gfSpect = gfSpect;
	float fRange = pfRange[1] - pfRange[0];
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
	float fInitVal = pfRange[0];
	//---------------------------
	for(int i=0; i<iNumSteps; i++)
	{	m_afNewParam[iParam] = fInitVal + fStep * (i - iCent);
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

void CFindDefocus2D::mGridSearchAA
(	float* pfAstMagRange,
	float* pfAstAngRange
)
{	memcpy(m_afOldParam, m_afNewParam, sizeof(m_afOldParam));
	//---------------------------
	int iNumStepsAM = 20;
	float fRangeAM = pfAstMagRange[1] - pfAstMagRange[0];
	float fStepAM = fRangeAM / iNumStepsAM;
	if(fStepAM <= 0) iNumStepsAM = 1;
	//---------------------------
	int iNumStepsAA = 30;
	float fRangeAA = pfAstAngRange[1] - pfAstAngRange[0];
	float fStepAA = fRangeAA / iNumStepsAA;
	if(fStepAA <= 0) iNumStepsAA = 1;
	//---------------------------
	int iSteps = iNumStepsAM * iNumStepsAA;
	if(iSteps == 1) return;
	//---------------------------
	float fBestAM = 0.0f;
	float fBestAA = 0.0f;
	float fBestCC = (float)-1e20;
	//---------------------------
	for(int i=0; i<iSteps; i++)
	{	int iAM = i % iNumStepsAM;
		int iAA = i / iNumStepsAM;
		m_afNewParam[1] = iAM * fStepAM + pfAstMagRange[0];
		m_afNewParam[2] = iAA * fStepAA + pfAstAngRange[0];
		float fCC = mCorrelate();
		if(fCC <= fBestCC) continue;
		//-------------------
		fBestAM = m_afNewParam[1];
		fBestAA = m_afNewParam[2];
		fBestCC = fCC;
	}
	m_afNewParam[1] = fBestAM;
	m_afNewParam[2] = fBestAA;
	m_afNewParam[4] = fBestCC;
	if(fBestCC > m_afOldParam[4]) return;
	//---------------------------
	memcpy(m_afNewParam, m_afOldParam, sizeof(m_afNewParam));
}

void CFindDefocus2D::mGridSearchFP
(	float* pfDfRange,
	float* pfPhaseRange
)
{       memcpy(m_afOldParam, m_afNewParam, sizeof(m_afOldParam));
	//---------------------------
	float fDfStep = 300.0f;
	float fPhStep = 3.0f;
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
	m_pGCC2D->m_fBFactor = m_pCtfParam->m_fBFactor;
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

