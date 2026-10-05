#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;
static float s_fD2R = 0.01745329f;

CCGradient::CCGradient(void)
{
	m_gfCtf2D = 0L;
	m_pGCC2D = 0L;
	m_pfScales = 0L;
}

CCGradient::~CCGradient(void)
{
	this->Clean();
}

void CCGradient::Clean(void)
{
	if(m_gfCtf2D != 0L) cudaFree(m_gfCtf2D);
	if(m_pGCC2D != 0L) delete m_pGCC2D;
	if(m_pfScales != 0L) delete[] m_pfScales;
	m_gfCtf2D = 0L;
	m_pGCC2D = 0L;
	m_pfScales = 0L;
}

void CCGradient::SetCtfParam(MD::CCtfParam* pCtfParam)
{
	m_pCtfParam = pCtfParam;
	m_aGCalcCtf2D.SetParam(m_pCtfParam);
}

void CCGradient::SetSpect
(	float* gfSpect,
	int* piCmpSize
)
{	m_gfSpect = gfSpect;
	memcpy(m_aiCmpSize, piCmpSize, sizeof(int) * 2);
	//---------------------------
	if(m_gfCtf2D != 0L) cudaFree(m_gfCtf2D);
	cudaMalloc(&m_gfCtf2D, sizeof(float) 
	   * m_aiCmpSize[0] * m_aiCmpSize[1]);
	//---------------------------
	if(m_pGCC2D != 0L) delete m_pGCC2D;
	m_pGCC2D = new GCC2D;
	m_pGCC2D->SetSize(m_aiCmpSize);	
	//---------------------------
	MD::CCtfInput* pCtfInput = MD::CCtfInput::GetInstance();
	m_pGCC2D->SetResRange(pCtfInput->m_afResRange,
	   m_pCtfParam->m_fPixelSize);
}

float CCGradient::DoIt
(	float* pfInitPoint,
	float* pfSearchRange,
	int iNumSteps
)
{	if(m_pfScales != 0L) delete[] m_pfScales;
	m_pfScales = new float[m_iDim];
	//---------------------------------------------------------
	// Scale both the point and the search range to avoid the
	// gradient determination is not biased to defocus because
	// of its much larger magnitude.
	//---------------------------------------------------------
	for(int i=0; i<m_iDim; i++)
	{	float fScale = pfInitPoint[i] + pfSearchRange[i] / 2;
		if(fScale == 0.0f) fScale = 1.0f;
		m_pfScales[i] = (float)fabs(fScale);
		pfInitPoint[i] /= m_pfScales[i];
		pfSearchRange[i] /= m_pfScales[i];
	}
	CPowell::DoIt(pfInitPoint, pfSearchRange, iNumSteps);
	//---------------------------------------------------------
	// Scale back to the input scales including the determined
	// best point.
	//---------------------------------------------------------
	for(int i=0; i<m_iDim; i++)
	{	pfInitPoint[i] *= m_pfScales[i];
		pfSearchRange[i] *= m_pfScales[i];
		m_pfBestPoint[i] *= m_pfScales[i];
	}
	//---------------------------
	if(m_pfScales != 0L) delete[] m_pfScales;
	m_pfScales = 0L;
	//---------------------------
	return m_fBestCC;
}

float CCGradient::Eval(float* pfPoint)
{
	//---------------------------------------------------------
	// pfPoint is scaled. We need to scale it back to original
	// scale first.
	//---------------------------------------------------------
	for(int i=0; i<m_iDim; i++)
	{	pfPoint[i] *= m_pfScales[i];
	}
	//---------------------------
	float fDfMean = pfPoint[0];
	float fAstRatio = pfPoint[1];
	float fAstAngle = pfPoint[2] * s_fD2R;
	float fExtPhase = (m_iDim >= 4) ? pfPoint[3] * s_fD2R : 0.0f;
	//---------------------------
	float fPixSize = m_pCtfParam->m_fPixelSize;
	//---------------------------
	float fDfMin = CFindCtfHelp::CalcDfMin(fDfMean, fAstRatio) / fPixSize;
	float fDfMax = CFindCtfHelp::CalcDfMax(fDfMean, fAstRatio) / fPixSize;
	//---------------------------
	m_aGCalcCtf2D.DoIt(fDfMin, fDfMax, 
	   fAstAngle, fExtPhase, 
	   m_gfCtf2D, m_aiCmpSize);
	m_fBestCC = m_pGCC2D->DoIt(m_gfCtf2D, m_gfSpect);
	//---------------------------
	for(int i=0; i<m_iDim; i++)
	{	pfPoint[i] /= m_pfScales[i];
	}
	//---------------------------------------------------------
	// CPowell minimizes the target function. Larger CCs are
	// better, so we need to convert CC to the best value.
	//---------------------------------------------------------
	float fBestVal = (float)exp(-m_fBestCC);
	//printf("%e  %e  %e  %e\n", fDfMin, fDfMax, fAstAngle, fExtPhase);
	//printf("CC, BestVal: %e  %e\n", fCC, fBestVal);
	return fBestVal;
}

float CCGradient::GetDfMin(void)
{
	float fDfMin = CFindCtfHelp::CalcDfMin(
	   m_pfBestPoint[0],
	   m_pfBestPoint[1]);
	return fDfMin;
}

float CCGradient::GetDfMax(void)
{
        float fDfMax = CFindCtfHelp::CalcDfMax(
	   m_pfBestPoint[0],
	   m_pfBestPoint[1]);
	return fDfMax;
}

float CCGradient::GetAstAngle(void)
{
	return m_pfBestPoint[2];
}

float CCGradient::GetExtPhase(void)
{
	if(m_iDim >= 4) return m_pfBestPoint[3];
	else return 0.0f;
}
