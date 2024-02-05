#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;

CRefineTiltAxis::CRefineTiltAxis(void)
{
	m_pfRotAngles = 0L;
	m_pfSearchRange = 0L;
	m_pfTerms = 0L;
	m_pfCoeff = 0L;
}

CRefineTiltAxis::~CRefineTiltAxis(void)
{
	this->Clean();
}

void CRefineTiltAxis::Clean(void)
{
	if(m_pfRotAngles != 0L) delete[] m_pfRotAngles;
	if(m_pfSearchRange != 0L) delete[] m_pfSearchRange;
	if(m_pfTerms != 0L) delete[] m_pfTerms;
	if(m_pfCoeff != 0L) delete[] m_pfCoeff;
	m_pfRotAngles = 0L;
	m_pfSearchRange = 0L;
	m_pfTerms = 0L;
	m_pfCoeff = 0L;
}

void CRefineTiltAxis::GetRotAngles(float* pfRotAngles)
{
	int iBytes = m_iNumProjs * sizeof(float);
	memcpy(pfRotAngles, m_pfRotAngles, iBytes);
}

void CRefineTiltAxis::Setup(int iDim, int iIterations, float fTol)
{
	this->Clean();
	Util_Powell::Setup(iDim, iIterations, fTol);
	//------------------------------------------
	m_pfSearchRange = new float[m_iDim];
	m_pfTerms = new float[m_iDim];
	m_pfCoeff = new float[m_iDim];
}

float CRefineTiltAxis::Refine
(	CPossibleLines* pPossibleLines,
	CLineSet* pLineSet
)
{	m_pPossibleLines = pPossibleLines;
	m_pLineSet = pLineSet;
	//--------------------
	m_iNumProjs = m_pPossibleLines->m_iNumProjs;
	m_iNumLines = m_pPossibleLines->m_iNumLines;
	if(m_pfRotAngles != 0L) delete[] m_pfRotAngles;
	m_pfRotAngles = new float[m_iNumProjs];
	//-------------------------------------
	float* pfRotAngles = m_pPossibleLines->m_pfRotAngles;
	m_fRefRot = (pfRotAngles[0] + pfRotAngles[m_iNumLines-1]) / 2;
	//------------------------------------------------------------
	float* pfTiltAngles = m_pPossibleLines->m_pfTiltAngles;
	m_fRefTilt = (pfTiltAngles[0] + pfTiltAngles[m_iNumProjs-1]) / 2;
	//---------------------------------------------------------------
	double dMaxTilt1 = fabs(pfTiltAngles[0] - m_fRefTilt);
	double dMaxTilt2 = fabs(pfTiltAngles[m_iNumProjs-1] - m_fRefTilt);
	if(dMaxTilt1 > dMaxTilt2) m_fNormTilt = (float)dMaxTilt1;
	else m_fNormTilt = (float)dMaxTilt2;
	//----------------------------------
	m_pfSearchRange[0] = pfRotAngles[m_iNumLines-1] - pfRotAngles[0];
	for(int i=1; i<m_iDim; i++)
	{	m_pfSearchRange[i] = m_pfSearchRange[i-1] * 0.8f;
	}
	//-------------------------------------------------------
	int iNumSteps = 101.0f;
	memset(m_pfCoeff, 0, sizeof(float) * m_iDim);
	this->DoIt(m_pfCoeff, m_pfSearchRange, iNumSteps);
	//------------------------------------------------
	mCalcRotAngles(m_pfBestPoint);
	return 1.0f - m_fBestVal;	
}

float CRefineTiltAxis::Eval(float* pfCoeff)
{
	mCalcRotAngles(pfCoeff);	
	//-----------------
	GInterpolateLineSet intLineSet;
	intLineSet.DoIt(m_pPossibleLines, m_pfRotAngles, m_pLineSet);
	//-----------------
	CSumLines sumLines;
	sumLines.DoIt(m_pLineSet);
	cufftComplex* gCmpSum = sumLines.GetSum(true);
	//-----------------
	CCalcScore calcScore;
	float fScore = calcScore.DoIt(m_pLineSet, gCmpSum);
	if(gCmpSum != 0L) cudaFree(gCmpSum);
	return 1.0f - fScore;
}

void CRefineTiltAxis::mCalcRotAngles(float* pfCoeff)
{
	m_pfTerms[0] = 1.0f;
	float fTiltBar = 0.0f;
	float* pfTiltAngles = m_pPossibleLines->m_pfTiltAngles;
	//-----------------------------------------------------
	for(int iProj=0; iProj<m_iNumProjs; iProj++)
	{	fTiltBar = (pfTiltAngles[iProj] - m_fRefTilt) / m_fNormTilt;
		float fRotA = m_fRefRot + m_pfCoeff[0] * m_pfTerms[0];
		for(int i=1; i<m_iDim; i++)
		{	m_pfTerms[i] = m_pfTerms[i-1] * fTiltBar;
			fRotA += (pfCoeff[i] * m_pfTerms[i]);
		}
		m_pfRotAngles[iProj] = fRotA;
	}
}

