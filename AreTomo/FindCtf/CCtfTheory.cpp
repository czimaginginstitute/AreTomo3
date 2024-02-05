#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>

using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.01745329f;

CCtfParam::CCtfParam(void)
{
	m_fPixelSize = 1.0f;
}

CCtfParam::~CCtfParam(void)
{
}

float CCtfParam::GetWavelength(bool bAngstrom)
{
	if(bAngstrom) return (m_fWavelength * m_fPixelSize);
	else return m_fWavelength;
}

float CCtfParam::GetDefocusMax(bool bAngstrom)
{
	if(bAngstrom) return (m_fDefocusMax * m_fPixelSize);
	else return m_fDefocusMax;
}

float CCtfParam::GetDefocusMin(bool bAngstrom)
{
	if(bAngstrom) return (m_fDefocusMin * m_fPixelSize);
	else return m_fDefocusMin;
}

CCtfParam* CCtfParam::GetCopy(void)
{
	CCtfParam* pCopy = new CCtfParam;
	memcpy(pCopy, this, sizeof(CCtfParam));
	return pCopy;
}

CCtfTheory::CCtfTheory(void)
{
	m_fPI = (float)(4.0 * atan(1.0));
	m_pCtfParam = new CCtfParam;
	memset(m_pCtfParam, 0, sizeof(CCtfParam));
}

CCtfTheory::~CCtfTheory(void)
{
	if(m_pCtfParam != 0L) delete m_pCtfParam;
}

void CCtfTheory::Setup
(	float fKv, // keV
	float fCs, // mm
	float fAmpContrast,
	float fPixelSize,    // A
	float fAstTol,       // A, negative means no tolerance
	float fExtPhase      // radian
)
{	m_pCtfParam->m_fWavelength = mCalcWavelength(fKv) / fPixelSize;
	m_pCtfParam->m_fCs = (float)(fCs * 1e7 / fPixelSize);
	m_pCtfParam->m_fAmpContrast = fAmpContrast;
	m_pCtfParam->m_fAmpPhaseShift = atan(fAmpContrast / sqrt(1 - 
	   fAmpContrast * fAmpContrast));
	m_pCtfParam->m_fPixelSize = fPixelSize;
	m_pCtfParam->m_fExtPhase = fmodf(fExtPhase, m_fPI);
	m_pCtfParam->m_fAstTol = fAstTol / m_pCtfParam->m_fPixelSize;
}

void CCtfTheory::SetExtPhase(float fExtPhase, bool bDegree)
{
	if(bDegree) fExtPhase *= s_fD2R; 
	m_pCtfParam->m_fExtPhase = fmodf(fExtPhase, m_fPI);
}

float CCtfTheory::GetExtPhase(bool bDegree)
{
	if(bDegree) return m_pCtfParam->m_fExtPhase / s_fD2R;
	else return m_pCtfParam->m_fExtPhase;
}

void CCtfTheory::SetPixelSize(float fPixSize)
{
	m_pCtfParam->m_fPixelSize = fPixSize;
}

void CCtfTheory::SetDefocus
(	float fDefocusMin,   // A
	float fDefocusMax,   // A
	float fAstAzimuth    // Rad
)
{	m_pCtfParam->m_fDefocusMin = fDefocusMin / m_pCtfParam->m_fPixelSize;
	m_pCtfParam->m_fDefocusMax = fDefocusMax / m_pCtfParam->m_fPixelSize;
	m_pCtfParam->m_fAstAzimuth = fAstAzimuth;
}

void CCtfTheory::SetDefocusInPixel
(	float fDefocusMin, // pixel
	float fDefocusMax, // pixel
	float fAstAzimuth  // Radian
)
{	m_pCtfParam->m_fDefocusMin = fDefocusMin;
	m_pCtfParam->m_fDefocusMax = fDefocusMax;
	m_pCtfParam->m_fAstAzimuth = fAstAzimuth;
}

void CCtfTheory::SetParam
(	CCtfParam* pCtfParam
)
{	if(m_pCtfParam == pCtfParam) return;
	memcpy(m_pCtfParam, pCtfParam, sizeof(CCtfParam));
}

CCtfParam* CCtfTheory::GetParam(bool bCopy)
{
	if(!bCopy) return m_pCtfParam;
	//----------------------------
	CCtfParam* pCopy = m_pCtfParam->GetCopy();
	return pCopy;
}

float CCtfTheory::Evaluate
(	float fFreq,   // relative frquency in [-0.5, 0.5]
	float fAzimuth
)
{	float fPhaseShift = CalcPhaseShift(fFreq, fAzimuth);
	return (float)(-sin(fPhaseShift));
}

//------------------------------------------------------------------------------
// 1. Return number of extrema before the given spatial frequency.
//    Eq. 11 of Rohou & Grigoriff 2015
// 2. fFrequency is relative frequency in [-0.5, +0.5].
//------------------------------------------------------------------------------
int CCtfTheory::CalcNumExtrema
(	float fFreq,
	float fAzimuth
)
{	float fPhaseShift = CalcPhaseShift(fFreq, fAzimuth);
	int iNumExtrema = (int)(fPhaseShift / m_fPI + 0.5f);
	return iNumExtrema;
}

//-----------------------------------------------------------------------------
// 1. Return the spatial frequency of Nth zero.
//    The returned frequency is in 1/pixel.
//-----------------------------------------------------------------------------
float CCtfTheory::CalcNthZero(int iNthZero, float fAzimuth)
{
	float fPhaseShift = iNthZero * m_fPI;
	float fFreq = CalcFrequency(fPhaseShift, fAzimuth);
	return fFreq;
}

//-----------------------------------------------------------------------------
//  1. Calculate defocus in pixel at the given azumuth angle.
//  2. fDefocusMin, fDefocusMax are the min, max defocus at the major and
//     minor axis.
//  3. fAstAzimuth is the angle of the astimatism, or the major axis.
//  4. fDefocusMax must be larger than fDefocusMin.
//-----------------------------------------------------------------------------
float CCtfTheory::CalcDefocus(float fAzimuth)
{
	float fSumDf = m_pCtfParam->m_fDefocusMax
		+ m_pCtfParam->m_fDefocusMin;
	float fDifDf = m_pCtfParam->m_fDefocusMax
		- m_pCtfParam->m_fDefocusMin;
	double dCosA = cos(2.0 * (fAzimuth - m_pCtfParam->m_fAstAzimuth));
	float fDefocus = (float)(0.5 * (fSumDf + fDifDf * dCosA));
	return fDefocus;
}

float CCtfTheory::CalcPhaseShift
(	float fFreq,
	float fAzimuth
)
{	float fS2 = fFreq * fFreq;
	float fW2 = m_pCtfParam->m_fWavelength
		* m_pCtfParam->m_fWavelength;
	float fDefocus = CalcDefocus(fAzimuth);
	float fPhaseShift = m_fPI * m_pCtfParam->m_fWavelength * fS2
		* (fDefocus - 0.5f * fW2 * fS2 * m_pCtfParam->m_fCs)
		+ m_pCtfParam->m_fAmpPhaseShift
		+ m_pCtfParam->m_fExtPhase;
	return fPhaseShift;
}

//-----------------------------------------------------------------------------
// 1. Returen spatial frequency in 1/pixel given phase shift and fAzimuth
//    in radian.
//-----------------------------------------------------------------------------
float CCtfTheory::CalcFrequency
(	float fPhaseShift,
	float fAzimuth
)
{	float fDefocus = CalcDefocus(fAzimuth);
	double dW3 = pow(m_pCtfParam->m_fWavelength, 3.0);
	double a = -0.5 * m_fPI * dW3 * m_pCtfParam->m_fCs;
	double b = m_fPI * m_pCtfParam->m_fWavelength * fDefocus;
	double c = m_pCtfParam->m_fExtPhase
		+ m_pCtfParam->m_fAmpPhaseShift;
	double dDet = b * b - 4.0 * a * (c - fPhaseShift);
	//------------------------------------------------
	if(m_pCtfParam->m_fCs == 0)
	{	double dFreq2 = (fPhaseShift - c) / b;
		if(dFreq2 > 0) return (float)sqrt(dFreq2);
		else return 0.0f;
	}
	else if(dDet < 0.0)
	{	return 0.0f;
	}
	else
	{	double dSln1 = (-b + sqrt(dDet)) / (2 * a);
		double dSln2 = (-b - sqrt(dDet)) / (2 * a);
		if(dSln1 > 0) return (float)sqrt(dSln1);
		else if(dSln2 > 0) return (float)sqrt(dSln2);
		else return 0.0f;
	}
}

//-----------------------------------------------------------------------------
// 1. Compare if this instance is almost equal to the argument pCtf in terms
//    of their member values.
// 2. fDfTol is the tolerance of defocus comparison.
// 3. Other tolerances are hard-coded.
//-----------------------------------------------------------------------------
bool CCtfTheory::EqualTo(CCtfTheory* pCtf, float fDfTol)
{
	bool bCopy = true;
	CCtfParam* pParam = pCtf->GetParam(!bCopy);
	float fDif = m_pCtfParam->m_fDefocusMax - pParam->m_fDefocusMax;
	if(fabs(fDif) > fDfTol) return false;
	//-----------------------------------
	fDif = m_pCtfParam->m_fDefocusMin - pParam->m_fDefocusMin;
	if(fabs(fDif) > fDfTol) return false;
	//-----------------------------------
	fDif = m_pCtfParam->m_fCs - pParam->m_fCs;
	if(fabs(fDif) > 0.01) return false;
	//---------------------------------
	fDif = m_pCtfParam->m_fWavelength - pParam->m_fWavelength;
	if(fabs(fDif) > 1e-4) return false;
	//---------------------------------
	fDif = m_pCtfParam->m_fAmpContrast - pParam->m_fAmpContrast;
	if(fabs(fDif) > 1e-4) return false;
	//---------------------------------
	float f5Degree = 0.0277f;
	double dDif = fabs(m_pCtfParam->m_fExtPhase - pParam->m_fExtPhase);
	dDif = fmod(fDif, 2.0 * m_fPI);
	if(dDif > f5Degree) return false;
	//-------------------------------
	dDif = fabs(m_pCtfParam->m_fAstAzimuth - pParam->m_fAstAzimuth);
	dDif = fmod(dDif, m_fPI);
	if(dDif > f5Degree) return false;
	//-------------------------------
	return true;
}

CCtfTheory* CCtfTheory::GetCopy(void)
{
	bool bCopy = true;
	CCtfTheory* pCopy = new CCtfTheory;
	pCopy->SetParam(this->GetParam(!bCopy));
	return pCopy;
}

float CCtfTheory::GetPixelSize(void)
{
	return m_pCtfParam->m_fPixelSize;
}

//-----------------------------------------------------------------------------
// Given acceleration voltage in keV, return the electron wavelength.
//-----------------------------------------------------------------------------
float CCtfTheory::mCalcWavelength(float fKv)
{
	double dWl = 12.26 / sqrt(fKv * 1000 + 0.9784 * fKv * fKv);
	return (float)dWl;
}

//-----------------------------------------------------------------------------
// Enforce that m_fDefocusMax > m_fDefocusMin and -90 < m_fAstAzimuth < 90.
//-----------------------------------------------------------------------------
void CCtfTheory::mEnforce(void)
{
	m_pCtfParam->m_fAstAzimuth -= m_fPI
	   * round(m_pCtfParam->m_fAstAzimuth / m_fPI);
	if(m_pCtfParam->m_fDefocusMax < m_pCtfParam->m_fDefocusMin)
	{	float fTemp = m_pCtfParam->m_fDefocusMax;
		m_pCtfParam->m_fDefocusMax = m_pCtfParam->m_fDefocusMin;
		m_pCtfParam->m_fDefocusMin = fTemp;
	}
}
