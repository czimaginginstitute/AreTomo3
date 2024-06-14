#include "CDataUtilInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>

using namespace McAreTomo::DataUtil;

static float s_fD2R = 0.01745329f;

CCtfParam::CCtfParam(void)
{
	m_fPixelSize = 1.0f;
	m_fExtPhase = 0.0f;
	m_fDefocusMin = 0.0f;
	m_fDefocusMax = 0.0f;
	m_fTilt = 0.0f;
	m_fScore = 0.0f;
	m_fCtfRes = 10.0f;
}

CCtfParam::~CCtfParam(void)
{
}

void CCtfParam::Setup(int iKv, float fCs, float fAC, float fPixSize)
{
	m_fAmpContrast = fAC;
	m_fPixelSize = fPixSize;
	//-----------------
	double dWl = 12.26 / sqrt(iKv * 1000 + 0.9784 * iKv * iKv);
	m_fWavelength = (float)(dWl / m_fPixelSize);
	//-----------------
	m_fCs = (float)(fCs * 1e7 / m_fPixelSize);
	//-----------------
	m_fAmpPhaseShift = atan(fAC / sqrt(1 - fAC * fAC));
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

float CCtfParam::GetDfMean(bool bAngstrom)
{
	float fDfMean = (m_fDefocusMin + m_fDefocusMax) * 0.5f;
	if(bAngstrom) fDfMean *= m_fPixelSize;
	return fDfMean;
}

float CCtfParam::GetDfSigma(bool bAngstrom)
{
	float fDfSigma = (m_fDefocusMax - m_fDefocusMin) * 0.5f;
	if(bAngstrom) fDfSigma *= m_fPixelSize;
	return fDfSigma;
}

void CCtfParam::SetParam(CCtfParam* pCtfParam)
{
	memcpy(this, pCtfParam, sizeof(CCtfParam));
}

CCtfParam* CCtfParam::GetCopy(void)
{
	CCtfParam* pCopy = new CCtfParam;
	memcpy(pCopy, this, sizeof(CCtfParam));
	return pCopy;
}

