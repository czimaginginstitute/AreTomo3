#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

CSpectrumImage::CSpectrumImage(void)
{
}

CSpectrumImage::~CSpectrumImage(void)
{
}

void CSpectrumImage::DoIt
(	float* gfHalfSpect,
	float* gfCtfBuf,
	int* piCmpSize,
	CCtfTheory* pCtfTheory,
	float* pfResRange,
	float* gfFullSpect
)
{	m_aiCmpSize[0] = piCmpSize[0];
	m_aiCmpSize[1] = piCmpSize[1];
	m_afResRange[0] = pfResRange[0];
	m_afResRange[1] = pfResRange[1];
	m_pCtfTheory = pCtfTheory;
	m_gfHalfSpect = gfHalfSpect;
	m_gfCtfBuf = gfCtfBuf;
	m_gfFullSpect = gfFullSpect;
	//--------------------------
	mGenFullSpectrum();
	mEmbedCTF();
}

void CSpectrumImage::mGenFullSpectrum(void)
{
	GCalcSpectrum gCalcSpect;
	bool bPadded = true;
	gCalcSpect.GenFullSpect(m_gfHalfSpect, m_aiCmpSize,
	   m_gfFullSpect, !bPadded);
	//-----------------
	MU::GCalcMoment2D gCalcMoment;
	bool bSync = true;
	gCalcMoment.SetSize(m_aiCmpSize, !bPadded);
	m_fMean = gCalcMoment.DoIt(m_gfHalfSpect, 1, bSync);
	m_fStd = gCalcMoment.DoIt(m_gfHalfSpect, 2, bSync);
	m_fStd = m_fStd - m_fMean * m_fMean;
	if(m_fStd < 0) m_fStd = 0.0f;
	else m_fStd = sqrt(m_fStd);
}

void CSpectrumImage::mEmbedCTF(void)
{
	float fPixelSize = m_pCtfTheory->GetPixelSize();
	float fMinFreq = fPixelSize / m_afResRange[0];
	float fMaxFreq = fPixelSize / m_afResRange[1];
	float fGain = m_fStd * 1.5f;
	//--------------------------
	GCalcCTF2D gCalcCtf2D;
	MD::CCtfParam* pCtfParam = m_pCtfTheory->GetParam(false);
	gCalcCtf2D.DoIt(pCtfParam, m_gfCtfBuf, m_aiCmpSize);
	gCalcCtf2D.EmbedCtf(m_gfCtfBuf, fMinFreq, fMaxFreq,
	   m_fMean, fGain, m_gfFullSpect, m_aiCmpSize);
}
