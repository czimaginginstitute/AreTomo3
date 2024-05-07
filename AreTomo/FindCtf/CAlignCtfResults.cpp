#include "CFindCtfInc.h"
#include <stdio.h>
#include <memory.h>

using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.017453f;

CAlignCtfResults::CAlignCtfResults(void)
{
}

CAlignCtfResults::~CAlignCtfResults(void)
{
}
		
void CAlignCtfResults::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CCtfResults* pCtfResults = 
	   MD::CCtfResults::GetInstance(m_iNthGpu);
	//-----------------------------------------------
	// 1) CTFs of dark frames have been removed in
	// CAreTomoMain::mDoFull and mSkipAlign
	//-----------------------------------------------
	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
	{	mAlignCtf(i);
	}
}

void CAlignCtfResults::mAlignCtf(int iImage)
{
	MD::CCtfResults* pCtfResults = 
	   MD::CCtfResults::GetInstance(m_iNthGpu);
	MAM::CAlignParam* pAlignParam =
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	float afShift[2] = {0.0f};
	pAlignParam->GetShift(iImage, afShift);
	float fTiltAxis = pAlignParam->GetTiltAxis(iImage);
	float fTilt = pAlignParam->GetTilt(iImage);
	//-----------------------------------------------
	// 1) Rotating the tilt axis to y axis is 
	// clockwise, the azimuth should be subtracted.
	//-----------------------------------------------
	float fAzimuth = pCtfResults->GetAzimuth(iImage);
	fAzimuth -= fTiltAxis;
	if(fAzimuth < 0) fAzimuth += 360.0f;
	pCtfResults->SetAzimuth(iImage, fAzimuth);
	//-----------------------------------------------
	// 1) Be consistent with CCorrImgCtf.cpp.
	// Positive delta z makes less under-focus.
	//-----------------------------------------------
	float fPixSize = pCtfResults->GetPixSize(iImage);
	float fCosTx = (float)cos(fTiltAxis * s_fD2R);
	float fSinTx = (float)sin(fTiltAxis * s_fD2R);
	float fX = afShift[0] * fCosTx + afShift[1] * fSinTx;
	float fZ = fX * (float)tan(fTilt * s_fD2R);
	float fDeltaF = (-fZ) * fPixSize;
	//-----------------
	float fDfMin = pCtfResults->GetDfMin(iImage);
	float fDfMax = pCtfResults->GetDfMax(iImage);
	float fDfMean = (fDfMin + fDfMax) * 0.5f + (float)1e-30;
	float fDfAst = (fDfMean - fDfMin) / fDfMean;
	//-----------------
	fDfMean += fDeltaF;
	fDeltaF = fDfMean * fDfAst;
	fDfMin = fDfMean - fDeltaF;
	fDfMax = fDfMean + fDeltaF;
	//-----------------
	pCtfResults->SetDfMin(iImage, fDfMin);
	pCtfResults->SetDfMax(iImage, fDfMax);
}
