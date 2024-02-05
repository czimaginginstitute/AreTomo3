#include "CAlignInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;

CAlignParam* CAlignParam::m_pInstance = 0L;

CAlignParam* CAlignParam::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CAlignParam;
	return m_pInstance;
}

void CAlignParam::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CAlignParam::CAlignParam(void)
{
}

CAlignParam::~CAlignParam(void)
{
}

bool CAlignParam::bCorrectMag(void)
{
	float afStretch[2] = {1.0f, 0.0f};
	this->GetMagStretch(afStretch);
	if(fabs(afStretch[0] - 1) < 1e-5) return false;
	else return true;
}

void CAlignParam::GetMagStretch(float* pfStretch)
{
	//----------------------------------------------------
	// mag_distortion_correct_1.0.0: 03/15/2017
	// 1. Its y-axis points downward. My y-axis points
	//    upward. Therefore, the angle of major axis
	//    is multiplied by -1.
	// 2. This is found by comparing the image with
	//    mag corrected by MotionCor4 and the one
	//    by mag_distortion_correct_1.0.0 using
	//    major scale 1.5 and minor scale 1.0
	//----------------------------------------------------
	CMcInput* pInput = CMcInput::GetInstance();
	pfStretch[0] = pInput->m_afMag[0] / pInput->m_afMag[1];
	pfStretch[1] = -1.0f * pInput->m_afMag[2];
	printf("Mag stretch: %.3f  %.3f\n\n", 
	   pfStretch[0], pfStretch[1]);
}

bool CAlignParam::bPatchAlign(void)
{
	CMcInput* pInput = CMcInput::GetInstance();
	if(pInput->m_aiNumPatches[0] <= 1) return false;
	if(pInput->m_aiNumPatches[1] <= 1) return false;
	return true;
}

int CAlignParam::GetNumPatches(void)
{	
	CMcInput* pInput = CMcInput::GetInstance();
	int iPatches = pInput->m_aiNumPatches[0]
	   * pInput->m_aiNumPatches[1];
	return iPatches;
}

int CAlignParam::GetFrameRef(int iNumFrames)
{
	CMcInput* pInput = CMcInput::GetInstance();
	if(pInput->m_iFmRef < 0) return iNumFrames / 2;
	//---------------------------------------------
	int iFrameRef = pInput->m_iFmRef - 1;
	if(iFrameRef < 0) iFrameRef = 0;
	else if(iFrameRef >= iNumFrames) iFrameRef = iNumFrames - 1;
	return iFrameRef;
}

