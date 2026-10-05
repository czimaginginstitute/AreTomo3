#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>

using namespace McAreTomo;
using namespace McAreTomo::DataUtil;

static float s_fD2R = 0.01745329f;
CCtfInput* CCtfInput::m_pInstance = 0L;

CCtfInput* CCtfInput::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CCtfInput;
	return m_pInstance;
}

void CCtfInput::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CCtfInput::CCtfInput(void)
{
	m_afResRange[0] = 30.0f;
	m_afResRange[1] = 4.0f;
	m_afDfRange[0] = 1000.0f;
	m_afDfRange[1] = 40000.0f;
	m_afAstMagRange[0] = 0.0f;
	m_afAstMagRange[1] = 0.2f;
	m_afAstAngRange[0] = 0.0f;
	m_afAstAngRange[1] = 180.0;
	//---------------------------
	CAtInput* pAtInput = CAtInput::GetInstance();
	float fCentVal = pAtInput->m_afExtPhase[0];
	m_afPhaseRange[0] = fCentVal - pAtInput->m_afExtPhase[1] * 0.5f;
	m_afPhaseRange[1] = fCentVal + pAtInput->m_afExtPhase[1] * 0.5f;
}

CCtfInput::~CCtfInput(void)
{
}

void CCtfInput::GetDfRange
(	float fCentVal,
	float fRange,
	float* pfAllowed
)
{	pfAllowed[0] = fCentVal - fRange * 0.5f;
	pfAllowed[1] = fCentVal + fRange;
	ClipRange(pfAllowed, m_afDfRange);
}

void CCtfInput::GetAstMagRange
(	float fCentVal,
	float fRange,
	float* pfAllowed
)
{	pfAllowed[0] = fCentVal - fRange * 0.5f;
        pfAllowed[1] = fCentVal + fRange;
        ClipRange(pfAllowed, m_afAstMagRange);
}

void CCtfInput::GetAstAngRange
(       float fCentVal,
        float fRange,
        float* pfAllowed
)
{       pfAllowed[0] = fCentVal - fRange * 0.5f;
        pfAllowed[1] = fCentVal + fRange;
        ClipRange(pfAllowed, m_afAstAngRange);
}

void CCtfInput::GetPhaseRange
(       float fCentVal,
        float fRange,
        float* pfAllowed
)
{       pfAllowed[0] = fCentVal - fRange * 0.5f;
        pfAllowed[1] = fCentVal + fRange;
        ClipRange(pfAllowed, m_afPhaseRange);
}

void CCtfInput::ClipRange(float* pfRange, float* pfMaxRange)
{
	pfRange[0] = fmax(pfRange[0], pfMaxRange[0]);
	pfRange[1] = fmin(pfRange[1], pfMaxRange[1]);
	if(pfRange[1] < pfRange[0]) pfRange[1] = pfRange[0];
}

bool CCtfInput::bExtPhase(void)
{
	float fRange = m_afPhaseRange[1] - m_afPhaseRange[0];
	if(fRange > 0) return true;
	else return false;
}
