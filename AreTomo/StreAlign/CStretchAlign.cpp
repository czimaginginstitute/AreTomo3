#include "CStreAlignInc.h"
#include "../Util/CUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::StreAlign;

static float m_fBFactor = 200.0f;
static float m_afBinning[] = {1.0f, 1.0f};

CStretchAlign::CStretchAlign(void)
{
	m_pcLog = 0L;
}

CStretchAlign::~CStretchAlign(void)
{
	if(m_pcLog != 0L) delete[] m_pcLog;
}

float CStretchAlign::DoIt
(	MD::CTiltSeries* pTiltSeries,
	MAM::CAlignParam* pAlignParam,
	float fBFactor,
	float* pfBinning
)
{	m_pTiltSeries = pTiltSeries;
	m_pAlignParam = pAlignParam;
	//-----------------
	m_fBFactor = fBFactor;
	m_afBinning[0] = pfBinning[0];
	m_afBinning[1] = pfBinning[1];
	//-----------------
	m_stretchXcf.Setup(m_pTiltSeries->m_aiStkSize, m_fBFactor); 
	m_fMaxErr = 0.0f;
	//-----------------
	if(m_pcLog != 0L) delete[] m_pcLog;
	int iSize = (m_pTiltSeries->m_aiStkSize[2] + 16) * 256;
	m_pcLog = new char[iSize];
	memset(m_pcLog, 0, sizeof(char) * iSize);
	strcpy(m_pcLog, "Stretching based alignment\n");
	//-----------------
	for(int i=0; i<m_pTiltSeries->m_aiStkSize[2]; i++)
	{	float fErr = mMeasure(i);
		if(fErr > m_fMaxErr) m_fMaxErr = fErr;
	}
	m_stretchXcf.Clean();
	//-----------------
	printf("%s\n", m_pcLog);
	if(m_pcLog != 0L) delete[] m_pcLog;
	m_pcLog = 0L;
	//-----------------
	return m_fMaxErr;
}

float CStretchAlign::mMeasure(int iProj)
{
	int iRefProj = mFindRefIndex(iProj);
	if(iRefProj == iProj) return 0.0f;
	//-----------------
	float fRefTilt = m_pAlignParam->GetTilt(iRefProj);
	float fTilt = m_pAlignParam->GetTilt(iProj);
	float fTiltAxis = m_pAlignParam->GetTiltAxis(iProj);
	//-----------------
	float* pfRefProj = (float*)m_pTiltSeries->GetFrame(iRefProj);
	float* pfProj = (float*)m_pTiltSeries->GetFrame(iProj);
	m_stretchXcf.DoIt(pfRefProj, pfProj, fRefTilt, fTilt, fTiltAxis);
	//-----------------
	float afShift[2] = {0.0f};
	m_stretchXcf.GetShift(m_afBinning[0], m_afBinning[1], afShift);
	m_pAlignParam->SetShift(iProj, afShift);
	//-----------------
	char acLog[256] = {'\0'};
	sprintf(acLog, " %3d  %8.2f %8.2f  %8.2f\n", iProj+1,
		fTilt, afShift[0], afShift[1]);
	strcat(m_pcLog, acLog);
	//-----------------
	float fErr = (float)sqrt(afShift[0] * afShift[0]
		+ afShift[1] * afShift[1]);
	return fErr;
}

int CStretchAlign::mFindRefIndex(int iProj)
{
	int iNumProjs = m_pTiltSeries->m_aiStkSize[2];
	int iProj0 = iProj - 1;
	int iProj2 = iProj + 1;
	if(iProj0 < 0) iProj0 = iProj;
	if(iProj2 >= iNumProjs) iProj2 = iProj;
	//-------------------------------------
	double dTilt0 = fabs(m_pAlignParam->GetTilt(iProj0));
	double dTilt = fabs(m_pAlignParam->GetTilt(iProj));
	double dTilt2 = fabs(m_pAlignParam->GetTilt(iProj2));
	if(dTilt0 < dTilt) return iProj0;
	else if(dTilt2 < dTilt) return iProj2;
	else return iProj;
}
