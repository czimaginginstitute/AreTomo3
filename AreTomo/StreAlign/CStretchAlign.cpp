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
	m_pbBadImgs = 0L;
}

CStretchAlign::~CStretchAlign(void)
{
	if(m_pcLog != 0L) delete[] m_pcLog;
	if(m_pbBadImgs != 0L) delete[] m_pbBadImgs;
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
	m_iZeroTilt = m_pAlignParam->GetFrameIdxFromTilt(0.0f);
	int iNumTilts = m_pTiltSeries->m_aiStkSize[2];
	if(m_pcLog != 0L) delete[] m_pcLog;
	int iSize = (iNumTilts + 16) * 256;
	m_pcLog = new char[iSize];
	memset(m_pcLog, 0, sizeof(char) * iSize);
	strcpy(m_pcLog, "Stretching based alignment\n");
	//-----------------
	m_pbBadImgs = new bool[iNumTilts];
	memset(m_pbBadImgs, 0, sizeof(bool) * iNumTilts);
	float fTol = 0.15f * m_pTiltSeries->m_aiStkSize[0] * m_afBinning[0];
	//-----------------
	for(int i=m_iZeroTilt+1; i<iNumTilts; i++)
	{	float fErr = mMeasure(i);
		if(fErr > fTol) m_pbBadImgs[i] = true; 
		if(fErr > m_fMaxErr) m_fMaxErr = fErr;
	}
	for(int i=m_iZeroTilt-1; i>=0; i--)
	{	float fErr = mMeasure(i);
		if(fErr > fTol) m_pbBadImgs[i] = true;
		if(fErr > m_fMaxErr) m_fMaxErr = fErr;
	}
	m_stretchXcf.Clean();
	//-----------------
	for(int i=0; i<iNumTilts; i++)
	{	printf("%s", &m_pcLog[i*256]);
	}
	printf("\n");
	//-----------------
	if(m_pcLog != 0L) delete[] m_pcLog;
	if(m_pbBadImgs != 0L) delete[] m_pbBadImgs;
	m_pcLog = 0L;
	m_pbBadImgs = 0L;
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
	char* pcLog = m_pcLog + iProj * 256;
	sprintf(pcLog, " %3d  %8.2f %8.2f  %8.2f\n", iProj+1,
		fTilt, afShift[0], afShift[1]);
	//-----------------
	float fErr = (float)sqrt(afShift[0] * afShift[0]
		+ afShift[1] * afShift[1]);
	return fErr;
}

int CStretchAlign::mFindRefIndex(int iProj)
{
	if(iProj == m_iZeroTilt) return iProj;
	//-----------------
	if(iProj > m_iZeroTilt)
	{	for(int i=iProj-1; i>m_iZeroTilt; i--)
		{	if(m_pbBadImgs[i]) continue;
			else return i;
		}
		return m_iZeroTilt;
	}
	else
	{	for(int i=iProj+1; i<m_iZeroTilt; i++)
		{	if(m_pbBadImgs[i]) continue;
			else return i;
		}
	}
	return iProj;
}
