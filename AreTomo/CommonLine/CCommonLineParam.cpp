#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;

CCommonLineParam* CCommonLineParam::m_pInstances = 0L;
int CCommonLineParam::m_iNumGpus = 0;

void CCommonLineParam::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CCommonLineParam[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CCommonLineParam::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CCommonLineParam* CCommonLineParam::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CCommonLineParam::CCommonLineParam(void)
{
	m_pfRotAngles = 0L;
}

CCommonLineParam::~CCommonLineParam(void)
{
	this->Clean();
}

void CCommonLineParam::Clean(void)
{
	if(m_pfRotAngles != 0L) delete[] m_pfRotAngles;
	m_pfRotAngles = 0L;
}

void CCommonLineParam::Setup
(	float fAngRange, 
	int iNumSteps
)
{	this->Clean();
	m_fAngRange = fAngRange;
	m_iNumLines = iNumSteps;
	//-----------------
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPackage->GetSeries(0);
	MAM::CAlignParam* pAlignParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	int iZeroTilt = pAlignParam->GetFrameIdxFromTilt(0.0f);
	m_fTiltAxis = pAlignParam->GetTiltAxis(iZeroTilt);	
	//-----------------
	m_pfRotAngles = new float[m_iNumLines];
	float fAngStep = fAngRange / (m_iNumLines - 1);
	m_pfRotAngles[0] = m_fTiltAxis - fAngStep * m_iNumLines / 2;
	for(int i=1; i<m_iNumLines; i++)
	{	m_pfRotAngles[i] = m_pfRotAngles[i-1] + fAngStep;
	}
	//-----------------
	int iSizeX = pTiltSeries->m_aiStkSize[0];
	int iSizeY = pTiltSeries->m_aiStkSize[1];
	double dRad = 3.1415926 / 180.0;
	float fMinSize = (float)1e20;
	for(int i=0; i<m_iNumLines; i++)
	{	float fSin = (float)fabs(sin(dRad * m_pfRotAngles[i]));
		float fCos = (float)fabs(cos(dRad * m_pfRotAngles[i]));
		float fSize1 = iSizeX / (fSin + 0.000001f);
		float fSize2 = iSizeY / (fCos + 0.000001f);
		if(fMinSize > fSize1) fMinSize = fSize1;
		if(fMinSize > fSize2) fMinSize = fSize2;
	}
	m_iLineSize = (int)fMinSize;
	m_iLineSize = m_iLineSize / 2 * 2 - 100;
	m_iCmpLineSize = m_iLineSize / 2 + 1;
	printf("CommonLine: Line Size = %d\n", m_iLineSize);
}
