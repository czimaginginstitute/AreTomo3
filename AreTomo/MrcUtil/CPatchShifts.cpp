#include "CMrcUtilInc.h"
#include <memory.h>
#include <math.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CPatchShifts* CPatchShifts::m_pInstances = 0L;
int CPatchShifts::m_iNumGpus = 0;

void CPatchShifts::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CPatchShifts[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CPatchShifts::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CPatchShifts* CPatchShifts::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CPatchShifts::CPatchShifts(void)
{
	m_ppPatAlnParams = 0L;
	m_pbBadShifts = 0L;
	m_iZeroTilt = -1;
}

CPatchShifts::~CPatchShifts(void)
{
	this->Clean();
}

void CPatchShifts::Clean(void)
{
	if(m_ppPatAlnParams != 0L)
	{	for(int i=0; i<m_iNumPatches; i++)
		{	if(m_ppPatAlnParams[i] == 0L) continue;
			delete m_ppPatAlnParams[i];
		}
		delete[] m_ppPatAlnParams;
		m_ppPatAlnParams = 0L;
	}
	//----------------------------
	if(m_pbBadShifts != 0L)
	{	delete[] m_pbBadShifts;
		m_pbBadShifts = 0L;
	}
	//-------------------------
	m_iZeroTilt = -1;
}

void CPatchShifts::Setup(int iNumPatches, int iNumTilts)
{	
	this->Clean();
	m_iNumPatches = iNumPatches;
	m_iNumTilts = iNumTilts;
	//----------------------
	m_ppPatAlnParams = new CAlignParam*[m_iNumPatches];
	memset(m_ppPatAlnParams, 0, sizeof(CAlignParam*) * m_iNumPatches);
	//----------------------------------------------------------------
	int iSize = m_iNumPatches * m_iNumTilts;
	m_pbBadShifts = new bool[iSize];
	memset(m_pbBadShifts, 0, sizeof(bool) * iSize);
}

void CPatchShifts::SetRawShift
(	int iPatch, 
	CAlignParam* pPatAlnParam
)
{	if(m_ppPatAlnParams[iPatch] != 0L) delete m_ppPatAlnParams[iPatch];
	m_ppPatAlnParams[iPatch] = pPatAlnParam;
	//--------------------------------------
	if(m_iZeroTilt >= 0) return;
	m_iZeroTilt = pPatAlnParam->GetFrameIdxFromTilt(0.0f);
}

void CPatchShifts::GetPatCenterXYs(float* pfCentXs, float* pfCentYs)
{
	float afCent[2] = {0.0f};
	for(int i=0; i<m_iNumPatches; i++)
	{	m_ppPatAlnParams[i]->GetShift(m_iZeroTilt, afCent);
		pfCentXs[i] = afCent[0];
		pfCentYs[i] = afCent[1];
	}
}

void CPatchShifts::GetPatShifts(float* pfShiftXs, float* pfShiftYs)
{
	int iNumImgs = m_ppPatAlnParams[0]->m_iNumFrames;
	int iBytes = sizeof(float) * iNumImgs;
	for(int i=0; i<m_iNumPatches; i++)
	{	CAlignParam* pAlnParam = m_ppPatAlnParams[i];
		float* pfSrcX = pAlnParam->GetShiftXs();
		float* pfSrcY = pAlnParam->GetShiftYs();
		float* pfDstX = pfShiftXs + i * iNumImgs;
		float* pfDstY = pfShiftYs + i * iNumImgs;
		memcpy(pfDstX, pfSrcX, iBytes);
		memcpy(pfDstY, pfSrcY, iBytes);
	}
}

void CPatchShifts::RotPatCenterXYs
(	float fRot, 
	float* pfCentXs, 
	float* pfCentYs
)
{	this->GetPatCenterXYs(pfCentXs, pfCentYs);
	//----------------------------------------
	double dRad = (atan(1.0) / 45.0) * fRot;
	float fCos = (float)cos(dRad);
	float fSin = (float)sin(dRad);
	//----------------------------
	float fX, fY;
	for(int i=0; i<m_iNumPatches; i++)
	{	fX = pfCentXs[i] * fCos - pfCentYs[i] * fSin;
		fY = pfCentYs[i] * fCos + pfCentXs[i] * fSin;
		pfCentXs[i] = fX;
		pfCentYs[i] = fY;
	}
}

CAlignParam* CPatchShifts::GetAlignParam(int iPatch)
{
	return m_ppPatAlnParams[iPatch];
}

void CPatchShifts::GetShift(int iPatch, int iTilt, float* pfShift)
{
	m_ppPatAlnParams[iPatch]->GetShift(iTilt, pfShift);
}

float CPatchShifts::GetTiltAxis(int iPatch, int iTilt)
{
	return m_ppPatAlnParams[iPatch]->GetTiltAxis(iTilt);
}

void CPatchShifts::GetRotCenter(int iPatch, float* pfRotCent)
{
	return m_ppPatAlnParams[iPatch]->GetRotationCenter(pfRotCent);
}

void CPatchShifts::SetRotCenterZ(int iPatch, float fCentZ)
{
	m_ppPatAlnParams[iPatch]->SetRotationCenterZ(fCentZ);
}

	
	
