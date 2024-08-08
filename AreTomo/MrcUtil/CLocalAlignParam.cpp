#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::MrcUtil;

CLocalAlignParam* CLocalAlignParam::m_pInstances = 0L;
int CLocalAlignParam::m_iNumGpus = 0;

void CLocalAlignParam::CreateInstances(int iNumGpus)
{
	if(iNumGpus == m_iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CLocalAlignParam[iNumGpus];
	//-----------------
	for(int i=0; i<iNumGpus; i++) m_pInstances[i].m_iNthGpu = i;
	m_iNumGpus = iNumGpus;
}

void CLocalAlignParam::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CLocalAlignParam* CLocalAlignParam::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CLocalAlignParam::CLocalAlignParam(void)
{
	m_pfCoordXs = 0L;
	m_iNumParams = 5;
	m_iNumPatches = 0;
}

CLocalAlignParam::~CLocalAlignParam(void)
{
	this->Clean();
}

void CLocalAlignParam::Clean(void)
{
	if(m_pfCoordXs == 0L) return;
	cudaFreeHost(m_pfCoordXs);
	m_pfCoordXs = 0L;
}

void CLocalAlignParam::Setup(int iNumTilts, int iNumPatches)
{
	this->Clean();
	m_iNumTilts = iNumTilts;
	m_iNumPatches = iNumPatches;
	if(m_iNumPatches <= 0) return;
	//-----------------
	int iSize = m_iNumTilts * m_iNumPatches;
	int iBytes = sizeof(float) * iSize * m_iNumParams;
	cudaMallocHost(&m_pfCoordXs, iBytes);
	m_pfCoordYs = m_pfCoordXs + iSize;
	m_pfShiftXs = m_pfCoordXs + iSize * 2;
	m_pfShiftYs = m_pfCoordXs + iSize * 3;
	m_pfGoodShifts = m_pfCoordXs + iSize * 4;
	//-----------------
	memset(m_pfCoordXs, 0, iBytes);
}

void CLocalAlignParam::GetParam(int iTilt, float* gfAlnParam)
{
	int iSize = m_iNumTilts * m_iNumPatches;
	int iOffset = iTilt * m_iNumPatches;
	int iBytes = m_iNumPatches * sizeof(float);
	for(int i=0; i<m_iNumParams; i++)
	{	float* pfSrc = m_pfCoordXs + i * iSize + iOffset;
		float* gfDst = gfAlnParam + i * m_iNumPatches;
		cudaMemcpy(gfDst, pfSrc, iBytes, cudaMemcpyDefault);
	}
}

void CLocalAlignParam::GetCoordXYs
(	int iTilt, 
	float* pfCoordXs, 
	float* pfCoordYs
)
{	int iBytes = m_iNumPatches * sizeof(float);
	int iOffset = iTilt * m_iNumPatches;
	memcpy(pfCoordXs, m_pfCoordXs + iOffset, iBytes);
	memcpy(pfCoordYs, m_pfCoordYs + iOffset, iBytes); 
}

void CLocalAlignParam::SetCoordXY(int iTilt, int iPatch, float fX, float fY)
{
	int i = iTilt * m_iNumPatches + iPatch;
	m_pfCoordXs[i] = fX;
	m_pfCoordYs[i] = fY;
}

void CLocalAlignParam::SetShift(int iTilt, int iPatch, float fSx, float fSy)
{
	int i = iTilt * m_iNumPatches + iPatch;
	m_pfShiftXs[i] = fSx;
	m_pfShiftYs[i] = fSy;
}

void CLocalAlignParam::SetBad(int iTilt, int iPatch, bool bBad)
{
	int i = iTilt * m_iNumPatches + iPatch;
	if(bBad) m_pfGoodShifts[i] = 0.0f;
	else m_pfGoodShifts[i] = 1.0f;
}

void CLocalAlignParam::GetCoordXY(int iTilt, int iPatch, float* pfCoord)
{
	int i = iTilt * m_iNumPatches + iPatch;
        pfCoord[0] = m_pfCoordXs[i];
        pfCoord[1] = m_pfCoordYs[i];
}

void CLocalAlignParam::GetShift(int iTilt, int iPatch, float* pfShift)
{
	int i = iTilt * m_iNumPatches + iPatch;
	pfShift[0] = m_pfShiftXs[i];
	pfShift[1] = m_pfShiftYs[i];
}

float CLocalAlignParam::GetGood(int iTilt, int iPatch)
{
	int i = iTilt * m_iNumPatches + iPatch;
	return m_pfGoodShifts[i];
}

float CLocalAlignParam::GetBadPercentage(float fMaxTilt)
{
	CAlignParam* pAlnParam = CAlignParam::GetInstance(m_iNthGpu);
	float fTiltCount = 0.0f;
	int iBadCount = 0;
	for(int i=0; i<m_iNumTilts; i++)
	{	if(fabs(pAlnParam->GetTilt(i)) > fMaxTilt) continue;
		int iNumBads = mGetNumBads(i);
		iBadCount += iNumBads;
		fTiltCount += 1.0f;
	}
	float fPercentage = iBadCount / (fTiltCount * m_iNumPatches);
	return fPercentage;
}

int CLocalAlignParam::mGetNumBads(int iTilt)
{
	int iBadCount = 0;
	float* pfGoodShifts = &m_pfGoodShifts[iTilt * m_iNumPatches];
	for(int i=0; i<m_iNumPatches; i++)
	{	if(pfGoodShifts[i] > 0.1f) continue;
		else iBadCount += 1;
	}
	return iBadCount;
}

