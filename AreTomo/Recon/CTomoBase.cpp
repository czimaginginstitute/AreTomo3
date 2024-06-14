#include "CReconInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Util/CUtilInc.h"
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Recon;

CTomoBase::CTomoBase(void)
{
	m_gfCosSin = 0L;
	m_gbNoProjs = 0L;
}

CTomoBase::~CTomoBase(void)
{
	this->Clean();
}

void CTomoBase::Clean(void)
{
	if(m_gfCosSin != 0L) cudaFree(m_gfCosSin);
	if(m_gbNoProjs != 0L) cudaFree(m_gbNoProjs);
	m_gfCosSin = 0L;
	m_gbNoProjs = 0L;
}

void CTomoBase::Setup
(	int iVolX,
	int iVolZ,
	MD::CTiltSeries* pTiltSeries,
	MAM::CAlignParam* pAlignParam
)
{	this->Clean();
	//-----------------
	m_aiVolSize[0] = iVolX;
	m_aiVolSize[1] = iVolZ;
	m_pTiltSeries = pTiltSeries;
	m_pAlignParam = pAlignParam;
	//-----------------
	m_iPadProjX = (m_pTiltSeries->m_aiStkSize[0] / 2 + 1) * 2;
	m_iNumProjs = m_pTiltSeries->m_aiStkSize[2];
	//-----------------
	size_t tBytes = sizeof(bool) * m_iNumProjs;
	cudaMalloc(&m_gbNoProjs, tBytes);
	//-----------------
	tBytes = sizeof(float) * m_iNumProjs * 2;
	cudaMalloc(&m_gfCosSin, tBytes);
	bool bCopy = true;
	float fRad = 3.1415926f / 180.0f;
	float* pfTilts = m_pAlignParam->GetTilts(!bCopy);
	float* pfCosSin = new float[m_iNumProjs * 2];
	for(int i=0; i<m_iNumProjs; i++)
	{	int j = 2 * i;
		float fAngle = fRad * pfTilts[i];
		pfCosSin[j] = (float)cos(fAngle);
		pfCosSin[j+1] = (float)sin(fAngle);
	}
	cudaMemcpy(m_gfCosSin, pfCosSin, tBytes, cudaMemcpyDefault);
	delete[] pfCosSin;
	//-----------------
	int aiPadProjSize[] = {m_iPadProjX, m_iNumProjs};
        m_aGBackProj.SetSize(aiPadProjSize, m_aiVolSize);
}

void CTomoBase::ExcludeTilts(float* pfTilts, int iNumTilts)
{
	bool* pbNoProjs = new bool[m_iNumProjs];
	memset(pbNoProjs, 0, sizeof(bool) * m_iNumProjs);
	for(int i=0; i<iNumTilts; i++)
	{	int j = m_pAlignParam->GetFrameIdxFromTilt(pfTilts[i]);
		pbNoProjs[j] = true;
	}
	cudaMemcpy(m_gbNoProjs, pbNoProjs, sizeof(bool) * m_iNumProjs,
	   cudaMemcpyDefault);
}

