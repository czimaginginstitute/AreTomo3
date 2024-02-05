#include "CReconInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Recon;

CTomoWbp::CTomoWbp(void)
{
	m_gfCosSin = 0L;
}

CTomoWbp::~CTomoWbp(void)
{
	this->Clean();
}

void CTomoWbp::Clean(void)
{
	if(m_gfCosSin != 0L) cudaFree(m_gfCosSin);
	m_gfCosSin = 0L;
	m_aGRWeight.Clean();
}

void CTomoWbp::Setup
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
	//-----------------------------------------
	size_t tBytes = sizeof(float) * m_iNumProjs * 2;
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
	//----------------
	int aiPadProjSize[] = {m_iPadProjX, m_iNumProjs};
	m_aGBackProj.SetSize(aiPadProjSize, m_aiVolSize);
	//-----------------
        m_aGRWeight.SetSize(m_iPadProjX, m_iNumProjs);
}

void CTomoWbp::DoIt(float* gfPadSinogram, float* gfVolXZ, cudaStream_t stream)
{
	m_gfPadSinogram = gfPadSinogram;
	m_gfVolXZ = gfVolXZ;
	m_stream = stream;
	//----------------
	bool bPadded = true;
	int aiProjSize[] = {m_iPadProjX, m_iNumProjs};
	m_aGWeightProjs.DoIt
	( m_gfPadSinogram, m_gfCosSin, aiProjSize, !bPadded, 
	  m_aiVolSize[1], m_stream
	);
	m_aGRWeight.DoIt(m_gfPadSinogram);
	//--------------------------------
	bool bSart = true;
	m_aGBackProj.DoIt(m_gfPadSinogram, m_gfCosSin, 0, m_iNumProjs, 
	   !bSart, 1.0f, m_gfVolXZ, m_stream);
}

