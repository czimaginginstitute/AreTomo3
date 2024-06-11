#include "CReconInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Recon;

CTomoWbp::CTomoWbp(void)
{
}

CTomoWbp::~CTomoWbp(void)
{
	this->Clean();
}

void CTomoWbp::Clean(void)
{
	m_aGRWeight.Clean();
}

void CTomoWbp::Setup
(	int iVolX,
	int iVolZ,
	MD::CTiltSeries* pTiltSeries,
	MAM::CAlignParam* pAlignParam
)
{	this->Clean();
	CTomoBase::Setup(iVolX, iVolZ, pTiltSeries, pAlignParam);
        m_aGRWeight.SetSize(m_iPadProjX, m_iNumProjs);
}

void CTomoWbp::DoIt(float* gfPadSinogram, float* gfVolXZ, cudaStream_t stream)
{
	m_gfPadSinogram = gfPadSinogram;
	m_gfVolXZ = gfVolXZ;
	m_stream = stream;
	//-----------------
	bool bPadded = true;
	int aiProjSize[] = {m_iPadProjX, m_iNumProjs};
	m_aGWeightProjs.DoIt(m_gfPadSinogram, m_gfCosSin, aiProjSize, 
	   bPadded, m_aiVolSize[1], m_stream);
	//-----------------
	m_aGRWeight.DoIt(m_gfPadSinogram);
	//-----------------
	bool bSart = true;
	m_aGBackProj.DoIt(m_gfPadSinogram, m_gfCosSin, m_gbNoProjs, 
	   !bSart, 1.0f, m_gfVolXZ, m_stream);
	cudaStreamSynchronize(m_stream);
}

