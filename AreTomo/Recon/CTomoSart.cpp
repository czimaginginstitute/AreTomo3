#include "CReconInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Util/CUtilInc.h"
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Recon;

CTomoSart::CTomoSart(void)
{
	m_gfPadForProjs = 0L;
}

CTomoSart::~CTomoSart(void)
{
	this->Clean();
}

void CTomoSart::Clean(void)
{
	if(m_gfCosSin != 0L) cudaFree(m_gfCosSin);
	if(m_gfPadForProjs != 0L) cudaFree(m_gfPadForProjs);
	m_gfCosSin = 0L;
	m_gfPadForProjs = 0L;
}

void CTomoSart::Setup
(	int iVolX,
	int iVolZ,
	int iNumSubsets,
	int iNumIters,
	MD::CTiltSeries* pTiltSeries,
	MAM::CAlignParam* pAlignParam,
	int iStartTilt,
	int iNumTilts
)
{	this->Clean();
	//------------
	m_aiVolSize[0] = iVolX;
	m_aiVolSize[1] = iVolZ;
	m_iNumSubsets = iNumSubsets;
	m_iNumIters = iNumIters;
	m_pTiltSeries = pTiltSeries;
	m_pAlignParam = pAlignParam;
	m_aiTiltRange[0] = iStartTilt;
	m_aiTiltRange[1] = iNumTilts;
	//---------------------------
	m_iPadProjX = (m_pTiltSeries->m_aiStkSize[0] / 2 + 1) * 2;
	m_iNumProjs = m_pTiltSeries->m_aiStkSize[2];
	//-----------------------------------------
	size_t tBytes = sizeof(float) * m_iPadProjX * m_iNumProjs;
	cudaMalloc(&m_gfPadForProjs, tBytes);
	//-----------------------------------
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
	//----------------
	int aiPadProjSize[] = {m_iPadProjX, m_iNumProjs};
	m_aGBackProj.SetSize(aiPadProjSize, m_aiVolSize);
	//-----------------------------------------------
	bool bPadded = true;
	m_aGForProj.SetVolSize(m_aiVolSize[0], !bPadded, m_aiVolSize[1]);
}

void CTomoSart::DoIt(float* gfPadSinogram, float* gfVolXZ, cudaStream_t stream)
{
	m_gfPadSinogram = gfPadSinogram;
	m_gfVolXZ = gfVolXZ;
	m_stream = stream;
	//----------------
	MAU::CSplitItems splitItems;
	splitItems.Create(m_aiTiltRange[1], m_iNumSubsets);
	//-------------------------------------------------
	bool bPadded = true;
	int aiProjSize[] = {m_iPadProjX, m_iNumProjs};
	m_aGWeightProjs.DoIt
	( m_gfPadSinogram, m_gfCosSin, aiProjSize, !bPadded, 
	  m_aiVolSize[1], m_stream
	);
	//-----------------------------------------------
	float fRelax = 1.0f;
	mBackProj(m_gfPadSinogram, 0, m_iNumProjs, fRelax);
	//-------------------------------------------------
	for(int iIter=0; iIter<m_iNumIters; iIter++)
	{	fRelax = 1.0f - 0.1f * iIter;
		if(fRelax < 0.1f) fRelax = 0.1f;
		for(int i=0; i<m_iNumSubsets; i++)
		{	int iStartProj = splitItems.GetStart(i);
			iStartProj += m_aiTiltRange[0];
			int iNumProjs = splitItems.GetSize(i);
			mForProj(iStartProj, iNumProjs);
			mDiffProj(iStartProj, iNumProjs);
			mBackProj(m_gfPadForProjs, iStartProj, iNumProjs, fRelax);
		}
	}
}

void CTomoSart::mForProj(int iStartProj, int iNumProjs)
{
	float* gfCosSin = m_gfCosSin + iStartProj * 2;
	float* gfForProjs = m_gfPadForProjs + iStartProj * m_iPadProjX;
	//-------------------------------------------------------------
	bool bPadded = true;
	int aiProjSize[] = {m_iPadProjX, iNumProjs};
	m_aGForProj.DoIt
	( m_gfVolXZ, gfCosSin, aiProjSize, bPadded, 
	  gfForProjs, m_stream
	);
}

void CTomoSart::mDiffProj(int iStartProj, int iNumProjs)
{
	int iOffset = iStartProj * m_iPadProjX;
	float* gfRawProjs = m_gfPadSinogram + iOffset;
	float* gfForProjs = m_gfPadForProjs + iOffset;
	//-----------------------------------------
	bool bPadded = true;
	int aiProjSize[] = {m_iPadProjX, iNumProjs};
	m_aGDiffProj.DoIt
	( gfRawProjs, gfForProjs, gfForProjs,
	  aiProjSize, bPadded, m_stream
	);
}

void CTomoSart::mBackProj
(	float* gfSinogram,
	int iStartProj, 
	int iNumProjs, 
	float fRelax
)
{	bool bSart = true;
	m_aGBackProj.DoIt( gfSinogram, m_gfCosSin, iStartProj, 
	   iNumProjs, bSart, fRelax, m_gfVolXZ, m_stream);
}

