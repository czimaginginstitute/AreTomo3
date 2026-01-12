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
	if(m_gfPadForProjs != 0L) cudaFree(m_gfPadForProjs);
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
	CTomoBase::Setup(iVolX, iVolZ, pTiltSeries, pAlignParam);
	int aiPadProjSize[] = {m_iPadProjX, m_iNumProjs};
        m_aGBackProj.SetSize(aiPadProjSize, m_aiVolSize);
	//-----------------
	m_iNumSubsets = iNumSubsets;
	m_iNumIters = iNumIters;
	m_aiTiltRange[0] = iStartTilt;
	m_aiTiltRange[1] = iNumTilts;
	//-----------------
	size_t tBytes = sizeof(float) * m_iPadProjX * m_iNumProjs;
	cudaMalloc(&m_gfPadForProjs, tBytes);
	//-----------------
	bool bPadded = true;
	m_aGForProj.SetVolSize(m_aiVolSize[0], !bPadded, m_aiVolSize[1]);
}

void CTomoSart::DoIt(float* gfPadSinogram, float* gfVolXZ, cudaStream_t stream)
{
	m_gfPadSinogram = gfPadSinogram;
	m_gfVolXZ = gfVolXZ;
	m_stream = stream;
	//-----------------
	MAU::CSplitItems splitItems;
	splitItems.Create(m_aiTiltRange[1], m_iNumSubsets);
	//-----------------
	bool bPadded = true;
	int aiProjSize[] = {m_iPadProjX, m_iNumProjs};
	m_aGWeightProjs.DoIt(m_gfPadSinogram, m_gfCosSin, aiProjSize, 
	   bPadded, m_aiVolSize[1], m_stream);
	//-----------------
	float fRelax = 1.0f;
	mBackProj(m_gfPadSinogram, 0, m_iNumProjs, fRelax);
	//-----------------
	fRelax = 1.0f / m_iNumSubsets;
	if(fRelax < 0.1f) fRelax = 0.1f;
	//-----------------
	for(int iIter=0; iIter<m_iNumIters; iIter++)
	{	for(int i=0; i<m_iNumSubsets; i++)
		{	int iStartProj = splitItems.GetStart(i);
			int iNumProjs = splitItems.GetSize(i);
			//---------------
			iStartProj += m_aiTiltRange[0];
			int iEndProj = iStartProj + iNumProjs;
			//---------------
			mForProj(iStartProj, iNumProjs);
			mDiffProj(iStartProj, iNumProjs);
			mBackProj(m_gfPadForProjs, iStartProj, iEndProj, fRelax);
		}
		fRelax *= 0.8f;
	}
	cudaStreamSynchronize(m_stream);
}

void CTomoSart::mForProj(int iStartProj, int iNumProjs)
{
	float* gfCosSin = m_gfCosSin + iStartProj * 2;
	float* gfForProjs = m_gfPadForProjs + iStartProj * m_iPadProjX;
	//-----------------
	bool bPadded = true;
	int aiProjSize[] = {m_iPadProjX, iNumProjs};
	m_aGForProj.DoIt(m_gfVolXZ, gfCosSin, aiProjSize, bPadded, 
	   gfForProjs, m_stream);
}

void CTomoSart::mDiffProj(int iStartProj, int iNumProjs)
{
	int iOffset = iStartProj * m_iPadProjX;
	float* gfRawProjs = m_gfPadSinogram + iOffset;
	float* gfForProjs = m_gfPadForProjs + iOffset;
	//-----------------
	bool bPadded = true;
	int aiProjSize[] = {m_iPadProjX, iNumProjs};
	m_aGDiffProj.DoIt(gfRawProjs, gfForProjs, gfForProjs,
	   aiProjSize, bPadded, m_stream);
}

void CTomoSart::mBackProj
(	float* gfSinogram,
	int iStartProj, 
	int iEndProj,
	float fRelax
)
{	bool bSart = true;
	m_aGBackProj.SetSubset(iStartProj, iEndProj);
	m_aGBackProj.DoIt(gfSinogram, m_gfCosSin, m_gbNoProjs,
	   bSart, fRelax, m_gfVolXZ, m_stream);
}

