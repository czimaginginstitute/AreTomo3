#include "CReconInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Recon;

CDoSartRecon::CDoSartRecon(void)
{
	m_iNumIters = 1;
	m_iNumSubsets = 1;
	m_pVolSeries = 0L;
}

CDoSartRecon::~CDoSartRecon(void)
{
	this->Clean();
}

void CDoSartRecon::Clean(void)
{
	CDoBaseRecon::Clean();
	m_aTomoSart.Clean();
	if(m_pVolSeries != 0L) delete m_pVolSeries;
	m_pVolSeries = 0L;
}

MD::CTiltSeries* CDoSartRecon::DoIt
(	MD::CTiltSeries* pTiltSeries,
	MAM::CAlignParam* pAlignParam,
	int iStartTilt,
	int iNumTilts,
	int iVolZ,
	int iIterations,
	int iNumSubsets
)
{	m_pTiltSeries = pTiltSeries;
	m_pAlignParam = pAlignParam;
	m_iVolZ = iVolZ;
	m_iStartTilt = iStartTilt;
	m_iNumTilts = iNumTilts;
	m_iNumIters = iIterations;
	m_iNumSubsets = iNumSubsets;
	//-----------------
	int aiVolSize[3] = {1, iVolZ, pTiltSeries->m_aiStkSize[1]};
	aiVolSize[0] = pTiltSeries->m_aiStkSize[0] / 2 * 2;
	m_pVolSeries = new MD::CTiltSeries;
	m_pVolSeries->Create(aiVolSize, aiVolSize[2]);
	//-----------------
	mDoIt();
	//-----------------
	MD::CTiltSeries* pVolSeries = m_pVolSeries;
	m_pVolSeries = 0L;
	return pVolSeries;		
}

void CDoSartRecon::mDoIt(void)
{
	int iPadX = (m_pTiltSeries->m_aiStkSize[0] / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * iPadX * m_pTiltSeries->m_aiStkSize[2];
	cudaMalloc(&m_gfPadSinogram, tBytes);
	cudaMallocHost(&m_pfPadSinogram, tBytes);	
	//-----------------
	tBytes = m_pVolSeries->GetPixels() * sizeof(float); 
	cudaMalloc(&m_gfVolXZ, tBytes);
	cudaMemset(m_gfVolXZ, 0, tBytes);
	cudaMallocHost(&m_pfVolXZ, tBytes);
	//-----------------
	m_aTomoSart.Setup
	( m_pVolSeries->m_aiStkSize[0], m_pVolSeries->m_aiStkSize[1],
	  m_iNumSubsets, m_iNumIters, m_pTiltSeries, m_pAlignParam,
	  m_iStartTilt, m_iNumTilts
	);
	//-----------------
	cudaStreamCreate(&m_stream);
	cudaEventCreate(&m_eventSino);
	//-----------------
	int iLastY = -1;
	for(int iY=0; iY<m_pTiltSeries->m_aiStkSize[1]; iY++)
	{	if(iY % 101 == 0)
		{	int iLeft = m_pTiltSeries->m_aiStkSize[1] - 1 - iY;
			printf("...... reconstruct slice %4d, "
			   "%4d slices left\n", iY+1, iLeft);
		}
		//----------------
		mExtractSinogram(iY);
		mGetReconResult(iLastY);
		mReconstruct(iY);
		iLastY = iY;
	}
	cudaStreamSynchronize(m_stream);
	mGetReconResult(iLastY);
	//-----------------
	cudaStreamDestroy(m_stream);
	cudaEventDestroy(m_eventSino);
}

void CDoSartRecon::mExtractSinogram(int iY)
{
	int iProjX = m_pTiltSeries->m_aiStkSize[0];
	int iPadX = (iProjX / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * iProjX;
	for(int i=0; i<m_pTiltSeries->m_aiStkSize[2]; i++)
	{	float* pfProj = (float*)m_pTiltSeries->GetFrame(i);
		float* pfSrc = pfProj + iY * iProjX;
		float* pfDst = m_pfPadSinogram + i * iPadX;
		memcpy(pfDst, pfSrc, tBytes);
	}
	//-----------------
	cudaStreamWaitEvent(m_stream, m_eventSino, 0);
	tBytes = sizeof(float) * iPadX * m_pTiltSeries->m_aiStkSize[2];
	cudaMemcpyAsync(m_gfPadSinogram, m_pfPadSinogram, tBytes,
		cudaMemcpyDefault, m_stream);
	cudaEventRecord(m_eventSino, m_stream);	
}

void CDoSartRecon::mGetReconResult(int iLastY)
{
	if(iLastY < 0) return;
	float* pfVolXZ = (float*)m_pVolSeries->GetFrame(iLastY);
	cudaStreamSynchronize(m_stream);
	//---------------------------------------
	//  Flip z axis to match IMOD convention
	//---------------------------------------
	int iBytes = m_pVolSeries->m_aiStkSize[0] * sizeof(float);
	int iLastZ = m_pVolSeries->m_aiStkSize[1] - 1;
	for(int z=0; z<=iLastZ; z++)
	{	float* pfSrc = m_pfVolXZ + z * m_pVolSeries->m_aiStkSize[0];
		float* pfDst = pfVolXZ + (iLastZ - z)
			* m_pVolSeries->m_aiStkSize[0];
		memcpy(pfDst, pfSrc, iBytes);
	}	
}

void CDoSartRecon::mReconstruct(int iY)
{
	size_t tBytes = m_pVolSeries->GetPixels() * sizeof(float);
	cudaMemsetAsync(m_gfVolXZ, 0, tBytes, m_stream);
	m_aTomoSart.DoIt(m_gfPadSinogram, m_gfVolXZ, m_stream);
	//-----------------
	cudaMemcpyAsync(m_pfVolXZ, m_gfVolXZ, tBytes,
	   cudaMemcpyDefault, m_stream);
}

