#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Recon;

CDoWbpRecon::CDoWbpRecon(void)
{
	m_pVolSeries = 0L;
}

CDoWbpRecon::~CDoWbpRecon(void)
{
        this->Clean();
}

void CDoWbpRecon::Clean(void)
{
	CDoBaseRecon::Clean();
	m_aTomoWbp.Clean();
	if(m_pVolSeries != 0L) delete m_pVolSeries;
	m_pVolSeries = 0L;
}

MD::CTiltSeries* CDoWbpRecon::DoIt
(	MD::CTiltSeries* pTiltSeries,
	MAM::CAlignParam* pAlignParam,
	int iVolZ
)
{	m_pTiltSeries = pTiltSeries;
	m_pAlignParam = pAlignParam;
	m_iVolZ = iVolZ;
	//-----------------
	int aiVolSize[3] = {1, iVolZ, pTiltSeries->m_aiStkSize[1]};
	aiVolSize[0] = pTiltSeries->m_aiStkSize[0] / 2 * 2;
	m_pVolSeries = new MD::CTiltSeries;
	m_pVolSeries->Create(aiVolSize);
	//-----------------
	mDoIt();
	//-----------------
	MD::CTiltSeries* pVolSeries = m_pVolSeries;
	m_pVolSeries = 0L;
	return pVolSeries;		
}

void CDoWbpRecon::mDoIt(void)
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
	m_aTomoWbp.Setup(m_pVolSeries->m_aiStkSize[0], 
	   m_pVolSeries->m_aiStkSize[1],
	   m_pTiltSeries, m_pAlignParam);
	//-----------------
	cudaStreamCreate(&m_stream);
	cudaEventCreate(&m_eventSino);
	int iLastY = -1;
	//-----------------
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
	//----------------------
	cudaStreamDestroy(m_stream);
	cudaEventDestroy(m_eventSino);
}

void CDoWbpRecon::mExtractSinogram(int iY)
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
	//-----------------------------------
	cudaStreamWaitEvent(m_stream, m_eventSino, 0);
	tBytes = sizeof(float) * iPadX * m_pTiltSeries->m_aiStkSize[2];
	cudaMemcpyAsync(m_gfPadSinogram, m_pfPadSinogram, tBytes,
	   cudaMemcpyDefault, m_stream);
	cudaEventRecord(m_eventSino, m_stream);	
}

void CDoWbpRecon::mGetReconResult(int iLastY)
{
	if(iLastY < 0) return;
	float* pfVolXZ = (float*)m_pVolSeries->GetFrame(iLastY);
	cudaStreamSynchronize(m_stream);
	//--------------------------------
	// Flip z to match IMOD handedness
	//--------------------------------
	int iBytes = m_pVolSeries->m_aiStkSize[0] * sizeof(float);
	int iLastZ = m_pVolSeries->m_aiStkSize[1] - 1;
	for(int z=0; z<=iLastZ; z++)
	{	float* pfSrc = m_pfVolXZ + z * m_pVolSeries->m_aiStkSize[0];
		float* pfDst = pfVolXZ + (iLastZ - z) *
		   m_pVolSeries->m_aiStkSize[0];
		memcpy(pfDst, pfSrc, iBytes);
	}	
}

void CDoWbpRecon::mReconstruct(int iY)
{
	size_t tBytes = m_pVolSeries->GetPixels() * sizeof(float);
	cudaMemsetAsync(m_gfVolXZ, 0, tBytes, m_stream);
	m_aTomoWbp.DoIt(m_gfPadSinogram, m_gfVolXZ, m_stream);
	//----------------------------------------------------
	cudaMemcpyAsync(m_pfVolXZ, m_gfVolXZ, tBytes,
		cudaMemcpyDefault, m_stream);
}

