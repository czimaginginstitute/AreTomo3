#include "CAlignInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;

CAlignedSum::CAlignedSum(void)
{
}

CAlignedSum::~CAlignedSum(void)
{
}

//--------------------------------------------------------------------
// piSumRange: 2-element array of the starting and ending frame
//    indices (inclusive) over which the sum is calculated. If
//    null, the entire stack is summed. 
//--------------------------------------------------------------------
void CAlignedSum::DoIt
(	int iBuffer,
	MMD::CStackShift* pStackShift,
	int* piSumRange,
	int iNthGpu
)
{	m_pStackShift = pStackShift;
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(iNthGpu);
	//-----------------
	m_pFrmBuffer = pBufferPool->GetBuffer(iBuffer);
	MD::CStackBuffer* pSumBuffer = pBufferPool->GetBuffer(MD::EBuffer::sum);
        MD::CStackBuffer* pTmpBuffer = pBufferPool->GetBuffer(MD::EBuffer::tmp);
	//-----------------
	m_gCmpSums[0] = pSumBuffer->GetFrame(0);
	m_gCmpSums[1] = pTmpBuffer->GetFrame(0);
	m_streams[0] = pBufferPool->GetCudaStream(0);
	m_streams[1] = pBufferPool->GetCudaStream(1);
	//-----------------
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
	cudaMemsetAsync(m_gCmpSums[0], 0, tBytes, m_streams[0]);
	cudaMemsetAsync(m_gCmpSums[1], 0, tBytes, m_streams[1]);
	//-----------------
	int aiSumRange[] = {0, m_pFrmBuffer->m_iNumFrames - 1};
	if(piSumRange != 0L) memcpy(aiSumRange, piSumRange, sizeof(int) * 2);
	//-----------------
	for(int i=aiSumRange[0]; i<=aiSumRange[1]; i++)
	{	mDoFrame(i);
	}
	cudaStreamSynchronize(m_streams[1]);
	//-----------------
	MU::GAddFrames addFrames;
	addFrames.DoIt(m_gCmpSums[0], 1.0f, m_gCmpSums[1], 1.0f,
	   m_gCmpSums[0], m_pFrmBuffer->m_aiCmpSize, m_streams[0]);
	cudaStreamSynchronize(m_streams[0]);
}

void CAlignedSum::mDoFrame(int iFrame)
{
	int iStream = iFrame % 2;
	cufftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(iFrame);
	cufftComplex* gCmpSum = m_gCmpSums[iStream];
	//-----------------
	float afShift[2] = {0.0f};
	m_pStackShift->GetShift(iFrame, afShift, -1.0f);
	//-----------------
	bool bSum = true;
	MU::GPhaseShift2D phaseShift;
	phaseShift.DoIt(gCmpFrm, m_pFrmBuffer->m_aiCmpSize,
           afShift, bSum, gCmpSum, m_streams[0]);
}
