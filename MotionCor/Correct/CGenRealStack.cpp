#include "CCorrectInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::MotionCor::Correct;
using namespace McAreTomo::MotionCor;

CGenRealStack::CGenRealStack(void)
{
}

CGenRealStack::~CGenRealStack(void)
{
}

void CGenRealStack::Setup
(	int iBuffer,
	bool bCorrBilinear,
	bool bMotionDecon,
	int iNthGpu
)
{	m_bCorrBilinear = bCorrBilinear;
	m_bMotionDecon = bMotionDecon;
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(iNthGpu);
	m_pFrmBuffer = pBufferPool->GetBuffer(iBuffer);
	m_pTmpBuffer = pBufferPool->GetBuffer(MD::EBuffer::tmp);
	//-----------------
	m_streams[0] = pBufferPool->GetCudaStream(0);
	m_streams[1] = pBufferPool->GetCudaStream(1);
	m_pCufft2D = pBufferPool->GetCufft2D(false);
	m_pCufft2D->CreateInversePlan(m_pFrmBuffer->m_aiCmpSize, true);
}

void CGenRealStack::DoIt(MMD::CStackShift* pStackShift)
{
	m_pStackShift = pStackShift;
	//-----------------
	mDoCpuFrames();
	mDoGpuFrames();
	cudaStreamSynchronize(m_streams[0]);
	cudaStreamSynchronize(m_streams[1]);
}

void CGenRealStack::mDoGpuFrames(void)
{
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	if(!m_pFrmBuffer->IsGpuFrame(i)) continue;
		m_iFrame = i;
		cufftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(i);
		//----------------
		mAlignFrame(gCmpFrm);
		mCorrectBilinear(gCmpFrm);
		mMotionDecon(gCmpFrm);
		m_pCufft2D->Inverse(gCmpFrm, m_streams[0]);
	}
}

void CGenRealStack::mDoCpuFrames(void)
{
	int iCount = 0;
	cufftComplex *gCmpBuf = 0L, *pCmpFrm = 0L;
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
	//-----------------
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	if(m_pFrmBuffer->IsGpuFrame(i)) continue;
		m_iFrame = i;
		int iStream = iCount % 2;
		//----------------
		pCmpFrm = m_pFrmBuffer->GetFrame(i);
		gCmpBuf = m_pTmpBuffer->GetFrame(iStream);
		//----------------
		cudaMemcpyAsync(gCmpBuf, pCmpFrm, tBytes,
		   cudaMemcpyDefault, m_streams[iStream]);
		if(iStream == 1) cudaStreamSynchronize(m_streams[1]);
		//----------------
		mAlignFrame(gCmpBuf);
		mCorrectBilinear(gCmpBuf);
		mMotionDecon(gCmpBuf);
		m_pCufft2D->Inverse(gCmpBuf, m_streams[0]);
		if(iStream == 1) cudaStreamSynchronize(m_streams[0]);
		//----------------
		cudaMemcpyAsync(pCmpFrm, gCmpBuf, tBytes,
		   cudaMemcpyDefault, m_streams[iStream]);
		iCount += 1;
	}
}

void CGenRealStack::mCorrectBilinear(cufftComplex* gCmpFrm)
{
	if(!m_bCorrBilinear) return;
	MU::GCorrLinearInterp gCorrLinearInterp;
	int* piCmpSize = m_pFrmBuffer->m_aiCmpSize;
	gCorrLinearInterp.DoIt(gCmpFrm, piCmpSize, m_streams[0]);
}

void CGenRealStack::mAlignFrame(cufftComplex* gCmpFrm)
{
	if(m_pStackShift == 0L) return;
	//-----------------------------	
	float afShift[2] = {0.0f};
	m_pStackShift->GetShift(m_iFrame, afShift, -1.0f);
	int* piCmpSize = m_pFrmBuffer->m_aiCmpSize;
	MU::GPhaseShift2D aGPhaseShift;
	aGPhaseShift.DoIt(gCmpFrm, piCmpSize, afShift, m_streams[0]);
}

void CGenRealStack::mMotionDecon(cufftComplex* gCmpFrm)
{	
     	if(!m_bMotionDecon) return;
	if(m_pStackShift == 0L) return;
	//-----------------
	m_aInFrameMotion.SetFullShift(m_pStackShift);
	m_aInFrameMotion.DoFullMotion(m_iFrame, gCmpFrm, 
	   m_pFrmBuffer->m_aiCmpSize, m_streams[0]);
}

