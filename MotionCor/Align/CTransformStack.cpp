#include "CAlignInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;

CTransformStack::CTransformStack(void)
{
}

CTransformStack::~CTransformStack(void)
{
}

void CTransformStack::Setup
(	MD::EBuffer eBuffer,
	bool bForward,
	bool bNorm,
	int iNthGpu
)
{	m_bForward = bForward;
	m_bNorm = bNorm;
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	m_pFrmBuffer = pBufferPool->GetBuffer(eBuffer);
	m_pTmpBuffer = pBufferPool->GetBuffer(MD::EBuffer::tmp);
	m_pCufft2D = pBufferPool->GetCufft2D(m_bForward);
	m_streams[0] = pBufferPool->GetCudaStream(0);
	m_streams[1] = pBufferPool->GetCudaStream(1);
	//-----------------
	int* piCmpSize = m_pFrmBuffer->m_aiCmpSize;
	int aiPadSize[] = {piCmpSize[0] * 2, piCmpSize[1]};
	if(m_bForward) m_pCufft2D->CreateForwardPlan(aiPadSize, true);
	else m_pCufft2D->CreateInversePlan(piCmpSize, true);
}

void CTransformStack::DoIt(void)
{
	nvtxRangePushA("CTransformStack::DoIt");
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	//-----------------
	mTransformCpuFrames();
	mTransformGpuFrames();
	nvtxRangePop();
}

void CTransformStack::mTransformGpuFrames(void)
{
	cufftComplex* gCmpFrm = 0L;
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	if(!m_pFrmBuffer->IsGpuFrame(i)) continue;
		gCmpFrm = m_pFrmBuffer->GetFrame(i);
		mTransformFrame(gCmpFrm);
	}
}

void CTransformStack::mTransformCpuFrames(void)
{
	int iCount = 0;
	cufftComplex *pCmpFrm, *gCmpBuf;
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
	//-----------------
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	if(m_pFrmBuffer->IsGpuFrame(i)) continue;
		int iStream = iCount % 2;
		pCmpFrm = m_pFrmBuffer->GetFrame(i);
		gCmpBuf = m_pTmpBuffer->GetFrame(iStream);
		//----------------
		cudaMemcpyAsync(gCmpBuf, pCmpFrm, tBytes, 
		   cudaMemcpyDefault, m_streams[iStream]);
		if(iStream == 1) cudaStreamSynchronize(m_streams[1]);
		//----------------
		mTransformFrame(gCmpBuf);
		if(iStream == 1) cudaStreamSynchronize(m_streams[0]);
		//----------------
		cudaMemcpyAsync(pCmpFrm, gCmpBuf, tBytes,
		   cudaMemcpyDefault, m_streams[iStream]);
		iCount += 1;
	}
}

void CTransformStack::mTransformFrame(cufftComplex* gCmpFrm) 
{
	if(m_bForward)
	{	float* gfPadFrm = reinterpret_cast<float*>(gCmpFrm);
		m_pCufft2D->Forward(gfPadFrm, m_bNorm, m_streams[0]);
	}
	else
	{	m_pCufft2D->Inverse(gCmpFrm, m_streams[0]);
	}
}
