#include "CMrcUtilInc.h"
#include "../CMotionCorInc.h"
#include "../Util/CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::MrcUtil;
using namespace McAreTomo::MotionCor;

CSumFFTStack::CSumFFTStack(void)
{
}

CSumFFTStack::~CSumFFTStack(void)
{
}

//------------------------------------------------------------------------------
// 1. This function calculates the simple sum(s) without correction of beam
//    induced motion.
// 2. If bSplitSum is true, odd and even sums will be calculated in addition
//    to the whole sum. (SZ: 08-10-2023)
//------------------------------------------------------------------------------ 
void CSumFFTStack::DoIt(int iBuffer, bool bSplitSum, int iNthGpu)
{
        nvtxRangePushA("CSumFFTStack::DoIt");
	m_iBuffer = iBuffer;
	m_bSplitSum = bSplitSum;
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pFrmBuffer = pBufPool->GetBuffer((MD::EBuffer)m_iBuffer);
	MD::CStackBuffer* pSumBuffer = pBufPool->GetBuffer(MD::EBuffer::sum);
	MD::CStackBuffer* pTmpBuffer = pBufPool->GetBuffer(MD::EBuffer::tmp);
	//-----------------
	m_streams[0] = pBufPool->GetCudaStream(0);
	m_streams[1] = pBufPool->GetCudaStream(1);
	//-----------------
	mSumFrames();
	if(m_bSplitSum) mSplitSums();
	//-----------------
	cudaStreamSynchronize(m_streams[1]);
	cudaStreamSynchronize(m_streams[0]);
        nvtxRangePop();
}

void CSumFFTStack::mSumFrames(void)
{
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(m_iNthGpu);
        MD::CStackBuffer* pFrmBuffer = pBufPool->GetBuffer((MD::EBuffer)m_iBuffer);
	MD::CStackBuffer* pSumBuffer = pBufPool->GetBuffer(MD::EBuffer::sum);
	//-----------------
	size_t tBytes = pFrmBuffer->m_tFmBytes;
	cufftComplex* gCmpSum0 = pSumBuffer->GetFrame(0);
	cufftComplex* gCmpSum1 = pSumBuffer->GetFrame(1);
	cudaMemsetAsync(gCmpSum0, 0, tBytes, m_streams[0]);
	if(m_bSplitSum) cudaMemsetAsync(gCmpSum1, 0, tBytes, m_streams[0]);
	//-----------------
	mSumCpuFrames();
	mSumGpuFrames();
}

void CSumFFTStack::mSumGpuFrames(void)
{
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(m_iNthGpu);
        MD::CStackBuffer* pFrmBuffer = pBufPool->GetBuffer((MD::EBuffer)m_iBuffer);
        MD::CStackBuffer* pSumBuffer = pBufPool->GetBuffer(MD::EBuffer::sum);
	//-----------------
	cufftComplex* gCmpSum0 = pSumBuffer->GetFrame(0);
	cufftComplex* gCmpSum1 = pSumBuffer->GetFrame(1);
	cufftComplex* gCmpFrm = 0L;
	int* piCmpSize = pFrmBuffer->m_aiCmpSize;
	//-----------------
	MU::GAddFrames addFrames;
	for(int i=0; i<pFrmBuffer->m_iNumFrames; i++)
	{	if(!pFrmBuffer->IsGpuFrame(i)) continue;
		gCmpFrm = pFrmBuffer->GetFrame(i);
		//-------------------------
		// calculate the whole sum
		//-------------------------
		addFrames.DoIt(gCmpSum0, 1.0f, gCmpFrm, 1.0f, gCmpSum0, 
		   piCmpSize, m_streams[0]);
		if(!m_bSplitSum) continue;
		//------------------------
		// calculate the even sum
		//------------------------
		if(i % 2 != 0) continue;
		addFrames.DoIt(gCmpSum1, 1.0f, gCmpFrm, 1.0f, gCmpSum1,
                   piCmpSize, m_streams[0]);
	}
}

void CSumFFTStack::mSumCpuFrames(void)
{
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(m_iNthGpu);
        MD::CStackBuffer* pFrmBuffer = pBufPool->GetBuffer((MD::EBuffer)m_iBuffer);
        MD::CStackBuffer* pSumBuffer = pBufPool->GetBuffer(MD::EBuffer::sum);
	MD::CStackBuffer* pTmpBuffer = pBufPool->GetBuffer(MD::EBuffer::tmp);
	//-----------------
	cufftComplex* gCmpSum0 = pSumBuffer->GetFrame(0);
	cufftComplex* gCmpSum1 = pSumBuffer->GetFrame(1);
	cufftComplex *gCmpBuf = 0L, *pCmpFrm = 0L;
	//-----------------
	int* piCmpSize = pFrmBuffer->m_aiCmpSize;
	size_t tBytes = pFrmBuffer->m_tFmBytes;
	MU::GAddFrames addFrames;
	int iCount = 0;
	//-----------------
	for(int i=0; i<pFrmBuffer->m_iNumFrames; i++)
	{	if(pFrmBuffer->IsGpuFrame(i)) continue;
		int iStream = iCount % 2;
		pCmpFrm = pFrmBuffer->GetFrame(i);
		gCmpBuf = pTmpBuffer->GetFrame(iStream);
		//----------------
		if(iStream == 1) cudaStreamSynchronize(m_streams[0]);
		cudaMemcpyAsync(gCmpBuf, pCmpFrm, tBytes,
		   cudaMemcpyDefault, m_streams[iStream]);
		if(iStream == 1) cudaStreamSynchronize(m_streams[1]);
		//-------------------------
		// calculate the whole sum
		//-------------------------
		addFrames.DoIt(gCmpSum0, 1.0f, gCmpBuf, 1.0f, gCmpSum0,
		   piCmpSize, m_streams[0]);
		if(!m_bSplitSum) continue;
		//---------------------
		// calculate even sums
		//---------------------
		if(i % 2 != 0) continue;
		addFrames.DoIt(gCmpSum1, 1.0f, gCmpBuf, 1.0f, gCmpSum1,
		   piCmpSize, m_streams[0]);
	}
}

void CSumFFTStack::mSplitSums(void)
{
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(m_iNthGpu);
        MD::CStackBuffer* pSumBuffer = pBufPool->GetBuffer(MD::EBuffer::sum);
        MU::GAddFrames addFrames;
        cufftComplex* gCmpSum0 = pSumBuffer->GetFrame(0);
        cufftComplex* gCmpSum1 = pSumBuffer->GetFrame(1);
        cufftComplex* gCmpSum2 = pSumBuffer->GetFrame(2);
        addFrames.DoIt(gCmpSum0, 1.0f, gCmpSum1, -1.0f, gCmpSum2,
           pSumBuffer->m_aiCmpSize, m_streams[0]);
}

