#include "CAlignInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;
namespace MMM = McAreTomo::MotionCor::MrcUtil;

CSimpleSum::CSimpleSum(void)
{
}

CSimpleSum::~CSimpleSum(void)
{
}

void CSimpleSum::DoIt(int iNthGpu)
{
	nvtxRangePushA ("CsimpleSum");
	CAlignBase::Clean();
	CAlignBase::DoIt(iNthGpu);
	//-----------------
	CMcInput* pInput = CMcInput::GetInstance();
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(iNthGpu);
	MD::CStackBuffer* pFrmBuffer = pBufferPool->GetBuffer(MD::EBuffer::frm);
	//-----------------
	m_streams[0] = pBufferPool->GetCudaStream(0);
	m_streams[1] = pBufferPool->GetCudaStream(1);
	//-----------------
	m_pForwardFFT = pBufferPool->GetCufft2D(true);
	m_pInverseFFT = pBufferPool->GetCufft2D(false);
	if(pInput->m_fMcBin > 1)
	{	int aiPadSize[] = {pFrmBuffer->m_aiCmpSize[0] * 2,
		   pFrmBuffer->m_aiCmpSize[1]};
		m_pForwardFFT->CreateForwardPlan(aiPadSize, true);
		m_pInverseFFT->CreateInversePlan(m_aiCmpSize, true);
	}	
	//-----------------
	mCalcSum();
	nvtxRangePop ();
}

void CSimpleSum::mCalcSum(void)
{
	CMcInput* pInput = CMcInput::GetInstance();
	MMM::CSumFFTStack sumFFTStack;
	bool bSplitSum = true;
	sumFFTStack.DoIt(MD::EBuffer::frm, bSplitSum, m_iNthGpu);
	//----------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pSumBuffer = pBufferPool->GetBuffer(MD::EBuffer::sum);
	cufftComplex* gCmpSum0 = pSumBuffer->GetFrame(0);
	cufftComplex* gCmpSum1 = pSumBuffer->GetFrame(1);
	cufftComplex* gCmpSum2 = pSumBuffer->GetFrame(2);
	//-----------------
	MD::CStackBuffer* pTmpBuffer = pBufferPool->GetBuffer(MD::EBuffer::tmp);
	cufftComplex* gCmpBuf0 = pTmpBuffer->GetFrame(0);
	cufftComplex* gCmpBuf1 = pTmpBuffer->GetFrame(1);
	cufftComplex* gCmpBuf2 = pTmpBuffer->GetFrame(2);
	//-----------------
	mCropFrame(gCmpSum0, gCmpBuf0, 0);
	if(bSplitSum)
	{	mCropFrame(gCmpSum1, gCmpBuf1, 0);
		mCropFrame(gCmpSum2, gCmpBuf2, 0);
	}
	//-----------------
	float *gfUnpad0 = 0L, *gfUnpad1 = 0L, *gfUnpad2 = 0L;
	gfUnpad0 = reinterpret_cast<float*>(gCmpSum0);
	mUnpad(gCmpBuf0, gfUnpad0, 0);
	if(bSplitSum)
	{	gfUnpad1 = reinterpret_cast<float*>(gCmpSum1);
		mUnpad(gCmpBuf1, gfUnpad1, 0);
		gfUnpad2 = reinterpret_cast<float*>(gCmpSum2);
		mUnpad(gCmpBuf2, gfUnpad2, 0);
	}
	cudaStreamSynchronize(m_streams[0]);
	//-----------------
	MD::CMcPackage* pMcPkg = MD::CMcPackage::GetInstance(m_iNthGpu);
	size_t tBytes = pMcPkg->m_pAlnSums->m_tFmBytes;
	void* pvCropped = pMcPkg->m_pAlnSums->GetFrame(0);
	cudaMemcpy(pvCropped, gfUnpad0, tBytes, cudaMemcpyDefault);
	if(bSplitSum)
	{	pvCropped = pMcPkg->m_pAlnSums->GetFrame(1);
		cudaMemcpy(pvCropped, gfUnpad1, tBytes, cudaMemcpyDefault);
		pvCropped = pMcPkg->m_pAlnSums->GetFrame(2);
		cudaMemcpy(pvCropped, gfUnpad2, tBytes, cudaMemcpyDefault);
	}
}

void CSimpleSum::mCropFrame
(	cufftComplex* gCmpFrm,
	cufftComplex* gCmpBuf,
	int iStream
)
{	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pFrmBuffer = pBufferPool->GetBuffer(MD::EBuffer::frm);
	//--------------------------------------------------------
	// no Fourier cropping is needed, simply copy gCmpFrm to
	// gCmpBuf and return.
	//--------------------------------------------------------
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(pMcInput->m_fMcBin == 1)
	{	cudaMemcpy(gCmpBuf, gCmpFrm, pFrmBuffer->m_tFmBytes,
		   cudaMemcpyDefault);
		return;
	}
	//-----------------
	float* gfPadFrm = reinterpret_cast<float*>(gCmpFrm);
	m_pForwardFFT->Forward(gfPadFrm, true, m_streams[iStream]);
	//-----------------
	bool bSum = true;
	MU::GFourierResize2D fourierResize;
	fourierResize.DoIt(gCmpFrm, pFrmBuffer->m_aiCmpSize, 
	   gCmpBuf, m_aiCmpSize, !bSum, m_streams[iStream]);
	//-----------------
	m_pInverseFFT->Inverse(gCmpBuf, m_streams[iStream]);
}

void CSimpleSum::mUnpad
(	cufftComplex* gCmpPad, 
	float* gfUnpad,
	int iStream
)
{ 	MU::GPad2D pad2D;
	float* gfPad = reinterpret_cast<float*>(gCmpPad);
	pad2D.Unpad(gfPad, m_aiPadSize, gfUnpad, m_streams[iStream]);
}
