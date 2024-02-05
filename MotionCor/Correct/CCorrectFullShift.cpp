#include "CCorrectInc.h"
#include "../Align/CAlignInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::MotionCor::Correct;
using namespace McAreTomo::MotionCor;

CCorrectFullShift::CCorrectFullShift(void)
{
	m_pForwardFFT = 0L;
	m_pInverseFFT = 0L;
}

CCorrectFullShift::~CCorrectFullShift(void)
{
	m_pForwardFFT = 0L;
	m_pInverseFFT = 0L;
}

void CCorrectFullShift::Setup
(	MMD::CStackShift* pStackShift,
	int iNthGpu
)
{	m_pFullShift = pStackShift;
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CMcPackage* pMcPkg = MD::CMcPackage::GetInstance(m_iNthGpu);
	m_aiOutCmpSize[0] = pMcPkg->m_pAlnSums->m_aiStkSize[0] / 2 + 1;
	m_aiOutCmpSize[1] = pMcPkg->m_pAlnSums->m_aiStkSize[1];
	m_aiOutPadSize[0] = m_aiOutCmpSize[0] * 2;
	m_aiOutPadSize[1] = m_aiOutCmpSize[1];
	//------------------------------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	m_pFrmBuffer = pBufferPool->GetBuffer(MD::EBuffer::frm);
	m_pSumBuffer = pBufferPool->GetBuffer(MD::EBuffer::sum);
        m_pTmpBuffer = pBufferPool->GetBuffer(MD::EBuffer::tmp);
	//-----------------
	m_aiInCmpSize[0] = m_pFrmBuffer->m_aiCmpSize[0];
	m_aiInCmpSize[1] = m_pFrmBuffer->m_aiCmpSize[1];
	m_aiInPadSize[0] = m_aiInCmpSize[0] * 2;
	m_aiInPadSize[1] = m_aiInCmpSize[1];
	//-----------------
	m_pForwardFFT = pBufferPool->GetCufft2D(true);
	m_pInverseFFT = pBufferPool->GetCufft2D(false);
	//-----------------
	m_streams[0] = pBufferPool->GetCudaStream(0);
        m_streams[1] = pBufferPool->GetCudaStream(1);
}

void CCorrectFullShift::DoIt(void)
{
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	//-----------------
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
	for(int i=0; i<pBufferPool->m_iNumSums; i++)
	{	cufftComplex* gCmpSum = m_pSumBuffer->GetFrame(i);
		cudaMemsetAsync(gCmpSum, 0, tBytes, m_streams[0]);
	}
	//-----------------
        mCorrectGpuFrames();
        mCorrectCpuFrames();
	mCorrectMag();
	mUnpadSums();
	//-----------------
	cudaStreamSynchronize(m_streams[0]);
        cudaStreamSynchronize(m_streams[1]);
}

void CCorrectFullShift::mCorrectMag(void)
{
        Align::CAlignParam* pAlignParam = Align::CAlignParam::GetInstance();
        float afStretch[] = {1.0f, 0.0f};
        pAlignParam->GetMagStretch(afStretch);
        if(fabs(afStretch[0] - 1) < 1e-5) return;
	//-----------------
	GStretch aGStretch;
        aGStretch.Setup(afStretch[0], afStretch[1]);
	//-----------------
	int aiPadSize[] = {0, m_pFrmBuffer->m_aiCmpSize[1]};
        aiPadSize[0] = m_pFrmBuffer->m_aiCmpSize[0] * 2;
	m_pForwardFFT->CreateForwardPlan(aiPadSize, true);
	m_pInverseFFT->CreateInversePlan(m_pFrmBuffer->m_aiCmpSize, true);
	//-----------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
        for(int i=0; i<pBufferPool->m_iNumSums; i++)
        {       cufftComplex* gCmpSum = m_pSumBuffer->GetFrame(i);
		cufftComplex* gCmpTmp = m_pTmpBuffer->GetFrame(0);
		m_pInverseFFT->Inverse(gCmpSum, (float*)gCmpTmp);
		//-----------------
		float* gfPadFrm = reinterpret_cast<float*>(gCmpTmp);
                bool bPadded = true;
                aGStretch.DoIt(gfPadFrm, bPadded, aiPadSize, (float*)gCmpSum);
		m_pForwardFFT->Forward((float*)gCmpSum, true);
        }
}

void CCorrectFullShift::mUnpadSums(void)
{
	cufftComplex* gCmpBuf = m_pTmpBuffer->GetFrame(0);
	MU::GFourierResize2D fftResize;
	MU::GPad2D pad2D;
	MU::GPositivity2D positivity2D;
	//-----------------
	m_pInverseFFT->CreateInversePlan(m_aiOutCmpSize, true);
	//-----------------
	MD::CMcPackage* pMcPkg = MD::CMcPackage::GetInstance(m_iNthGpu);
	MD::CMrcStack* pAlnSums = pMcPkg->m_pAlnSums;
	//-----------------
	for(int i=0; i<pAlnSums->m_aiStkSize[2]; i++)
	{	cufftComplex* gCmpSum = m_pSumBuffer->GetFrame(i);
		fftResize.DoIt(gCmpSum, m_pSumBuffer->m_aiCmpSize,
		   gCmpBuf, m_aiOutCmpSize, false);
		//----------------
		m_pInverseFFT->Inverse(gCmpBuf);
		float* gfPadBuf = reinterpret_cast<float*>(gCmpBuf);
		//----------------
		positivity2D.DoIt(gfPadBuf, m_aiOutPadSize);
		//----------------
		float* gfImg = reinterpret_cast<float*>(gCmpSum);
		pad2D.Unpad(gfPadBuf, m_aiOutPadSize, gfImg);
		//----------------
		positivity2D.DoIt(gfImg, pAlnSums->m_aiStkSize);
		//----------------
		void* pvImg = pAlnSums->GetFrame(i);
		cudaMemcpy(pvImg, gfImg, pAlnSums->m_tFmBytes,
		   cudaMemcpyDefault);
	}
}

void CCorrectFullShift::mCorrectGpuFrames(void)
{
        for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
        {       if(!m_pFrmBuffer->IsGpuFrame(i)) continue;
		else m_iFrame = i;
		//----------------
                cufftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(i);
		mAlignFrame(gCmpFrm);
                mGenSums(gCmpFrm);
        }
}

void CCorrectFullShift::mCorrectCpuFrames(void)
{
	int iCount = 0;
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
	cufftComplex *pCmpFrm, *gCmpBuf;
	//------------------------------
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	if(m_pFrmBuffer->IsGpuFrame(i)) continue;
		else m_iFrame = i;
		//----------------
		int iStream = iCount % 2;
		gCmpBuf = m_pTmpBuffer->GetFrame(iStream);
		pCmpFrm = m_pFrmBuffer->GetFrame(i);
		//-----------------
		cudaMemcpyAsync(gCmpBuf, pCmpFrm, tBytes, 
		   cudaMemcpyDefault, m_streams[iStream]);
		if(iStream == 1) cudaStreamSynchronize(m_streams[1]);
		//----------------
		mAlignFrame(gCmpBuf);
		mGenSums(gCmpBuf);
		//----------------
		if(iStream == 1) cudaStreamSynchronize(m_streams[0]);
		iCount += 1;
	}
}

//--------------------------------------------------------------------
// 1. generate three sums: dose-unweighted sum, even-frame sum, and
//    odd-frame sums.
// 2. dose-weighting will be done in AreTomo scope.
//--------------------------------------------------------------------
void CCorrectFullShift::mGenSums(cufftComplex* gCmpFrm)
{	
	mMotionDecon(gCmpFrm);
	mSum(gCmpFrm, 0);
	if(m_iFrame % 2 == 0) mSum(gCmpFrm, 1);
	else mSum(gCmpFrm, 2);
}

void CCorrectFullShift::mAlignFrame(cufftComplex* gCmpFrm)
{	
	float afShift[2] = {0};
	m_pFullShift->GetShift(m_iFrame, afShift, -1.0f);
	MU::GPhaseShift2D phaseShift2D;
	phaseShift2D.DoIt(gCmpFrm, m_pFrmBuffer->m_aiCmpSize,
	   afShift, m_streams[0]);
}

void CCorrectFullShift::mMotionDecon(cufftComplex* gCmpFrm)
{	
	CMcInput* pInput = CMcInput::GetInstance();
	if(pInput->m_iInFmMotion == 0) return;
	//------------------------------------
	int* piCmpSize = m_pFrmBuffer->m_aiCmpSize;
	m_aInFrameMotion.SetFullShift(m_pFullShift);
	m_aInFrameMotion.DoFullMotion(m_iFrame, gCmpFrm, 
	   piCmpSize, m_streams[0]);
}

void CCorrectFullShift::mSum(cufftComplex* gCmpFrm, int iNthSum)
{      
	cufftComplex* gCmpSum = m_pSumBuffer->GetFrame(iNthSum);
	MU::GAddFrames addFrames;
	addFrames.DoIt(gCmpFrm, 1.0f, gCmpSum, 1.0f, gCmpSum,
	   m_pFrmBuffer->m_aiCmpSize, m_streams[0]);
}

