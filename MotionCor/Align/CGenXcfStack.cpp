#include "CAlignInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;

CGenXcfStack::CGenXcfStack(void)
{
}

CGenXcfStack::~CGenXcfStack(void)
{
}

void CGenXcfStack::DoIt(MMD::CStackShift* pStackShift, int iNthGpu)
{	
	nvtxRangePushA ("CGenXcfStack::DoIt");
	m_pStackShift = pStackShift;
	//-----------------
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(iNthGpu);
	m_stream = pBufferPool->GetCudaStream(0);
	//-----------------
	m_pFrmBuffer = pBufferPool->GetBuffer(MD::EBuffer::frm);
	m_pSumBuffer = pBufferPool->GetBuffer(MD::EBuffer::sum);
	m_pTmpBuffer = pBufferPool->GetBuffer(MD::EBuffer::tmp);
	m_pXcfBuffer = pBufferPool->GetBuffer(MD::EBuffer::xcf);
	//-----------------
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	mDoXcfFrame(i);
	}
	cudaStreamSynchronize(m_stream);
        m_pStackShift = 0L;
	//-----------------
	nvtxRangePop();
}

void CGenXcfStack::mDoXcfFrame(int iFrm)
{
	int iStream = iFrm % 2;
	//-----------------
	cufftComplex *gCmpXcf, *gCmpFrm, *gCmpTmp, *gCmpBuf;
	gCmpXcf = m_pXcfBuffer->GetFrame(iFrm);
	gCmpTmp = m_pTmpBuffer->GetFrame(iStream);
	gCmpFrm = m_pFrmBuffer->GetFrame(iFrm);
	//-----------------
	MU::GFourierResize2D fourierResize;
	MU::GPhaseShift2D phaseShift;
	float afShift[2] = {0.0f};
	if(m_pStackShift != 0L) 
	{	m_pStackShift->GetShift(iFrm, afShift, -1.0f);
	}
	//-----------------
	if(afShift[0] == 0 && afShift[1] == 0)
	{	fourierResize.DoIt(gCmpFrm, m_pFrmBuffer->m_aiCmpSize,
		   gCmpXcf, m_pXcfBuffer->m_aiCmpSize, 
		   false, m_stream);
	}
	else
	{	phaseShift.DoIt(gCmpFrm, m_pFrmBuffer->m_aiCmpSize,
		   afShift, false, gCmpTmp, m_stream);
		fourierResize.DoIt(gCmpTmp, m_pFrmBuffer->m_aiCmpSize,
		   gCmpXcf, m_pXcfBuffer->m_aiCmpSize, 
		   false, m_stream);
	}
}

