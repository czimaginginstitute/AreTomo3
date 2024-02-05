#include "CAlignInc.h"
#include "../CMotionCorInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;

CAlignBase::CAlignBase(void)
{
	m_pFullShift = 0L;
}

CAlignBase::~CAlignBase(void)
{
	this->Clean();
}

void CAlignBase::Clean(void)
{
	if(m_pFullShift == 0L) return;
	delete m_pFullShift;
	m_pFullShift = 0L;
}

void CAlignBase::DoIt(int iNthGpu)
{
	nvtxRangePushA ("CAlignBase::DoIt");
	m_iNthGpu = iNthGpu;
	//-----------------
	CMcInput* pInput = CMcInput::GetInstance();
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pFrmBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::frm);
	//-----------------
	MU::GFourierResize2D::GetBinnedCmpSize
	( pFrmBuffer->m_aiCmpSize,
	  pInput->m_fMcBin, m_aiCmpSize
	);
	//-----------------
	m_aiImgSize[0] = (m_aiCmpSize[0] - 1) * 2;
	m_aiImgSize[1] = m_aiCmpSize[1];
	m_aiPadSize[0] = m_aiCmpSize[0] * 2;
	m_aiPadSize[1] = m_aiCmpSize[1];
	//-----------------
	mCreateAlnSums();
	nvtxRangePop();
}

void CAlignBase::LogShift(char* pcLogFile)
{
}

void CAlignBase::mCreateAlnSums(void)
{
	MD::CMcPackage* pMcPkg = MD::CMcPackage::GetInstance(m_iNthGpu);
	pMcPkg->m_pAlnSums->Create(m_aiImgSize);
	//-----------------
	CMcInput* pInput = CMcInput::GetInstance();
	pMcPkg->m_pAlnSums->m_fPixSize = pInput->GetFinalPixelSize();
}

