#include "CEerUtilInc.h"
#include "../Util/CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::MotionCor::EerUtil;

CRenderMrcStack::CRenderMrcStack(void)
{
}

CRenderMrcStack::~CRenderMrcStack(void)
{
}

void CRenderMrcStack::DoIt
(	CLoadEerHeader* pLoadHeader,
	CLoadEerFrames* pLoadFrames,
	int iNthGpu
)
{	m_pLoadHeader = pLoadHeader;
	m_pLoadFrames = pLoadFrames;
	m_iNthGpu = iNthGpu;
	//-----------------
	m_aDecodeEerFrame.Setup(m_pLoadHeader->m_aiCamSize, 
	   m_pLoadHeader->m_iEerSampling);
	//-----------------
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	m_pRawStack = pPackage->m_pRawStack;
	//-----------------
	m_pFmIntParam = MMD::CFmIntParam::GetInstance(m_iNthGpu);
	if(m_pFmIntParam->bIntegrate()) mRenderInt();
	else mRender();
}

void CRenderMrcStack::mRender(void)
{
	unsigned char* pucMrcFrm = 0L;
	for(int i=0; i<m_pRawStack->m_aiStkSize[2]; i++)
	{	pucMrcFrm = (unsigned char*)m_pRawStack->GetFrame(i);
		memset(pucMrcFrm, 0, m_pRawStack->m_tFmBytes);
		int iEerFrm = m_pFmIntParam->GetIntFmStart(i);
		mDecodeFrame(iEerFrm, pucMrcFrm);
	}
}

void CRenderMrcStack::mRenderInt(void)
{
	for(int i=0; i<m_pRawStack->m_aiStkSize[2]; i++)
	{	int iFmSize = m_pFmIntParam->GetIntFmSize(i);
		if(iFmSize == 1) 
		{	void* pvMrcFrm = m_pRawStack->GetFrame(i);
			memset(pvMrcFrm, 0, m_pRawStack->m_tFmBytes);
			int iFmStart = m_pFmIntParam->GetIntFmStart(i);
			mDecodeFrame(iFmStart, (unsigned char*)pvMrcFrm);
		}
		else mRenderFrame(i);
	}
}

void CRenderMrcStack::mRenderFrame(int iIntFrm)
{
	int iIntFmStart = m_pFmIntParam->GetIntFmStart(iIntFrm);
	int iIntFmSize = m_pFmIntParam->GetIntFmSize(iIntFrm);
	//-----------------
	unsigned char* pucFrm = 0L;
	pucFrm = (unsigned char*)m_pRawStack->GetFrame(iIntFrm);
	memset(pucFrm, 0, m_pRawStack->m_tFmBytes);
	//-----------------
	for(int i=0; i<iIntFmSize; i++)
	{	int iEerFrame = iIntFmStart + i;
		mDecodeFrame(iEerFrame, pucFrm);
	}
}	

void CRenderMrcStack::mDecodeFrame
(	int iEerFrame,
	unsigned char* pucDecodedFrm
)
{	int iEerFmBytes = m_pLoadFrames->GetEerFrameSize(iEerFrame);
	int iEerBits = m_pLoadHeader->m_iNumBits;
	unsigned char* pEerFrm = m_pLoadFrames->GetEerFrame(iEerFrame);
	if(iEerBits == 7)
	{	m_aDecodeEerFrame.Do7Bits(pEerFrm, 
		   iEerFmBytes, pucDecodedFrm);
	}
	else
	{	m_aDecodeEerFrame.Do8Bits(pEerFrm, 
		   iEerFmBytes, pucDecodedFrm);
	}
}

