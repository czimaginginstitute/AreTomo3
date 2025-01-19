#include "CAlignInc.h"
#include "../Correct/CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;

CIterativeAlign::CIterativeAlign(void)
{
	m_iMaxIterations = 7;
	m_iIterations = 0;
	m_fTol = 0.5f;
	m_fBFactor = 150.0f;
	m_bPhaseOnly = false;
	//-------------------
	m_pfErrors = new float[m_iMaxIterations];
	memset(m_pfErrors, 0, sizeof(float) * m_iMaxIterations);
}

CIterativeAlign::~CIterativeAlign(void)
{
	if(m_pfErrors != 0L) delete[] m_pfErrors;
}

void CIterativeAlign::Setup
(	int iBuffer,
	int iNthGpu
)
{	m_iBuffer = iBuffer;
	m_iNthGpu = iNthGpu;
	//-----------------
	m_fBFactor = (iBuffer == MD::EBuffer::xcf) ? 500.0f : 150.0f;
	m_bPhaseOnly = false;
	//-----------------
	CMcInput* pMcInput = CMcInput::GetInstance();	
	m_iMaxIterations = pMcInput->m_iMcIter;
	//-----------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	float fBin = fmaxf(pBufferPool->m_afXcfBin[0], 
	   pBufferPool->m_afXcfBin[1]);
	m_fTol = pMcInput->m_fMcTol / fBin;
	//-----------------		
	if(m_pfErrors != 0L) delete[] m_pfErrors;
	int iSize = m_iMaxIterations * 100;
	m_pfErrors = new float[iSize];
	memset(m_pfErrors, 0, sizeof(float) * iSize);
}

void CIterativeAlign::DoIt(MMD::CStackShift* pStackShift)
{	
	nvtxRangePushA("CIterativeAlign::DoIt");
	//-----------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pStackBuffer = pBufferPool->GetBuffer(m_iBuffer);
	//-----------------
	bool bPatch = (m_iBuffer == MD::EBuffer::pat) ? true : false;
	MMD::CFmGroupParam* pFmGroupParam = 
	   MMD::CFmGroupParam::GetInstance(m_iNthGpu, bPatch);
	//-----------------
	pStackShift->Reset();
	pStackShift->m_bConverged = false;
	//-----------------
	m_pAlignStack = new CAlignStack;
	m_pAlignStack->Set1(m_iBuffer, m_iNthGpu);
	//-----------------
	m_iIterations = 0;
	float fWeight = 0.3f + 0.1f * pFmGroupParam->m_iGroupSize;
	if(fWeight > 0.9f) fWeight = 0.9f;
	//-----------------
	MMD::CStackShift* pResShift = mAlignStack(pStackShift);
	pStackShift->SetShift(pResShift);
	if(pResShift != 0L) delete pResShift;
	pStackShift->Smooth(fWeight);
	//-----------------
	if(pFmGroupParam->m_iGroupSize >= 2)
	{	CEarlyMotion* pEarlyMotion = new CEarlyMotion;
		pEarlyMotion->Setup(m_iBuffer, m_fBFactor, m_iNthGpu);
		pEarlyMotion->DoIt(pStackShift);
		delete pEarlyMotion;
	}
	delete m_pAlignStack;
	//-----------------
        nvtxRangePop();
}

MMD::CStackShift* CIterativeAlign::mAlignStack(MMD::CStackShift* pInitShift)
{	
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	int aiSumRange[] = {0, pInitShift->m_iNumFrames};
	bool bPatch = (m_iBuffer == MD::EBuffer::pat) ? true : false;
	MMD::CFmGroupParam* pFmGroupParam = 
	   MMD::CFmGroupParam::GetInstance(m_iNthGpu, bPatch);
	int iNumGroups = pFmGroupParam->m_iNumGroups;
	//-----------------
	MMD::CStackShift* pTmpShift = pInitShift->GetCopy();
	MMD::CStackShift* pGroupShift = new MMD::CStackShift;
	MMD::CStackShift* pTotalShift = new MMD::CStackShift;
	pGroupShift->Setup(iNumGroups);
	pTotalShift->Setup(iNumGroups);
	//-----------------
	float fBFactor = m_fBFactor;
	MMD::CStackShift* pIntShift = 0L;
	float fMaxErr = 0.0f;
	CInterpolateShift aIntpShift;
	CAlignedSum alignedSum;
	//-----------------
	for(int i=0; i<m_iMaxIterations; i++)
	{	alignedSum.DoIt(m_iBuffer, pTmpShift, 0L, m_iNthGpu);
		m_pAlignStack->Set2(fBFactor, m_bPhaseOnly);
		m_pAlignStack->DoIt(pTmpShift, pGroupShift);
		//----------------
		fBFactor -= 2;
		if(fBFactor < 20) fBFactor = 20.0f;
		//----------------
		fMaxErr = 0.0f;
		m_pAlignStack->WaitStreams();
		float fErr = m_pAlignStack->m_fErr;
		if(m_pAlignStack->m_fErr > fMaxErr) 
		{	fMaxErr = m_pAlignStack->m_fErr;
		}
		//----------------
		pTotalShift->AddShift(pGroupShift);
		pIntShift = aIntpShift.DoIt(pGroupShift, 
		   m_iNthGpu, bPatch, false);
                pTmpShift->AddShift(pIntShift);
		if(pIntShift != 0L) delete pIntShift;
		//----------------
                m_pfErrors[m_iIterations] = fMaxErr;
                m_iIterations += 1;
		if(fMaxErr < m_fTol && i > 0) break;
        }
	pTotalShift->RemoveSpikes(true);
	pIntShift = aIntpShift.DoIt(pTotalShift, m_iNthGpu, bPatch, false);
	pIntShift->RemoveSpikes(false);
	pIntShift->Smooth(0.5f);
	//-----------------
	if(pGroupShift != 0L) delete pGroupShift;
	if(pTotalShift != 0L) delete pTotalShift;
	if(pTmpShift != 0L) delete pTmpShift;
	if(fMaxErr < m_fTol) pIntShift->m_bConverged = true;
	return pIntShift;	
}

char* CIterativeAlign::GetErrorLog(void)
{
	int iSize = m_iMaxIterations * 1024;
	char* pcLog = new char[iSize];
	memset(pcLog, 0, sizeof(char) * iSize);
	//-------------------------------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	float fXcfBin = fmaxf(pBufferPool->m_afXcfBin[0], 
		pBufferPool->m_afXcfBin[1]);
	char acBuf[80] = {0};
	//-------------------
	for(int i=0; i<m_iIterations; i++)
	{	float fErr = m_pfErrors[i] * fXcfBin;
		sprintf(acBuf, "Iteration %2d  Error: %9.3f\n", i+1, fErr);
		strcat(pcLog, acBuf);
	}
	return pcLog;
}

