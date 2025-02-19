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
	m_fBFactor = (iBuffer == MD::EBuffer::xcf) ? 500.0f : 50.0f;
	m_bPhaseOnly = false;
	//-----------------
	CMcInput* pMcInput = CMcInput::GetInstance();	
	m_iMaxIterations = pMcInput->m_iMcIter;
	//-----------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	m_afXcfBin[0] = pBufferPool->m_afXcfBin[0];
	m_afXcfBin[1] = pBufferPool->m_afXcfBin[1];
	m_fTol = pMcInput->m_fMcTol / (m_afXcfBin[0] + m_afXcfBin[1]) * 2.0f;
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
	MMD::CStackShift* pGroupShift = new MMD::CStackShift;
	MMD::CStackShift* pTotalShift = new MMD::CStackShift;
	pGroupShift->Setup(iNumGroups);
	pTotalShift->Setup(iNumGroups);
	//-----------------
	float fBFactor = m_fBFactor;
	float fMaxErr = bPatch ? 10.0f : 100.0f;
	fMaxErr = fMaxErr * 2.0f / (m_afXcfBin[0] + m_afXcfBin[1]);
	CAlignedSum alignedSum;
	//-----------------
	float fBestBF = 0.0f;
	float fMinErr = (float)1e20;
	for(int i=0; i<m_iMaxIterations; i++)
        {       fBFactor = i * 5.0f; 
		alignedSum.DoIt(m_iBuffer, pTotalShift, 0L, m_iNthGpu);
                m_pAlignStack->Set2(fBFactor, m_bPhaseOnly);
                m_pAlignStack->DoIt(pTotalShift, pGroupShift);
                //----------------
                m_pAlignStack->WaitStreams();
                float fErr = m_pAlignStack->m_fErr;
		if(fErr < fMinErr)
		{	fMinErr = fErr;
			fBestBF = fBFactor;
		}
	}
	if(fMinErr > fMaxErr)
	{	delete pGroupShift;
		printf("Inaccurate measurement, skip.\n\n");
		return pTotalShift;
	}
	fBFactor = fBestBF;
	//---------------------------
	for(int i=0; i<m_iMaxIterations; i++)
	{	alignedSum.DoIt(m_iBuffer, pTotalShift, 0L, m_iNthGpu);
		m_pAlignStack->Set2(fBFactor, m_bPhaseOnly);
		m_pAlignStack->DoIt(pTotalShift, pGroupShift);
		//-------------------
		m_pAlignStack->WaitStreams();
		float fErr = m_pAlignStack->m_fErr;
		if(fErr < fMaxErr) 
		{	fMaxErr = fErr;
			pTotalShift->AddShift(pGroupShift);
		}
		else 
		{	fBFactor -= 1;
			if(fBFactor < 1) fBFactor = 1.0f;
			break;
		}
                m_pfErrors[m_iIterations] = fErr;
                m_iIterations += 1;
		if(fMaxErr < m_fTol && i > 0) break;
        }
	pTotalShift->RemoveSpikes(false);
	pTotalShift->Smooth(0.4f);
	//-----------------
	if(pGroupShift != 0L) delete pGroupShift;
	if(fMaxErr < m_fTol) pTotalShift->m_bConverged = true;
	return pTotalShift;	
}

char* CIterativeAlign::GetErrorLog(void)
{
	int iSize = m_iMaxIterations * 1024;
	char* pcLog = new char[iSize];
	memset(pcLog, 0, sizeof(char) * iSize);
	//-------------------------------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	float fXcfBin = fmaxf(m_afXcfBin[0], m_afXcfBin[1]);
	char acBuf[80] = {0};
	//-------------------
	for(int i=0; i<m_iIterations; i++)
	{	float fErr = m_pfErrors[i] * fXcfBin;
		sprintf(acBuf, "Iteration %2d  Error: %9.3f\n", i+1, fErr);
		strcat(pcLog, acBuf);
	}
	return pcLog;
}

