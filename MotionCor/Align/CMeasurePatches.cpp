#include "CAlignInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;

CMeasurePatches::CMeasurePatches(void)
{
}

CMeasurePatches::~CMeasurePatches(void)
{
}

void CMeasurePatches::DoIt
(	MMD::CPatchShifts* pPatchShifts,
	int iNthGpu
)
{	m_pPatchShifts = pPatchShifts;
	m_iNthGpu = iNthGpu;
	//-----------------
	CMcInput* pInput = CMcInput::GetInstance();
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	//-----------------
	float fBin = fmaxf(pBufferPool->m_afXcfBin[0], 
	   pBufferPool->m_afXcfBin[1]);
	m_fTol = pInput->m_fMcTol / fBin;
	//-----------------
	m_iterAlign.Setup(MD::EBuffer::pat, m_iNthGpu);
	//-----------------
	bool bForward = true, bNorm = false;
	m_transformStack.Setup(MD::EBuffer::pat, bForward, bNorm, m_iNthGpu);
	//-----------------
	CExtractPatch extractPatch;
	int iNumPatches = pInput->m_aiNumPatches[0] 
	   * pInput->m_aiNumPatches[1];
	for(int i=0; i<iNumPatches; i++)
	{	extractPatch.DoIt(i, m_iNthGpu);
		mCalcPatchShift(i);
	}
	//m_pPatchShifts->MakeRelative();
	m_pPatchShifts->DetectBads();
}

void CMeasurePatches::mCalcPatchShift(int iPatch)
{
	CMcInput* pInput = CMcInput::GetInstance();	
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pFrmBuffer = pBufferPool->GetBuffer(MD::EBuffer::frm);
	MD::CStackBuffer* pPatBuffer = pBufferPool->GetBuffer(MD::EBuffer::pat);
	//-----------------
	m_transformStack.DoIt();
	//-----------------
	MMD::CStackShift* pPatShift = new MMD::CStackShift;
	pPatShift->Setup(pPatBuffer->m_iNumFrames);
	//-----------------
	m_iterAlign.DoIt(pPatShift);
	//-----------------
	int iNumPatches = pInput->m_aiNumPatches[0] 
	   * pInput->m_aiNumPatches[1];
	char* pcErrLog = m_iterAlign.GetErrorLog();
	int iLeft = iNumPatches - 1 - iPatch;
	printf("Align patch %d  %d left\n%s\n", iPatch+1, iLeft, pcErrLog);
	if(pcErrLog != 0L) delete[] pcErrLog;
	//-----------------
	int aiCenter[2] = {0};
	CPatchCenters* pPatchCenters = CPatchCenters::GetInstance(m_iNthGpu);
	pPatchCenters->GetCenter(iPatch, aiCenter);
	float fCentX = aiCenter[0] * pBufferPool->m_afXcfBin[0];
	float fCentY = aiCenter[1] * pBufferPool->m_afXcfBin[1];
	pPatShift->SetCenter(fCentX, fCentY);
	//-----------------
	pPatShift->Multiply(pBufferPool->m_afXcfBin[0],
	   pBufferPool->m_afXcfBin[1]);
	m_pPatchShifts->SetRawShift(pPatShift, iPatch);
	if(pPatShift != 0L) delete pPatShift;
}

