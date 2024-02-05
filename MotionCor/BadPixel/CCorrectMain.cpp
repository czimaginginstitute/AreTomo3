#include "CBadPixelInc.h"
#include "../CMotionCorInc.h"
#include "../Util/CUtilInc.h"
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::BadPixel;
namespace MMU = McAreTomo::MotionCor::Util;

CCorrectMain* CCorrectMain::m_pInstances = 0L;
int CCorrectMain::m_iNumGpus = 0;

void CCorrectMain::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	delete[] m_pInstances;
	m_pInstances = new CCorrectMain[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CCorrectMain::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CCorrectMain* CCorrectMain::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CCorrectMain::CCorrectMain(void)
{
}

CCorrectMain::~CCorrectMain(void)
{
}

void CCorrectMain::DoIt(int iDefectSize, int iNthGpu)
{		
	nvtxRangePushA("CCorrectMain::DoIt");
	m_iDefectSize = iDefectSize;
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pFrmBuffer = 	
	   pBufferPool->GetBuffer(MD::EBuffer::frm);
	//-----------------
	m_aiPadSize[0] = pFrmBuffer->m_aiCmpSize[0] * 2;
	m_aiPadSize[1] = pFrmBuffer->m_aiCmpSize[1];
	//-----------------
	mCorrectFrames();
        nvtxRangePop();
}

void CCorrectMain::mCorrectFrames(void)
{
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pFrmBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::frm);
	MD::CStackBuffer* pTmpBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::tmp);
	//-----------------
	void* pvBuf = pBufferPool->GetPinnedBuf(0);
	size_t tBytes = m_aiPadSize[0] * m_aiPadSize[1] * sizeof(char);
	cufftComplex* gCmpBuf = pTmpBuffer->GetFrame(0);
	cudaMemcpy(gCmpBuf, pvBuf, tBytes, cudaMemcpyDefault);
	//-----------------
	for(int i=0; i<pFrmBuffer->m_iNumFrames; i++)
	{	mCorrectFrame(i);
	}
}

void CCorrectMain::mCorrectFrame(int iFrame)
{
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	//-----------------
	int iStream = iFrame % 2;
	cudaStream_t stream = pBufferPool->GetCudaStream(iStream);
	//-----------------
	MD::CStackBuffer* pFrmBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::frm);
	MD::CStackBuffer* pTmpBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::tmp);
	//-----------------
	cufftComplex* gCmpFrm = pFrmBuffer->GetFrame(iFrame);
	cufftComplex* gCmpMap = pTmpBuffer->GetFrame(0);
	//-----------------
	float* gfPadFrm = reinterpret_cast<float*>(gCmpFrm);
	unsigned char* gucMap = reinterpret_cast<unsigned char*>(gCmpMap);
	//-----------------
	GCorrectBad aGCorrectBad;
	aGCorrectBad.SetWinSize(m_iDefectSize);
	aGCorrectBad.GDoIt(gfPadFrm, gucMap, m_aiPadSize, true, stream);
}

