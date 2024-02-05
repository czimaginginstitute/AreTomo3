#include "CBadPixelInc.h"
#include "../CMotionCorInc.h"
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::BadPixel;

CLocalCCMap::CLocalCCMap(void)
{
}

CLocalCCMap::~CLocalCCMap(void)
{
}

void CLocalCCMap::DoIt(int* piModSize, int iNthGpu)
{	
	nvtxRangePushA("CLocalCCMap::DoIt");
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(iNthGpu);
	MD::CStackBuffer* pSumBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::sum);
	float* gfPadSum = (float*)pSumBuffer->GetFrame(0);
	//-----------------
	MD::CStackBuffer* pTmpBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::tmp);
	float* gfPadCC = (float*)pTmpBuffer->GetFrame(0);
	//-----------------
	MD::CStackBuffer* pFrmBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::frm);
	int aiPadSize[2] = {0, pFrmBuffer->m_aiCmpSize[1]};
	aiPadSize[0] = pFrmBuffer->m_aiCmpSize[0] * 2;
	//-----------------------------------------------------------
	// pvPinnedBuf is also used to store bad pixel map with it's
	// first aiPadSize[0] * aiPadSize[1] bytes in CDetectMain.
	//-----------------------------------------------------------
	char* pcPinnedBuf = (char*)pBufferPool->GetPinnedBuf(0);
	float* pfMod = (float*)pBufferPool->GetPinnedBuf(1);
	int iPadSize = aiPadSize[0] * aiPadSize[1];
	//-----------------
	CTemplate aTemplate;
	aTemplate.Create(piModSize, pfMod);
	//-----------------
	cudaStream_t aStream = pBufferPool->GetCudaStream(0);
	GLocalCC aGLocalCC;
	aGLocalCC.SetRef(pfMod, piModSize);
	aGLocalCC.DoIt(gfPadSum, aiPadSize, 0, iPadSize, gfPadCC, aStream);
	cudaStreamSynchronize(aStream);
	//-----------------
	nvtxRangePop();
}
