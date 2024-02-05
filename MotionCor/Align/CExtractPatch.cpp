#include "CAlignInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::MotionCor::Align;
namespace MU = McAreTomo::MaUtil;
	
CExtractPatch::CExtractPatch(void)
{
}

CExtractPatch::~CExtractPatch(void)
{
}

void CExtractPatch::DoIt(int iPatch, int iNthGpu)
{
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(iNthGpu);
	m_streams[0] = pBufferPool->GetCudaStream(0);
	m_streams[1] = pBufferPool->GetCudaStream(1);
	//-----------------
	m_pPatBuffer = pBufferPool->GetBuffer(MD::EBuffer::pat);
	m_pXcfBuffer = pBufferPool->GetBuffer(MD::EBuffer::xcf);
	m_pSumBuffer = pBufferPool->GetBuffer(MD::EBuffer::sum);
	//-----------------
	CPatchCenters* pPatchCenters = CPatchCenters::GetInstance(iNthGpu);
	pPatchCenters->GetStart(iPatch, m_aiPatStart);
	//-----------------
	for(int i=0; i<m_pPatBuffer->m_iNumFrames; i++)
	{	mProcessFrame(i);
	}
	//-----------------
	cudaStreamSynchronize(m_streams[0]);
	cudaStreamSynchronize(m_streams[1]);
}

void CExtractPatch::mProcessFrame(int iFrame)
{
	cufftComplex* gPatCmp = m_pPatBuffer->GetFrame(iFrame);
	cufftComplex* gXcfCmp = m_pXcfBuffer->GetFrame(iFrame);
	float* gfPatFrm = reinterpret_cast<float*>(gPatCmp);
	float* gfXcfFrm = reinterpret_cast<float*>(gXcfCmp);
	//-----------------
	int iXcfPadX = m_pXcfBuffer->m_aiCmpSize[0] * 2;
	int iOffset = m_aiPatStart[1] * iXcfPadX + m_aiPatStart[0];
	float* gfSrc = gfXcfFrm + iOffset;
	//-----------------
	int iCpySizeX = (m_pPatBuffer->m_aiCmpSize[0] - 1) * 2;
	int iPatPadX = m_pPatBuffer->m_aiCmpSize[0] * 2;
	int iPatPadY = m_pPatBuffer->m_aiCmpSize[1];
	int aiPatPadSize[] = {iPatPadX, iPatPadY};
	//-----------------
	MU::GPartialCopy::DoIt(gfSrc, iXcfPadX, gfPatFrm, 
	   iCpySizeX, aiPatPadSize, m_streams[iFrame % 2]);
}

