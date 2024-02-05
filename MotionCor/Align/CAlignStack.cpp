#include "CAlignInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::MotionCor;
using namespace McAreTomo::MotionCor::Align;

CAlignStack::CAlignStack(void)
{
	m_iNthGpu = -1;
}

CAlignStack::~CAlignStack(void)
{
}

void CAlignStack::Set1
(	int iBuffer, 
	int iNthGpu
)
{	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	m_pFrmBuffer = pBufferPool->GetBuffer(iBuffer);
	m_pTmpBuffer = pBufferPool->GetBuffer(MD::EBuffer::tmp);
	m_pSumBuffer = pBufferPool->GetBuffer(MD::EBuffer::sum);
	//-----------------
	m_streams[0] = pBufferPool->GetCudaStream(0);
	m_streams[1] = pBufferPool->GetCudaStream(1);
	//-----------------
	bool bPatch = (iBuffer == MD::EBuffer::pat);
	m_pFmGroupParam = 
	   MMD::CFmGroupParam::GetInstance(m_iNthGpu, bPatch);
	//-----------------
	int* piCmpSize = m_pFrmBuffer->m_aiCmpSize;
	int iImgSizeX = (piCmpSize[0] - 1) * 2;
	m_aiSeaSize[0] = (64 < iImgSizeX) ? 64 : iImgSizeX;
	m_aiSeaSize[1] = (64 < piCmpSize[1]) ? 64 : piCmpSize[1];
	m_aGCorrelateSum.SetSize(m_pFrmBuffer->m_aiCmpSize, m_aiSeaSize);
	//-----------------
	bool bForward = true;
	m_pInverseFFT = pBufferPool->GetCufft2D(!bForward);
	m_pInverseFFT->CreateInversePlan(m_pFrmBuffer->m_aiCmpSize, true);
}

void CAlignStack::Set2(float fBFactor, bool bPhaseOnly)
{
	m_aGCorrelateSum.SetFilter(fBFactor, bPhaseOnly);
}

void CAlignStack::DoIt
(	MMD::CStackShift* pStackShift, 
	MMD::CStackShift* pGroupShift
)
{	m_pStackShift = pStackShift;
	m_pGroupShift = pGroupShift;
	//-----------------
	m_gCmpSum = m_pSumBuffer->GetFrame(0);
	m_fErr = 0.0f;
	mDoGroups();
}

void CAlignStack::WaitStreams(void)
{
	cudaStreamSynchronize(m_streams[0]);
	cudaStreamSynchronize(m_streams[1]);
	mFindPeaks();
}

void CAlignStack::mDoGroups(void)
{
	int iStream = 0;	
	for(int i=0; i<m_pFmGroupParam->m_iNumGroups; i++)
	{	mDoGroup(i, iStream);
		iStream = (iStream + 1) % 2;
	}
}

void CAlignStack::mDoGroup(int iGroup, int iStream)
{
	int iGpStart = m_pFmGroupParam->GetGroupStart(iGroup);
	int iGpSize = m_pFmGroupParam->GetGroupSize(iGroup);
	//-----------------
	for(int i=0; i<iGpSize; i++)
	{	m_iFrame = iGpStart + i;
		bool bSum = (i > 0) ? true : false;
		mPhaseShift(iStream, bSum);
	}
	mCorrelate(iGroup, iStream);
}

void CAlignStack::mPhaseShift(int iStream, bool bSum)
{
	float afShift[2] = {0.0f};
	m_pStackShift->GetShift(m_iFrame, afShift, -1.0f);
	//-----------------
	cufftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(m_iFrame);
	cufftComplex* gCmpTmp = m_pTmpBuffer->GetFrame(iStream);
	MU::GPhaseShift2D phaseShift;
	//-----------------
	cufftComplex* gCmpIn = gCmpFrm;
	if(!m_pFrmBuffer->IsGpuFrame(m_iFrame))
	{	int iCmpSize = m_pFrmBuffer->m_aiCmpSize[0] *
		   m_pFrmBuffer->m_aiCmpSize[1];
		gCmpIn = &gCmpTmp[iCmpSize];
		cudaMemcpyAsync(gCmpIn, gCmpFrm, m_pFrmBuffer->m_tFmBytes,
		   cudaMemcpyDefault, m_streams[iStream]);
	}
	//-----------------
	phaseShift.DoIt(gCmpIn, m_pFrmBuffer->m_aiCmpSize,
	   afShift, bSum, gCmpTmp, m_streams[iStream]);
}

void CAlignStack::mCorrelate
(	int iGroup, 
	int iStream
)
{	int iSeaSize = m_aiSeaSize[0] * m_aiSeaSize[1];	
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	//-----------------
	float* pfPinnedBuf = (float*)pBufferPool->GetPinnedBuf(0);
	float* pfXcfBuf = pfPinnedBuf + iSeaSize * iGroup;
	//-----------------
	cufftComplex* gCmpTmp = m_pTmpBuffer->GetFrame(iStream);
	m_aGCorrelateSum.DoIt(m_gCmpSum, gCmpTmp, pfXcfBuf,
	   m_pInverseFFT, m_streams[iStream]);
}

void CAlignStack::mFindPeaks(void)
{
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	void* pvPinnedBuf = pBufferPool->GetPinnedBuf(0);
	//-----------------
	float* pfPinnedBuf = reinterpret_cast<float*>(pvPinnedBuf);
	int iSeaSize = m_aiSeaSize[0] * m_aiSeaSize[1];
	MU::CPeak2D peak2D;
	//-----------------
	for(int i=0; i<m_pGroupShift->m_iNumFrames; i++)
	{	float* pfXcfBuf = pfPinnedBuf + i * iSeaSize;
		peak2D.DoIt(pfXcfBuf, m_aiSeaSize, false, 0L);
		m_pGroupShift->SetShift(i, peak2D.m_afShift);
		mUpdateError(peak2D.m_afShift);
	}
}

void CAlignStack::mUpdateError(float* pfShift)
{
	double dS = sqrtf(pfShift[0] * pfShift[0] + pfShift[1] * pfShift[1]);
	if(m_fErr < dS) m_fErr = (float)dS;
}
