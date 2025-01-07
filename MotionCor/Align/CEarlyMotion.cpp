#include "CAlignInc.h"
#include "../Util/CUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Align;
using namespace McAreTomo::MotionCor;
namespace MMD = McAreTomo::MotionCor::DataUtil;
namespace MAU = McAreTomo::MaUtil;

CEarlyMotion::CEarlyMotion(void)
{
	m_fBFactor = 500.0f;
	m_gCmpRef = 0L;
	m_iNthGpu = -1;
}

CEarlyMotion::~CEarlyMotion(void)
{
	if(m_gCmpRef != 0L) cudaFree(m_gCmpRef);
}

void CEarlyMotion::Setup(int iBuffer, float fBFactor, int iNthGpu)
{	
	m_iBuffer = iBuffer;
	m_fBFactor = fBFactor;
	m_iNthGpu = iNthGpu;
	//-----------------
	CMcInput* pMcInput = CMcInput::GetInstance();	
	if(iBuffer == MD::EBuffer::xcf 
	   && pMcInput->m_aiGroup[0] == 1) return;
	if(iBuffer == MD::EBuffer::pat 
	   && pMcInput->m_aiGroup[1] == 1) return;
	//-----------------
	m_pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pStackBuffer = 
	   m_pBufferPool->GetBuffer(iBuffer);
	memcpy(m_aiCmpSize, pStackBuffer->m_aiCmpSize, sizeof(int) * 2);
	//-----------------
	if(m_gCmpRef != 0L) cudaFree(m_gCmpRef);
	size_t tBytes = sizeof(cufftComplex) *
	   m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gCmpRef, tBytes);
	//-----------------
	m_aiSeaSize[0] = 16; m_aiSeaSize[1] = 16;
	m_aGCorrelateSum.SetSubtract(false);
	m_aGCorrelateSum.SetFilter(m_fBFactor, false);
	m_aGCorrelateSum.SetSize(m_aiCmpSize, m_aiSeaSize);
	//-----------------
	m_pInverseFFT = m_pBufferPool->GetCufft2D(false);
	m_pInverseFFT->CreateInversePlan(m_aiCmpSize, true);
}

void CEarlyMotion::DoIt(MMD::CStackShift* pStackShift)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(m_iBuffer == MD::EBuffer::xcf 
	   && pMcInput->m_aiGroup[0] == 1) return;
	if(m_iBuffer == MD::EBuffer::pat 
	   && pMcInput->m_aiGroup[1] == 1) return;
	if(pStackShift->m_iNumFrames < 2) return;
	//-----------------
	m_pStackShift = pStackShift; 
	//-----------------
	bool bPatch = (m_iBuffer == MD::EBuffer::pat) ? true : false;
	MMD::CFmGroupParam* pFmGrpParam = 
	   MMD::CFmGroupParam::GetInstance(m_iNthGpu, bPatch);
	if(pFmGrpParam->m_iNumGroups < 2) return;
	//-------------------------------------------------------
	// 1) GroupCent[0] is at the 1st int frame. GroupCent[1]
	//    is at the center of the 1st group. GroupCent[2]
	//    if at the center of the 2nd group.
	// 2) This is why we need at least two group sums.
	//-------------------------------------------------------
	m_afGroupCent[0] = 0.0f;
	m_afGroupCent[1] = pFmGrpParam->GetGroupCenter(0);
	m_afGroupCent[2] = pFmGrpParam->GetGroupCenter(1);
	//-------------------------------------------------------
	// 1. calculate the sum from the 1st frame in the second
	//    group to the frame at 2/3 of all frames.
	// 2. This sum is the reference to align the sum of the
	//    first group.
	//-------------------------------------------------------
	m_aiSumRange[0] = pFmGrpParam->GetGroupSize(0);
	m_aiSumRange[1] = m_pStackShift->m_iNumFrames * 2 / 3;
	CAlignedSum alignedSum;
	alignedSum.DoIt(m_iBuffer, m_pStackShift, m_aiSumRange, m_iNthGpu);
	//-----------------
	MD::CStackBuffer* pSumBuffer = 
	   m_pBufferPool->GetBuffer(MD::EBuffer::sum);
	cufftComplex* gCmpSum = pSumBuffer->GetFrame(0);
	size_t tBytes = sizeof(cufftComplex) * m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMemcpy(m_gCmpRef, gCmpSum, tBytes, cudaMemcpyDefault);
	//-----------------
	mDoIt();
}

void CEarlyMotion::mDoIt(void)
{
	float afRefPeak[3] = {0.0f};
	mCorrelate(0, m_pStackShift);	
	mFindPeak(0, afRefPeak);
	//-----------------
	m_iNumSteps = 40;
	m_fStepSize = 0.1f;
	MMD::CStackShift* pStackShift = m_pStackShift->GetCopy();
	float fIncX = mIterate(pStackShift, 0);
	float fIncY = mIterate(pStackShift, 1);
	//printf("Inc: %8.2f  %8.2f\n", fIncX, fIncY);
	//-----------------
	float afOptPeak[3] = {0.0f};
	mCorrelate(0, pStackShift);
	mFindPeak(0, afOptPeak);
	if(afRefPeak[2] < afOptPeak[2]) 
	{	m_pStackShift->SetShift(pStackShift);
	}
	//-------------------------------------------
	/*printf("Ref: %8.2f  %8.2f  %10.7e\n",
	   afRefPeak[0], afRefPeak[1], afRefPeak[2]); 
	printf("Opt: %8.2f  %8.2f  %10.7e\n",
	   afOptPeak[0], afOptPeak[1], afOptPeak[2]);
	*/
	delete pStackShift;
}

float CEarlyMotion::mIterate(MMD::CStackShift* pStackShift, int iAxis)
{
	float* pfIncs = new float[m_iNumSteps];
	float* pfPeaks = new float[m_iNumSteps * 3];
	float afShifts[3] = {0.0f}, afCoeffs[3] = {0.0f};
	mGetNodeShifts(pStackShift, iAxis, afShifts);	
	//-------------------------------------------
	float* pfCoeffXs = (iAxis == 0) ? &afCoeffs[0] : 0L;
	float* pfCoeffYs = (iAxis == 1) ? &afCoeffs[0] : 0L;
	//--------------------------------------------------	
	for(int i=0; i<m_iNumSteps; i++)
        {       pfIncs[i] = 1.0f + (i - 0.5f * m_iNumSteps) * m_fStepSize;
                mCalcCoeff(pfIncs[i], afShifts, &afCoeffs[0]);
                mCalcShift(pfCoeffXs, pfCoeffYs, pStackShift);
                mCorrelate(i, pStackShift);
        }
        mFindPeaks(pfPeaks);
	//------------------
	int iOptimal = -1;
	float fPeak = (float)-1e30;
	for(int i=0; i<m_iNumSteps; i++)
	{	float* pfPeak = pfPeaks + i * 3;
		//printf("%3d  %3d  %8.2f  %8.2f  %10.7e\n", i, iOptimal,
		//   pfPeak[0], pfPeak[1], pfPeak[2]);
		if(fPeak >= pfPeak[2]) continue;
		fPeak = pfPeak[2];
		iOptimal = i;
	}
	//printf("\n");
	//---------------------------------------------
	float fInc = pfIncs[iOptimal];
	mCalcCoeff(fInc, afShifts, &afCoeffs[0]);
	mCalcShift(pfCoeffXs, pfCoeffYs, pStackShift);
	//--------------------------------------------
	float* pfExtraShift = &pfPeaks[iOptimal * 3];
	for(int i=0; i<m_aiSumRange[0]; i++)
	{	pStackShift->AddShift(i, pfExtraShift);
	}
	delete[] pfIncs;
	delete[] pfPeaks;
	return fInc;
}

void CEarlyMotion::mGetNodeShifts
(	MMD::CStackShift* pStackShift, 
	int iAxis, float* pfShift
)
{	float afShifts[6] = {0.0f};
	pStackShift->GetShift((int)m_afGroupCent[0], &afShifts[0]);
	pStackShift->GetShift((int)m_afGroupCent[1], &afShifts[2]);
	pStackShift->GetShift((int)m_afGroupCent[2], &afShifts[4]);
	pfShift[0] = afShifts[0 + iAxis];
	pfShift[1] = afShifts[2 + iAxis];
	pfShift[2] = afShifts[4 + iAxis];
}
	
void CEarlyMotion::mCalcCoeff(float fGain, float* pfShift, float* pfCoeff)
{
	pfCoeff[0] = pfShift[0] + fGain;
	//------------------
	float x1_2 = m_afGroupCent[1] * m_afGroupCent[1];
	float x2_2 = m_afGroupCent[2] * m_afGroupCent[2];
	float fDelta = m_afGroupCent[1] * x2_2 - m_afGroupCent[2] * x1_2;
	//------------------
	pfCoeff[1] = ((pfShift[1] - pfCoeff[0]) * x2_2 - 
	   (pfShift[2] - pfCoeff[0]) * x1_2) / fDelta;
	pfCoeff[2] = ((pfShift[2] - pfCoeff[0]) * m_afGroupCent[1] -
	   (pfShift[1] - pfCoeff[0]) * m_afGroupCent[2]) / fDelta;
}

void CEarlyMotion::mCalcShift
(	float* pfCoeffXs,
	float* pfCoeffYs,
	MMD::CStackShift* pStackShift
)
{	float afShift[2] = {0.0f};
	for(int i=0; i<m_aiSumRange[0]; i++)
	{	pStackShift->GetShift(i, afShift);
		int x = i;
		int x2 = x * x;
		if(pfCoeffXs != 0L)
		{	afShift[0] = pfCoeffXs[0] + pfCoeffXs[1] * x 
			   + pfCoeffXs[2] * x2;
		}
		if(pfCoeffYs != 0L)
		{	afShift[1] = pfCoeffYs[0] + pfCoeffYs[1] * x 
			   + pfCoeffYs[2] * x2;
		}
		pStackShift->SetShift(i, afShift);
	}
}

void CEarlyMotion::mCorrelate(int iStep, MMD::CStackShift* pStackShift)
{
	//------------------------------------------------
	// 1) generated the aligned sum of the 1st group.
	//------------------------------------------------
	int aiSumRange[2] = {0, 1};
	aiSumRange[1] = m_aiSumRange[0] - 1;
	CAlignedSum alignedSum;
	alignedSum.DoIt(m_iBuffer, pStackShift, aiSumRange, m_iNthGpu);
	//-----------------
	MD::CStackBuffer* pSumBuffer = 
	   m_pBufferPool->GetBuffer(MD::EBuffer::sum);
	cufftComplex* gCmpSum = pSumBuffer->GetFrame(0);
	//-----------------
	int iSeaSize = m_aiSeaSize[0] * m_aiSeaSize[1];
	float* pfPinnedBuf = (float*)m_pBufferPool->GetPinnedBuf(0);
	float* pfXcfBuf = pfPinnedBuf + iSeaSize * iStep;
	//------------------------------------------------
	// 1) correlate the aligned sum of the 1st group
	//    against the reference to see if there is
	//    any residual shift in the 1st group.
	//------------------------------------------------
	m_aGCorrelateSum.DoIt(m_gCmpRef, gCmpSum, pfXcfBuf, m_pInverseFFT, 0);	
}

void CEarlyMotion::mFindPeaks(float* pfPeaks)
{
	cudaStreamSynchronize((cudaStream_t)0);
	//-----------------
	float* pfPinnedBuf = (float*)m_pBufferPool->GetPinnedBuf(0);
	int iSeaSize = m_aiSeaSize[0] * m_aiSeaSize[1];
	//-----------------
	MU::CPeak2D peak2D;
	bool bPadded = false;
	for(int i=0; i<m_iNumSteps; i++)
	{	float* pfXcfBuf = pfPinnedBuf + i * iSeaSize;
		peak2D.DoIt(pfXcfBuf, m_aiSeaSize, bPadded, 0L);
		float* pfPeak = pfPeaks + i * 3;
		memcpy(pfPeak, peak2D.m_afShift, sizeof(float) * 2);
		pfPeak[2] = peak2D.m_fPeakInt;
	}
}

void CEarlyMotion::mFindPeak(int iPeak, float* pfPeak)
{
	cudaStreamSynchronize((cudaStream_t)0);
	//-----------------
        float* pfPinnedBuf = (float*)m_pBufferPool->GetPinnedBuf(0);
        int iSeaSize = m_aiSeaSize[0] * m_aiSeaSize[1];
	float* pfXcfImg = pfPinnedBuf + iPeak * iSeaSize;
	//-----------------
	MU::CPeak2D peak2D;
	peak2D.DoIt(pfXcfImg, m_aiSeaSize, false, 0L);
	memcpy(pfPeak, peak2D.m_afShift, sizeof(float) * 2);
	pfPeak[2] = peak2D.m_fPeakInt;
}
