#include "CAlignInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::MotionCor;
using namespace McAreTomo::MotionCor::Align;

//--------------------------------------------------------------------
// Number of terms per frame returned by GGradCC2D.
//--------------------------------------------------------------------
static const int s_iTerms = 7;

CGradAlignStack::CGradAlignStack(void)
{
	m_pFrmBuffer = 0L;
	m_pSumBuffer = 0L;
	m_pFmGroupParam = 0L;
	m_pfShifts = 0L;
	m_pfTrials = 0L;
	m_iNumFrames = 0;
	m_afBFactors[0] = 4000.0f;
	m_afBFactors[1] = 20.0f;
	m_fTol = 0.05f;
	m_iMaxIterations = 10;
	m_iNthGpu = -1;
	m_fCC = 0.0f;
	m_fErr = 0.0f;
	m_iIterations = 0;
}

CGradAlignStack::~CGradAlignStack(void)
{
	if(m_pfShifts != 0L) delete[] m_pfShifts;
	if(m_pfTrials != 0L) delete[] m_pfTrials;
}

//--------------------------------------------------------------------
// The B-factors are in the unit of GCorrelateSum2D. The largest one
// sets the capture range: the cross correlation peak has a width of
// sigma = sqrt(B) / (2 PI) pixels. The smallest one sets the final
// accuracy. The tolerance is in pixels of the given buffer.
//--------------------------------------------------------------------
void CGradAlignStack::Setup(int iBuffer, int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	MD::CBufferPool* pBufferPool =
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	m_pFrmBuffer = pBufferPool->GetBuffer(iBuffer);
	m_pSumBuffer = pBufferPool->GetBuffer(MD::EBuffer::sum);
	m_stream = pBufferPool->GetCudaStream(0);
	//-----------------
	bool bPatch = (iBuffer == MD::EBuffer::pat);
	m_pFmGroupParam =
	   MMD::CFmGroupParam::GetInstance(m_iNthGpu, bPatch);
	//-----------------
	if(iBuffer == MD::EBuffer::xcf) this->SetBFactors(16000.0f, 20.0f);
	else if(bPatch) this->SetBFactors(500.0f, 50.0f);
	else this->SetBFactors(4000.0f, 20.0f);
	//-----------------
	CMcInput* pMcInput = CMcInput::GetInstance();
	float* pfXcfBin = pBufferPool->m_afXcfBin;
	float fBin = (iBuffer == MD::EBuffer::frm) ? 1.0f :
	   0.5f * (pfXcfBin[0] + pfXcfBin[1]);
	m_fTol = pMcInput->m_fMcTol / fBin;
	m_iMaxIterations = pMcInput->m_iMcIter;
}

void CGradAlignStack::SetBFactors(float fMaxBFactor, float fMinBFactor)
{
	m_afBFactors[0] = fMaxBFactor;
	m_afBFactors[1] = fMinBFactor;
	if(m_afBFactors[1] > m_afBFactors[0])
	{	m_afBFactors[1] = m_afBFactors[0];
	}
}

void CGradAlignStack::SetTolerance(float fTol)
{
	m_fTol = fTol;
}

void CGradAlignStack::SetMaxIterations(int iMaxIterations)
{
	m_iMaxIterations = iMaxIterations;
}

//--------------------------------------------------------------------
// pStackShift: one shift per frame of the buffer. It provides the
//    initial shifts and receives the refined ones. As in CAlignStack
//    the frames are corrected by the opposite of the stored shifts.
//--------------------------------------------------------------------
void CGradAlignStack::DoIt(MMD::CStackShift* pStackShift)
{
	m_iNumFrames = pStackShift->m_iNumFrames;
	m_gCmpSum = m_pSumBuffer->GetFrame(0);
	m_aGGradCC2D.SetSize(m_pFrmBuffer->m_aiCmpSize, m_iNumFrames);
	//-----------------
	if(m_pfShifts != 0L) delete[] m_pfShifts;
	if(m_pfTrials != 0L) delete[] m_pfTrials;
	m_pfShifts = new float[m_iNumFrames * 2];
	m_pfTrials = new float[m_iNumFrames * 2];
	memcpy(m_pfShifts, pStackShift->GetShifts(),
	   sizeof(float) * m_iNumFrames * 2);
	//-----------------
	m_iIterations = 0;
	m_fErr = 0.0f;
	//-----------------
	// Target function of the initial shifts at the final B-factor
	// to guard against the coarse levels making things worse.
	//-----------------
	m_aGGradCC2D.SetBFactor(m_afBFactors[1]);
	mCalcSum(m_pfShifts);
	mCalcGradient();
	double dInitTarget = m_aGGradCC2D.DoPower(m_gCmpSum, m_stream)
	   - m_dPowSum;
	//-----------------
	float fBFactor = m_afBFactors[0];
	while(true)
	{	mAlignAtBFactor(fBFactor);
		if(fBFactor <= m_afBFactors[1]) break;
		fBFactor = fmaxf(fBFactor * 0.25f, m_afBFactors[1]);
	}
	if(m_dTarget < dInitTarget) 
	{	memcpy(m_pfShifts, pStackShift->GetShifts(),
		   sizeof(float) * m_iNumFrames * 2);
		m_dTarget = dInitTarget;
		double dDenom = (m_iNumFrames - 1) * m_dPowSum;
		m_fCC = (dDenom > 0) ? (float)(m_dTarget / dDenom) : 0.0f;
		m_fErr = 0.0f;
	}
	//-----------------
	float* pfShiftYs = &m_pfShifts[m_iNumFrames];
	for(int i=0; i<m_iNumFrames; i++)
	{	float afShift[] = {m_pfShifts[i], pfShiftYs[i]};
		pStackShift->SetShift(i, afShift);
	}
}

//--------------------------------------------------------------------
// Gradient ascent at a fixed B-factor. Each iteration takes a step
// along the gradient scaled per frame by the local curvature and
// backtracks until the target function increases.
//--------------------------------------------------------------------
void CGradAlignStack::mAlignAtBFactor(float fBFactor)
{
	m_aGGradCC2D.SetBFactor(fBFactor);
	bool bFinal = (fBFactor <= m_afBFactors[1]);
	//-----------------
	float fSigma = (float)(sqrt(fBFactor) / (8.0 * atan(1.0)));
	float fMaxStep = fmaxf(fSigma, 1.0f);
	float fTol = bFinal ? m_fTol : fmaxf(m_fTol, 0.1f * fSigma);
	//-----------------
	mCalcSum(m_pfShifts);
	mCalcGradient();
	m_dTarget = m_aGGradCC2D.DoPower(m_gCmpSum, m_stream) - m_dPowSum;
	//-----------------
	float* pfSteps = new float[m_iNumFrames * 2];
	float* pfStepYs = &pfSteps[m_iNumFrames];
	float* pfTrialYs = &m_pfTrials[m_iNumFrames];
	//-----------------
	for(int it=0; it<m_iMaxIterations; it++)
	{	mCalcStep(fMaxStep, pfSteps);
		//----------------
		bool bAccepted = false;
		float fAlpha = 1.0f;
		for(int t=0; t<5; t++)
		{	for(int i=0; i<m_iNumFrames*2; i++)
			{	m_pfTrials[i] = m_pfShifts[i] + fAlpha * pfSteps[i];
			}
			double dTarget = mCalcTarget(m_pfTrials);
			if(dTarget > m_dTarget)
			{	m_dTarget = dTarget;
				bAccepted = true;
				break;
			}
			fAlpha *= 0.5f;
		}
		if(!bAccepted) break;
		//----------------
		m_fErr = 0.0f;
		for(int i=0; i<m_iNumFrames; i++)
		{	float fDx = fAlpha * pfSteps[i];
			float fDy = fAlpha * pfStepYs[i];
			float fS = sqrtf(fDx * fDx + fDy * fDy);
			if(fS > m_fErr) m_fErr = fS;
		}
		//----------------
		// The sum of the accepted trial is in m_gCmpSum.
		//----------------
		memcpy(m_pfShifts, m_pfTrials, sizeof(float) * m_iNumFrames * 2);
		mCalcGradient();
		m_iIterations += 1;
		if(m_fErr < fTol) break;
	}
	delete[] pfSteps;
	//-----------------
	double dDenom = (m_iNumFrames - 1) * m_dPowSum;
	m_fCC = (dDenom > 0) ? (float)(m_dTarget / dDenom) : 0.0f;
}

//--------------------------------------------------------------------
// S = sum_k F_k shifted by -s_k. It is stored in m_gCmpSum.
//--------------------------------------------------------------------
void CGradAlignStack::mCalcSum(float* pfShifts)
{
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
	cudaMemsetAsync(m_gCmpSum, 0, tBytes, m_stream);
	//-----------------
	float* pfShiftYs = &pfShifts[m_iNumFrames];
	MU::GPhaseShift2D phaseShift;
	for(int i=0; i<m_iNumFrames; i++)
	{	float afShift[] = {-pfShifts[i], -pfShiftYs[i]};
		cufftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(i);
		phaseShift.DoIt(gCmpFrm, m_pFrmBuffer->m_aiCmpSize,
		   afShift, true, m_gCmpSum, m_stream);
	}
}

//--------------------------------------------------------------------
// Must be called right after mCalcSum with the shifts in m_pfShifts.
// It also calculates m_dPowSum that depends on the B-factor only.
// Frames not in GPU memory are in pinned host memory and are read
// directly by the kernel, as done in CAlignedSum.
//--------------------------------------------------------------------
void CGradAlignStack::mCalcGradient(void)
{
	float* pfShiftYs = &m_pfShifts[m_iNumFrames];
	for(int i=0; i<m_iNumFrames; i++)
	{	float afShift[] = {m_pfShifts[i], pfShiftYs[i]};
		cufftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(i);
		m_aGGradCC2D.DoFrame(gCmpFrm, m_gCmpSum, afShift,
		   i, m_stream);
	}
	m_pdGrad = m_aGGradCC2D.Wait(m_iNumFrames, m_stream);
	//-----------------
	m_dPowSum = 0.0;
	for(int i=0; i<m_iNumFrames; i++)
	{	m_dPowSum += m_pdGrad[i * s_iTerms + 6];
	}
}

//--------------------------------------------------------------------
// E = sum_q W |S|^2 - sum_k sum_q W |G_k|^2. The second term does not
// depend on the shifts and has been calculated in mCalcGradient. The
// sum of the trial shifts is left in m_gCmpSum.
//--------------------------------------------------------------------
double CGradAlignStack::mCalcTarget(float* pfShifts)
{
	mCalcSum(pfShifts);
	double dPower = m_aGGradCC2D.DoPower(m_gCmpSum, m_stream);
	return dPower - m_dPowSum;
}

//--------------------------------------------------------------------
// The step of frame k is h^-1 g where g is the gradient and h the
// negative 2x2 Hessian of the target function with respect to s_k.
// When the frames are grouped, g and h are pooled over the frames
// of the group centered at k, assuming they move together. When h
// is not positive definite, the step falls back to g scaled by the
// mean curvature of all frames. Steps are capped at fMaxStep and
// their mean is removed since E is invariant to a global shift.
//--------------------------------------------------------------------
void CGradAlignStack::mCalcStep(float fMaxStep, float* pfSteps)
{
	float* pfStepYs = &pfSteps[m_iNumFrames];
	//-----------------
	double dMeanCurv = 0.0;
	for(int i=0; i<m_iNumFrames; i++)
	{	double* pdTerms = &m_pdGrad[i * s_iTerms];
		dMeanCurv += (pdTerms[2] + pdTerms[3]);
	}
	dMeanCurv = dMeanCurv / m_iNumFrames;  // 2 * 0.5 * (hxx + hyy)
	if(dMeanCurv <= 0) dMeanCurv = 1e-30;
	//-----------------
	bool bGroup = (m_pFmGroupParam != 0L &&
	   m_pFmGroupParam->m_iNumGroups == m_iNumFrames &&
	   m_pFmGroupParam->m_iGroupSize > 1);
	int iGpSize = bGroup ? m_pFmGroupParam->m_iGroupSize : 1;
	//-----------------
	double adMean[2] = {0.0};
	for(int i=0; i<m_iNumFrames; i++)
	{	int* piIdxs = bGroup ? m_pFmGroupParam->GetGroupIdxs(i) : &i;
		double adG[2] = {0.0}, adH[3] = {0.0};
		for(int j=0; j<iGpSize; j++)
		{	double* pdTerms = &m_pdGrad[piIdxs[j] * s_iTerms];
			adG[0] += 2.0 * pdTerms[0];
			adG[1] += 2.0 * pdTerms[1];
			adH[0] += 2.0 * pdTerms[2];
			adH[1] += 2.0 * pdTerms[3];
			adH[2] += 2.0 * pdTerms[4];
		}
		//----------------
		double dDamp = 0.01 * dMeanCurv * iGpSize;
		double dHxx = adH[0] + dDamp;
		double dHyy = adH[1] + dDamp;
		double dDet = dHxx * dHyy - adH[2] * adH[2];
		double dSx, dSy;
		if(dHxx > 0 && dHyy > 0 && dDet > 0)
		{	dSx = (dHyy * adG[0] - adH[2] * adG[1]) / dDet;
			dSy = (dHxx * adG[1] - adH[2] * adG[0]) / dDet;
		}
		else
		{	dSx = adG[0] / (dMeanCurv * iGpSize);
			dSy = adG[1] / (dMeanCurv * iGpSize);
		}
		//----------------
		double dS = sqrt(dSx * dSx + dSy * dSy);
		if(dS > fMaxStep)
		{	dSx *= (fMaxStep / dS);
			dSy *= (fMaxStep / dS);
		}
		pfSteps[i] = (float)dSx;
		pfStepYs[i] = (float)dSy;
		adMean[0] += dSx;
		adMean[1] += dSy;
	}
	//-----------------
	adMean[0] /= m_iNumFrames;
	adMean[1] /= m_iNumFrames;
	for(int i=0; i<m_iNumFrames; i++)
	{	pfSteps[i] -= (float)adMean[0];
		pfStepYs[i] -= (float)adMean[1];
	}
}
