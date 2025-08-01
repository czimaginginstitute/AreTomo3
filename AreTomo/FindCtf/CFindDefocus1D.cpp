#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.0174533f;

CFindDefocus1D::CFindDefocus1D(void)
{
	m_gfCtf1D = 0L;
	m_pGCC1D = 0L;
}

CFindDefocus1D::~CFindDefocus1D(void)
{
	this->Clean();
}

void CFindDefocus1D::Clean(void)
{
	if(m_gfCtf1D != 0L) 
	{	cudaFree(m_gfCtf1D);
		m_gfCtf1D = 0L;
	}
	if(m_pGCC1D != 0L)
	{	delete m_pGCC1D;
		m_pGCC1D = 0L;
	}
}

void CFindDefocus1D::Setup(MD::CCtfParam* pCtfParam, int iCmpSize)
{
	this->Clean();
	//-----------------
	m_pCtfParam = pCtfParam;
	m_iCmpSize = iCmpSize;
	//-----------------
	m_aGCalcCtf1D.SetParam(m_pCtfParam);
	cudaMalloc(&m_gfCtf1D, sizeof(float) * m_iCmpSize);
	//-----------------
	m_pGCC1D = new GCC1D;
	m_pGCC1D->SetSize(m_iCmpSize);	
}

void CFindDefocus1D::SetResRange(float afRange[2])
{
	m_afResRange[0] = afRange[0];
	m_afResRange[1] = afRange[1];
}

void CFindDefocus1D::DoIt
(	float afDfRange[2],
	float afPhaseRange[2],
	float* gfRadialAvg
)
{	memcpy(m_afDfRange, afDfRange, sizeof(float) * 2);
	memcpy(m_afPhaseRange, afPhaseRange, sizeof(float) * 2);
	m_gfRadialAvg = gfRadialAvg;
	//--------------------------
	m_fMaxCC = (float)-1e20;
	float afResult[3] = {0.0f};
	mBrutalForceSearch(afResult);
	m_fBestDf = afResult[0];
	m_fBestPhase = afResult[1];
	m_fMaxCC = afResult[2];
}

//--------------------------------------------------------------------
// Search both defocus and phase shift.
//--------------------------------------------------------------------
void CFindDefocus1D::mBrutalForceSearch(float afResult[3])
{	
	float fTiny = (float)1e-30;
	int iDfSteps = 501;
	float fDfRange = m_afDfRange[1] - m_afDfRange[0];
	float fDfStep = fDfRange / (iDfSteps - 1);
	if(fDfStep < 50) fDfStep = 50.0f;
	iDfSteps = (int)(fDfRange / (fDfStep + fTiny)) / 2 * 2 + 1;
	//---------------------------
	int iPsSteps = 37;
	float fPsStep = m_afPhaseRange[1] / (iPsSteps - 1);
	iPsSteps = (int)(m_afPhaseRange[1] / (fPsStep + fTiny)) / 2 * 2 + 1;
	//-----------------
	int iPoints = iDfSteps * iPsSteps;
	float* pfCCs = new float[iPoints];
	//-----------------
	float fDefocus, fPhase;
	int iFocus = 0, iPhase = 0;
	afResult[2] = (float)-1e20;
	//-----------------
	for(int i=0; i<iPoints; i++)
	{	iFocus = i % iDfSteps;
		iPhase = i / iDfSteps;
		fDefocus = m_afDfRange[0] + iFocus * fDfStep;
		fPhase = m_afPhaseRange[0] + (iPhase - iPsSteps / 2) * fPsStep;
		if(fPhase < 0) fPhase = 0.0f;
		else if(fPhase > 150.0f) fPhase = 150.0f;
		//----------------
		mCalcCTF(fDefocus, fPhase);
		pfCCs[i] = mCorrelate();
		if(pfCCs[i] > afResult[2])
		{	afResult[0] = fDefocus;
			afResult[1] = fPhase;
			afResult[2] = pfCCs[i];
		}
		/*				
		printf("%3d  %8.2f  %8.2f  %8.4f  %8.2f %8.2f  %8.4f\n", i,
		   fDefocus, fPhase, pfCCs[i],
		   afResult[0], afResult[1], afResult[2]);
		*/
	}
	if(pfCCs != 0L) delete[] pfCCs;
}

void CFindDefocus1D::mCalcCTF(float fDefocus, float fExtPhase)
{
	fExtPhase *= s_fD2R;
	float fPixDefocus = fDefocus / m_pCtfParam->m_fPixelSize;
	m_aGCalcCtf1D.DoIt(fPixDefocus, fExtPhase, m_gfCtf1D, m_iCmpSize);
}

float CFindDefocus1D::mCorrelate(void)
{
	float fRes1 = ((m_iCmpSize - 1) * 2) * m_pCtfParam->m_fPixelSize;
	float fMinFreq = fRes1 / m_afResRange[0];
	float fMaxFreq = fRes1 / m_afResRange[1];
	//---------------------------------------
	m_pGCC1D->Setup(fMinFreq, fMaxFreq, 0.0f);
	float fCC = m_pGCC1D->DoIt(m_gfCtf1D, m_gfRadialAvg);
	return fCC;
}
