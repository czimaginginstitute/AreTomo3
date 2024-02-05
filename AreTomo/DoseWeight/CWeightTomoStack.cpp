#include "CDoseWeightInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::DoseWeight;
namespace MD = McAreTomo::DataUtil;

CWeightTomoStack::CWeightTomoStack(void)
{
	m_pGDoseWeightImg = 0L;
	m_gCmpImg = 0L;
	m_pfDose = 0L;
}

CWeightTomoStack::~CWeightTomoStack(void)
{
	this->Clean();
}

void CWeightTomoStack::Clean(void)
{
	if(m_pGDoseWeightImg != 0L) delete m_pGDoseWeightImg;
	if(m_gCmpImg != 0L) cudaFree(m_gCmpImg);
	if(m_pfDose != 0L) delete[] m_pfDose;
	m_pGDoseWeightImg = 0L;
	m_gCmpImg = 0L;
	m_pfDose = 0L;
}

void CWeightTomoStack::DoIt(int iNthGpu)
{
	this->Clean();
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	m_pTiltSeries = pTsPkg->GetSeries(0);
	m_pfDose = m_pTiltSeries->GetAccDose();
	//-----------------
	m_aiCmpSize[0] = m_pTiltSeries->m_aiStkSize[0] / 2 + 1;
	m_aiCmpSize[1] = m_pTiltSeries->m_aiStkSize[1];
	//-----------------
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gCmpImg, sizeof(cufftComplex) * iCmpSize);
	//-----------------
	bool bPad = true;
	m_aForwardFFT.CreateForwardPlan(m_pTiltSeries->m_aiStkSize, !bPad);
	m_aInverseFFT.CreateInversePlan(m_pTiltSeries->m_aiStkSize, !bPad);	
	//-----------------
	CInput* pInput = CInput::GetInstance();
	float fKv = (float)pInput->m_iKv;
	if(m_pfDose != 0L)
	{	m_pGDoseWeightImg = new GDoseWeightImage;
		m_pGDoseWeightImg->BuildWeight(m_pTiltSeries->m_fPixSize,
		   fKv, m_pfDose, m_pTiltSeries->m_aiStkSize);
	}
	for(int i=0; i<m_pTiltSeries->m_aiStkSize[2]; i++)
	{	mCorrectProj(i);
		printf("  image %4d has been processed.\n", i);
	}
	//-----------------
	m_aForwardFFT.DestroyPlan();
	m_aInverseFFT.DestroyPlan();
	//-----------------
	this->Clean();
}

void CWeightTomoStack::mCorrectProj(int iProj)
{
	mForwardFFT(iProj);
	mDoseWeight(iProj);
	mInverseFFT(iProj);
}

void CWeightTomoStack::mForwardFFT(int iProj)
{
	float* pfProj = (float*)m_pTiltSeries->GetFrame(iProj);
	MU::CPad2D pad2D;
	pad2D.Pad(pfProj, m_pTiltSeries->m_aiStkSize, (float*)m_gCmpImg);
	//-----------------
	bool bNorm = true;
	m_aForwardFFT.Forward((float*)m_gCmpImg, bNorm);
}

void CWeightTomoStack::mDoseWeight(int iProj)
{
	if(m_pGDoseWeightImg == 0L) return;
	m_pGDoseWeightImg->DoIt(m_gCmpImg, m_pfDose[iProj]);
}

void CWeightTomoStack::mInverseFFT(int iProj)
{
	m_aInverseFFT.Inverse(m_gCmpImg);
	//-------------------------------
	MU::CPad2D aPad2D;
	float* pfProj = (float*)m_pTiltSeries->GetFrame(iProj);
	int aiPadSize[] = {2 * m_aiCmpSize[0], m_aiCmpSize[1]};
	aPad2D.Unpad((float*)m_gCmpImg, aiPadSize, pfProj);
}
	
