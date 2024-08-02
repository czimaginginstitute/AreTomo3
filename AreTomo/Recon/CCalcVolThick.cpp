#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Recon;

CCalcVolThick::CCalcVolThick(void)
{
	m_pVolSeries = 0L;
	m_gLocalCC2D = 0L;
	m_gfImg1 = 0L;
	m_gfImg2 = 0L;
}

CCalcVolThick::~CCalcVolThick(void)
{
	mClean();
}

void CCalcVolThick::DoIt
(	MD::CTiltSeries* pTiltSeries,
	MAM::CAlignParam* pAlnParam
)
{	mClean();
	//-----------------
	CDoWbpRecon* pDoWbpRecon = new CDoWbpRecon;
	int iVolZ = pTiltSeries->m_aiStkSize[0] / 2;
	MD::CTiltSeries* pVolSeries = pDoWbpRecon->DoIt(pTiltSeries,
	   pAlnParam, iVolZ);
	if(pDoWbpRecon != 0L) delete pDoWbpRecon;
	//-----------------
	m_pVolSeries = pVolSeries->FlipVol(true);
	if(pVolSeries != 0L) delete pVolSeries;
	//-----------------
	mSetup();
	int aiStart[2] = {0};
	aiStart[0] = (m_pVolSeries->m_aiStkSize[0] - m_aiTileSize[0]) / 2;
	aiStart[1] = (m_pVolSeries->m_aiStkSize[1] - m_aiTileSize[1]) / 2;
	//-----------------
	int iEndZ = m_pVolSeries->m_aiStkSize[2] - 1;
	float* pfCCs = new float[iEndZ];
	//-----------------
	for(int z=0; z<iEndZ; z++)
	{	pfCCs[z] = mMeasure(z, aiStart);
	}
}

float CCalcVolThick::mMeasure(int iZ, int* piStart)
{
	int iPixels = m_pVolSeries->GetPixels();
	size_t tBytes = iPixels * sizeof(float);
	float* pfImg1 = (float*)m_pVolSeries->GetFrame(iZ);
	float* pfImg2 = (float*)m_pVolSeries->GetFrame(iZ+1);
	//-----------------
	if(iZ == 0) 
	{	cudaMemcpy(m_gfImg1, pfImg1, tBytes, cudaMemcpyDefault);
		cudaMemcpy(m_gfImg2, pfImg2, tBytes, cudaMemcpyDefault);
		float fCC = m_gLocalCC2D->DoIt(m_gfImg1, m_gfImg2, piStart);
		return fCC;
	}
	//-----------------
	if((iZ % 2) == 0)
	{	cudaMemcpy(m_gfImg2, pfImg2, tBytes, cudaMemcpyDefault);
	}
	else cudaMemcpy(m_gfImg1, pfImg2, tBytes, cudaMemcpyDefault);
	//-----------------
	float fCC = m_gLocalCC2D->DoIt(m_gfImg1, m_gfImg2, piStart);
	return fCC;
}

void CCalcVolThick::mSetup(void)
{
	m_aiTileSize[0] = (m_pVolSeries->m_aiStkSize[0] * 9 / 10) / 5 * 4;
	m_aiTileSize[1] = (m_pVolSeries->m_aiStkSize[1] * 9 / 10) / 5 * 4;
	//-----------------`
	int iPixels = m_pVolSeries->GetPixels();
	size_t tBytes = sizeof(float) * iPixels * 2;
	cudaMalloc(&m_gfImg1, tBytes);
	m_gfImg2 = m_gfImg1 + iPixels;
	//-----------------
	m_gLocalCC2D = new MAU::GLocalCC2D;
	m_gLocalCC2D->SetSizes(m_pVolSeries->m_aiStkSize, m_aiTileSize);
}

void CCalcVolThick::mClean(void)
{
	if(m_pVolSeries != 0L) delete m_pVolSeries;
	if(m_gLocalCC2D != 0L) delete m_gLocalCC2D;
	if(m_gfImg1 != 0L) cudaFree(m_gfImg1);
	m_pVolSeries = 0L;
	m_gLocalCC2D = 0L;
	m_gfImg1 = 0L;
	m_gfImg2 = 0L;
}
