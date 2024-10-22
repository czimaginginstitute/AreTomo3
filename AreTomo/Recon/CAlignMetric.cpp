#include "CReconInc.h"
#include "../Correct/CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Recon;

CAlignMetric::CAlignMetric(void)
{
	m_pVolSeries = 0L;
	m_gfImg1 = 0L;
	m_gfImg2 = 0L;
	m_fBinning = 8.0f;
	m_fPixSize = 1.0f;
}

CAlignMetric::~CAlignMetric(void)
{
	mClean();
}

void CAlignMetric::Calculate(int iNthGpu, int iThickness)
{	
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(iNthGpu);
	float fTiltAxis = pAlnParam->GetTiltAxis(0);
	//------------------------
	// 1) align tilt series 0
	//------------------------
	MAC::CCorrTomoStack* pCorrTomoStack = 
	   new MAC::CCorrTomoStack;
	bool bFFTCrop = true, bRandFill = true, bForRecon = true;
       	bool bShiftOnly = true, bIntCor = true, bRWeight = true;
	pCorrTomoStack->Set0(iNthGpu);
	pCorrTomoStack->Set1(0, fTiltAxis);
        pCorrTomoStack->Set2(m_fBinning, bFFTCrop, bRandFill);
        pCorrTomoStack->Set3(!bShiftOnly, !bIntCor, !bRWeight);
        pCorrTomoStack->Set4(bForRecon);
	pCorrTomoStack->DoIt(0, 0L);
	MD::CTiltSeries* pAlnSeries = pCorrTomoStack->GetCorrectedStack(true);
	if(pCorrTomoStack != 0L) delete pCorrTomoStack;
	//-----------------
	mSetup(pAlnSeries->m_aiStkSize);
	m_fPixSize = pAlnSeries->m_fPixSize / m_fBinning;
	//------------------------------------
	// 2) move the zero-tilt image to GPU
	//------------------------------------
	int iZeroTilt = pAlnParam->GetFrameIdxFromTilt(0.0f);
	float* pfImg = (float*)pAlnSeries->GetFrame(iZeroTilt);
	size_t tBytes = pAlnSeries->GetPixels() * sizeof(float);
	cudaMemcpy(m_gfImg1, pfImg, tBytes, cudaMemcpyDefault);
	//------------------------------------------------
	// 3) reconstruct the aligned tilt series by SART
	//------------------------------------------------
	CDoSartRecon* pDoSartRecon = new CDoSartRecon;
	int iVolZ = (int)(iThickness / m_fBinning) / 2 * 2;
	int iNumTilts = pAlnSeries->m_aiStkSize[2];
	int iNumSubsets = iNumTilts / 5;
	int iIters = 20;
	MD::CTiltSeries* pVolSeries = pDoSartRecon->DoIt(pAlnSeries,
	   pAlnParam, 0, iNumTilts, iVolZ, iIters, iNumSubsets);
	if(pAlnSeries != 0L) delete pAlnSeries;
	if(pDoSartRecon != 0L) delete pDoSartRecon;	
	//---------------------------------
	// 4) flip the volume to xyz view.
	//---------------------------------
	m_pVolSeries = pVolSeries->FlipVol(true);
	if(pVolSeries != 0L) delete pVolSeries;
	//--------------------------------------------
	// Forward-project the volume at zero degree.
	//--------------------------------------------
	mReproj();
	//-----------------
	int aiStart[2] = {0};
	aiStart[0] = (m_pVolSeries->m_aiStkSize[0] - m_aiTileSize[0]) / 2;
	aiStart[1] = (m_pVolSeries->m_aiStkSize[1] - m_aiTileSize[1]) / 2;
	//-----------------
	//MAU::GLocalRms2D gLocalRms;
	//gLocalRms.SetSizes(m_pVolSeries->m_aiStkSize, m_aiTileSize);
	//m_fRms = gLocalRms.DoIt(m_gfImg1, m_gfImg2, aiStart);
	
	MAU::GLocalCC2D gLocalCC;
	gLocalCC.SetSizes(m_pVolSeries->m_aiStkSize, m_aiTileSize);
	m_fRms = gLocalCC.DoIt(m_gfImg1, m_gfImg2, aiStart);
	//-----------------
	mClean();
}

void CAlignMetric::mReproj(void)
{
	int iPixels = m_pVolSeries->GetPixels();
	size_t tBytes = iPixels * sizeof(float);	
	float* pfImg = (float*)m_pVolSeries->GetFrame(0);
	cudaMemcpy(m_gfImg2, pfImg, tBytes, cudaMemcpyDefault);
	//-----------------
	MU::GAddFrames gAddFrames;
	//-----------------
	for(int z=1; z<m_pVolSeries->m_aiStkSize[2]; z++)
	{	pfImg = (float*)m_pVolSeries->GetFrame(z);
		cudaMemcpy(m_gfImg3, pfImg, tBytes, cudaMemcpyDefault);
		//----------------
		gAddFrames.DoIt(m_gfImg2, 1.0f, m_gfImg3, 1.0f,
		   m_gfImg2, m_pVolSeries->m_aiStkSize);
	}

	MU::CSaveTempMrc saveMrc;
	saveMrc.SetFile("/home/shawn.zheng/szheng/Temp/TestReproj", ".mrc");
	saveMrc.GDoIt(m_gfImg2, m_pVolSeries->m_aiStkSize);
	printf("CAlignMetric: done");
}

void CAlignMetric::mSetup(int* piImgSize)
{
	m_aiTileSize[0] = piImgSize[0] * 2 / 8 * 2;
	m_aiTileSize[1] = piImgSize[1] * 2 / 8 * 2;
	//-----------------`
	int iPixels = piImgSize[0] * piImgSize[1];
	size_t tBytes = sizeof(float) * iPixels * 3;
	cudaMalloc(&m_gfImg1, tBytes);
	m_gfImg2 = m_gfImg1 + iPixels;
	m_gfImg3 = m_gfImg2 + iPixels;
}

void CAlignMetric::mClean(void)
{
	if(m_pVolSeries != 0L) delete m_pVolSeries;
	if(m_gfImg1 != 0L) cudaFree(m_gfImg1);
	m_pVolSeries = 0L;
	m_gfImg1 = 0L;
	m_gfImg2 = 0L;
	m_gfImg3 = 0L;
}

