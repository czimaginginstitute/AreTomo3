#include "CReconInc.h"
#include "../Correct/CCorrectInc.h"
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
	m_fBinning = 10.0f;
	m_fPixSize = 1.0f;
}

CCalcVolThick::~CCalcVolThick(void)
{
	mClean();
}

float CCalcVolThick::GetThickness(bool bAngstrom)
{
	float fThick = m_aiSampleEdges[1] - m_aiSampleEdges[0];
	if(bAngstrom) fThick *= m_fPixSize;
	return fThick;
}

float CCalcVolThick::GetLowEdge(bool bAngstrom)
{
	float fScale = bAngstrom ? m_fPixSize : 1.0f;
	float fEdge = m_aiSampleEdges[0] * fScale;
	return fEdge;
}

float CCalcVolThick::GetHighEdge(bool bAngstrom)
{
	float fScale = bAngstrom ? m_fPixSize : 1.0f;
	float fEdge = m_aiSampleEdges[1] * fScale;
	return fEdge;
}

void CCalcVolThick::DoIt(int iNthGpu)
{	
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(iNthGpu);
	float fTiltAxis = pAlnParam->GetTiltAxis(0);
	//-------------------------------------------------
	// 1) align tilt series 0 and then 2) reconstruct.
	//-------------------------------------------------
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
	m_fPixSize = pAlnSeries->m_fPixSize / m_fBinning;
	//--------------------------------------------------
	// 3) reconstruct the aligned tilt series by WBP.
	//--------------------------------------------------
	/*
	CDoWbpRecon* pDoWbpRecon = new CDoWbpRecon;
	int iVolZ = pAlnSeries->m_aiStkSize[0] / 2;
	MD::CTiltSeries* pVolSeries = pDoWbpRecon->DoIt(pAlnSeries,
	   pAlnParam, iVolZ);
	if(pAlnSeries != 0L) delete pAlnSeries;
	if(pDoWbpRecon != 0L) delete pDoWbpRecon;
	*/
	CDoSartRecon* pDoSartRecon = new CDoSartRecon;
	int iVolZ = pAlnSeries->m_aiStkSize[0] * 3 / 8 * 2;
	int iNumTilts = pAlnSeries->m_aiStkSize[2];
	int iNumSubsets = iNumTilts / 5;
	if(iNumSubsets == 0) iNumSubsets = 1;
	int iIters = 20;
	MD::CTiltSeries* pVolSeries = pDoSartRecon->DoIt(pAlnSeries,
	   pAlnParam, 0, iNumTilts, iVolZ, iIters, iNumSubsets);
	if(pAlnSeries != 0L) delete pAlnSeries;
	if(pDoSartRecon != 0L) delete pDoSartRecon;	
	//--------------------------------------------------
	// 4) flip the volume to xyz view.
	//--------------------------------------------------
	m_pVolSeries = pVolSeries->FlipVol(true);
	if(pVolSeries != 0L) delete pVolSeries;

	MU::CSaveTempMrc saveMrc;
	saveMrc.SetFile("/home/shawn.zheng/szheng/Temp/TestVolThick", ".mrc");
	saveMrc.DoMany(m_pVolSeries->GetFrames(), 2, m_pVolSeries->m_aiStkSize);
	//--------------------------------------------------
	// 5) meaure sample thickness inside the volume.
	//--------------------------------------------------
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
		//printf(" %4d  %.4f\n", z, pfCCs[z]);
	}
	//-----------------
	mDetectEdges(pfCCs, iEndZ);
	if(pfCCs != 0L) delete[] pfCCs;
	//-----------------
	int iThick = (int)this->GetThickness(false);
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(iNthGpu);
	pPackage->m_iThickness = iThick;
	//-----------------
	mClean();
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
	m_aiTileSize[0] = m_pVolSeries->m_aiStkSize[0] * 3 / 8 * 2;
	m_aiTileSize[1] = m_pVolSeries->m_aiStkSize[1] * 3 / 8 * 2;
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

void CCalcVolThick::mDetectEdges(float* pfCCs, int iSize)
{
	//-----------------------------------------------
	// 1) local min CCs from left and right sides
	// respectively.
	//-----------------------------------------------
	int iHalfZ = iSize / 2;
	float afMinCCs[] = {100.0f, 100.0f};
	int aiMinLocs[] = {-1, -1};
	for(int i=0; i<iHalfZ; i++)
	{	if(pfCCs[i] < afMinCCs[0])
		{	afMinCCs[0] = pfCCs[i];
			aiMinLocs[0] = i;
		}
		//----------------
		int j = iSize - 1 - i;
		if(pfCCs[j] < afMinCCs[1])
		{	afMinCCs[1] = pfCCs[j];
			aiMinLocs[1] = j;
		}
	}
	//-----------------------------------------------
	// 1) search the location of the maximum CC
	//-----------------------------------------------
	float fMaxCC = -1000.0f;
	int iMaxCC = -1;
	for(int i=aiMinLocs[0]; i<aiMinLocs[1]; i++)
	{	if(pfCCs[i] > fMaxCC)
		{	fMaxCC = pfCCs[i];
			iMaxCC = i;
		}
	}
	//-----------------------------------------------
	// 1) seach the location of the second maximum
	// CC in another half of the volume.
	//-----------------------------------------------
	int iStart, iEnd;
	if(iMaxCC < iHalfZ)
	{	iStart = iHalfZ; 
		iEnd = aiMinLocs[1];
	}
	else
	{	iStart = aiMinLocs[0];
		iEnd = iHalfZ;
	}
	float fMaxCC2 = -1000.0;
	int iMaxCC2 = -1;
	for(int i=iStart; i<iEnd; i++)
	{	if(pfCCs[i] > fMaxCC2)
		{	fMaxCC2 = pfCCs[i];
			iMaxCC2 = i;
		}
	}
	//-----------------------------------------------
	// 1) find which of fMaxCC fMaxCC2 is at left
	// and which at right
	//-----------------------------------------------
	int aiMaxLocs[] = {0, 0};
	if(iMaxCC < iMaxCC2)
	{	aiMaxLocs[0] = iMaxCC;
		aiMaxLocs[1] = iMaxCC2;
	}
	else
	{	aiMaxLocs[0] = iMaxCC2;
		aiMaxLocs[1] = iMaxCC;
	}
	//-----------------------------------------------
	// 1) Determine the true minimums that are free
	// from SART artifact.
	//-----------------------------------------------
	int iPoints = (aiMaxLocs[0] - aiMinLocs[0]) / 5;
	if(iPoints < 5) iPoints = 5;
	float fMeanCC1 = 0.0f;
	for(int i=1; i<=iPoints; i++)
	{	int j = aiMinLocs[0] + i;
		fMeanCC1 += pfCCs[j];
	}
	fMeanCC1 /= iPoints;
	//-----------------
	iPoints = (aiMinLocs[1] - aiMaxLocs[1]) / 5;
	if(iPoints < 5) iPoints = 5;
	float fMeanCC2 = 0.0f;
	for(int i=1; i<=iPoints; i++)
	{	int j = aiMinLocs[1] - i;
		fMeanCC2 += pfCCs[j];
	}
	fMeanCC2 /= iPoints;
	//-----------------------------------------------
	// 1) The sample edges are in the middle between
	// true minimum and maximum
	//-----------------------------------------------
	float fEdgeCC1 = (pfCCs[aiMaxLocs[0]] + fMeanCC1) * 0.5f;
	float fEdgeCC2 = (pfCCs[aiMaxLocs[1]] + fMeanCC2) * 0.5f;
	float fMinDif = (float)1e20;
	for(int i=aiMaxLocs[0]; i>aiMinLocs[0]; i--)
	{	float fDif = fabs(pfCCs[i] - fEdgeCC1);
		if(fDif < fMinDif)
		{	fMinDif = fDif;
			m_aiSampleEdges[0] = i;
		}
	}
	//-----------------
	fMinDif = (float)1e20;
	for(int i=aiMaxLocs[1]; i<aiMinLocs[1]; i++)
	{	float fDif = fabs(pfCCs[i] - fEdgeCC2);
		if(fDif < fMinDif)
		{	fMinDif = fDif;
			m_aiSampleEdges[1] = i;
		}
	}
	m_aiSampleEdges[0] *= m_fBinning;
	m_aiSampleEdges[1] *= m_fBinning;
	//-----------------
	printf("Sample edges: %6d  %6d\n", m_aiSampleEdges[0], m_aiSampleEdges[1]);
}
