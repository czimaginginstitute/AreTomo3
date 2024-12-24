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
	m_iNthGpu = 0;
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
	m_iNthGpu = iNthGpu;
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
	mSaveTmpVol(); // for debugging 	
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
	}
	mSmooth(pfCCs, iEndZ);
	//-----------------
	mDetectEdges(pfCCs, iEndZ);
	mSaveTmpCCs(pfCCs, iEndZ); // for debugging
	if(pfCCs != 0L) delete[] pfCCs;
	//-----------------
	mClean();
}

void CCalcVolThick::mSmooth(float* pfCCs, int iSize)
{
	int iWin = 11;
	float* pfBuf = new float[iSize];
	for(int i=0; i<iSize; i++)
	{	int iStart = i - iWin / 2;
		double dSum = 0;
		for(int j=0; j<iWin; j++)
		{	int k = j + iStart;
			if(k < 0) k = 0;
			else if(k >= iSize) k = iSize -1;
			dSum += pfCCs[k];
		}
		pfBuf[i] = (float)dSum / iWin;
	}
	memcpy(pfCCs, pfBuf, sizeof(float) * iSize);
	delete[] pfBuf;
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
	m_aiTileSize[0] = (int)(m_pVolSeries->m_aiStkSize[0] * 3.5) / 8 * 2;
	m_aiTileSize[1] = (int)(m_pVolSeries->m_aiStkSize[1] * 3.5) / 8 * 2;
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
	float fMinCC0 = pfCCs[aiMinLocs[0]];
	float fMinCC1 = pfCCs[aiMinLocs[1]];
	//-----------------------------------------------
	// 1) The sample edges are in the middle between
	// true minimum and maximum
	//-----------------------------------------------
	float fW = 0.55f;
	fMaxCC = (pfCCs[aiMaxLocs[0]] + pfCCs[aiMaxLocs[1]]) * 0.5f;
	float fEdgeCC1 = fMaxCC * (1 - fW) + fMinCC0 * fW;
	float fEdgeCC2 = fMaxCC * (1 - fW) + fMinCC1 * fW;
	float fEdgeCC = (fEdgeCC1 + fEdgeCC2) * 0.5f;
	//-----------------------------------------------
	// 1) This is initialization just in case
	//-----------------------------------------------
	m_aiSampleEdges[0] = aiMinLocs[0];
	m_aiSampleEdges[1] = aiMinLocs[1];
	//-----------------
	for(int i=aiMaxLocs[0]; i>aiMinLocs[0]; i--)
	{	if(pfCCs[i] < fEdgeCC)
		{	m_aiSampleEdges[0] = i;
			break;
		}
	}
	//-----------------
	for(int i=aiMaxLocs[1]; i<aiMinLocs[1]; i++)
	{	if(pfCCs[i] < fEdgeCC)
		{	m_aiSampleEdges[1] = i;
			break;
		}
	}
	//------------------
	m_aiSampleEdges[0] *= m_fBinning;
	m_aiSampleEdges[1] *= m_fBinning;
	//-----------------
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	pAlnParam->m_iThickness = m_aiSampleEdges[1] - m_aiSampleEdges[0];
	//-----------------
	int iSampleCent = (m_aiSampleEdges[0] + m_aiSampleEdges[1]) / 2;
	int iVolCent = (int)(iHalfZ * m_fBinning);
	pAlnParam->m_iOffsetZ = iSampleCent - iVolCent;
	//-----------------
	printf("Sample edges: %6d  %6d\n\n", 
	   m_aiSampleEdges[0], m_aiSampleEdges[1]);
}

void CCalcVolThick::mSaveTmpVol(void)
{
	char* pcMrcName = mGenTmpName();
	if(pcMrcName == 0L) return;
	//-----------------
	MU::CSaveTempMrc saveMrc;
        saveMrc.SetFile(pcMrcName, ".mrc");
        saveMrc.DoMany(m_pVolSeries->GetFrames(), 2, m_pVolSeries->m_aiStkSize);
	//-----------------
	delete[] pcMrcName;
}

void CCalcVolThick::mSaveTmpCCs(float* pfCCs, int iSize)
{
	char* pcCCName = mGenTmpName();
	if(pcCCName == 0L) return;
	//-----------------
	float fBotEdge = m_aiSampleEdges[0] / m_fBinning;
        float fTopEdge = m_aiSampleEdges[1] / m_fBinning;
	//-----------------
	strcat(pcCCName, "_CC.csv");
	FILE* pFile = fopen(pcCCName, "w");
	if(pFile != 0L)
	{	for(int i=0; i<iSize; i++)
		{	fprintf(pFile, "%d,%.5f,%.1f,%.1f\n", 
			   i, pfCCs[i], fBotEdge, fTopEdge);
		}
		fclose(pFile);
	}
	//-----------------
	if(pcCCName != 0L) delete[] pcCCName;
}

char* CCalcVolThick::mGenTmpName(void)
{
	CInput* pInput = CInput::GetInstance();
	if(strlen(pInput->m_acTmpDir) == 0) return 0L;
	//-----------------
	char* pcTmpName = new char[256];
	memset(pcTmpName, 0, sizeof(char) * 256);
	strcpy(pcTmpName, pInput->m_acTmpDir);
	//-----------------
	MD::CTsPackage* pPackage = 0L; 
        pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
        strcat(pcTmpName, pPackage->m_acMrcMain);
        strcat(pcTmpName, "_Thick");
	//-----------------
	return pcTmpName;
}
