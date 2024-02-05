#include "CCommonLineInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;

void mFindMMM(float* gfPadLine, int iPadSize)
{
	int iSize = (iPadSize / 2 - 1) * 2;
	float* pfLine = new float[iSize];
	cudaMemcpy(pfLine, gfPadLine, iSize * sizeof(float), cudaMemcpyDefault);
	float fMin = pfLine[0];
	float fMax = pfLine[0];
	float fMean = 0;
	for(int i=0; i<iSize; i++)
	{	if(pfLine[i] < fMin) fMin = pfLine[i];
		else if(pfLine[i] > fMax) fMax = pfLine[i];
		if(pfLine[i] > 0) fMean += (pfLine[i] / iSize);
	}
	delete[] pfLine;
}

void mFindImageMMM(float* gfImg, int* piImgSize)
{
	int iPixels = piImgSize[0] * piImgSize[1];
	size_t tBytes = sizeof(float) * iPixels;
	float* pfImg = new float[iPixels];
	cudaMemcpy(pfImg, gfImg, tBytes, cudaMemcpyDefault);
	//--------------------------------------------------
	float fMin = pfImg[0];
	float fMax = pfImg[1];
	float fMean = 0.0f;
	for(int i=0; i<iPixels; i++)
	{	if(pfImg[i] < fMin) fMin = pfImg[i];
		else if(pfImg[i] > fMax) fMax = pfImg[i];
		if(pfImg[i] > 0) fMean += (pfImg[i] / iPixels);
	}
	delete[] pfImg;
}

CGenLines::CGenLines(void)
{
	m_gCmpPlane = 0L;
}

CGenLines::~CGenLines(void)
{
}

void CGenLines::mClean(void)
{
	if(m_gCmpPlane != 0L) cudaFree(m_gCmpPlane);
	m_gCmpPlane = 0L;
	//---------------
	m_calcComRegion.Clean();
	m_genComLine.Clean();
	m_fft1D.DestroyPlan();
}

CPossibleLines* CGenLines::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance(iNthGpu);
	m_iNumLines = pClParam->m_iNumLines;
	m_iLineSize = pClParam->m_iLineSize;
	//-----------------
	m_pPossibleLines = new CPossibleLines;
	m_pPossibleLines->Setup(m_iNthGpu);
	//-----------------
	m_calcComRegion.DoIt(iNthGpu);
	mGenLines();
	this->mClean();
	//-----------------
	CPossibleLines* pPossibleLines = m_pPossibleLines;
	m_pPossibleLines = 0L;
	return pPossibleLines;
}

void CGenLines::mGenLines(void)
{
	if(m_gCmpPlane != 0L) cudaFree(m_gCmpPlane);
	int iCmpLineSize = m_iLineSize / 2 + 1;
	size_t tBytes = sizeof(cufftComplex) * m_iNumLines * iCmpLineSize;
	cudaMalloc(&m_gCmpPlane, tBytes);
	//-------------------------------
	bool bForward = true;
	m_fft1D.CreatePlan(m_iLineSize, m_iNumLines, bForward);
	//-----------------------------------------------------
	bool bPadded = true;
	m_genComLine.Setup(m_iNthGpu);
	//-------------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	mGenProjLines(i);
	}
}

void CGenLines::mGenProjLines(int iProj)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	float* pfProj = (float*)pTiltSeries->GetFrame(iProj);
	float fTiltAngle = pAlnParam->GetTilt(iProj);
	float afShift[2] = {0.0f};
	pAlnParam->GetShift(iProj, afShift);
	//-----------------
	int* giComRegion = m_calcComRegion.m_giComRegion;
	m_genComLine.DoIt(iProj, giComRegion, (float*)m_gCmpPlane);
	//-----------------
	mForwardFFT();
	m_pPossibleLines->SetPlane(iProj, m_gCmpPlane);
}

void CGenLines::mForwardFFT(void)
{
	int iPadSize = (m_iLineSize / 2 + 1) * 2;
	float* gfPadLines = (float*)m_gCmpPlane;
	GRemoveMean removeMean;
	for(int i=0; i<m_iNumLines; i++)
	{	float* gfPadLine = gfPadLines + i * iPadSize;
		removeMean.DoIt(gfPadLine, iPadSize);
	}
	//-----------------
	bool bNorm = true;
	m_fft1D.Forward(gfPadLines, !bNorm);
}
