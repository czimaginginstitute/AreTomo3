#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CRemoveDarkFrames::CRemoveDarkFrames(void)
{
	m_pfMeans = 0L;
	m_pfStds = 0L;
}

CRemoveDarkFrames::~CRemoveDarkFrames(void)
{
	if(m_pfMeans != 0L) delete[] m_pfMeans;
}

void CRemoveDarkFrames::Setup(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPackage->GetSeries(0);
	m_iAllFrms = pTiltSeries->m_aiStkSize[2];
	//-----------------
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance(m_iNthGpu);
	pDarkFrames->Setup(pTiltSeries);
}

void CRemoveDarkFrames::Detect(float fThreshold)
{
	m_fThreshold = fThreshold;
	mCalcStats();
	mDetect();
}

void CRemoveDarkFrames::Remove(void)
{
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance(m_iNthGpu);
	int iNumSeries = MD::CAlnSums::m_iNumSums;
	//-----------------
	for(int i=0; i<iNumSeries; i++)
	{	mRemoveSeries(i);
	}
	//-----------------
	/*
	for(int i=pDarkFrames->m_iNumDarks-1; i>=0; i--)
	{	int iFrmIdx = pDarkFrames->GetDarkIdx(i);
		pAlnParam->RemoveFrame(iFrmIdx);
	}
	*/
}

void CRemoveDarkFrames::mDetect(void)
{
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPackage->GetSeries(0);
	//-----------------
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance(m_iNthGpu);
	pDarkFrames->Setup(pTiltSeries);
	//-----------------
	int iSize = (m_iAllFrms + 16) * 64;
	char* pcLog = new char[iSize];
	memset(pcLog, 0, sizeof(char) * iSize);
	//-----------------
	sprintf(pcLog, "GPU %d: Detect dark images in the tilt "
	   "series.\n", m_iNthGpu);
	strcat(pcLog, "# index  tilt    mean         std      ratio\n");
	char acBuf[64] = {'\0'};
	//-----------------
	for(int i=0; i<m_iAllFrms; i++)
	{	float fTilt = pTiltSeries->m_pfTilts[i];
		float fMean = (float)fabs(m_pfMeans[i]);
		float fRatio = fMean / (m_pfStds[i] + 0.000001f);
		//----------------
		sprintf(acBuf, " %3d  %8.2f  %8.2f  %8.2f  %8.2f\n", 
		   i, fTilt, m_pfMeans[i], m_pfStds[i], fRatio);
		strcat(pcLog, acBuf);
	}
	printf("%s\n", pcLog);
	if(pcLog != 0L) delete[] pcLog;
	//-----------------
	int iZeroTilt = pTiltSeries->GetTiltIdx(0.0f);
	float fTol = m_fThreshold * (float)fabs(m_pfMeans[iZeroTilt])
	   / (m_pfStds[iZeroTilt] + 0.000001f);
	//-----------------
	for(int i=0; i<m_iAllFrms; i++)
	{	float fRatio = (float)fabs(m_pfMeans[i]) / 
		   (m_pfStds[i] + 0.000001f);
		//----------------
		if(fRatio > fTol) continue;
		else pDarkFrames->AddDark(i);
	}

	/*   Debugging Code
	pDarkFrames->AddDark(0);
	pDarkFrames->AddDark(1);
	*/

	if(pDarkFrames->m_iNumDarks <= 0) 
	{	printf("GPU %d: no dark images detected.\n\n", m_iNthGpu);
	}
}

void CRemoveDarkFrames::mRemoveSeries(int iSeries)
{
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPackage->GetSeries(iSeries);
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance(m_iNthGpu);
	//-----------------
	int iSize = (pDarkFrames->m_iNumDarks + 16) * 64;
	char* pcLog = new char[iSize];
	memset(pcLog, 0, sizeof(char) * iSize);
	sprintf(pcLog, "GPU %d: removed dark frames of series %d\n", 
	   m_iNthGpu, iSeries);
	//-----------------
	char acBuf[64] = {'\0'};
	for(int i=pDarkFrames->m_iNumDarks-1; i>=0; i--)
	{	int iDarkIdx = pDarkFrames->GetDarkIdx(i);
		float fTilt = pTiltSeries->m_pfTilts[iDarkIdx];
		pTiltSeries->RemoveFrame(iDarkIdx);
		sprintf(acBuf, "Remove image at %.2f deg \n", fTilt);
		strcat(pcLog, acBuf);
	}
	//-----------------
	printf("%s\n", pcLog);
	if(pcLog != 0L) delete[] pcLog;
}

void CRemoveDarkFrames::mCalcStats(void)
{
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPackage->GetSeries(0);
	//-----------------
	if(m_pfMeans != 0L) delete[] m_pfMeans;
	m_pfMeans = new float[m_iAllFrms * 2];
	m_pfStds = &m_pfMeans[m_iAllFrms];
	//-----------------
	int iPixels = pTiltSeries->GetPixels();
	size_t tBytes = sizeof(float) * iPixels;
	float *gfImg = 0L, *gfBuf = 0L;
	cudaMalloc(&gfImg, tBytes);
	cudaMalloc(&gfBuf, tBytes);
	//-----------------
	MU::GCalcMoment2D calcMoment2D;
	bool bPadded = true;
	calcMoment2D.SetSize(pTiltSeries->m_aiStkSize, !bPadded);
	//-----------------
	float afMeanStd[2] = {0.0f};
	for(int i=0; i<m_iAllFrms; i++)
	{	float* pfFrame = (float*)pTiltSeries->GetFrame(i);
		cudaMemcpy(gfImg, pfFrame, tBytes, cudaMemcpyDefault);
		m_pfMeans[i] = calcMoment2D.DoIt(gfImg, 1, true);
		m_pfStds[i] = calcMoment2D.DoIt(gfImg, 2, true)
		   - m_pfMeans[i] * m_pfMeans[i];
		if(m_pfStds[i] <= 0) m_pfStds[i] = 0.0f;
		else m_pfStds[i] = (float)sqrtf(m_pfStds[i]);
	}
	if(gfImg != 0L) cudaFree(gfImg);
	if(gfBuf != 0L) cudaFree(gfBuf);
}
