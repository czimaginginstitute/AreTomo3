#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CRemoveDarkFrames::CRemoveDarkFrames(void)
{
}

CRemoveDarkFrames::~CRemoveDarkFrames(void)
{
}

void CRemoveDarkFrames::DoIt(int iNthGpu, float fThreshold)
{
	m_iNthGpu = iNthGpu;
	m_fThreshold = fThreshold;
	//-----------------
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPackage->GetSeries(0);
	m_iAllFrms = pTiltSeries->m_aiStkSize[2];
	//-----------------
	float* pfMeans = new float[m_iAllFrms];
	float* pfStds = new float[m_iAllFrms];
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
		pfMeans[i] = calcMoment2D.DoIt(gfImg, 1, true);
		pfStds[i] = calcMoment2D.DoIt(gfImg, 2, true)
		   - pfMeans[i] * pfMeans[i];
		if(pfStds[i] <= 0) pfStds[i] = 0.0f;
		else pfStds[i] = (float)sqrtf(pfStds[i]);
	}
	if(gfImg != 0L) cudaFree(gfImg);
	if(gfBuf != 0L) cudaFree(gfBuf);
	//-----------------
	mRemove(pfMeans, pfStds);
	delete[] pfMeans;
	delete[] pfStds;
}

void CRemoveDarkFrames::mRemove(float* pfMeans, float* pfStds)
{
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPackage->GetSeries(0);
	CAlignParam* pAlnParam = CAlignParam::GetInstance(m_iNthGpu);
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
	{	float fMean = (float)fabs(pfMeans[i]);
		float fRatio = fMean / (pfStds[i] + 0.000001);
		float fTilt = pAlnParam->GetTilt(i);
		sprintf(acBuf, " %3d  %8.2f  %8.2f  %8.2f  %8.2f\n", 
		   i, fTilt, pfMeans[i], pfStds[i], fRatio);
		strcat(pcLog, acBuf);
	}
	printf("%s\n", pcLog);
	if(pcLog != 0L) delete[] pcLog;
	//-----------------
	int iZeroTilt = pAlnParam->GetFrameIdxFromTilt(0.0f);
	float fTol = m_fThreshold * (float)fabs(pfMeans[iZeroTilt])
	   / (pfStds[iZeroTilt] + 0.000001f);
	//-----------------
	for(int i=0; i<m_iAllFrms; i++)
	{	float fRatio = (float)fabs(pfMeans[i]) / (pfStds[i] + 0.000001f);
		if(fRatio > fTol) continue;
		else pDarkFrames->AddDark(i);
	}
	if(pDarkFrames->m_iNumDarks <= 0) 
	{	printf("GPU %d: no dark images detected.\n\n", m_iNthGpu);
		return;
	}
	//-----------------
	int iNumSeries = MD::CAlnSums::m_iNumSums;
	for(int i=0; i<iNumSeries; i++)
	{	mRemoveSeries(i);
	}
	//-----------------
	for(int i=pDarkFrames->m_iNumDarks-1; i>=0; i--)
	{	int iFrmIdx = pDarkFrames->GetDarkIdx(i);
		pAlnParam->RemoveFrame(iFrmIdx);
	}
}

void CRemoveDarkFrames::mRemoveSeries(int iSeries)
{
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPackage->GetSeries(iSeries);
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance(m_iNthGpu);
	CAlignParam* pAlnParam = CAlignParam::GetInstance(m_iNthGpu);
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
		float fTilt = pAlnParam->GetTilt(iDarkIdx);
		pTiltSeries->RemoveFrame(iDarkIdx);
		sprintf(acBuf, "Remove image at %.2f deg \n", fTilt);
		strcat(pcLog, acBuf);
	}
	//-----------------
	printf("%s\n", pcLog);
	if(pcLog != 0L) delete[] pcLog;
}
