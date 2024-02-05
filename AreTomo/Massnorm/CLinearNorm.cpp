#include "CMassNormInc.h"
#include "../CAreTomoInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::MassNorm;

CLinearNorm::CLinearNorm(void)
{
	m_fMissingVal = (float)-1e20;
}

CLinearNorm::~CLinearNorm(void)
{
}

void CLinearNorm::DoIt(int iNthGpu)
{	
	printf("GPU %d: linear mass normalization\n", iNthGpu);
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(iNthGpu);
	//-----------------
	m_aiStart[0] = pTiltSeries->m_aiStkSize[0] * 1 / 6;
	m_aiStart[1] = pTiltSeries->m_aiStkSize[1] * 1 / 6;
	m_aiSize[0] = pTiltSeries->m_aiStkSize[0] * 4 / 6;
	m_aiSize[1] = pTiltSeries->m_aiStkSize[1] * 4 / 6;
	m_iNumFrames = pTiltSeries->m_aiStkSize[2];
	//-----------------
	size_t tBytes = sizeof(float) * m_aiSize[0] * m_aiSize[1];
	cudaMalloc(&m_gfSubImg, tBytes);
	m_pfMeans = new float[m_iNumFrames];
	//-----------------
	m_iZeroTilt = pAlnParam->GetFrameIdxFromTilt(0.0f);
	int iNumSeries = MD::CAlnSums::m_iNumSums;
	for(int i=0; i<iNumSeries; i++)
	{	mCalcMeans(i);
		mScale(i);
	}
	//-----------------
	if(m_pfMeans != 0L) delete[] m_pfMeans;
	if(m_gfSubImg != 0L) cudaFree(m_gfSubImg);
	m_pfMeans = 0L;
	m_gfSubImg = 0L;
	//-----------------
	printf("GPU %d: linear mass normalization done.\n\n");
}

void CLinearNorm::mSmooth(float* pfMeans)
{
	double dMean = 0, dStd = 0;
	for(int i=0; i<m_iNumFrames; i++)
	{	dMean += pfMeans[i];
		dStd += (pfMeans[i] * pfMeans[i]);
	}
	dMean /= m_iNumFrames;
	dStd = dStd / m_iNumFrames - dMean * dMean;
	if(dStd <= 0) return;
	else dStd = sqrtf(dStd);
	//----------------------
	float fMin = (float)(dMean - 3 * dStd);	
	if(fMin <= 0) return;
	//-------------------
	for(int i=0; i<m_iNumFrames; i++)
	{	if(pfMeans[i] > fMin) continue;
		else pfMeans[i] = (float)dMean;
	}
}

void CLinearNorm::FlipInt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	m_iNumFrames = pTiltSeries->m_aiStkSize[2];
	for(int i=0; i<m_iNumFrames; i++)
	{	mFlipInt(i);
	}
}
/*
float CLinearNorm::mCalcMean(int iFrame)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
        MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	double dMean = 0.0;
	int iCount = 0;
	float* pfFrame = (float*)pTiltSeries->GetFrame(iFrame);
	int iOffset = m_aiStart[1] * pTiltSeries->m_aiStkSize[0]
		+ m_aiStart[0];
	for(int y=0; y<m_aiSize[1]; y++)
	{	int i = y * pTiltSeries->m_aiStkSize[0] + iOffset;
		for(int x=0; x<m_aiSize[0]; x++)
		{	float fVal = pfFrame[i+x];
			if(fVal <= m_fMissingVal) continue;
			dMean += fVal;
			iCount++;
		}
	}
	if(iCount == 0) return 0.0f;
	dMean = dMean / iCount;
	return (float)dMean;
}
*/

void CLinearNorm::mCalcMeans(int iSeries)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(iSeries);
	//-----------------
	bool bPadded = true;
	MU::GCalcMoment2D calcMoment;
	calcMoment.SetSize(m_aiSize, !bPadded);
	//-----------------
	for(int i=0; i<m_iNumFrames; i++)
	{	float* pfImg = (float*)pTiltSeries->GetFrame(i);
		mExtractSubImg(pfImg, pTiltSeries->m_aiStkSize);
		m_pfMeans[i] = calcMoment.DoIt(m_gfSubImg, 1, true);
	}
}

void CLinearNorm::mExtractSubImg(float* pfImg, int* piImgSize)
{
	float* pfSrc = pfImg + m_aiStart[1] * piImgSize[0] + m_aiStart[0];
        int iBytes = sizeof(float) * m_aiSize[0];
        for(int y=0; y<m_aiSize[1]; y++)
        {       float* gfDst = m_gfSubImg + y * m_aiSize[0];
                cudaMemcpy(gfDst, pfSrc, iBytes, cudaMemcpyDefault);
        }
}

void CLinearNorm::mScale(int iSeries)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(iSeries);
	//-----------------
	float fRefMean = m_pfMeans[m_iZeroTilt];
        if(fRefMean > 1000.0f) fRefMean = 1000.0f;
	//-----------------
	int iPixels = pTiltSeries->GetPixels();
        for(int i=0; i<m_iNumFrames; i++)
        {       float fScale = fRefMean / (m_pfMeans[i] + 0.00001f);
		float* pfImg = (float*)pTiltSeries->GetFrame(i);
		for(int j=0; j<iPixels; j++)
		{	if(pfImg[j] <= m_fMissingVal) continue;
			pfImg[j] *= fScale;
		}
	}
}

void CLinearNorm::mFlipInt(int iFrame)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
        MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	float* pfFrame = (float*)pTiltSeries->GetFrame(iFrame);
	int iPixels = pTiltSeries->m_aiStkSize[0] *
	   pTiltSeries->m_aiStkSize[1];
	float fMin = (float)1e20;
	float fMax = (float)-1e20;
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= m_fMissingVal) continue;
		if(pfFrame[i] < fMin) fMin = pfFrame[i];
		else if(pfFrame[i] > fMax) fMax = pfFrame[i];
		pfFrame[i] *= (-1);
	}
	//-------------------------
	float fOffset = fMax + fMin;
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= m_fMissingVal) continue;
		pfFrame[i] += fOffset;
	}
}
