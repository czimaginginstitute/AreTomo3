#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CCalcStackStats::CCalcStackStats(void)
{
}

CCalcStackStats::~CCalcStackStats(void)
{
}

void CCalcStackStats::DoIt
(	MD::CTiltSeries* pTiltSeries,
	float* pfStats
)
{	int iPixels = pTiltSeries->GetPixels();
	size_t tBytes = sizeof(float) * iPixels;
	float *gfImg = 0L, *gfBuf = 0L;
	cudaMalloc(&gfImg, tBytes);
	cudaMalloc(&gfBuf, tBytes);
	//-----------------
	bool bPadded = true;
	MU::GCalcMoment2D calcMoment2D;
	calcMoment2D.SetSize(pTiltSeries->m_aiStkSize, !bPadded);
	MU::GFindMinMax2D findMinMax2D;
	findMinMax2D.SetSize(pTiltSeries->m_aiStkSize, !bPadded);
	//-----------------
	float* pfMin = pfStats;
	float* pfMax = pfStats + pTiltSeries->m_aiStkSize[2];
	float* pfMean = pfStats + pTiltSeries->m_aiStkSize[2] * 2;
	float* pfMean2 = pfStats + pTiltSeries->m_aiStkSize[2] * 3;
	//-----------------
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float* pfFrame = (float*)pTiltSeries->GetFrame(i);
		cudaMemcpy(gfImg, pfFrame, tBytes, cudaMemcpyDefault);
		//----------------
		pfMin[i] = findMinMax2D.DoMin(gfImg, true);
		pfMax[i] = findMinMax2D.DoMax(gfImg, true);
		pfMean[i] = calcMoment2D.DoIt(gfImg, 1, true);
		pfMean2[i] = calcMoment2D.DoIt(gfImg, 2, true);	
	}
	if(gfImg != 0L) cudaFree(gfImg);
	if(gfBuf != 0L) cudaFree(gfBuf);
}
