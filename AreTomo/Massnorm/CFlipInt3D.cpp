#include "CMassNormInc.h"
#include "../CAreTomoInc.h"
#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::MassNorm;

CFlipInt3D::CFlipInt3D(void)
{
}

CFlipInt3D::~CFlipInt3D(void)
{
}

void CFlipInt3D::DoIt(int iNthGpu)
{
	printf("Flip volume intensity ...\n");
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	int* piStkSize = pTiltSeries->m_aiStkSize;
	float *gfFrm = 0L;
	size_t tBytes = sizeof(float) * piStkSize[0] * piStkSize[1];
	cudaMalloc(&gfFrm, tBytes);
	//-----------------
	bool bPadded = true;
	float afMinMax[2] = {(float)1e30, (float)-1e30};
	MU::GFindMinMax2D aFindMinMax;
	aFindMinMax.SetSize(piStkSize, false);
	//-----------------
	for(int i=0; i<piStkSize[2]; i++)
	{	float* pfFrm = (float*)pTiltSeries->GetFrame(i);
		cudaMemcpy(gfFrm, pfFrm, tBytes, cudaMemcpyDefault);
		//----------------
		float fMin = aFindMinMax.DoMin(gfFrm, true);
		float fMax = aFindMinMax.DoMax(gfFrm, true);
		//----------------
		if(fMin < afMinMax[0]) afMinMax[0] = fMin;
		if(fMax > afMinMax[1]) afMinMax[1] = fMax;
	}
	//-----------------
	GFlipInt2D aFlipInt;
	for(int i=0; i<piStkSize[2]; i++)
	{	float* pfFrm = (float*)pTiltSeries->GetFrame(i);
		cudaMemcpy(gfFrm, pfFrm, tBytes, cudaMemcpyDefault);
		aFlipInt.DoIt(gfFrm, piStkSize, !bPadded, 
		   afMinMax[0], afMinMax[1]);
		cudaMemcpy(pfFrm, gfFrm, tBytes, cudaMemcpyDefault);
	}
	cudaFree(gfFrm);
	printf("Flip volume intensity: done.\n\n");
}

