#pragma once
#include "../CAreTomoInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <cufft.h>

namespace McAreTomo::AreTomo::DoseWeight 
{

class GDoseWeightImage
{
public:
	GDoseWeightImage(void);
	~GDoseWeightImage(void);
	void Clean(void);
	void BuildWeight
	( float fPixelSize, float fKv,
	  float* pfImgDose, int* piStkSize, 
	   cudaStream_t stream = 0
	);
	void DoIt
	( cufftComplex* gCmpImg, float fDose,
	  cudaStream_t stream = 0
	);
	float* m_gfWeightSum;
	int m_aiCmpSize[2];
};

class CWeightTomoStack 
{
public:
	CWeightTomoStack(void);
	~CWeightTomoStack(void);
	void Clean(void);
	void DoIt(int iNthGpu);
private:
	void mCorrectProj(int iProj);
	void mForwardFFT(int iProj);
	void mInverseFFT(int iProj);
	void mDoseWeight(int iProj);
	//-----------------
	MD::CTiltSeries* m_pTiltSeries;
	float* m_pfDose;
	//-----------------
	cufftComplex* m_gCmpImg;
	int m_aiCmpSize[2];
	MU::CCufft2D m_aForwardFFT;
	MU::CCufft2D m_aInverseFFT;
	GDoseWeightImage* m_pGDoseWeightImg;
};
}

namespace MAW = McAreTomo::AreTomo::DoseWeight;
