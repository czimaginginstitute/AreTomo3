#pragma once
#include "../CAreTomoInc.h"

namespace McAreTomo::AreTomo::MassNorm
{
class GFlipInt2D
{
public:
	GFlipInt2D(void);
	~GFlipInt2D(void);
	void DoIt
	( float* gfImg, int* piSize, bool bPadded,
	  float fMin, float fMax, 
	  cudaStream_t stream = 0
	);
};

class CLinearNorm
{
public:
	CLinearNorm(void);
	~CLinearNorm(void);
	void DoIt(int iNthGpu);
	void FlipInt(int iNthGpu);
private:
	void mCalcMeans(int iScales);
	void mExtractSubImg(float* pfImg, int* piImgSize);
	void mScale(int iSeries);
	void mSmooth(float* pfMeans);
	void mFlipInt(int iFrame);
	//-----------------
	int m_aiStart[2];
	int m_aiSize[2];
	int m_iNumFrames;
	float m_fMissingVal;
	float m_fRefMean;
	int m_iZeroTilt;
	int m_iNthGpu;
	float* m_pfMeans;
	float* m_gfSubImg;
};

class CPositivity
{
public:
	CPositivity(void);
	~CPositivity(void);
	void DoIt(int iNthGpu);
private:
	float mCalcMin(int iFrame);
	void mSetPositivity(int iFrame);
	MD::CTiltSeries* m_pTiltSeries;
	float m_fMissingVal;
	float m_fMin;
	int m_iNthGpu;
};

class CFlipInt3D
{
public:
	CFlipInt3D(void);
	~CFlipInt3D(void);
	void DoIt(int iNthGpu);
};

class GPositivity
{
public:
	GPositivity(void);
	~GPositivity(void);
	void DoIt(int iNthGpu);
};
}
