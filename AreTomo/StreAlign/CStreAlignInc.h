#pragma once
#include "../CAreTomoInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilFwd.h"
#include "../Correct/CCorrectFwd.h"
#include <cufft.h>

namespace McAreTomo::AreTomo::StreAlign
{
class CStretchXcf
{
public:
	CStretchXcf(void);
	~CStretchXcf(void);
	void Clean(void);
	void Setup(int* piImgSize, float fBFactor);
	void DoIt
	( float* pfRefImg, float* pfImg,
	  float fRefTilt, float fTilt, float fTiltAxis
	);
	void GetShift(float fFactX, float fFactY, float* pfShift);
private:
	void mPadImage(float* pfImg, float* gfPadImg);
	void mNormalize(float* gfPadImg);
	void mRoundEdge(void);
	void mForwardFFT(void);
	cufftComplex* m_gCmpRef;
	cufftComplex* m_gCmpBuf;
	cufftComplex* m_gCmp;
	float m_fBFactor;
	MU::CCufft2D* m_pForwardFFT;
	MU::CCufft2D* m_pInverseFFT;
	MAU::GXcf2D* m_pGXcf2D;
	int m_aiImgSize[2];
	int m_aiPadSize[2];
	int m_aiCmpSize[2];
	float m_afShift[2];
};

class CStretchCC2D
{
public:
	CStretchCC2D(void);
	~CStretchCC2D(void);
	void Clean(void);
	void SetSize(int* piSize, bool bPadded);
	float DoIt
	( float* pfRefImg,// lower tilt image
	  float* pfImg,   // higher tilt image to be stretched
	  float fRefTilt,
	  float fTilt,
	  float fTiltAxis
	);
private:
	int m_aiSize[2];
	bool m_bPadded;
	float* m_gfRefImg;
	float* m_gfImg;
	float* m_gfBuf;
};

class CStretchAlign
{
public:
	CStretchAlign(void);
	~CStretchAlign(void);
	float DoIt
	( MD::CTiltSeries* pTiltSeries,
	  MAM::CAlignParam* pAlignParam,
	  float fBFactor,
	  float* pfBinning
	);
	float m_fMaxErr;
private:
	float mMeasure(int iProj);
	int mFindRefIndex(int iProj);
	//-----------------
	float m_fBFactor;
	float m_afBinning[2];
	//-----------------
	MD::CTiltSeries* m_pTiltSeries;
	MAM::CAlignParam* m_pAlignParam;
	CStretchXcf m_stretchXcf;
	bool* m_pbBadImgs;
	int m_iZeroTilt;
	char* m_pcLog;
};

class CStreAlignMain
{
public:
	CStreAlignMain(void);
	~CStreAlignMain(void);
	void Clean(void);
	void Setup(int iNthGpu);
	void DoIt(void);
private:
	float mMeasure(void);
	void mUpdateShift(void);
	void mUnstretch
	( int iLowTilt, int iHighTilt, 
	  float* pfShift
	);
	MAC::CCorrTomoStack* m_pCorrTomoStack;
	MD::CTiltSeries* m_pTiltSeries;
	MAM::CAlignParam* m_pAlignParam;
	MD::CTiltSeries* m_pBinSeries;
	MAM::CAlignParam* m_pMeaParam;
	float m_afBinning[2];
};

}

namespace MAS = McAreTomo::AreTomo::StreAlign;
