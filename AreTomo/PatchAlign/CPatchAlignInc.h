#pragma once
#include "../CAreTomoInc.h"
#include "../ProjAlign/CProjAlignFwd.h"
#include "../MrcUtil/CMrcUtilFwd.h"
#include <cuda.h>
#include <cufft.h>
#include <pthread.h>

namespace McAreTomo::AreTomo::PatchAlign
{

class GRandom2D
{
public:
	GRandom2D(void);
	~GRandom2D(void);
	void DoIt
	( float* gfInImg,
	  float* gfOutImg,
	  int* piImgSize,
	  bool bPadded,
	  cudaStream_t stream = 0
	);
};

class GExtractPatch
{
public:
	GExtractPatch(void);
	~GExtractPatch(void);
	void SetSizes
	( int* piInSize, bool bInPadded,
	  int* piOutSize, bool bOutPadded
	);
	void DoIt
	( float* gfInImg, int* piShift,
	  bool bRandomFill, float* gfOutImg
	);
private:
	int m_iInImgX;
	int m_iOutImgX;
	int m_aiOutSize[2];
};

class GCommonArea
{
public:
	GCommonArea(void);
	~GCommonArea(void);
	void DoIt
	( float* gfImg1,
	  float* gfImg2,
	  float* gf2Buf2,
	  int* piImgSize,
	  bool bPadded,
	  float* gfCommArea,
	  cudaStream_t stream = 0
	);
private:
	void mFindCommonArea(void);
	void mCenterCommonArea(float* gfCommArea);
	int m_aiImgSize[2];
	int m_iPadX;
	float* m_gfImg1;
	float* m_gfImg2;
	float* m_gfBuf1;
	float* m_gfBuf2;
	cudaStream_t m_stream;
};

class GGenXcfImage
{
public:
	GGenXcfImage(void);
	~GGenXcfImage(void);
	void Clean(void);
	void Setup(int* piCmpSize);
	void DoIt
	( cufftComplex* gCmp1,
	  cufftComplex* gCmp2,
	  float* gfXcfImg,
	  float fBFactor,
	  cudaStream_t stream = 0
	);
	int m_aiXcfSize[2];
private:
	MU::CCufft2D* m_pInverseFFT2D;
};

class GPartialCopy
{
public:
	GPartialCopy(void);
	~GPartialCopy(void);
	void DoIt
	( float* gfSrc, int iSrcSizeX,
	  float* gfDst, int* piDstSize,
	  int iCpySizeX,
	  cudaStream_t stream = 0
	);
	void DoIt
	( float* gfSrcImg, int* piSrcSize, int* piSrcStart,
	  float* gfPatch, int* piPatSize, bool bPadded,
	  cudaStream_t stream = 0
	);
};

class GNormByStd2D
{
public:
	GNormByStd2D(void);
	~GNormByStd2D(void);
	void DoIt(float* gfImg, int* piImgSize, bool bPadded,
	   int* piWinSize, cudaStream_t stream = 0);
};

class CCalcXcfImage
{
public:
	CCalcXcfImage(void);
	~CCalcXcfImage(void);
	void Clean(void);
	void Setup(int* piPatSize); 
	void DoIt
	( float* gfImg1,
	  float* gfImg2,
	  float* gfBuf,
	  int* piImgSize,
	  int* piStart,
	  float* gfXcfImg,
	  int* piXcfSize,
	  float* gfCommArea, 
	  cudaStream_t stream = 0
	);
private:
	void mExtractPatches
	( float* gfImg1, float* gfImg2, 
	  int* piImgSize, int* piStart
	);
	void mNormalize(float* gfPatImg);
	void mRoundEdge(float* gfPatImg);
	//-------------------------------
	int m_aiPatSize[2];
	int m_aiPatPad[2]; // patch padded size
	GGenXcfImage m_GGenXcfImage;
	MU::CCufft2D m_Gfft2D;
	float* m_gfPatImg1;
	float* m_gfPatImg2;
	float* m_gfBuf;
	cudaStream_t m_stream;
};

class CFitPatchShifts
{
public:
	CFitPatchShifts(void);
	~CFitPatchShifts(void);
	void Clean(void);
	void Setup
	( MAM::CAlignParam* pFullParam,
	  int iNumPatches
	);
	float DoIt
	( MAM::CPatchShifts* pPatchShifts,
	  MAM::CLocalAlignParam* pLocalAlnParam
	);
	int m_iNumPatches;
	int m_iNumTilts;
private:
	void mCalcPatCenters(void);
	//void mCalcXs(void);
	float mCalcZs(void);
	float mCalcPatchZ(int iPatch);
	float mRefineTiltAxis(void);
	float mCalcTiltAxis(float fDelta);
	void mCalcSinCosRots(void);
	void mCalcLocalShifts(void);
	void mCalcPatchLocalShifts(int iPatch);
	void mScreenPatchLocalShifts(int iPatch);
	void mScreenTiltLocalShifts(int iTilt);
	//-------------------------------------
	MAM::CPatchShifts* m_pPatchShifts;
	MAM::CAlignParam* m_pFullParam;
	MAM::CLocalAlignParam* m_pLocalParam;
	float* m_pfMeasuredUs;
	float* m_pfMeasuredVs;
	float* m_pfPatCentUs;
	float* m_pfPatCentVs;
	float* m_pfCosTilts;
	float* m_pfSinTilts;
	float* m_pfCosRots;
	float* m_pfSinRots;
	float* m_pfDeltaRots;
	//float* m_pfXs;
	float m_fErr;
	int m_iZeroTilt;
};

class CLocalAlign
{
public:
	CLocalAlign(void);
	~CLocalAlign(void);
	void Setup(int iNthGpu);
	void DoIt(MAM::CAlignParam* pAlignParam, int* piRoi);
private:
	MAJ::CProjAlignMain* m_pProjAlignMain;
	int m_iNthGpu;
};

class CDetectFeatures
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CDetectFeatures* GetInstance(int iNthGpu);
	static void DeleteInstance(void);
	~CDetectFeatures(void);
	void SetSize(int* piImgSize, int* piNumPatches);
	void DoIt(float* pfImg);
	void GetCenter(int iPatch, int* piCent);
private:
	CDetectFeatures(void);
	void mClean(void);
	void mFindCenters(void);
	bool mCheckFeature(float fCentX, float fCentY);
	void mFindCenter(float* pfCenter);
	void mSetUsed(float fCentX, float fCentY);
	int mCheckRange(int iStartX, int iSize, int* piRange);
	int m_aiImgSize[2];
	int m_aiBinnedSize[2];
	int m_aiNumPatches[2];
	float m_afPatSize[2];
	int m_aiSeaRange[4];
	bool* m_pbFeatures;
	bool* m_pbUsed;
	float* m_pfCenters;
	static CDetectFeatures* m_pInstances;
	static int m_iNumGpus;
};

class CPatchTargets
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CPatchTargets* GetInstance(int iNthGpu);
	~CPatchTargets(void);
	void Clean(void);
	void Detect(void);
	void GetTarget(int iTgt, int* piTgt);
	int m_iNumTgts;
	int m_iNthGpu;
private:
	CPatchTargets(void);
	int* m_piTargets;
	int m_iTgtImg;
	static CPatchTargets* m_pInstances;
	static int m_iNumGpus;
};

class CPatchAlignMain
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CPatchAlignMain* GetInstance(int iNthGpu);
	//-----------------
	~CPatchAlignMain(void);
	void DoIt(float fTiltOffset);
private:
	CPatchAlignMain(void);
	void mAlignStack(int iPatch);
	//-----------------
	MD::CTiltSeries* m_pTiltSeries;
	MAM::CAlignParam* m_pFullParam;
	float m_fTiltOffset;
	//-----------------
	MAM::CPatchShifts* m_pPatchShifts;
	MAM::CLocalAlignParam* m_pLocalParam;
	CLocalAlign* m_pLocalAlign;
	int m_iNthGpu;
	//-----------------
	static CPatchAlignMain* m_pInstances;
	static int m_iNumGpus;
};
}

namespace MAP = McAreTomo::AreTomo::PatchAlign;
