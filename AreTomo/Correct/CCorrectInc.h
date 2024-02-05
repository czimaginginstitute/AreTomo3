#pragma once
#include "../CAreTomoInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Recon/CReconFwd.h"
#include <cufft.h>
#include <cuda_runtime.h>

namespace McAreTomo::AreTomo::Correct
{
class CCorrectUtil
{
public:
	static void CalcAlignedSize
	( int* piRawSize, float fTiltAxis, 
	  int* piAlnSize
	);
	static void CalcBinnedSize
	( int* piRawSize, float fBinning, bool bFourierCrop,
	  int* piBinnedSize
	);
};

class CBinStack
{
public:
	CBinStack(void);
	~CBinStack(void);
	MD::CTiltSeries* DoReal
	( MD::CTiltSeries* pTiltSeries, 
	  int iBin, int iNthGpu
	);
	MD::CTiltSeries* DoFFT
	( MD::CTiltSeries* pTiltSeries, 
	  float fBin, int iNthGpu
	);
};

class CFourierCropImage
{
public:
	CFourierCropImage(void);
	~CFourierCropImage(void);
	void Setup(int iNthGpu, int* piImgSize, float fBin);
	void DoPad(float* gfPadImgIn, float* gfImgPadOut);
	int m_aiImgSizeIn[2];
	int m_aiImgSizeOut[2];
private:
	void mCropFFT(float* gfPadImgIn, float* gfPadImgOut);
	void mCopy(float* gfPadImgIn, float* gfPadImgOut);
	//-----------------
	MU::CCufft2D* m_pForward2D;
	MU::CCufft2D* m_pInverse2D;
	bool m_bSameSize;
};

class GCorrPatchShift
{
public:
	GCorrPatchShift(void);
	~GCorrPatchShift(void);
	void SetSizes
	( int* piInSize,
	  bool bInPadded,
	  int* piOutSize,
	  bool bOutPadded,
	  int iNumPatches
	);
	void DoIt
	( float* gfInImg,
	  float* pfGlobalShift,
	  float fRotAngle,
	  float* gfLocalAlnParams,
	  bool bRandomFill,
	  float* gfOutImg
	);
private:
	float m_fD2R;
	int m_iInImgX;
	int m_iOutImgX;
	int m_iOutImgY;
};

class CCorrProj
{
public:
	CCorrProj(void);
	~CCorrProj(void);
	void Clean(void);
	void Setup
	( int* piInSize, bool bInPadded,
	  bool bRandomFill, bool bFourierCrop,
	  float fTiltAxis, float fBinning,
	  int iNthGpu
	);
	void SetProj(float* pfInProj);
	void DoIt(float* pfGlobalShift, float fTiltAxis);
	void GetProj(float* pfCorProj, int* piSize, bool bPadded);
private:
	int m_aiInSize[2];
	int m_iInImgX;
	bool m_bInPadded;
	bool m_bRandomFill;
	bool m_bFourierCrop;
	float m_fBinning;
	int m_iNthGpu;
	//------------
	float* m_gfRawProj;
	float* m_gfCorProj;
	float* m_gfRetProj;
	int m_aiCorSize[2];
	int m_aiRetSize[2];
	//-----------------
	MAU::GBinImage2D m_aGBinImg2D;
	CFourierCropImage m_aFFTCropImg;
	GCorrPatchShift m_aGCorrPatchShift;
};

class CCorrTomoStack 
{
public:
	CCorrTomoStack(void);
	~CCorrTomoStack(void);
	//-----------------------------------------------------------
	// In case of shift only, fTiltAxis must be zero.
	//-----------------------------------------------------------
	void Set0(int iNthGpu);
	void Set1(int iNumPatches, float fTiltAxis);
	void Set2(float fOutBin, bool bFourierCrop, bool bRandFill);
	void Set3(bool bShiftOnly, bool bCorrInt, bool bRWeight);
	void Set4(bool bForRecon);
	void DoIt(int iNthSeries, MAM::CAlignParam* pAlignParam);
	MD::CTiltSeries* GetCorrectedStack(bool bClean);
	void GetBinning(float* pfBinning);
	void Clean(void);
private:
	void mCorrectProj(int iProj);
	float* m_gfRawProj;
	float* m_gfCorrProj;
	float* m_gfBinProj;
	float* m_gfLocalParam;
	//-----------------
	MAM::CAlignParam* m_pAlignParam;
	GCorrPatchShift m_aGCorrPatchShift;
	Util::GBinImage2D m_aGBinImg2D;
	CFourierCropImage m_aFFTCropImg;
	MD::CTiltSeries* m_pOutSeries;
	MAR::GRWeight* m_pGRWeight;
	//-----------------
	float m_fOutBin;
	float m_afBinning[2];
	int m_aiStkSize[3];
	int m_aiAlnSize[3];
	int m_aiBinnedSize[3];
	bool m_bShiftOnly;
	bool m_bRandomFill;
	bool m_bFourierCrop;
	int m_iNthGpu;
	int m_iSeries;
	bool m_bForRecon;
};

}

namespace MAC = McAreTomo::AreTomo::Correct;
