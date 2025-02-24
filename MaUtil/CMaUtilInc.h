#pragma once
#include <cufft.h>

namespace McAreTomo::MaUtil
{
size_t GetGpuMemory(int iGpuId);
void PrintGpuMemoryUsage(const char* pcInfo);
float GetGpuMemoryUsage(void);
void CheckCudaError(const char* pcLocation);
void CheckRUsage(const char* pcLocaltion);
void* GetGpuBuf(size_t tBytes, bool bZero);
//------------------
const char* GetHomeDir(void); // do not free
bool GetCurrDir(char* pcRet, int iSize);
void UseFullPath(char* pcPath);


class CParseArgs
{
public:
        CParseArgs(void);
        ~CParseArgs(void);
        void Set(int argc, char* argv[]);
        bool FindVals(const char* pcTag, int aiRange[2]);
        void GetVals(int aiRange[2], float* pfVals);
        void GetVals(int aiRange[2], int* piVal);
        void GetVal(int iArg, char* pcVal);
        void GetVals(int aiRange[2], char** ppcVals);
private:
        char** m_argv;
        int m_argc;
};

class CCufft2D
{
public:
	CCufft2D(void);
	~CCufft2D(void);
	void CreateForwardPlan(int* piSize, bool bPad);
	void CreateInversePlan(int* piSize, bool bCmp);
	void DestroyPlan(void);
	//-----------------
	bool Forward
	( float* gfPadImg, cufftComplex* gCmpImg,
	  bool bNorm, cudaStream_t stream=0
	);
	bool Forward
	( float* gfPadImg, bool bNorm, 
	  cudaStream_t stream=0
	);
	cufftComplex* ForwardH2G(float* pfImg, bool bNorm);
	//-----------------
	bool Inverse
	( cufftComplex* gCom, float* gfPadImg, 
	  cudaStream_t stream=0
	);
	bool Inverse
	( cufftComplex* gCom, 
	  cudaStream_t stream=0
	);
	float* InverseG2H(cufftComplex* gCmp);
	//-----------------
	void SubtractMean(cufftComplex* gComplex);
private:
	bool mCheckError(cufftResult* pResult, const char* pcFormat);
	const char* mGetErrorEnum(cufftResult error);
	//-----------------
	cufftHandle m_aPlan;
	cufftType m_aType;
	int m_iFFTx;
	int m_iFFTy;
};

class CFileName
{
public:
	CFileName(const char* pcFileName);
	CFileName(void);
	~CFileName(void);
	void Setup(const char* pcFileName);
	void GetFolder(char* pcFolder);
	void GetName(char* pcName); // file name no path & no extension
	void GetExt(char* pcExt);
	//-----------------
	char m_acFolder[256];
	char m_acFileName[128];
	char m_acFileExt[32];
};

class CPad2D
{
public:
	CPad2D(void);
	~CPad2D(void);
	void Pad(float* pfImg, int* piImgSize, float* pfPad);
	void Unpad(float* pfPad, int* piPadSize, float* pfImg);

};

class CPeak2D
{
public:
	CPeak2D(void);
	~CPeak2D(void);
	void GetShift(float fXcfBin, float* pfShift);
	void GetShift(float* pfXcfBin, float* pfShift);
	void DoIt
	( float* pfImg, int* piImgSize, bool bPadded,
	  int* piSeaSize = 0L
	);
	float m_afShift[2];
	float m_fPeakInt;
private:
	void mSearchIntPeak(void);
	void mSearchFloatPeak(void);
	float* m_pfImg;
	int m_aiImgSize[2];
	int m_iPadX;
	int m_aiSeaSize[2];
	int m_aiPeak[2];
	float m_afPeak[2];
};

class CSaveTempMrc
{
public:
	CSaveTempMrc(void);
	~CSaveTempMrc(void);
	void SetFile(char* pcMain, const char* pcMinor);
	void GDoIt(cufftComplex* gCmp, int* piCmpSize);
        void GDoIt(float* gfImg, int* piSize);
        void GDoIt(unsigned char* gucImg, int* piSize);
        void DoIt(void* pvImg, int iMode, int* piSize);
	void DoMany(void** pvImgs, int iMode, int* piSize);
	void DoMany(float* pfImgs, int* piSize);
private:
        char m_acMrcFile[256];
};	//CSaveTempMrc

class GAddFrames
{
public:
	GAddFrames(void);
	~GAddFrames(void);
	void DoIt
	( float* gfFrame1, float fFactor1,
	  float* gfFrame2, float fFactor2,
	  float* gfSum, int* piFrmSize,
	  cudaStream_t stream = 0
	);
	void DoIt
	( cufftComplex* gCmp1, float fFactor1,
	  cufftComplex* gCmp2, float fFactor2,
	  cufftComplex* gCmpSum, int* piCmpSize,
	  cudaStream_t stream = 0
	);
	void DoIt
	( unsigned char* gucFrm1,
	  unsigned char* gucFrm2,
	  unsigned char* gucSum,
	  int* piFrmSize,
	  cudaStream_t stream = 0
	);
};

class GCalcMoment2D
{
public:
	GCalcMoment2D(void);
	~GCalcMoment2D(void);
	void Clean(void);
	void SetSize(int* piImgSize, bool bPadded);
	float DoIt
	( float* gfImg, int iExponent, bool bSync,
	  cudaStream_t stream = 0
	);
	float GetResult(void);
	void Test(float* gfImg, float fExp);
private:
	float* m_gfBuf;
	dim3 m_aBlockDim;
	dim3 m_aGridDim;
	//-----------------
	int m_aiImgSize[2];
	int m_iPadX;
};

class GCorrLinearInterp
{
public:
	GCorrLinearInterp(void);
	~GCorrLinearInterp(void);
	void DoIt
	( cufftComplex* gCmpFrm, int* piCmpSize,
	  cudaStream_t stream = 0
	);
};

class GFFT1D
{
public:
	GFFT1D(void);
	~GFFT1D(void);
	void DestroyPlan(void);
	void CreatePlan
	( int iFFTSize, 
	  int iNumLines, 
	  bool bForward
	);
	void Forward
	( float* gfPadLines,
	  bool bNorm
	);
	void Inverse
	( cufftComplex* gCmpLines 
	);
private:
	int m_iFFTSize;
	int m_iNumLines;
	cufftType m_cufftType;
	cufftHandle m_cufftPlan;
};

class GPad2D
{
public:
	void Pad
	( float* gfImg, 
	  int* piImgSize, 
	  float* gfPadImg,
	  cudaStream_t stream
	);
	void Unpad
	( float* gfPadImg, 
	  int* piPadSize, 
	  float* gfImg,
	  cudaStream_t stream = 0
	);
};

class GRoundEdge2D
{
public:
	GRoundEdge2D(void);
	~GRoundEdge2D(void);
	void SetMask(float* pfCent, float* pfSize);
	void DoIt
	( float* gfImg, int* piSize, bool bPadded,
	  float fPower, cudaStream_t stream = 0
	);
private:
	float m_afMaskCent[2];
	float m_afMaskSize[2];
};

class GRoundEdge1D
{
public:
	GRoundEdge1D(void);
	~GRoundEdge1D(void);
	void DoIt(float* gfData, int iSize);
	void DoPad(float* gfPadData, int iPadSize);
};

class GNormalize2D
{
public:
	GNormalize2D(void);
	~GNormalize2D(void);
	void DoIt
	( float* gfImg, int* piSize, bool bPadded,
	  float fMean, float fStd,
	  cudaStream_t stream = 0
	);
};

class GThreshold2D
{
public:
	GThreshold2D(void);
	~GThreshold2D(void);
	void DoIt
	( float* gfImg, int* piImgSize, bool bPadded,
	  float fMin, float fMax
	);
};

class GFourierResize2D
{
public:
	GFourierResize2D(void);
	~GFourierResize2D(void);
	//-----------------
	static void GetBinnedCmpSize
	(  int* piCmpSize,// cmp size before binning
	   float fBin,
	   int* piNewSize // cmp size after binning
	);
	static void GetBinnedImgSize
	(  int* piImgSize, // img size before binning
	   float fBin,
	   int* piNewSize
	);
	static float CalcPixSize
	(  int* piImgSize, // img size before binning
	   float fBin,
	   float fPixSize  // before binning
	);
	static void GetBinning
	(  int* piCmpSize,  // cmp size before binning
	   int* piNewSize,  // cmp size after binning
	   float* pfBinning
	);
	void DoIt
	( cufftComplex* gCmpIn, 
	  int* piSizeIn,
	  cufftComplex* gCmpOut, 
	  int* piSizeOut,
	  bool bSum,
	  cudaStream_t stream = 0
	);
	//-----------------
	void Clean(void);
	void Setup(int* piInImgSize, int* piOutImgSize);
	void DoIt(float* pfInImg, float* pfOutImg);
private:
	float* m_gfInImg;
	float* m_gfOutImg;
	int m_aiInImgSize[2];
	int m_aiOutImgSize[2];
	CCufft2D* m_pForwardFFT;
	CCufft2D* m_pInverseFFT;
};

class GFtResize2D
{
public:
        GFtResize2D(void);
        ~GFtResize2D(void);
        //-----------------
        static void GetBinnedCmpSize
        (  int* piCmpSize,// cmp size before binning
           float fBin,
           int* piNewSize // cmp size after binning
        );
        static void GetBinnedImgSize
        (  int* piImgSize, // img size before binning
           float fBin,
           int* piNewSize
        );
        static float CalcPixSize
        (  int* piImgSize, // img size before binning
           float fBin,
           float fPixSize  // before binning
        );
        static void GetBinning
        (  int* piCmpSize,  // cmp size before binning
           int* piNewSize,  // cmp size after binning
           float* pfBinning
        );
	void DownSample
        ( cufftComplex* gCmpIn, int* piSizeIn,
          cufftComplex* gCmpOut, int* piSizeOut,
          bool bSum, cudaStream_t stream = 0
        );
	void UpSample
        ( cufftComplex* gCmpIn, int* piSizeIn,
          cufftComplex* gCmpOut, int* piSizeOut,
          cudaStream_t stream = 0
        );
};

class GPositivity2D
{
public:
	GPositivity2D(void);
	~GPositivity2D(void);
	void DoIt(float* gfImg, int* piImgSize, cudaStream_t stream = 0);
	void AddVal(float* gfImg, int* piImgSize, float fVal,
	   cudaStream_t stream = 0);
};

class GFFTUtil2D
{
public:
	GFFTUtil2D(void);
	~GFFTUtil2D(void);
	void Multiply
	( cufftComplex* gComp,
	  int* piCmpSize,
	  float fFactor,
          cudaStream_t stream=0
	);
	void GetAmp
	( cufftComplex* gComp,
	  int* piCmpSize,
	  float* pfAmpRes,
	  bool bGpuRes,
          cudaStream_t stream=0
	);
	void Shift
	( cufftComplex* gComp,
	  int* piCmpSize,
	  float* pfShift,
          cudaStream_t stream=0
	);
	void Lowpass
	( cufftComplex* gInCmp, cufftComplex* gOutCmp,
	  int* piCmpSize, float fBFactor
	);
	//-----------------------------------------------
	// This is 1 minus lowpass filter.
	//-----------------------------------------------
	void Highpass
	( cufftComplex* gInCmp, cufftComplex* gOutCmp,
	  int* piCmpSize, float fBFactor
	);
};

class GFindMinMax2D
{
public:
	GFindMinMax2D(void);
	~GFindMinMax2D(void);
	void Clean(void);
	void SetSize(int* piImgSize, bool bPadded);
	float DoMin(float* gfImg, bool bSync, cudaStream_t stream = 0);
	float DoMax(float* gfImg, bool bSync, cudaStream_t stream = 0);
	float GetResult(void);
	void Test(float* gfImg);
private:
	int m_aiImgSize[2];
	int m_iPadX;
	dim3 m_aBlockDim;
	dim3 m_aGridDim;
	float* m_gfBuf;
};

class GPartialCopy
{
public:
  	static void DoIt
	( float* pSrc,
	  int iSrcSizeX,
	  float* pDst,
	  int iCpySizeX,
	  int* piDstSize,
	  cudaStream_t stream = 0
	);
	static void DoIt
	( cufftComplex* pSrc,
	  int iSrcSizeX,
	  cufftComplex* pDst,
	  int iCpySizeX,
	  int* piDstSize,
	  cudaStream_t stream = 0
	);
};

class GPhaseShift2D
{
public:
	GPhaseShift2D(void);
	~GPhaseShift2D(void);
	void DoIt
	( cufftComplex* gInCmp,
	  int* piCmpSize,
	  float* pfShift,
	  bool bSum,
	  cufftComplex* gOutCmp,
	  cudaStream_t stream = 0
	);
	void DoIt
	( cufftComplex* gCmpFrm,
	  int* piCmpSize,
	  float* pfShift,
	  cudaStream_t stream = 0
	);
};

class GGriddingCorrect
{
public:
	GGriddingCorrect(void);
	~GGriddingCorrect(void);
	void DoCmp
	( cufftComplex* gCmp, 
	  int* piCmpSize,
          cudaStream_t stream=0
	);
};

class GCalcFRC
{
public:
	GCalcFRC(void);
	~GCalcFRC(void);
	void DoIt
	( cufftComplex* gCmp1, 
	  cufftComplex* gCmp2,
	  float* gfFRC, int iRingWidth, // in pixel
	  int* piCmpSize,
	  cudaStream_t stream = 0
	);
	float* DoIt
	( float* pfImg1, float* pfImg2,
	  int* piImgSize, bool bPadded,
	  int iRingWidth
	);
	float m_fRes;
private:
	float* mHost2Gpu(float* pfImg, int* piImgSize);
	void mCalcFFT(float* gfPadImg1, float* gfPadImg2);
	int m_aiCmpSize[2];
};
}

namespace MU = McAreTomo::MaUtil;
