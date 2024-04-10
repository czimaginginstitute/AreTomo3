#pragma once
#include "../CAreTomoInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <cufft.h>
#include <cuda_runtime.h>

namespace McAreTomo::AreTomo::FindCtf
{
class CCtfTheory
{
public:
	CCtfTheory(void);
	~CCtfTheory(void);
	void Setup
	( float fKv, // keV
	  float fCs, // mm
	  float fAmpContrast,
	  float fPixelSize,    // A
	  float fAstTil,       // A, negative means no tolerance
	  float fExtPhase      // Radian
	);
	void SetExtPhase(float fExtPhase, bool bDegree);
	float GetExtPhase(bool bDegree);
	void SetPixelSize(float fPixSize);
	void SetDefocus
	( float fDefocusMin, // A
	  float fDefocusMax, // A
	  float fAstAzimuth  // deg
	);
	void SetDefocusInPixel
	( float fDefocusMaxPixel, // pixel
	  float fDefocusMinPixel, // pixel
	  float fAstAzimuthRadian // radian
	);
	//-----------------
	void SetParam(MD::CCtfParam* pCTFParam); // copy values
	MD::CCtfParam* GetParam(bool bCopy);     // do not free
	//-----------------
	float Evaluate
	( float fFreq, // relative frequency in [-0.5, +0.5]
	  float fAzimuth
	);
	int CalcNumExtrema
	( float fFreq, // relative frequency in [-0.5, +0.5]
	  float fAzimuth
	);
	float CalcNthZero
	( int iNthZero,
	  float fAzimuth
	);
	float CalcDefocus
	( float fAzimuth
	);
	float CalcPhaseShift
	( float fFreq, // relative frequency [-0.5, 0.5]
	  float fAzimuth
	);
	float CalcFrequency
	( float fPhaseShift,
	  float fAzimuth
	);
	bool EqualTo
	( CCtfTheory* pCTFTheory,
	  float fDfTol
	);
	float GetPixelSize(void);
	CCtfTheory* GetCopy(void);
private:
	float mCalcWavelength(float fKv);
	void mEnforce(void);
	MD::CCtfParam* m_pCtfParam;
	float m_fPI;
};

class GCalcCTF1D
{
public:
	GCalcCTF1D(void);
	~GCalcCTF1D(void);
	void SetParam(MD::CCtfParam* pCtfParam);
	void DoIt
	( float fDefocus,  // in pixel
	  float fExtPhase, // phase in radian from phase plate
	  float* gfCTF1D,
	  int iCmpSize
	);
private:
	float m_fAmpPhase;
};

class GCalcCTF2D
{
public:
	GCalcCTF2D(void);
	~GCalcCTF2D(void);
	void SetParam(MD::CCtfParam* pCtfParam);
	void DoIt
	( float fDfMin, float fDfMax, float fAzimuth, 
	  float fExtPhase, // phase in radian from phase plate
	  float* gfCTF2D, int* piCmpSize
	);
	void DoIt
	( MD::CCtfParam* pCtfParam,
	  float* gfCtf2D,
	  int* piCmpSize
	);
	void EmbedCtf
	( float* gfCtf2D,
	  float fMinFreq,
	  float fMaxFreq, // relative freq
	  float fMean, float fGain, // for scaling
	  float* gfFullSpect,
	  int* piCmpSize  // size of gfCtf2D
	); 

private:
	float m_fAmpPhase; // phase from amplitude contrast
};

class GCalcSpectrum
{
public:
	GCalcSpectrum(void);
	~GCalcSpectrum(void);
	void Clean(void);
	void SetSize(int* piSize, bool bPadded);
	void DoPad
	( float* gfPadImg,   // image already padded
	  float* gfSpectrum, // GPU buffer
	  bool bLog
	);
	//-----------------
	void DoIt
	( cufftComplex* gCmp,
	  float* gfSpectrum,
	  int* piCmpSize,
	  bool bLog
	);
	void Logrithm
	( float* gfSpectrum,
	  int* piSize
	);
	void GenFullSpect
	( float* gfHalfSpect,
	  int* piCmpSize,
	  float* gfFullSpect,
	  bool bFullPadded
	);
private:
	MU::CCufft2D* m_pCufft2D;
	int m_aiPadSize[2];
	int m_aiCmpSize[2];
};

class GBackground1D
{
public:
	GBackground1D(void);
	~GBackground1D(void);
	void SetBackground(float* gfBackground, int iStart, int iSize);
	void Remove1D(float* gfSpectrum, int iSize);
	void Remove2D(float* gfSpectrum, int* piSize);
	void DoIt(float* pfSpectrum, int iSize);
	int m_iSize;
	int m_iStart;
private:
	int mFindStart(float* pfSpectrum);
	float* m_gfBackground;
};

class GRemoveMean
{
public:
	GRemoveMean(void);
	~GRemoveMean(void);
	void DoIt
	(  float* pfImg,  // 2D image
	   bool bGpu,     // if the image is in GPU memory
	   int* piImgSize // image x and y sizes
	);
	void DoPad
	(  float* pfPadImg, // 2D image with x dim padded
	   bool bGpu,       // if the image is in GPU memory
	   int* piPadSize   // x size is padded size
	);
private:
	float* mToDevice(float* pfImg, int* piSize);
	float mCalcMean(float* gfImg);
	void mRemoveMean(float* gfImg, float fMean);
	int m_iPadX;
	int m_aiImgSize[2];
};

class GRmBackground2D
{
public:
	GRmBackground2D(void);
	~GRmBackground2D(void);
	void DoIt
	( float* gfInSpect, // half spact
	  float* gfOutSpect,
	  int* piCmpSize,
	  float fMinFreq // relative frequency[0, 0.5]
	);
};

class GRadialAvg
{
public:
	GRadialAvg(void);
	~GRadialAvg(void);
	void DoIt(float* gfSpect, float* gfAverage, int* piCmpSize);
};

class GRoundEdge
{
public:
	GRoundEdge(void);
	~GRoundEdge(void);
	void SetMask
	(  float* pfCent,
	   float* pfSize
	);
	void DoIt
	(  float* gfImg,
	   int* piImgSize
	);

private:
	float m_afMaskCent[2];
	float m_afMaskSize[2];
};

class GCC2D
{
public:
	GCC2D(void);
	~GCC2D(void);
	void Setup
	(  float fFreqLow,  // relative freq [0, 0.5]
	   float fFreqHigh, // relative freq [0, 0.5]
	   float fBFactor
	);
	void SetSize(int* piCmpSize); // half spectrum
	float DoIt(float* gfCTF, float* gfSpectrum);
private:
	float m_fFreqLow;
	float m_fFreqHigh;
	float m_fBFactor;
	int m_aiCmpSize[2];
	int m_iGridDimX;
	int m_iBlockDimX;
	float* m_gfRes;
};

class GCC1D
{
public:
	GCC1D(void);
	~GCC1D(void);
	void SetSize(int iSize);
	void Setup
	(  float fFreqLow,   // relative freq [0, 0.5]
	   float fFreqHigh,  // relative freq [0, 0.5]
	   float fBFactor
	);
	float DoIt(float* gfCTF, float* gfSpectrum);
	float DoCPU
	(  float* gfCTF,
	   float* gfSpectrum,
	   int iSize
	);
private:
	int m_iSize;
	float* m_gfRes;
	float m_fFreqLow;
	float m_fFreqHigh;
	float m_fBFactor;
};

//--------------------------------------------------------------------
// 1. It calculates the Thon resolution and return the first shell
//    at which the CC drops to 0.143.
// 2. gfSpect is the half spectrum whose DC is at (0, iSpectY / 2).
//    The Nyqust in x is at iSpectX - 1.
// 3. If the real image has size of (Nx, Ny), iSpectX = iNx / 2 + 1,
//    iSpectY = Ny.
//--------------------------------------------------------------------
class GSpectralCC2D
{
public:
	GSpectralCC2D(void);
	~GSpectralCC2D(void);
	void SetSize(int* piSpectSize);
	int DoIt(float* gfCTF, float* gfSpect);
private:
	int m_aiSpectSize[2];
	float* m_gfCC;
	float* m_pfCC;
};

class GCorrCTF2D
{
public:
	GCorrCTF2D(void);
	~GCorrCTF2D(void);
	void SetParam(MD::CCtfParam* pCtfParam);
	void SetPhaseFlip(bool bValue);
	void DoIt
	( float fDfMin, float fDfMax, // pixel
	  float fAzimuth, float fExtPhase, // rad
	  float fTilt, cufftComplex* gCmp, // dgree 
	  int* piCmpSize,
	  cudaStream_t stream = 0
	);
	void DoIt
	( MD::CCtfParam* pCtfParam, float fTilt,
	  cufftComplex* gCmp, int* piCmpSize,
	  cudaStream_t stream = 0
	);
private:
	bool m_bPhaseFlip;
	float m_fAmpPhase;
	float* m_gfNoise2;
};

class CTile
{
public:
	CTile(void);
	~CTile(void);
	void Clean(void);
	void SetTileSize(int iTileSize);
	void SetCoreSize(int iCoreSize);
	void SetTileStart(int iTileStartX, int iTileStartY);
	void SetCoreStart(int iCoreStartX, int iCoreStartY);
	void SetImgSize(int* piImgSize);
	//-----------------
	void Extract(float* pfImage);
	void PasteCore(float* pfImage);
	void PasteCore(float* gfTile, float* pfImage);
	//-----------------
	void GetTileCenter(float* pfCent);
	void GetCoreCenter(float* pfCent);
	void GetTileStart(int* piStart);
	void GetCoreStart(int* piStart);
	int GetTileSize(void);
	int GetTileBytes(void);
	//-----------------
	float* m_pfTile;  // [m_iPadSize, m_iTileSize], do not free
	int m_iTileSize;
	int m_iPadSize;
private:
	int m_aiTileStart[2];
	int m_aiCoreStart[2];
	int m_iCoreSize;
	int m_aiImgSize[2];
};

class CExtractTiles
{
public:
	CExtractTiles(void);
	~CExtractTiles(void);
	void Setup(int iTileSize, int iCoreSize, int* piImgSize);
	void DoIt(float* pfImage);
	CTile* GetTile(int iNthTile);
	bool bEdgeTile(int iNthTile);
	int m_iNumTiles;
private:
	void mCalcTileLocations(void);
	void mCheckTileBound(int* piVal, int iImgSize);
	void mClean(void);
	//-----------------
	int m_aiNumTiles[2];
	int m_aiImgSize[2];
	int m_iTileSize;
	int m_iCoreSize;
	CTile* m_pTiles;
};

class CCorrImgCtf
{
public:
	CCorrImgCtf(void);
	~CCorrImgCtf(void);
	void Setup(int* piImgSize, int iNthGpu);
	void DoIt
	( float* pfImage, float fTilt, 
	  float fTiltAxis, bool bPhaseFlip
	);
private:
	void mTileToGpu(int iTile);
	void mGpuToTile(int iTile);
	float mCalcDeltaZ(int iTile);
	void mCorrectCTF(int iTile);
	//-----------------
	GCorrCTF2D* m_pGCorrCTF2D;
	CExtractTiles* m_pExtractTiles;
	MU::CCufft2D* m_pForwardFFT;
	MU::CCufft2D* m_pInverseFFT;
	MD::CCtfParam* m_pImgCtfParam;
	//-----------------
	cudaStream_t m_streams[2];
	float* m_ggfTiles[2];
	//-----------------
	int m_iTileSize;
	int m_iCoreSize;
	//-----------------
	float* m_pfImage;
	int m_aiImgSize[2];
	float m_fTilt;
	float m_fTiltAxis;
	//-----------------
	int m_iNthGpu;
};

class CTileSpectra
{
public:
	static CTileSpectra* GetInstance(int iNthGpu);
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
        //-----------------
        ~CTileSpectra(void);
        void Clean(void);
        void Create(int iTilesPerTilt, int iNumTilts, int iTileSize);
        void Adjust(int iTilesPerTilt, int iNumTilts);
        //-----------------
        void SetSpect(int iTile, int iTilt, float* gfSpect);
        void SetAvgSpect(int iTilt, float* gfSpect);
        //-----------------
        void SetStat(int iTile, int iTilt, float* pfStat);
        void SetXY(int iTile, int iTilt, float* pfXY);
        //-----------------
        float* GetSpect(int iTile, int iTilt);
        float* GetAvgSpect(int iTilt);
        //-----------------
        float GetWeight(int iTile, int iTilt);
        void GetXYZ(int iTile, int iTilt, float* pfXYZ);
        void GetXY(int iTile, int iTilt, float* pfXY);
        float GetZ(int iTile, int iTilt);
        //-----------------
        void Screen(void);
        int GetSpectSize(void);
        //-----------------
        void CalcZs
        ( float* pfTilts, float fTiltAxis,
          float fTiltOffset, float fBetaOffset
        );
        //-----------------
        int m_iTilesPerTilt;  // number of tiles per tilt
        int m_iNumTilts;      // include dark images
        int m_iSeriesTiles;   // m_iTilesPerTilt * m_iNumTilts
	int m_aiSpectSize[2]; // [iTileSize/2+1, iTileSize]
private:
        CTileSpectra(void);
        void mExpandBuf(void);
        void mCalcStat(float* pfVals, float* pfStat);
        void mCalcTiltZs(int iTilt, float fTilt);
        //-----------------
        float** m_ppfSpectra;
        float* m_pfTileXs;
        float* m_pfTileYs;
        float* m_pfTileZs;
        //-----------------
        float* m_pfMeans;
        float* m_pfStds;
        float* m_pfWeights;
        //-----------------
        int m_iBufSize;
        //-----------------
        float m_fTiltAxis;
        float m_fTiltOffset;
        float m_fBetaOffset;
        //-----------------
	int m_iNthGpu;
	static int m_iNumGpus;
	static CTileSpectra* m_pInstances;
};

class CGenAvgSpectrum
{
public:
	CGenAvgSpectrum(void);
	~CGenAvgSpectrum(void);
	void Clean(void);
	void SetSizes(int* piImgSize,int iTileSize);
	void DoIt(float* pfImage, float* gfAvgSpect);
	int m_aiCmpSize[2];
private:
	void mGenAvgSpectrum(void);
	void mCalcTileSpectrum(int iTile);
	void mExtractPadTile(int iTile);
	//-----------------
	MU::GCalcMoment2D* m_pGCalcMoment2D;
	GCalcSpectrum* m_pGCalcSpectrum;
	float* m_pfImage;
	int m_aiImgSize[2];
	int m_iTileSize;
	int m_aiPadSize[2];
	int m_aiNumTiles[2];
	int m_aiOffset[2];
	int m_iOverlap;
	float m_fOverlap;
	float* m_gfAvgSpect;
	float* m_gfTileSpect;
	float* m_gfPadTile;
};

class CSpectrumImage
{
public:
	CSpectrumImage(void);
	~CSpectrumImage(void);
     	void DoIt
	( float* gfHalfSpect,
	  float* gfCtfBuf,
	  int* piCmpSize,
	  CCtfTheory* pCtfTheory,
	  float* pfResRange,
	  float* gfFullSpect
	);
private:
	void mGenFullSpectrum(void);
	void mEmbedCTF(void);
	float* m_gfHalfSpect;
	float* m_gfCtfBuf;
	float* m_gfFullSpect;
	CCtfTheory* m_pCtfTheory;
	int m_aiCmpSize[2];
	float m_afResRange[2];
	float m_fMean;
	float m_fStd;     
};

class CFindDefocus1D
{
public:
	CFindDefocus1D(void);
	~CFindDefocus1D(void);
	void Clean(void);
	void Setup(MD::CCtfParam* pCtfParam, int iCmpSize);
	void SetResRange(float afRange[2]); // angstrom
	void DoIt
	( float afDfRange[2],    // f0, delta angstrom
	  float afPhaseRange[2], // p0, delta degree
	  float* gfRadiaAvg
	);
	float m_fBestDf;
	float m_fBestPhase;
	float m_fMaxCC;
private:
	void mBrutalForceSearch(float afResult[3]);
	void mCalcCTF(float fDefocus, float fExtPhase);
	float mCorrelate(void);
	//-----------------
	MD::CCtfParam* m_pCtfParam;
	GCC1D* m_pGCC1D;
	GCalcCTF1D m_aGCalcCtf1D;
	//-----------------
	float m_afResRange[2];
	float m_afDfRange[2];    // f0, delta in angstrom
	float m_afPhaseRange[2]; // p0, delta in degree
	float* m_gfRadialAvg;
	int m_iCmpSize;
	float* m_gfCtf1D;
};

class CFindDefocus2D 
{
public:
	CFindDefocus2D(void);
	~CFindDefocus2D(void);
	void Clean(void);
	void Setup1(MD::CCtfParam* pCtfParam, int* piCmpSize);
	void Setup2(float afResRange[2]); // angstrom
	void Setup3
	( float fDfMean, float fAstRatio,
	  float fAstAngle, float fExtPhase
	);
	//-----------------
	void DoIt
	( float* gfSpect,
	  float* pfPhaseRange
	);
	void Refine
	( float* gfSpect, float fDfMeanRange,
	  float fAstRange, float fAngRange,
	  float fPhaseRange
	);
	//-----------------
	float GetDfMin(void);    // angstrom
	float GetDfMax(void);    // angstrom
	float GetAstRatio(void);
	float GetAngle(void);    // degree
	float GetExtPhase(void); // degree
	float GetScore(void);
	float GetCtfRes(void); // angstrom
private:
	void mIterate(void);
	void mDoIt
	( float* pfDfRange,
	  float* pfAstRange,
	  float* pfAngRange,
	  float* pfPhaseRange
	);
	float mGridSearch
	( float* pfAstRange,
	  float* pfAngRange
	);
	float mRefineDfMean(float* pfDfRange);
	float mRefinePhase(float* pfPhaseRange);
        //-----------------
        float mCorrelate
	( float fAzimu, float fAstig, 
	  float fExtPhase
	);
	void mCalcCtfRes(void);
	//-----------------
        void mGetRange
        ( float fCentVal, float fRange,
          float* pfMinMax, float* pfRange
        );
        //-----------------
        float* m_gfSpect;
        float* m_gfCtf2D;
        int m_aiCmpSize[2];
        GCC2D* m_pGCC2D;
        GCalcCTF2D m_aGCalcCtf2D;
	MD::CCtfParam* m_pCtfParam;
        //-----------------
        float m_fDfMean;
        float m_fAstRatio;
        float m_fAstAngle;
        float m_fExtPhase;
	float m_fCtfRes; // angstrom
        float m_fCCMax;
        //-----------------
        float m_afDfRange[2];
        float m_afAstRange[2];
        float m_afAngRange[2];
        float m_afPhaseRange[2];
};

class CFindCtfBase
{
public:
	CFindCtfBase(void);
	virtual ~CFindCtfBase(void);
	void Clean(void);
	void Setup1(CCtfTheory* pCtfTheory);
	void Setup2(int* piImgSize);
	void SetPhase(float fInitPhase, float fPhaseRange); // degree
	void SetHalfSpect(float* pfCtfSpect);
	float* GetHalfSpect(bool bRaw, bool bToHost);
	void GetSpectSize(int* piSize, bool bHalf);
	void GenHalfSpectrum(float* pfImage);
	float* GenFullSpectrum(void); // clean by caller
	void ShowResult(void);
	//-----------------
	float m_fDfMin;
	float m_fDfMax;
	float m_fAstAng;   // degree
	float m_fExtPhase; // degree
	float m_fScore;
	float m_fCtfRes;
protected:
	void mRemoveBackground(void);
	void mInitPointers(void);
	void mLowpass(void);
	//-----------------
	CCtfTheory* m_pCtfTheory;
	CGenAvgSpectrum* m_pGenAvgSpect;
	float* m_gfFullSpect;
	float* m_gfRawSpect;
	float* m_gfCtfSpect;
	int m_aiCmpSize[2];
	int m_aiImgSize[2];
	float m_afResRange[2];
	float m_afPhaseRange[2]; // for searching extra phase in degree
};

class CFindCtf1D : public CFindCtfBase
{
public:
	CFindCtf1D(void);
	virtual ~CFindCtf1D(void);
	void Clean(void);
	void Setup1(CCtfTheory* pCtfTheory);
	void Do1D(void);
	void Refine1D(float fInitDf, float fDfRange);
protected:
	void mFindDefocus(void);
	void mRefineDefocus(float fDfRange);
	void mCalcRadialAverage(void);
	CFindDefocus1D* m_pFindDefocus1D;
	float* m_gfRadialAvg;
};

class CFindCtf2D : public CFindCtf1D
{
public:
	CFindCtf2D(void);
	virtual ~CFindCtf2D(void);
	void Clean(void);
	void Setup1(CCtfTheory* pCtfTheory);
	void Do2D(void);
	void Refine
	( float afDfMean[2], 
	  float afAstRatio[2],
	  float afAstAngle[2],
	  float afExtPhase[2]
	);
private:
	void mGetResults(void);
	CFindDefocus2D* m_pFindDefocus2D;
};

class CFindCtfHelp
{
public:
	static float CalcAstRatio(float fDfMin, float fDfMax);
	static float CalcDfMin(float fDfMean, float fAstRatio);
	static float CalcDfMax(float fDfMean, float fAstRatio);
};

class CSaveCtfResults
{
public:
	CSaveCtfResults(void);
	~CSaveCtfResults(void);
	static void GenFileName(int iNthGpu, char* pcCtfFile);
	void DoIt(int iNthGpu);
private:
	void mSaveImages(const char* pcCtfFile);
	void mSaveFittings(const char* pcCtfFile);
	int m_iNthGpu;
};

class CLoadCtfResults
{
public:
	CLoadCtfResults(void);
	~CLoadCtfResults(void);
	bool DoIt(int iNthGpu);
	//-----------------
	bool m_bLoaded;
	int m_iNthGpu;
private:
	bool mLoadFittings(const char* pcCtfFile);
};

class CFindCtfMain
{
public:
	CFindCtfMain(void);
	~CFindCtfMain(void);
	static bool bCheckInput(void);
	void Clean(void);
	void DoIt(int iNthGpu);
	//-----------------
	static int m_aiSpectSize[2];
private:
	void mGenSpectrums(void);
	void mDoZeroTilt(void);
	void mDo2D(void);
	void mSaveSpectFile(void);
	float mGetResults(int iTilt);
	char* mGenSpectFileName(void);
	//-----------------
	float** m_ppfHalfSpects;
	CFindCtf2D* m_pFindCtf2D;
	int m_iNumTilts;
	int m_iRefTilt;
	//-----------------
	MD::CTiltSeries* m_pTiltSeries;
	int m_iNthGpu;
};

class CCorrCtfMain
{
public:
	CCorrCtfMain(void);
	~CCorrCtfMain(void);
	void DoIt(int iNthGpu, bool bPhaseFlip);
private:
	void mCorrTiltSeries(int iSeries);
	//-----------------
	CCorrImgCtf* m_pCorrImgCtf;
	int m_iNthGpu;
	bool m_bPhaseFlip;
};

}

namespace MAF = McAreTomo::AreTomo::FindCtf;
