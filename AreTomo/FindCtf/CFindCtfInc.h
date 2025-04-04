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

class GRmSpikes
{
public:
	GRmSpikes(void);
	~GRmSpikes(void);
	void DoIt(float* gfSpect, int* piSpectSize, int iBoxSize);
};

class GRadialAvg
{
public:
	GRadialAvg(void);
	~GRadialAvg(void);
	void DoIt(float* gfSpect, float* gfAverage, int* piCmpSize);
};

class GExtractTile
{
public:
	GExtractTile(void);
	~GExtractTile(void);
	void SetImg(float* gfImg, int* piImgSize, bool bPadded);
	void SetTileSize(int* piTileSize, bool bPadded);
	void DoIt(float* gfTile, int* piStart, cudaStream_t stream = 0);
private:
	float* m_gfImg;
	int m_aiImgSize[2];
	int m_aiTileSize[2];
	int m_iImgPadX;
	int m_iTilePadX;
};

class GRoundEdge
{
public:
	GRoundEdge(void);
	~GRoundEdge(void);
	void SetMask(float* pfCent, float* pfSize);
	void DoIt
	( float* gfImg,
	  int* piImgSize,
	  bool bKeepCenter
	);

private:
	float m_afMaskCent[2];
	float m_afMaskSize[2];
};

class GScaleSpect2D
{
public:
	GScaleSpect2D(void);
	~GScaleSpect2D(void);
	void Clean(void);
	//-----------------
	void DoIt
	( float* gfInSpect, float* gfOutSpect,
	  float fScale, int* piSpectSize,
	  cudaStream_t stream = 0
	);
	//-----------------
	void Setup(int* piSpectSize);
	void DoIt
	( float* pfInSpect, 
	  float* gfOutSpect,
	  float fScale
	);
private:
	float* m_gfInSpect;
	int m_aiSpectSize[2];
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
	void SetLowpass(int iBfactor);
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
	float m_fBFactor;
};

class CTile
{
public:
	CTile(void);
	virtual ~CTile(void);
	void Clean(void);
	void SetSize(int* piTileSize);
	void SetCentX(float fCentX) { m_afCenter[0] = fCentX; }
	void SetCentY(float fCentY) { m_afCenter[1] = fCentY; }
	void SetCentZ(float fCentZ) { m_afCenter[2] = fCentZ; }
	void SetTilt(float fTilt) { m_fTilt = fTilt; }
	void SetPixSize(float fPixSize) { m_fPixSize = fPixSize; }
	void SetGood(bool bGood) { m_bGood = bGood; }
	//-----------------
	int* GetSize(void) { return m_aiTileSize; }
	int GetSizeX(void) { return m_aiTileSize[0]; }
	int GetSizeY(void) { return m_aiTileSize[1]; }
	int GetPixels(void) { return m_aiTileSize[0] * m_aiTileSize[1]; };
	//-----------------
	float* GetCenter(void) { return m_afCenter; }
	float GetCentX(void) { return m_afCenter[0]; }
	float GetCentY(void) { return m_afCenter[1]; }
	float GetCentZ(void) { return m_afCenter[2]; }
	//-----------------
	float* GetTile(void)   { return m_qfTile; }
	//-----------------
	float GetTilt(void) { return m_fTilt; }
	float GetPixSize(void) { return m_fPixSize; }
	bool IsGood(void) { return m_bGood; }
protected:
	float* m_qfTile;
	int m_aiTileSize[2];
	float m_afCenter[3];
	float m_fTilt;
	float m_fPixSize;
	bool m_bGood;
};

class CCoreTile : public CTile
{
public:
	CCoreTile(void);
	virtual ~CCoreTile(void);
	void SetSize(int iTileSize, int iCoreSize);
	void SetCoreStart(int iCoreStartX, int iCoreStartY);
	//-----------------
	void PasteCore(float* pfImage, int* piImgSize);
	//-----------------
	void GetCoreCenter(float* pfCent);
	void GetCoreCenterInTile(float* pfCent);
	//-----------------
	void GetTileStart(int* piStart);
	void GetCoreStart(int* piStart);
	//-----------------
	int GetCoreSize(void) { return m_iCoreSize; }
	//-----------------
	int GetTileBytes(void);
private:
	int m_aiTileStart[2];
	int m_aiCoreStart[2];
	int m_iCoreSize; // unpadded size
};

class CTsTiles
{
public:
	static CTsTiles* GetInstance(int iNthGpu);
	static void DeleteInstance(int iNthGpu);
	static void DeleteAll(void);
	//-----------------
	~CTsTiles(void);
	void Clean(void);
	void Generate(int iTileSize);
	//-----------------
	CTile* GetTile(int iTile);
	CTile* GetTile(int iTilt, int iImgTile);
	int GetTileSize(void) { return m_iTileSize; }
	//-----------------
	int GetAllTiles(void);
	int GetImgTiles(void);
	//-----------------
	int GetCentX(int iTilt, int iImgTile);
	int GetCentY(int iTilt, int iImgTile);
	int GetCentZ(int iTilt, int iImgTile);
	void GetCent(int iTilt, int iImgTile, int* piCentXYZ);
	void SetCent(int iTilt, int iImgTile, int* piCentXYZ);
	//-----------------
	void GetImgCent(int* piImgCent);
	int GetNumTilts(void) { return m_iNumTilts; }
	float GetTilt(int iTilt);
	int GetTiltIdx(float fTilt);
private:
	CTsTiles(void);
	void mSetSize(void);
	void mCalcBinning(void);
	void mDoTilt(int iTilt);
	void mDoBinning(int iTilt, float* pfBinnedImg);
	float mGenTileSpect(int iTilt, int iTile);
	void mExtractPadTile(int iTilt, int iTile, float* pfImg);
	//-----------------
	int m_iNthGpu;
	CTile* m_pTiles;
	float* m_gfTileSpect;
	float* m_gfPadTile;
	MU::GCalcMoment2D* m_pGCalcMoment2D;
	GCalcSpectrum* m_pGCalcSpectrum;	
	//-----------------
	int m_aiImgSize[2];
	int m_iNumTilts;
	int m_aiImgTiles[2];
	//-----------------
	int m_iTileSize;
	int m_iOverlap;
	float m_fOverlap;
	int m_aiOffset[2];
	float m_fPixSize;
	float m_fBinning;
	//-----------------
	static CTsTiles* m_pInstances[64];
};

class CExtractTiles
{
public:
	CExtractTiles(void);
	~CExtractTiles(void);
	void Setup(int iTileSize, int iCoreSize, int* piImgSize);
	void DoIt(float* pfImage);
	CCoreTile* GetTile(int iNthTile);
	int m_iNumTiles;
private:
	void mDoIt(float* gfImg, int iTile);
	void mCalcTileLocations(void);
	void mClean(void);
	//-----------------
	int m_aiNumTiles[2];
	int m_aiImgSize[2];
	int m_iTileSize;
	int m_iCoreSize;
	CCoreTile* m_pTiles;
};

class CTiltInducedZ
{
public:
	CTiltInducedZ(void);
	~CTiltInducedZ(void);
	void Setup
	( float fTilt, float fTiltAxis, // degree
	  float fTilt0, float fBeta0
	);
	float DoIt(float fDeltaX, float fDeltaY);
private:
	float m_fTanAlpha;
	float m_fCosTilt;
	float m_fTanBeta;
	float m_fCosTiltAxis;
	float m_fSinTiltAxis;
};

class CCorrImgCtf
{
public:
	CCorrImgCtf(void);
	~CCorrImgCtf(void);
	void Setup(int* piImgSize, int iNthGpu);
	void SetLowpass(int iBFactor);
	void DoIt
	( float* pfImage, 
	  float fTilt, float fTiltAxis,
	  float fAlpha0, float fBeta0,
	  bool bPhaseFlip
	);
private:
	void mTileToGpu(int iTile);
	void mGpuToTile(int iTile);
	//-----------------
	void mRoundEdge(int iTile);
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
	float m_fAlpha0;
	float m_fBeta0;
	int m_iDfHand;
	int m_iBFactor;
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
	void Create(int* piStkSize, int iTileSize);
	void Extract(void);
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
	void SetTiltOffsets(float fTiltOffset, float fBetaOffset);
	void DoIt
	( int iTilt,
	  float fTiltAxis,
	  float fCentDF,
	  int iHandedness, // 1 or -1
	  float* gfAvgSpect,
	  int iNthGpu
	);
private:
	void mDoNoScaling(void);
	void mDoScaling(void);
	void mScaleTile(int iTile, float* gfScaled);
	void mCalcTileCentZs(void);
	//-----------------
	int m_iTilt;
	float m_fTiltOffset;
	float m_fBetaOffset;
	float m_fTiltAxis;
	float m_fCentDF;
	int m_iHandedness;
	float* m_gfAvgSpect;
	int m_iNthGpu;
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
	  float fPhaseRange
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
	float mFindAstig
	( float* pfAstRange,
	  float* pfAngRange
	);
	float mRefineAstMag(float fAstRange);
	float mRefineAstAng(float fAngRange);
	float mRefineDfMean(float fDfRange);
	float mRefinePhase(float fPhaseRange);
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
	void SetGpu(int iNthGpu) { m_iNthGpu = iNthGpu; }
	void Setup1(CCtfTheory* pCtfTheory);
	void SetPhase(float fInitPhase, float fPhaseRange); // degree
	void SetDefocus(float fInitDF, float fDfRange);     // angstrom
	void SetHalfSpect(float* pfCtfSpect);
	//-----------------
	float* GetHalfSpect(bool bRaw, bool bToHost);
	void GetSpectSize(int* piSize, bool bHalf);
	//-----------------
	void GenHalfSpectrum(int iTilt, float fTiltOffset, 
	   float fBetaOffset);
	void GenFullSpectrum(float* pfFullSpect);
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
	void mLowpass(void);
	void mHighpass(void);
	//-----------------
	CCtfTheory* m_pCtfTheory;
	float* m_gfFullSpect;
	float* m_gfRawSpect;
	float* m_gfCtfSpect;
	int m_aiCmpSize[2];
	float m_afResRange[2];
	float m_afPhaseRange[2]; // for searching extra phase in degree
	float m_afDfRange[2];    // min and max defocus in angstrom 
	int m_iNthGpu;
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
	void mRefinePhase(float fPhaseRange);
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
	static void GenFileName(int iNthGpu, bool bInDir, char* pcCtfFile);
	void DoIt(int iNthGpu);
	void DoFittings(int iNthGpu);
private:
	void mSaveImages(const char* pcCtfFile);
	void mSaveFittings(const char* pcCtfFile);
	void mSaveImod(const char* pcCtfFile);
	int m_iNthGpu;
};

class CLoadCtfResults
{
public:
	CLoadCtfResults(void);
	~CLoadCtfResults(void);
	bool DoIt(int iNthGpu, bool bInputDir);
	//-----------------
	bool m_bLoaded;
	int m_iNthGpu;
private:
	bool mLoadFittings(const char* pcCtfFile);
};

class CAlignCtfResults
{
public:
	CAlignCtfResults(void);
	~CAlignCtfResults(void);
	void DoIt(int iNthGpu);
private:
	void mAlignCtf(int iImage);
	int m_iNthGpu;
};

class CFindCtfMain
{
public:
	CFindCtfMain(void);
	~CFindCtfMain(void);
	//-----------------
	static bool bCheckInput(void);
	void Clean(void);
	void DoIt(int iNthGpu);
protected:
	void mInit(bool bRefine);
	void mGenAvgSpects
	( float fTiltOffset, 
	  float fBetaOffset, 
	  float fMaxTilt
	);
	void mCleanSpects(void);
	//-----------------
	void mSaveSpectFile(void);
	float mGetResults(int iTilt);
	//-----------------
	float** m_ppfHalfSpects;
	CFindCtf2D* m_pFindCtf2D;
	int m_iNumTilts;
	MD::CCtfResults* m_pBestCtfRes;
        //-----------------
        int m_iNthGpu;
private:
	void mDoLowTilts(void);
	void mDoHighTilts(void);
	//-----------------
	float m_fLowTilt;
	float m_fDfMean;
	float m_fDfStd;
};

class CRefineCtfMain : public CFindCtfMain
{
public:
	CRefineCtfMain(void);
	virtual ~CRefineCtfMain(void);
	//---------------------------
	void Clean(void);
	void DoIt(int iNthGpu);
private:
	void mFindHandedness(void);
	void mRefineOffset(float fStep, int iNumSteps, bool bBeta);
	float mRefineCTF(int iKind);
	//---------------------------
	float m_fTiltOffset;
	float m_fBetaOffset;
	//---------------------------
	float m_fBestScore;
	MD::CCtfResults* m_pBestCtfRes;
	float m_fLowTilt;
};

class CCorrCtfMain
{
public:
	CCorrCtfMain(void);
	~CCorrCtfMain(void);
	void DoIt(int iNthGpu, bool bPhaseFlip, int iLowpass);
private:
	void mCorrTiltSeries(int iSeries);
	//-----------------
	CCorrImgCtf* m_pCorrImgCtf;
	int m_iNthGpu;
	bool m_bPhaseFlip;
};

}

namespace MAF = McAreTomo::AreTomo::FindCtf;
