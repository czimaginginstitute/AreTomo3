#pragma once
#include "../CAreTomoInc.h"
#include "../Util/CUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <queue>

namespace McAreTomo::AreTomo::MrcUtil
{
//-------------------------------------------------------------------
// CAlignParam stores the alignment parameters of all tilt images in
// an input tilt series.
// 1. Since tilt images may not be in order in the input MRC file,
//    CAlignParam maintains two indices: 1) Frame index orders
//    frames according to tilt angles; 2) Section index tracks
//    the order of tilt images in MRC files.
// 2. Dark images are removed after loading the MRC file. There are
//    no entries for the dark frames.
//-------------------------------------------------------------------
class CAlignParam
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CAlignParam* GetInstance(int iNthGpu);
	//-----------------
	CAlignParam(void);
	~CAlignParam(void);
	void Clean(void);
	void Create(int iNumFrames);
	void SetSecIndex(int iFrame, int iSecIdx);
	void SetTilt(int iFrame, float fTilt);
	void SetTiltAxis(int iFrame, float fTiltAxis);
	void SetTiltAxisAll(float fTiltAxis);
	void SetShift(int iFrame, float* pfShift);
	void SetCenter(float fCentX, float fCentY);
	void SetTiltRange(float fEndAng1, float fEndAng2);
	//-----------------
	int GetSecIndex(int iFrame);
	float GetTilt(int iFrame);
	float GetTiltAxis(int iFrame);
	void GetShift(int iFrame, float* pfShift);
	float* GetShiftXs(void) { return m_pfShiftXs; }
	float* GetShiftYs(void) { return m_pfShiftYs; }
	int GetFrameIdxFromTilt(float fTilt);
	float* GetTilts(bool bCopy);
	float GetMinTilt(void);
	float GetMaxTilt(void);
	//-----------------
	void AddAlphaOffset(float fTiltOffset);
	void AddBetaOffset(float fTiltOffset);
	void AddShift(int iFrame, float* pfShift);
	void AddShift(float* pfShift);
	void MultiplyShift(float fFactX, float fFactY);
	//-----------------
	void RotateShift(int iFrame, float fAngle);
	static void RotShift(float* pfInShift,
	  float fRotAngle, float* pfOutShift);
	//-----------------
	void FitRotCenterX(void);
	void FitRotCenterZ(void);
	void GetRotationCenter(float* pfCenter); // x, y, z
	void SetRotationCenterZ(float fCentZ) { m_fZ0 = fCentZ; }
	void RemoveOffsetZ(float fFact); // -1 remove, 1 restore
	void CalcZInducedShift(int iFrame, float* pfShift);	
	//-----------------
	void CalcRotCenter(void);
	//-----------------
	int FindRefFrame(void); // See m_iRefIdx definition below
	int GetRefFrame(void);
	void MakeRelative(int iRefFrame);
	void ResetShift(void);
	void SortByTilt(void);
	void SortBySecIndex(void);
	void RemoveDarkFrames(void);
	//-----------------
	CAlignParam* GetCopy(void);
	CAlignParam* GetCopy(int iStartFm, int iNumFms);
	CAlignParam* GetCopy(float fStartTilt, float fEndTilt);
	void Set(CAlignParam* pAlignParam);
	//-----------------
	bool bZeroBased(void);
	void ToOneBased(void);
	//-----------------
	void LogShift(char* pcLogFile);
	void DisplayShift(void);
	//-----------------
	float m_fAlphaOffset;
	float m_fBetaOffset;
	int m_iThickness; // in raw tilt-series pixel
	int m_iOffsetZ;   // between sample center and volume center
	int m_iRefFrm;    // Reference frame ID which x and y shifts are
			  // closest to the mean shifts in +/- 20 degree
	//-----------------
	int m_iNumFrames;
	int m_iNthGpu;
private:
	void mSwap(int iFrame1, int iFrame2);
	//-----------------
	int* m_piSecIndex;
	float* m_pfTilts;
	float* m_pfTiltAxis;
	float* m_pfShiftXs;
	float* m_pfShiftYs;
	float* m_pfDoses; // accumulated dose
	//-----------------
	float m_afCenter[2];
	float m_afTiltRange[2];
	float m_fX0;
	float m_fY0;
	float m_fZ0;
	static CAlignParam* m_pInstances;
	static int m_iNumGpus;
};

class CLocalAlignParam
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CLocalAlignParam* GetInstance(int iNthGpu);
	~CLocalAlignParam(void);
	void Clean(void);
	void Setup(int iNumTilts, int iNumPatches);
	void GetParam(int iTilt, float* gfAlnParam);
	void GetCoordXYs(int iTilt, float* pfCoordXs, float* pfCoordYs);
	void SetCoordXY(int iTilt, int iPatch, float fX, float fY);
	void SetShift(int iTilt, int iPatch, float fSx, float fSy);
	void SetBad(int iTilt, int iPatch, bool bBad);	
	//-----------------
	void GetCoordXY(int iTilt, int iPatch, float* pfCoord);
	void GetShift(int iTilt, int iPatch, float* pfShift);
	float GetGood(int iTilt, int iPatch);
	float GetBadPercentage(float fMaxTilt);
	//-----------------
	float* m_pfCoordXs;
	float* m_pfCoordYs;
	float* m_pfShiftXs;
	float* m_pfShiftYs;
	float* m_pfGoodShifts;
	int m_iNumTilts;
	int m_iNumPatches;
	int m_iNumParams; // x,y,sx,sy,bad per tilt
private:
	CLocalAlignParam(void);
	int mGetNumBads(int iTilt);
	//-----------------
	int m_iNthGpu;
	static CLocalAlignParam* m_pInstances;
	static int m_iNumGpus;
};

class CPatchShifts
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CPatchShifts* GetInstance(int iNthGpu);
	//-----------------
	~CPatchShifts(void);
	void Clean(void);
	void Setup(int iNumPatches, int iNumTilts);
	void SetRawShift(int iPatch, CAlignParam* pPatAlnParam);
	void GetPatCenterXYs(float* pfCentXs, float* pfCentYs);
	void GetPatShifts(float* pfShiftXs, float* pfShiftYs);
	void RotPatCenterXYs(float fRot, float* pfCentXs, float* pfCentYs);
	//-----------------------------------------------------------------
	CAlignParam* GetAlignParam(int iPatch); // do not free
	void GetShift(int iPatch, int iTilt, float* pfShift);
	float GetTiltAxis(int iPatch, int iTilt);
	void GetRotCenter(int iPatch, float* pfRotCent); // x, y, z
	void SetRotCenterZ(int iPatch, float fCentZ);
	//-------------------------------------------
	int m_iNumPatches;
	int m_iNumTilts;
	bool* m_pbBadShifts;
	int m_iNthGpu;
private:
	CPatchShifts(void);
	CAlignParam** m_ppPatAlnParams;
	int m_iZeroTilt;
	static CPatchShifts* m_pInstances;
	static int m_iNumGpus;
};

class CRemoveDarkFrames
{
public:
	CRemoveDarkFrames(void);
	~CRemoveDarkFrames(void);
	void Setup(int iNthGpu);
	void Detect(float fThreshold);
	void Remove(void);
	int m_iNthGpu;
private:
	void mDetect(void);
	void mRemove(float* pfMeans, float* pfStds);
	void mRemoveSeries(int iSeries);
	void mCalcStats(void);
	//-----------------
	int m_iAllFrms;
	float m_fThreshold;
	float* m_pfMeans;
	float* m_pfStds;
};

//-------------------------------------------------------------------
// GBinBandpass2D: Fourier-bins a host image and applies a Gaussian
// bandpass. Output is a host image of the binned size.
//-------------------------------------------------------------------
class GBinBandpass2D
{
public:
	GBinBandpass2D(void);
	~GBinBandpass2D(void);
	void Clean(void);
	void Setup
	( int* piRawSize, int* piBinSize,
	  float fLowSigma,  // cycles / binned pixel
	  float fHighSigma  // cycles / binned pixel
	);
	void DoIt(float* pfRawImg, float* pfBinImg);
private:
	int m_aiRawSize[2];
	int m_aiBinSize[2];
	float m_fLowSigma;
	float m_fHighSigma;
	float* m_gfRawPad;
	cufftComplex* m_gBinCmp;
	MU::CCufft2D* m_pForwardFFT;
	MU::CCufft2D* m_pInverseFFT;
};

//-------------------------------------------------------------------
// GMaskedNCC2D: overlap-normalized cross correlation (Padfield,
// IEEE TIP 2012) of two images that are zero outside a common
// rectangular mask. The FFT size is twice the image size so that
// there is no wrap-around and every shift is scored only on the
// pixels where both images overlap. Shifts whose overlap is less
// than fMinOverlap of the mask area are excluded.
//-------------------------------------------------------------------
class GMaskedNCC2D
{
public:
	GMaskedNCC2D(void);
	~GMaskedNCC2D(void);
	void Clean(void);
	void Setup
	( int* piImgSize, int* piMaskStart, int* piMaskSize,
	  float fMinOverlap
	);
	void SetRef(float* pfRefImg);  // host image
	//-----------------------------------------------
	// Returns peak NCC. pfShift is the position of
	// pfImg relative to the reference in pixels:
	// pfImg(x + shift) ~ pfRefImg(x).
	//-----------------------------------------------
	float DoIt(float* pfImg, float* pfShift);
private:
	void mLoad(float* pfImg, float* gfPad);
	void mSquare(float* gfIn, float* gfOut);
	void mConjMul
	( cufftComplex* gA, cufftComplex* gB,
	  cufftComplex* gOut
	);
	float mSearchPeak(float* pfShift);
	//-----------------
	int m_aiImgSize[2];
	int m_aiFftSize[2];
	int m_aiCmpSize[2];
	int m_iPadX;
	float m_fMinOverlap;
	float m_fMaxO;
	cufftComplex* m_gBuf;  // all GPU buffers below
	cufftComplex *m_gFm, *m_gFr, *m_gFr2, *m_gS1, *m_gQ1;
	cufftComplex *m_gO, *m_gP, *m_gS2, *m_gQ2;
	float* m_pfNCC;
	MU::CCufft2D* m_pForwardFFT;
	MU::CCufft2D* m_pInverseFFT;
};

//-------------------------------------------------------------------
// CDetectBadShifts: in a given (low) tilt range, find tilt images whose
// shifts relative to their neighbors are too large or cannot be
// measured consistently. Flagged images are added to CBadImages
// and are removed together with the dark images.
// 1. All pairs of low-tilt images are cross-correlated with the
//    overlap-normalized NCC. No stretching is applied since the
//    stretch factor is at most 1/cos(20) = 1.06.
// 2. Absolute shifts are solved by robust (Huber then Tukey IRLS) weighted
//    least squares. No image is the reference (the zero-tilt image
//    is often the bad one); the mean shift is set to zero.
// 3. Each image is tested by leave-one-out against a smooth
//    (quadratic in tilt) trend fit to the other images.
// 4. Must be called after dark detection and before removal, on
//    the full tilt series sorted by tilt angle.
//-------------------------------------------------------------------
class CDetectBadShifts
{
public:
	CDetectBadShifts(void);
	~CDetectBadShifts(void);
	//-----------------------------------------------
	// pfTiltRange: [min, max] tilt in degrees to be
	// searched. The search is skipped if max <= min.
	//-----------------------------------------------
	void DoIt(int iNthGpu, float* pfTiltRange);
private:
	void mClean(void);
	bool mFindLowTilts(void);
	void mBinImages(void);
	void mMeasurePairs(void);
	void mSolveShifts(void);
	void mTestTrend(void);
	void mFlagImages(void);
	void mFitTrend
	( float* pfWeights, int iSkip,
	  double* pdCoeffX, double* pdCoeffY
	);
	float mCalcTrend(double* pdCoeff, float fTilt);
	float mNormTilt(float fTilt);
	//-----------------
	int m_iNthGpu;
	MD::CTiltSeries* m_pTiltSeries;
	float m_afTiltRange[2];
	int m_iNumImgs;      // number of low-tilt images
	int* m_piFrmIdxs;    // frame indices in the full series
	float* m_pfTilts;
	//-----------------
	int m_aiBinSize[2];
	float m_afBinning[2];
	int m_aiMaskStart[2];
	int m_aiMaskSize[2];
	float** m_ppfBinImgs;
	//-----------------
	float* m_pfPairShifts; // [N*N*2] pos of j rel. to i, raw px
	float* m_pfPairNCCs;   // [N*N]
	float* m_pfPairWgts;   // [N*N] robust weights
	float* m_pfShifts;     // [N*2] solved positions, raw px
	float* m_pfInliers;    // [N] fraction of consistent pairs
	float* m_pfLooErrs;    // [N*2] leave-one-out trend error x, y
	float m_fTrendSigma;
	bool* m_pbBadImgs;
};

class CCalcStackStats
{
public:
	CCalcStackStats(void);
	~CCalcStackStats(void);
	void DoIt( MD::CTiltSeries* pTiltSeries, float* pfStats);
};

//--------------------------------------------------------------------
// 1. m_pfTilts in CBadImages should be sorted in ascending order.
// 2. m_piAcqIdxs stores the acquisition index at each tilt angle.
//    This allows us to generate the ordered list needed by Relion4.
// 3. m_piSecIdxs stores the mrc index of each tilt image since tilt
//    images can be ordered in a MRC file according to tilt angle
//    or acquisition index. This allows to save the section indices
//    in .aln file where entries are ordered according to tilt angle.
// 4. IMPORTANT: The tilt series (pSeries) passed into Setup must
//    be sorted by tilt angle first!!! (Done in CProcessThread).
//--------------------------------------------------------------------
class CBadImages
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CBadImages* GetInstance(int iNthGpu);
	//-----------------
	~CBadImages(void);
	void Setup                 // including dark images
	( MD::CTiltSeries* pSeries  // sorted by tilt angles
	);                         // hence frame idx is angle idx
	void Setup(int iNthGpu);
	//-----------------
	void AddBadImg(int iFrmIdx, int iSecIdx, float fTilt);
	void AddTiltOffset(float fTiltOffset); 
	//-----------------
	int GetAcqIdx(int iFrame);
	int GetSecIdx(int iFrame);
	float GetTilt(int iFrame);
	//-----------------
	int GetDarkIdx(int iDark); // iDark in [0, m_iNumBads)
	int GetDarkSec(int iDark);
	float GetDarkTilt(int iDark);
	int GetNumAlnTilts(void);
	//-----------------
	bool IsDarkImg(int iFrame);
	void GenImodExcludeList(char* pcLine, int iSize);
	//-----------------
	bool bZeroBased(void);
	void ToOneBased(void);
	//-----------------
	int* m_piDarkIdxs;
	int* m_piDarkSecs;
	float* m_pfDarkTilts;
	//-----------------
	int m_aiRawStkSize[3];
	int m_iNumBads;
	int m_iNthGpu;
private:
	CBadImages(void);
	void mClean(void);
	int* m_piAcqIdxs; // ordered chronologically in acquisition, or z-value
	int* m_piSecIdxs; // indices of images in MRC files
	float* m_pfTilts; // tilt angles of all images
	bool* m_pbDarkImgs; // flag of dark images
	static CBadImages* m_pInstances;
	static int m_iNumGpus;
};

class CSaveAlignFile
{
public:
	CSaveAlignFile(void);
	~CSaveAlignFile(void);
	void DoIt(int iNthGpu);
	//-----------------
	static void GenFileName
	( int iNthGpu,
	  bool bSave, // false for load 
	  char* pcAlnFile
	);
	//-----------------
	static char m_acRawSizeTag[32];
	static char m_acNumPatchesTag[32];
	static char m_acDarkFrameTag[32];
	static char m_acAlphaOffsetTag[32];
	static char m_acBetaOffsetTag[32];
	static char m_acThicknessTag[32];
	static char m_acOffsetZTag[32];
	static char m_acLocalAlignTag[32];
private:
	void mSaveHeader(void);
	void mSaveGlobal(void);
	void mSaveLocal(void);
	void mCloseFile(void);
	//-----------------
	CAlignParam* m_pAlignParam;
	CLocalAlignParam* m_pLocalParam;
	//-----------------
	FILE* m_pFile;
	int m_iNumTilts;
	int m_iNumPatches;
	int m_iNthGpu;
};

class CLoadAlignFile
{
public:
	CLoadAlignFile(void);
	~CLoadAlignFile(void);
	bool DoIt(int iNthGpu);
	//-----------------
	bool m_bLoaded;
	int m_iNthGpu;
private:
	bool mParseHeader(void);
	bool mParseRawSize(char* pcLine);
	bool mParseDarkFrame(char* pcLine);
	bool mParseNumPatches(char* pcLine);
	bool mParseAlphaOffset(char* pcLine);
	bool mParseBetaOffset(char* pcLine);
	bool mParseThickness(char* pcLine);
	//-----------------
	void mLoadGlobal(void);
	void mLoadLocal(void);
	void mClean(void);
	//-----------------
	void mToOneBased(void);
	//-----------------
	int m_aiRawSize[3];
	int m_iNumPatches;
	float m_fAlphaOffset;
	float m_fBetaOffset;
	int m_iThickness;
	//-----------------
	std::queue<char*> m_aHeaderQueue;
	std::queue<char*> m_aDataQueue;
};

class CSaveStack
{
public:
	CSaveStack(void);
	~CSaveStack(void);
	bool OpenFile(char* pcMrcFile);
	void DoIt
	( MD::CTiltSeries* pTiltSeries,
	  CAlignParam* pAlignParam,
	  float fPixelSize,
	  float* pfStats,
	  bool bVolume
	);
private:
	void mDrawTiltAxis(float* pfImg, int* piSize, float fTiltAxis);
	Mrc::CSaveMrc m_aSaveMrc;
	char m_acMrcFile[256];
};

class GExtractPatch
{
public:
	GExtractPatch(void);
	~GExtractPatch(void);
	void SetStack
	( MD::CTiltSeries* pTiltSeries,
	  CAlignParam* pAlignParam
	);
	MD::CTiltSeries* DoIt
	( int* piStart, // 3 elements
	  int* piSize   // 3 elements
	);
private:
	void mExtract(void);
	void mCalcCenter
	( float* pfCent0,
	  float fTilt,
	  float fTiltAxis,
	  float* pfCent
	);
	void mExtractProj
	( int iProj,
	  float* pfCent,
	  float* pfPatch
	);
	int m_aiStart[3];
	int m_aiSize[3];
	MD::CTiltSeries* m_pTiltSeries;
	CAlignParam* m_pAlignParam;
	MD::CTiltSeries* m_pPatSeries;
};

class CCropVolume
{
public:
	CCropVolume(void);
	~CCropVolume(void);
	void Clean(void);
	MD::CTiltSeries* DoIt
	( MD::CTiltSeries* pInVol, float fOutBin,
	  CAlignParam* pFullParam,
	  CLocalAlignParam* pLocalParam,
	  int* piOutSizeXY
	);
private:
	void mCalcOutCenter(void);
	void mCreateOutVol(void);
	void mCalcOutVol(void);	
	MD::CTiltSeries* m_pInVol; // must be xzy x is fastest y slowest
	CAlignParam* m_pFullParam;
	CLocalAlignParam* m_pLocalParam;
	float m_fOutBin;
	int m_aiOutSize[2];
	int m_aiOutCent[2];
	MD::CTiltSeries* m_pOutVol;
};

class CMuInstances
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
};
}

namespace MAM = McAreTomo::AreTomo::MrcUtil;
