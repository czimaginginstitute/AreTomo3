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
	float* GetShiftXs(void);  // do not free
	float* GetShiftYs(void);  // do not free
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
	void RemoveOffsetX(float fFact); // -1 remove, 1 restore
	void RemoveOffsetZ(float fFact); // -1 remove, 1 restore
	void CalcZInducedShift(int iFrame, float* pfShift);	
	//-----------------
	void CalcRotCenter(void);
	//-----------------
	void MakeRelative(int iRefFrame);
	void ResetShift(void);
	void SortByTilt(void);
	void SortBySecIndex(void);
	void RemoveFrame(int iFrame);
	//-----------------
	CAlignParam* GetCopy(void);
	CAlignParam* GetCopy(int iStartFm, int iNumFms);
	CAlignParam* GetCopy(float fStartTilt, float fEndTilt);
	void Set(CAlignParam* pAlignParam);
	//-----------------
	void LogShift(char* pcLogFile);
	//-----------------
	float m_fAlphaOffset;
	float m_fBetaOffset;
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
	void DoIt(int iNthGpu, float fThreshold);
private:
	void mRemove(float* pfMeans, float* pfStds);
	void mRemoveSeries(int iSeries);
	//-----------------
	int m_iAllFrms;
	int m_iNthGpu;
	float m_fThreshold;
};

class CCalcStackStats
{
public:
	CCalcStackStats(void);
	~CCalcStackStats(void);
	void DoIt( MD::CTiltSeries* pTiltSeries, float* pfStats);
};

//--------------------------------------------------------------------
// 1. m_pfTilts in CDarkFrames should be sorted in ascending order.
// 2. m_piAcqIdxs stores the acquisition index at each tilt angle.
//    This allows us to generate the ordered list needed by Relion4.
// 3. m_piSecIdxs stores the mrc index of each tilt image since tilt
//    images can be ordered in a MRC file according to tilt angle
//    or acquisition index. This allows to save the section indices
//    in .aln file where entries are ordered according to tilt angle.
// 4. IMPORTANT: The tilt series (pSeries) passed into Setup must
//    be sorted by tilt angle first!!! (Done in CProcessThread).
//--------------------------------------------------------------------
class CDarkFrames
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CDarkFrames* GetInstance(int iNthGpu);
	//-----------------
	~CDarkFrames(void);
	void Setup                 // including dark images
	( MD::CTiltSeries* pSeries  // sorted by tilt angles
	);                         // hence frame idx is angle idx
	void AddDark(int iFrmIdx);
	void AddTiltOffset(float fTiltOffset);
	//-----------------
	int GetAcqIdx(int iFrame);
	int GetSecIdx(int iFrame);
	float GetTilt(int iFrame);
	int GetDarkIdx(int iDark); // iDark in [0, m_iNumDarks)
	//-----------------
	bool IsDarkFrame(int iFrame);
	void GenImodExcludeList(char* pcLine, int iSize);
	int m_aiRawStkSize[3];
	int m_iNumDarks;
	int m_iNthGpu;
private:
	CDarkFrames(void);
	void mClean(void);
	int* m_piAcqIdxs; // ordered chronologically in acquisition, or z-value
	int* m_piSecIdxs; // indices of images in MRC files
	float* m_pfTilts; // tilt angles of all images
	bool* m_pbDarkImgs; // flag of dark images
	int* m_piDarkIdxs;
	static CDarkFrames* m_pInstances;
	static int m_iNumGpus;
};

class CSaveAlignFile
{
public:
	CSaveAlignFile(void);
	~CSaveAlignFile(void);
	void DoIt(int iNthGpu);
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
