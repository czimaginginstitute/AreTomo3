#pragma once

namespace McAreTomo::MotionCor::DataUtil
{
class CEntry
{
public:
	CEntry(void);
        ~CEntry(void);
        void Set(int iGroupSize, int iIntSize, float fFmDose);
        int m_iGroupSize;  // number of raw frame in this group
        int m_iIntSize;    // num of raw frames to be integrated
        float m_fFmDose;   // raw frame dose e/A2
};


class CReadFmIntFile
{
public:
	static CReadFmIntFile* GetInstance(void);
	static void DeleteInstance(void);
	~CReadFmIntFile(void);
	bool HasDose(void);
	bool NeedIntegrate(void);
	int GetGroupSize(int iEntry);
	int GetIntSize(int iEntry);
	float GetDose(int iEntry);
	void DoIt(void);
	int m_iNumEntries;
private:
	CReadFmIntFile(void);
	void mClean(void);
	CEntry** m_ppEntries;
	static CReadFmIntFile* m_pInstance;
};

class CFmIntParam
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CFmIntParam* GetInstance(int iNthGpu);
        ~CFmIntParam(void);
	bool bDoseWeight(void);
	bool bIntegrate(void);
	//-----------------
        void Setup(int iNumRawFms, int iMrcMode);// All frames in input movie
        int GetIntFmStart(int iIntFrame);
        int GetIntFmSize(int iIntFrame);
        int GetNumIntFrames(void);
        float GetAccruedDose(int iIntFrame);
	float GetTotalDose(void);
	//-----------------
	int m_iNthGpu;
        int m_iNumIntFms;
        float* m_pfIntFmDose;  // Dose within int. frame exposure
        float* m_pfAccFmDose;  // Acumulated dose of each int. frame
	float* m_pfIntFmCents; // Centers of int. frames for shift
	                       // interpolation.
private:
	CFmIntParam(void);
        void mSetup(void);
        void mSetupFile(void);    // FmIntFile has variable dose, intSize = 1
	void mSetupFileInt(void); // FmIntFile has fixed dose, intSize > 1
        void mClean(void);
        void mAllocate(void);
	void mCalcIntFmCenters(void);
        int* m_piIntFmStart;
        int* m_piIntFmSize;
        int m_iNumRawFms;   // All frames in the input movie file
        int m_iMrcMode;
	//-----------------
	static CFmIntParam* m_pInstances;
	static int m_iNumGpus; 
};

class CFmGroupParam
{
public:
        static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CFmGroupParam* GetInstance(int iNthGpu, bool bLocal);
	//-----------------
	~CFmGroupParam(void);
	void Setup(int iGroupSize);
        int GetGroupStart(int iGroup);
        int GetGroupSize(int iGroup);
        float GetGroupCenter(int iGroup);
	//-----------------
        int m_iNumGroups;
        int m_iNumIntFms;
	bool m_bGrouping;
	int m_iGroupSize;
	int m_iNthGpu;
private:
	CFmGroupParam(void);
        void mGroupByRawSize(void);
        void mGroupByDose(void);
        void mClean(void);
        void mAllocate(void);
        int* m_piGroupStart;
        int* m_piGroupSize;
        float* m_pfGroupCenters;
	//-----------------
	static CFmGroupParam* m_pInstances;
	static int m_iNumGpus;
};

class CStackShift
{
public:
	CStackShift(void);
	~CStackShift(void);
	void Setup(int iNumFrames);
	void SetCenter	 // region where measurement is performed
	( int* piStart, // 2 elements
	  int* piSize	 // 2 elements, (0, 0) is full size
	);
	void SetCenter(float fCentX, float fCentY);
	void Clear(void);
	void Reset(void);
	void SetShift(int iFrame, float* pfShift);
	void SetShift(CStackShift* pSrcShift);
	void AddShift(int iFrame, float* pfShift);
	void AddShift(CStackShift* pIncShift);
	//-----------------
	void MakeRelative(int iRefFrame);
	void Multiply(float fFactX, float fFactY);
	void TruncateDecimal(void);
	//-----------------
	void GetShift(int iFrame, float* pfShift, float fFact=1.0f);
	void GetRelativeShift(int iFrame, float* pfShift, int iRefFrame);
	float* GetShifts(void);  // do not free
	CStackShift* GetCopy(void);
	void GetCenter(float* pfLoc);
	int GetCentralFrame(void);
	void RemoveSpikes(bool bSingle);
	void DisplayShift(const char* pcHeader, int iRow=-1);
	void Smooth(float fWeight);
	//-----------------
	int m_iNumFrames;
	float m_afCenter[3];
	bool m_bConverged;
private:
	float* m_pfShiftXs;
	float* m_pfShiftYs;
};	//CStackShift

class CPatchShifts
{
public:
	CPatchShifts(void);
	~CPatchShifts(void);
	void Setup
	( int iNumPatches, // number of patches
	  int* piFullSize  // full stack size 3 elements
	);
	void Setup
	( int iPatchesX,   // number of patches in x axis
	  int iPatchesY,   // number of patches in y axis
	  int* piFullSize  // full stack size 3 elements
	);
	void SetFullShift
	( CStackShift* pFullShift
	);
	void SetRawShift  // buffer pStackShift
	( CStackShift* pStackShift,
	  int iPatch
	);
	void GetLocalShift
	( int iFrame,
	  int iPatch,
	  float* pfShift
	);
	void GetPatchCenter
	( int iPatch,
	  float* pfCenter
	);
	void CalcShiftSigma
	( int iFrame,
	  float* pfSigmaXY
	);
	void LogFullShifts
	( char* pcLogFile
	);
	void LogPatchShifts // show side by side raw and fit shifts
	( char* pcLogFile	    // of each patch
	);
	void LogFrameShifts // show side by side raw and fit shifts
	( char* pcLogFile      // of each frame
	);
	void CopyCentersToGpu
	( float* gfPatCenters
	);
	//-----------------
	void CopyShiftsToGpu(float* gfPatShifts);
	void CopyFlagsToGpu(bool* gbBadShifts);
	void MakeRelative(void);
	void DetectBads(void);
	//-----------------
	CStackShift* m_pFullShift; // shifts of full frames
	int m_iNumPatches;
	int m_aiFullSize[3];
private:
	void mClean(void);
	void mDetectBadOnFrame(int iFrame);
	void mCalcMeanStd(int iFrame, float* pfMeanStd);
	float mCalcLocalRms(int iFrame, int iPatch);
	float* m_pfPatCenters;
	float* m_pfPatShifts;
	bool* m_pbBadShifts;
};	//CPatchShifts
}

namespace MMD = McAreTomo::MotionCor::DataUtil;
