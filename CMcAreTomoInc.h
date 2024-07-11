#pragma once
#include "MaUtil/CMaUtilInc.h"
#include "DataUtil/CDataUtilInc.h"
#include <Util/Util_Thread.h>
#include <unordered_map>
#include <string>
#include <stdio.h>
#include <cufft.h>

namespace MD = McAreTomo::DataUtil;

namespace McAreTomo 
{
class CInput
{
public:
	static CInput* GetInstance(void);
	static void DeleteInstance(void);
	~CInput(void);
	void ShowTags(void);
	void Parse(int argc, char* argv[]);
	//-----------------
	char m_acInMdoc[256];
	char m_acInSuffix[256];
	char m_acInSkips[256];
	char m_acTmpFile[256];
	char m_acLogDir[256];
	char m_acOutDir[256];
	char m_acInDir[256]; // extract from m_acInMdoc
	//-----------------
	int* m_piGpuIDs;
	int m_iNumGpus;
	//-----------------
	int m_iKv;
	float m_fCs;
	float m_fPixSize;
	float m_fFmDose;
	//-----------------
	int m_iCmd;
	int m_iResume;
	int m_iSerial;
	//-----------------
	char m_acInMdocTag[32];
	char m_acInSuffixTag[32];
	char m_acInSkipsTag[32];
	char m_acTmpFileTag[32];
	char m_acLogDirTag[32];
	char m_acOutDirTag[32];
	//-----------------
	char m_acGpuIDTag[32];
	//-----------------
	char m_acKvTag[32];
	char m_acCsTag[32];
	char m_acPixSizeTag[32];
	char m_acFmDoseTag[32];
	//-----------------
	char m_acCmdTag[32];
	char m_acResumeTag[32];
	char m_acSerialTag[32];
private:
        CInput(void);
	void mExtractInDir(void);
	void mAddEndSlash(char* pcDir);
        void mPrint(void);
	int m_argc;
        char** m_argv;
        static CInput* m_pInstance;
};

class CMcInput
{
public:
	static CMcInput* GetInstance(void);
	static void DeleteInstance(void);
	~CMcInput(void);
	void ShowTags(void);
	void Parse(int argc, char* argv[]);
	void GetBinnedSize(int* piImgSize, int* piBinnedSize);
	float GetFinalPixelSize(void); // after binning & mag correction
	bool bLocalAlign(void);
	//-----------------
	char m_acGainFile[256];
	char m_acDarkMrc[256];
	char m_acDefectFile[256];
	char m_acFmIntFile[256];
	int m_aiNumPatches[3];
	int m_iMcIter;
	float m_fMcTol;
	float m_fMcBin;
	int m_aiGroup[2];
	int m_iFmRef;
	int m_iRotGain;
	int m_iFlipGain;
	int m_iInvGain;
	float m_afMag[3];
	int m_iInFmMotion;
	int m_iEerSampling;
	int m_iTiffOrder;
	int m_iCorrInterp;
	//-----------------
	char m_acGainFileTag[32];
	char m_acDarkMrcTag[32];
	char m_acDefectFileTag[32];
	char m_acFmIntFileTag[32];
	char m_acPatchesTag[32];
	char m_acIterTag[32];
	char m_acTolTag[32];
	char m_acMcBinTag[32];
	char m_acGroupTag[32];
	char m_acFmRefTag[32];
	char m_acRotGainTag[32];
	char m_acFlipGainTag[32];
	char m_acInvGainTag[32];
	char m_acMagTag[32];
	char m_acInFmMotionTag[32];
	char m_acEerSamplingTag[32];
	char m_acTiffOrderTag[32];
	char m_acCorrInterpTag[32];
private:
        CMcInput(void);
        void mPrint(void);
        int m_argc;
        char** m_argv;
        static CMcInput* m_pInstance;
};

//---------------------------------------------------------------
// Input parameter for tomographic alignment and reconstruction.
//---------------------------------------------------------------
class CAtInput
{
public:
	static CAtInput* GetInstance(void);
	static void DeleteInstance(void);
	~CAtInput(void);
	void Parse(int argc, char* argv[]);
	void ShowTags(void);
	bool bLocalAlign(void);
	int GetNumPatches(void);
	//-----------------
	float m_fTotalDose;
	float m_afTiltAxis[2];
	int m_iAlignZ;
	int m_iVolZ;
	float m_fAtBin;
	float m_afTiltCor[2];
	float m_afReconRange[2];
	float m_fAmpContrast;
	float m_afExtPhase[2];
	int m_iFlipVol;
	int m_iFlipInt;
	int m_aiSartParam[2];
	int m_iWbp;
	int m_aiAtPatches[2];
	int m_aiCropVol[2];
	int m_iOutXF;
	int m_iAlign;
	int m_iOutImod;
	float m_fDarkTol;
	bool m_bIntpCor;
	int m_iCtfTileSize;
	int m_aiCorrCTF[2];
	//-----------------
	char m_acTotalDoseTag[32];
	char m_acTiltAxisTag[32];
	char m_acAlignZTag[32];
	char m_acVolZTag[32];
	char m_acAtBinTag[32];
	char m_acTiltCorTag[32];
	char m_acReconRangeTag[32];
	char m_acAmpContrastTag[32];
	char m_acExtPhaseTag[32];
	char m_acFlipVolTag[32];
	char m_acFlipIntTag[32];
	char m_acSartTag[32];
	char m_acWbpTag[32];
	char m_acAtPatchTag[32];
	char m_acOutXFTag[32];
	char m_acAlignTag[32];
	char m_acCropVolTag[32];
	char m_acOutImodTag[32];
	char m_acDarkTolTag[32];
	char m_acBFactorTag[32];
	char m_acIntpCorTag[32];
	char m_acCorrCTFTag[32];
private:
        CAtInput(void);
        void mPrint(void);
        int m_argc;
        char** m_argv;
        static CAtInput* m_pInstance;
};

class CAreTomo3Json
{
public:
	CAreTomo3Json(void);
	~CAreTomo3Json(void);
	void Create(char* pcVersion);
private:
	void mAddInput(void);
	void mAddMcInput(void);
	void mAddAtInput(void);
	//-----------------
	void mAddLine(char* pcLine);
	void mAddLine(const char* pcKey, const char* pcVal, bool bList=false);
	void mAddLine(char* pcKey, int iVal);
	void mAddLine(char* pcKey, float fVal);
	void mAddLine(char* pcKey, int* piVals, int iNumVals);
	void mAddLine(char* pcKey, float* pfVals, int iNumVals);
	//-----------------
	void mCreateVal(int* piVals, int iNumVals, char* pcVal);
	void mCreateVal(float* pfVals, int iNumVals, char* pcVal);
	void mCreateVal(char** pcToks, int iNumToks, char* pcVal);
	//-----------------
	int m_iNumLines;
	int m_iLineSize;
	int m_iLineCount;
	int m_iIndent;
	char* m_pcJson;
};

class CCheckFreeGpus
{
public:
	CCheckFreeGpus(void);
	~CCheckFreeGpus(void);
	void SetAllGpus(int* piGpuIds, int iNumGpus);
	int GetFreeGpus(int* piFreeGpus, int iNumFreeGpus);
	void FreeGpus(void);
private:
	void mClean(void);
	int mOpenFile(void);
	int mLockFile(int iFd);
	int mUnlockFile(int iFd);
	void mReadGpus(FILE* pFile);
	void mWriteGpus(FILE* pFile);
	bool mCheckActivePid(int iPid);
	char m_acGpuFile[256];
	int* m_piGpuIds;
	int* m_piPids;
	int m_iNumGpus;
	int m_iPid;
};

class CProcessThread : public Util_Thread
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CProcessThread* GetFreeThread(void);
	static bool WaitExitAll(float fSeconds);
	~CProcessThread(void);
	int DoIt(void);
	void ThreadMain(void);
	int m_iNthGpu;
private:
	CProcessThread(void);
	void mProcessTsPackage(void);
	void mProcessMovies(void);
	bool mLoadTiltSeries(void);
	//-----------------
	void mProcessMovie(int iTilt);
	void mAssembleTiltSeries(int iTilt);
	void mProcessTiltSeries(void);
	//-----------------
	static CProcessThread* m_pInstances;
	static int m_iNumGpus;
	static std::unordered_map<std::string, int> *m_pMdocFiles;
};

class CGenStarFile
{
public:
	static CGenStarFile* GetInstance(void);
	static void DeleteInstance(void);
	~CGenStarFile(void);
	void OpenFile(char* pcInFile);
	void CloseFile(void);
	void SetStackSize(int* piStkSize);
	void SetHotPixels
	( unsigned char* pucBadMap, 
	  int* piMapSize, bool bPadded
	);
	void SetGlobalShifts(float* pfShifts, int iSize);
private:
	CGenStarFile(void);
	FILE* m_pFile;
	char m_acInMain[256];
	pthread_mutex_t m_aMutex;
	static CGenStarFile* m_pInstance;
};

class CMcAreTomoMain
{
public:
	CMcAreTomoMain(void);
	~CMcAreTomoMain(void);
	bool DoIt(void);
private:
	void mProcess(void);
};

}
