#pragma once
#include "../MaUtil/CMaUtilFwd.h"
#include <Util/Util_Thread.h>
#include <Mrcfile/CMrcFileInc.h>
#include <queue>
#include <unordered_map>
#include <string>
#include <cuda.h>

namespace McAreTomo::DataUtil
{

class CMrcStack
{
public:
	CMrcStack(void);
	virtual ~CMrcStack(void);
	void Create(int iMode, int* piStkSize);
	void* GetFrame(int iFrame);
	void** GetFrames(void) { return m_ppvFrames; }
	void RemoveFrame(int iFrame);
	int GetPixels(void);
	//-----------------
	int m_aiStkSize[3];
	int m_iMode;
	size_t m_tFmBytes;
	float m_fPixSize;
	float m_fStkDose;
protected:
	void mExpandBuf(int iNumFrms);
	void mCleanFrames(void);
	void** m_ppvFrames;
	int m_iBufSize;
};

class CTiltSeries : public CMrcStack
{
public:
	CTiltSeries(void);
	virtual ~CTiltSeries(void);
	void Create(int* piStkSize);
	void Create(int* piImgSize, int iNumTilts);
	void SortByTilt(void);
	void SortByAcq(void);
	//-----------------
	void SetTilts(float* pfTilts);
	void SetAcqs(int* piAcqIndices);
	void SetSecs(int* piSecIndices);
	//-----------------
	void SetImage(int iTilt, void* pvImage);
	void SetCenter(int iFrame, float* pfCent);
	void GetCenter(int iFrame, float* pfCent);
	//-----------------
	CTiltSeries* GetSubSeries(int* piStart, int* piSize);
	void RemoveFrame(int iFrame);
	void GetAlignedSize(float fTiltAxis, int* piAlnSize);
	float* GetAccDose(void);
	float** GetImages(void); // do not free;
	//-----------------
	void ResetSecIndices(void); // make sec indices ascending
	//-----------------
	float* m_pfTilts;
	int* m_piAcqIndices; // acquistion index, same as mdoc z value.
	int* m_piSecIndices; // section index in input MRC file.
	float m_fImgDose;
private:
	void mSwap(int iIdx1, int iIdx2);
	void mCleanCenters(void);
	float** m_ppfCenters;
	float** m_ppfImages;
};

class CAlnSums : public CMrcStack
{
public:
	CAlnSums(void);
	virtual ~CAlnSums(void);
	void Create(int* piImgSize);
	void* GetSum(void);
	void* GetSumOdd(void);
	void* GetSumEvn(void);
	//-----------------
	static int m_iNumSums;
};

class CGpuBuffer
{
public:
	CGpuBuffer(void);
	~CGpuBuffer(void);
	void Clean(void);
	void Create
	( size_t tFrmBytes,
	  int iNumFrames,
	  int iGpuID
	);
	void AdjustBuffer(int iNumFrames);
	void* GetFrame(int iFrame); // do not free
	//----------------------------------------
	size_t m_tFmBytes;
	int m_iNumFrames;
	int m_iNumGpuFrames;
	int m_iGpuID;
private:
	void mCalcGpuFrames(void);
	void mCreateCpuBuf(int iNumFrms);
	void mPrintAllocTimes(float* pfGBs, float* pfTimess);
	//---------------------------------------------------
	void* m_pvGpuFrames;
	void** m_ppvCpuFrames;
	int m_iMaxGpuFrms;
	int m_iMaxCpuFrms;
};

class CStackBuffer
{
public:
	CStackBuffer(void);
	~CStackBuffer(void);
	void Clean(void);
	void Create
	( int* piCmpSize,
	  int iNumFrames,
	  int iGpuID
	);
	void Adjust(int iNumFrames);
	//-----------------
	bool IsGpuFrame(int iFrame);
	cufftComplex* GetFrame(int iFrame);
	//------------------
	int m_iGpuID;
	int m_aiCmpSize[2];
	int m_iNumFrames;  // all stack frames
	size_t m_tFmBytes;
private:
	CGpuBuffer* m_pGpuBuffer;
};

enum EBuffer {tmp, sum, frm, xcf, pat};
enum EStkSize {img, cmp, pad};

class CBufferPool
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CBufferPool* GetInstance(int iNthGpu);
	//-----------------
	~CBufferPool(void);
	void Clean(void);
	void Create(int* piStkSize);
	void Adjust(int iNumFrames);
	CStackBuffer* GetBuffer(int iBuffer);
	//-----------------
	void* GetPinnedBuf(int iFrame);
	//-----------------
	MU::CCufft2D* GetCufft2D(bool bForward);
	//-----------------
	cudaStream_t GetCudaStream(int iStream);
	//-----------------
	int m_iNumSums;
	int m_iNthGpu;
	int m_iGpuID;
	float m_afXcfBin[2];
	int m_aiStkSize[3];
private:
	CBufferPool(void);
	void mCreateSumBuffer(void);
	void mCreateTmpBuffer(void);
	void mCreateFrmBuffer(void);
	void mCreateXcfBuffer(void);
	void mCreatePatBuffer(void);
	void mInit(void);
	CStackBuffer* m_pTmpBuffer;
	CStackBuffer* m_pSumBuffer;
	CStackBuffer* m_pFrmBuffer;
	CStackBuffer* m_pXcfBuffer;
	CStackBuffer* m_pPatBuffer;
	void* m_avPinnedBuf[2];
	MU::CCufft2D* m_pCufft2Ds;
	cudaStream_t* m_pCudaStreams;
	bool m_bCreated;
	//-----------------
	static CBufferPool* m_pInstances;
	static int m_iNumGpus;
};

class CCtfResults
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CCtfResults* GetInstance(int iNthGpu);
	~CCtfResults(void);
	void Clean(void);
	void Setup(int iNumImgs, int* piSpectSize);
	void SetTilt(int iImage, float fTilt);
	void SetDfMin(int iImage, float fDfMin);
	void SetDfMax(int iImage, float fDfMax);
	void SetAzimuth(int iImage, float fAzimuth);
	void SetExtPhase(int iImage, float fExtPhase);
	void SetScore(int iImage, float fScore);
	void SetSpect(int iImage, float* pfSpect);
	//----------------------------------------
	float GetTilt(int iImage);
	float GetDfMin(int iImage);
	float GetDfMax(int iImage);
	float GetAzimuth(int iImage);
	float GetExtPhase(int iImage);
	float GetScore(int iImage);
	float* GetSpect(int iImage, bool bClean);
	void SaveImod(const char* pcCtfTxtFile);
	void Display(int iNthCtf, char* pcLog);
	void DisplayAll(void);
	//-----------------------
	int m_aiSpectSize[2];
	int m_iNumImgs;
	int m_iNthGpu;
private:
	CCtfResults(void);
	void mInit(void);
	float* m_pfDfMins;
	float* m_pfDfMaxs;
	float* m_pfAzimuths;
	float* m_pfExtPhases;
	float* m_pfScores;
	float* m_pfTilts;
	float** m_ppfSpects;
	static CCtfResults* m_pInstances;
	static int m_iNumGpus;
};

class CMcPackage
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CMcPackage* GetInstance(int iNthGpu);
	//-----------------
	~CMcPackage(void);
	void SetMovieName(char* pcMovieName);
	bool bTiffFile(void);
	bool bEerFile(void);
	//-----------------
	int* GetMovieSize(void); // do NOT free
	int GetMovieMode(void);
	//-----------------
	char m_acMoviePath[256];
	CMrcStack* m_pRawStack;
	CAlnSums* m_pAlnSums;
	//-------------------
	int m_iAcqIdx;
	float m_fTilt;
	float m_fPixSize;
	int m_iNthGpu;
private:
	CMcPackage(void);
	static CMcPackage* m_pInstances;
	static int m_iNumGpus;
};

class CReadMdoc
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CReadMdoc* GetInstance(int iNthGpu);
	~CReadMdoc(void);
	bool DoIt(const char* pcMdocFile);
	char* GetFramePath(int iTilt);     // do not free
	char* GetFrameFileName(int iTilt); // do not free
	int GetAcqIdx(int iTilt);
	float GetTilt(int iTilt);
	int m_iNumTilts;
	int m_iNthGpu;
	char m_acMdocFile[256];
private:
	CReadMdoc(void);
	void mClean(void);
	int mExtractValZ(char* pcLine);
	bool mExtractTilt(char* pcLinei, float* pfTilt);
	char* mExtractFramePath(char* pcLine);
	//-----------------
	char** m_ppcFrmPath;
	int* m_piAcqIdxs;
	float* m_pfTilts;
	int m_iBufSize;
	static CReadMdoc* m_pInstances;
	static int m_iNumGpus;
};

class CTsPackage // tilt series package
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CTsPackage* GetInstance(int iNthGpu);
	//-----------------
	~CTsPackage(void);
	void SetMdoc(char* pcMdocFile);
	void CreateTiltSeries(void);
	void SetTiltAngle(int iTilt, float fTiltAngle);
	void SetAcqIdx(int iTilt, int iAcqIdx);
	void SetSecIdx(int iTilt, int iSecIdx);
	void SetSums(int iTilt, CAlnSums* pAlnSums);
	void SetImgDose(float fImgDose);
	//-----------------
	CTiltSeries* GetSeries(int iSeries); // 0 - raw, 1 - evn, 2 - odd
	//-----------------
	void SortTiltSeries(int iOrder); // 0 - by tilt, 1 - by acq
	void ResetSectionIndices(void);
	void SaveTiltSeries(void);
	//-----------------
	void SaveVol(CTiltSeries* pVol, int iVol);
	//-----------------
	char* m_pcMdocFile;
	char m_acMrcMain[256];
	int m_iNthGpu;
private:
	void mSaveTiltFile
	( const char* pcFileName,
	  CTiltSeries* pTiltSeries
	);
	void mSaveMrc
	( const char* pcFileName, 
	  const char* pcExt,
	  CTiltSeries* pTiltSeries
	); 
	//-----------------
	CTsPackage(void);
	CTiltSeries** m_ppTsStacks;
	CTiltSeries** m_ppVolStacks;
	static CTsPackage* m_pInstances;
	static int m_iNumGpus;
};

class CStackFolder : public Util_Thread
{
public:
	static CStackFolder* GetInstance(void);
	static void DeleteInstance(void);
	~CStackFolder(void);
	void PushFile(char* pcMdocFile);
	char* GetFile(bool bPop);
	void DeleteFront(void);
	int GetQueueSize(void);
	//---------------------
	bool ReadFiles(void);
	void ThreadMain(void);
private:
        CStackFolder(void);
        bool mReadSingle(void);
        bool mGetDirName(void);
        bool mOpenDir(void);
        int mReadFolder(void);
        bool mAsyncReadFolder(void);
	//-----------------
	bool mCheckSkips(const char* pcString);
	bool mFileExist(const char* pcFile);
	//-----------------
        void mClean(void);
	//-----------------
        char m_acDirName[256];
        char m_acPrefix[256];
        char m_acSuffix[256];
	char m_acSkips[256];
	//-----------------
	std::queue<char*> m_aFileQueue;
        std::unordered_map<std::string, int> m_aReadFiles;
        //-----------------
	int m_iNumChars;
	static CStackFolder* m_pInstance;
};

class CLogFiles
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CLogFiles* GetInstance(int iNthGpu);
	~CLogFiles(void);
	void Create(const char* pcMdocFile);
	FILE* m_pMcGlobalLog;
	FILE* m_pMcLocalLog;
	FILE* m_pAtGlobalLog;
	FILE* m_pAtLocalLog;
private:
	CLogFiles(void);
	void mCreateMcLogs(const char* pcPrefix);
	void mCreateAtLogs(const char* pcPrefix);
	void mCreatePath
	( const char* pcPrefix,
	  const char* pcSuffix,
	  char* pcPath
	);
	void mCloseLogs(void);
	//-----------------
	char m_acLogDir[256];
	int m_iNthGpu;
	//-----------------
	static CLogFiles* m_pInstances;
	static int m_iNumGpus;
};

class CDuInstances
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
};
}
namespace MD = McAreTomo::DataUtil;
