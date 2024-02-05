#pragma once
#include "../CMotionCorInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Util/CUtilInc.h"
#include <Util/Util_Thread.h>
#include <cuda.h>
#include <cufft.h>

namespace MMD = McAreTomo::MotionCor::DataUtil;

namespace McAreTomo::MotionCor::Align
{

class CAlignParam
{
public:
	static CAlignParam* GetInstance(void);
	static void DeleteInstance(void);
	~CAlignParam(void);
	bool bCorrectMag(void);
	void GetMagStretch(float* pfStretch);
	bool bPatchAlign(void);
	bool bSplitSum(void);
	int GetNumPatches(void);
	int GetFrameRef(int iNumFrames);
	//------------------------------
	int m_iTomoAcqIndex;
	int m_iTomoFrmsPerStack;
private:
	CAlignParam(void);
	static CAlignParam* m_pInstance;
};

class CAlignedSum 
{
public:
	CAlignedSum(void);
	~CAlignedSum(void);
	void DoIt
	( int iBuffer, 
	  MMD::CStackShift* pStackShift,
	  int* piSumRange,
	  int iNthGpu
	);
private:
	void mDoFrame(int iFrame);
	//------------------------
	MD::CStackBuffer* m_pFrmBuffer;
	MMD::CStackShift* m_pStackShift;
	cudaStream_t m_streams[2];
	cufftComplex* m_gCmpSums[2];
	int m_aiSumRange[2];
};
	
class CTransformStack 
{
public:
	CTransformStack(void);
	~CTransformStack(void);
	void Setup
	( MD::EBuffer eBuf,
	  bool bForward,
	  bool bNorm,
	  int iNthGpu 
	);
	void DoIt(void);
private:
	void mTransformGpuFrames(void);
	void mTransformCpuFrames(void);
	void mTransformFrame(cufftComplex* gCmpFrm);
	//------------------------------------------
	MD::CStackBuffer* m_pFrmBuffer;
	MD::CStackBuffer* m_pTmpBuffer;
	bool m_bForward;
	bool m_bNorm;
	int m_iNthGpu;
	cudaStream_t m_streams[2];
	MU::CCufft2D* m_pCufft2D;
};

class CGenXcfStack 
{
public:
	CGenXcfStack(void);
	~CGenXcfStack(void);
	void DoIt(MMD::CStackShift* pStackShift, int iNthGpu);
private:
	void mDoIt(int iNthGpu);
	void mDoXcfFrame(int iFrm);
	//-------------------------
	MMD::CStackShift* m_pStackShift;
	MD::CStackBuffer* m_pXcfBuffer;
	MD::CStackBuffer* m_pFrmBuffer;
	MD::CStackBuffer* m_pTmpBuffer;
	MD::CStackBuffer* m_pSumBuffer;
	cudaStream_t m_stream;
};

class CInterpolateShift
{
public:
	CInterpolateShift(void);
	~CInterpolateShift(void);
	MMD::CStackShift* DoIt
	( MMD::CStackShift* pGroupShift,
	  int iNthGpu,
	  bool bPatch,
	  bool bNearest
	);
	void DoIt
	( MMD::CStackShift* pGroupShift, 
	  MMD::CStackShift* pOutShift,
	  int iNthGpu,
	  bool bPatch,
	  bool bNearest
	);
private:
	void mInterpolate
	( float* pfGpCents,
	  float* pfFmCents,
	  float* pfGpShifts,
	  float* pfFmShifts
	);
	int mFindGroup(float* pfGpCents, float fCent);
	int m_iNumFrames;
	int m_iNumGroups;
};

//---------------------------------------------------------
class GCorrelateSum2D
{
public:
	GCorrelateSum2D(void);
	~GCorrelateSum2D(void);
	void SetFilter(float fBFactor, bool bPhaseOnly);
	void SetSize(int* piCmpSize, int* piSeaSize);
	void SetSubtract(bool bSubtract);
	void DoIt
	( cufftComplex* gCmpSum, 
	  cufftComplex* gCmpXcf,
	  float* pfPinnedXcf,
	  MU::CCufft2D* pInverseFFT,
	  cudaStream_t stream = 0
	);
private:
	float m_fBFactor;
	bool m_bPhaseOnly;
	int m_aiCmpSize[2];
	int m_aiSeaSize[2];
	bool m_bSubtract;
};	

class GCC2D
{
public:
	GCC2D(void);
	~GCC2D(void);
	void SetSize(int* piCmpSize);
	void SetBFactor(float fBFactor);
	float DoIt(cufftComplex* gCmp1, cufftComplex* gCmp2,
	   cudaStream_t stream);
private:
	void mTest(cufftComplex* gCmp1, cufftComplex* gCmp2);
	dim3 m_aGridDim;
	dim3 m_aBlockDim;
	int m_aiCmpSize[2];
	float m_fBFactor;
	float* m_gfCC;
	float* m_pfCC;
};

class CEarlyMotion
{
public:
	CEarlyMotion(void);
	~CEarlyMotion(void);
	void Setup(int iBuffer, float fBFactor, int iNthGpu);
	void DoIt(MMD::CStackShift* pStackShifti);	
	
private:
	void mDoIt(void);
	float mIterate(MMD::CStackShift* pStackShift, int iAxis);
	void mGetNodeShifts
	( MMD::CStackShift* pStackShift, 
	  int iAxis, float* pfShift
	);
	void mCalcCoeff(float fGain, float* pfShift, float* pfCoeff);
	void mCalcShift
	( float* pfCoeffXs, float* pfCoeffYs,
	  MMD::CStackShift* pfStackShift
	);
	void mCorrelate(int iStep, MMD::CStackShift* pStackShift);
	void mFindPeaks(float* pfPeaks);
	void mFindPeak(int iPeak, float* pfPeak);
	//-----------------
	cufftComplex* m_gCmpRef;
	int m_aiCmpSize[2];
	GCorrelateSum2D m_aGCorrelateSum;
	MMD::CStackShift* m_pStackShift;
	MU::CCufft2D* m_pInverseFFT;
	int m_aiCent[3];
	int m_aiSeaSize[2];
	int m_aiSumRange[2];
	int m_iNumSteps;
	float m_fStepSize;
	//-----------------
	MD::CBufferPool* m_pBufferPool;
	int m_iBuffer;
	float m_fBFactor;
	int m_iNthGpu;
};

class CAlignStack
{
public:
	CAlignStack(void);
	~CAlignStack(void);
	void Set1(int iBuffer, int iNthGpu);
	void Set2(float fBFactor, bool bPhaseOnly);
	void DoIt
	( MMD::CStackShift* pStackShift,
	  MMD::CStackShift* pGroupShift
	);
	void WaitStreams(void);
	float m_fErr;
private:
	void mDoGroups(void);
	void mDoGroup(int iGroup, int iStream);
	void mPhaseShift(int iStream, bool bSum);
	void mCorrelate(int iFrame, int iStream);
	void mFindPeaks(void);
	void mUpdateError(float* pfShift);
	//-----------------
	MD::CStackBuffer* m_pFrmBuffer;
	MD::CStackBuffer* m_pTmpBuffer;
	MD::CStackBuffer* m_pSumBuffer;
	//-----------------
	MMD::CStackShift* m_pStackShift;
	MMD::CStackShift* m_pGroupShift;
	MMD::CFmGroupParam* m_pFmGroupParam;
	//-----------------
	int m_aiSeaSize[2];
	MU::CCufft2D* m_pInverseFFT;
	GCorrelateSum2D m_aGCorrelateSum;
	cufftComplex* m_gCmpSum;
	cudaStream_t m_streams[2];
	int m_iFrame;
	//-----------------
	int m_iNthGpu;
};

class CIterativeAlign
{
public:
	CIterativeAlign(void);
	~CIterativeAlign(void);
	void Setup(int iBuffer, int iNthGpu);
	void DoIt(MMD::CStackShift* pStackShift);
	char* GetErrorLog(void);
private:
	MMD::CStackShift* mAlignStack(MMD::CStackShift* pInitShift); 
	CAlignStack* m_pAlignStack;
	int m_iMaxIterations;
	int m_iIterations;
	float m_fTol;
	float m_fBFactor;
	bool m_bPhaseOnly;
	float m_afXcfBin[2];
	float* m_pfErrors;
	int m_iBuffer;
	int m_iNthGpu;
};

//---------------------------------------------------------
class CAlignBase
{
public:
	CAlignBase(void);
	virtual ~CAlignBase(void);
	void Clean(void);
	virtual void DoIt(int iNthGpu);
	virtual void LogShift(char* pcLogFile);
	int m_aiImgSize[2]; // after Fourier cropping
	int m_aiPadSize[2]; // after Fourier cropping
	int m_aiCmpSize[2]; // after Fourier cropping
protected:
	void mCreateAlnSums(void);
	MMD::CStackShift* m_pFullShift;
	int m_iNthGpu;
};	//CAlignBase

class CSimpleSum : public CAlignBase
{
public:
	CSimpleSum(void);
	virtual ~CSimpleSum(void);
	void DoIt(int iNthGpu);
private:
	void mCalcSum(void);
	void mCropFrame
	( cufftComplex* gCmpFrm,
	  cufftComplex* gCmpBuf,
	  int iStream
	);
	void mUnpad
	( cufftComplex* gCmpPad,
	  float* gfUnpad,
	  int iStream
	);
	cudaStream_t m_streams[2];
	MU::CCufft2D* m_pForwardFFT;
	MU::CCufft2D* m_pInverseFFT;
};

class CFullAlign : public CAlignBase
{
public:
	CFullAlign(void);
	virtual ~CFullAlign(void);
	void Align(int iNthGpu);
	virtual void DoIt(int iNthGpu);
protected:
	void mFourierTransform(bool bForward);
	void mDoAlign(void);
	virtual void mCorrect(void);
private:
	void mLogShift(void);
};	//CFullAlign;

//--------------------------------------------------------------------
//
//--------------------------------------------------------------------
class CPatchAlign : public CFullAlign
{
public:
	CPatchAlign(void);
	virtual ~CPatchAlign(void);
	void DoIt(int iNthGpu);
	void LogShift(char* pcLogFile);
private:
	void mCorrectFullShift(void);
	void mCalcPatchShifts(void);
	void mChoose(void);
	void mCalcMeanStd
	( float* pfData, int iSize,
	  double* pdMean, double* pdStd
	);
	void mFindGraphenePeaks
	( cufftComplex* gCmp,
	  int* piCmpSize
	);
	void mLogShift(void);
	//-----------------
	MMD::CPatchShifts* m_pPatchShifts;
};

class CPatchCenters
{
public:
	static void CreateInstances(int iNumGpus);
	static CPatchCenters* GetInstance(int iNthGpu);
	static void DeleteInstances(void);
	~CPatchCenters(void);
	void Calculate(void);
	void GetCenter(int iPatch, int* piCenter);
	void GetStart(int iPatch, int* piStart);
	int m_iNumPatches;
	int m_aiXcfSize[2];
	int m_aiPatSize[2];
	int m_iNthGpu;
private:
	CPatchCenters(void);
	int mCalcStart(float fCent, int iPatSize, int iXcfSize);
	int* m_piPatStarts;
	static CPatchCenters* m_pInstances;
	static int m_iNumGpus;
};

class CExtractPatch 
{
public:
	CExtractPatch(void);
	~CExtractPatch(void);
	void DoIt(int iPatch, int iNthGpu);
private:
	void mProcessFrame(int iFrame);
	MD::CStackBuffer* m_pPatBuffer;
	MD::CStackBuffer* m_pXcfBuffer;
	MD::CStackBuffer* m_pSumBuffer;
	int m_aiPatStart[2];
	cudaStream_t m_streams[2];
};
     

//-------------------------------------------------------------------
// CMeasurePatches: Measure the motion of each patch until all
// patches have been measured.
// 1. Each thread checks first if there is any patch waiting to
//    be measured.
// 2. If found, the thread will extract the patch and perform
//    alignment. 
//-------------------------------------------------------------------
class CMeasurePatches 
{
public:
	CMeasurePatches(void);
	~CMeasurePatches(void);
	void DoIt
	( MMD::CPatchShifts* pPatchShifts,
	  int iNthGpu
	);
private:
	void mCalcPatchShift(int iPatch);
	MMD::CPatchShifts* m_pPatchShifts;
	CIterativeAlign m_iterAlign;
	CTransformStack m_transformStack;
	float m_fTol;
	bool m_bPhaseOnly;
	int m_iNthGpu;
};

class CSaveAlign
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CSaveAlign* GetInstance(int iNthGpu);
	~CSaveAlign(void);
	bool Open(char* pcFileName);
	bool Open(char* pcDirName, char* pcStackFile);
	void SaveSetting(int* piStkSize, int* piPatches, int* piThrow);
	void DoGlobal(MMD::CStackShift* pStackShift);
	void DoLocal(MMD::CPatchShifts* pPatchShifts);
	char* GetStartTag(char* pcTagName);
	char* GetEndTag(char* pcTagName);
	char m_acSetting[64];
	char m_acStackSize[64];
	char m_acPatches[64];
	char m_acThrow[64];
	char m_acGlobalShift[64];
	char m_acLocalShift[64];
	char m_acConverge[64];
	char m_acStackID[64];
	char m_acPatchID[64];
	int m_iNthGpu;
private:
	CSaveAlign(void);
	void mSaveGlobal(MMD::CStackShift* pFullShift);
	void mSaveLocal(MMD::CPatchShifts* pPatchShifts);
	FILE* m_pFile;
	static CSaveAlign* m_pInstances;
	static int m_iNumGpus;
};

class GNormByStd2D
{
public:
	GNormByStd2D(void);
	~GNormByStd2D(void);
	void DoIt(float* gfImg, int* piImgSize, bool bPadded,
	   int* piWinSize, cudaStream_t stream = 0);
};

class CDetectFeatures
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CDetectFeatures* GetInstance(int iNthGpu);
	//-----------------	
	~CDetectFeatures(void);
	void DoIt
	( MMD::CStackShift* pXcfStackShift, 
	  int* piNumPatches
	);
	void GetCenter
	( int iPatch, int* piImgSize, 
	  float* pfCent
	);
	void FindNearest
	( float* pfLoc, int* piImgSize, 
	  int* piPatSize, float* pfNewLoc
	);
	int m_iNthGpu;
private:
	CDetectFeatures(void);
	void mClean(void);
	void mFindCenters(void);
	void mFindCenter(float* pfCenter);
	bool mCheckFeature(float fCentX, float fCentY);
	void mSetUsed(float fCentX, float fCentY);
	int mCheckRange(int iStart, int iSize, int* piRange);
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

//--------------------------------------------------------------------
// CAlignMain: The entry point to start alignment and correction.
//--------------------------------------------------------------------
class CAlignMain
{
public:
	CAlignMain(void);
	~CAlignMain(void);
	void DoIt(int iNthGpu);
private:
	char* mCreateLogFile(void);
	int m_iNthGpu;
};

}

namespace MMA = McAreTomo::MotionCor::Align;
