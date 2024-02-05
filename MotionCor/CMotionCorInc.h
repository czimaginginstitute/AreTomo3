#pragma once
#include "../CMcAreTomoInc.h"
#include "../MaUtil/CMaUtilInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include <Util/Util_Thread.h>
#include <stdio.h>
#include <cufft.h>

namespace McAreTomo::MotionCor
{
class CLoadRefs
{
public:
	static CLoadRefs* GetInstance(void);
	static void DeleteInstance(void);
	~CLoadRefs(void);
	void CleanRefs(void);
	bool LoadGain(char* pcGainFile);
	bool LoadDark(char* pcDarkFile);
	void PostProcess(int iRotFact, int iFlip, int iInverse);
	bool AugmentRefs(int* piImgSize);
	float* m_pfGain;
	float* m_pfDark;
	int m_aiRefSize[2];
private:
	int mGetFileType(char* pcRefFile);
	void mClearGain(void);
	void mClearDark(void);
	void mLoadGainMrc(char* pcMrcFile);
	void mLoadGainTiff(char* pcTiffFile);
	void mRotate(float* gfRef, int* piRefSize, int iRotFact);
	void mFlip(float* gfRef, int* piRefSize, int iFlip);
	void mInverse(float* gfRef, int* piRefSize, int iInverse);
	float* mToFloat(void* pvRef, int iMode, int* piSize);
	void mCheckDarkRef(void);
	float* mAugmentRef(float* pfRef, int iFact);
	CLoadRefs(void);
	//-----------------
	int m_aiDarkSize[2];
	bool m_bAugmented;
	pthread_mutex_t m_mutex;
	//-----------------
	static CLoadRefs* m_pInstance;
};

class CProcessMovie
{
public:
	CProcessMovie(void);
	~CProcessMovie(void);
	bool DoIt(void* pvMvPackage, int iNthGpu);
private:
	bool mCheckGain(void);
	void mApplyRefs(void);
	void mDetectBadPixels(void);
	void mCorrectBadPixels(void);
	void mAlignStack(void);
	//-----------------
	int m_iNthGpu;
};

class CProcessThread : public Util_Thread
{
public:
	static void CreateInstances(void);
	static void DeleteInstances(void);
	static CProcessThread* GetFreeThread(void);
	~CProcessThread(void);
	void DoIt(void* pvTsPackage);
private:
	CProcessThread(void);
	void mProcessTsPackage(void);
	bool mLoadMovie(int iTilt);
	bool mProcessMovie(void);
	void mAssembleTiltSeries(int iTilt);
	bool mProcessTiltSeries(void);
	//-----------------
	void* m_pvTsPackage;
	void* m_pvMvPackage;
	int m_iNthGpu;
	void* m_pvReadMdoc;
	static CProcessThread* m_pInstances;
	static int m_iNumGpus;
};

class CMcInstances
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
};

class CMotionCorMain
{
public:
	static void LoadRefs(void);
	static void LoadFmIntFile(void);
	//-----------------
	CMotionCorMain(void);
	~CMotionCorMain(void);
	bool DoIt(int iNthGpu);
	//-----------------
private:
	bool mLoadStack(void);
	bool mCheckGain(void);
	void mCreateBuffer(void);
	void mApplyRefs(void);
	void mDetectBadPixels(void);
	void mCorrectBadPixels(void);
	void mAlignStack(void);
	//-----------------
	int m_iNthGpu;
};

}
