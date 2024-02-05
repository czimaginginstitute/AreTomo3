#pragma once
#include "../CMotionCorInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace McAreTomo::MotionCor::BadPixel
{

class CTemplate
{
public:
	CTemplate(void);
	~CTemplate(void);
	void Create(int* piSize, float* pfMod);
        int m_aiSize[2];
private:
        void mNormalize(float* pfTemplate);
};	//CTemplate


class GLocalCC
{
public:
	GLocalCC(void);
	~GLocalCC(void);
	void SetRef(float* gfRef, int* piRefSize);
	void DoIt
	( float* gfPadImg, 
	  int* piPadSize,
	  int iOffset, 
	  int iPartSize,
	  float* gfPadCC,
	  cudaStream_t stream
	);
private:
	float* m_gfRef;
};	//GLocalCC

//-------------------------------------------------------------------
// 1. Given bad pixel patch template (m_pfMod), it is used to perform
//    local correlation on the input image.
// 2. The output is the correlation map m_pfCC. It has the same
//    size of the input image.
//-------------------------------------------------------------------
class CLocalCCMap 
{
public:
	CLocalCCMap(void);
	~CLocalCCMap(void);
	void DoIt(int* piModSize, int iNthGpu);
private:
};

class GDetectPatch
{
public:
	GDetectPatch(void);
	~GDetectPatch(void);
	void DoIt
	( float* gfPadSum, 
	  float* gfPadCC,
	  float* gfPadBuf,
	  unsigned char* pucBadMap,
	  int* piPadSize,
	  int* piModSize,
	  float fStdThreshold
	);
private:
	void mUpdateBadMap
	( float* gfCCMap,
	  unsigned char* pucBadMap,
	  int* piPadSize,
	  int* piModSize
	);
	float m_fCCThreshold;
};


class GDetectHot
{
public:
        GDetectHot(void);
        ~GDetectHot(void);
        void DoIt
	( float* gfPadSum,
	  float* gfPadBuf, 
	  int* piPadSize, 
	  float fStdThreshold,
	  unsigned char* pucBadMap
	);
        int m_iNumHots;
};

class GCombineMap
{
public:
	GCombineMap(void);
	~GCombineMap(void);
	
	void GDoIt
	(  unsigned char* gucMap1,
	   unsigned char* gucMap2,
	   unsigned char* gucResMap,
	   int* piMapSize
	);
	unsigned char* GCopyMap
	(  unsigned char* pucMap,
	   int* piMapSize
	);
};

class GLabelPatch
{
public:
        GLabelPatch(void);
        ~GLabelPatch(void);
        void SetLabelSize(int iRadius);
        void DoIt
	(  float* gfImg, 
	   int* piImgSize,
	   int* piPatchList, 
	   int iNumPatches
	);
        int m_aiImgSize[2];
private:
        int m_iRadius;
};	//CLabelDefect


class CSumStack
{
public:
        CSumStack(void);
        ~CSumStack(void);
	float* DoIt
	(  MD::CMrcStack* pMrcStack,
	   int* piGpuIds, int iNumGpus
	);
private:
        void mAddUChar(void* pvFrame);
        void mAddFloat(void* pvFrame);
        int m_iMrcMode;
};


class CDetectMain
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CDetectMain* GetInstance(int iNthGpu);
	//-----------------
	~CDetectMain(void);
	void DoIt(int iNthGpu);
	unsigned char* GetDefectMap(bool bClean);
	int m_iNthGpu;
private:
	CDetectMain(void);
	void mDetectPatch(void);
	void mDetectHot(void);
	void mLabelDefects(float* gfImg, int* piDefects, int iNumDefects);
	void mLoadDefectFile(void);
	unsigned char* m_pucBadMap;
	float m_fThreshold;
	int m_aiPadSize[2];
	int m_aiDefectSize[2];
	//-----------------
	static CDetectMain* m_pInstances;
	static int m_iNumGpus;
};

class GCorrectBad
{
public:
        GCorrectBad(void);
        ~GCorrectBad(void);
        void SetWinSize(int iSize);
        void GDoIt
	( float* gfFrame,
	  unsigned char* gucBadMap,
	  int* piFrmSize,
	  bool bPadded,
          cudaStream_t stream=0
	);
private:
        int m_iWinSize;
};	//GCorrectBad

class CCorrectMain
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CCorrectMain* GetInstance(int iNthGpu);
	//-----------------
	CCorrectMain(void);
	~CCorrectMain(void);
	void DoIt(int iDefectSize, int iNthGpu);
	int m_iNthGpu;
private:
	void mCorrectFrames(void);
	void mCorrectFrame(int iFrame);
	int m_aiPadSize[2];
	int m_iDefectSize;
	//-----------------
	static CCorrectMain* m_pInstances;
	static int m_iNumGpus;
};

}

namespace MMB = McAreTomo::MotionCor::BadPixel;
