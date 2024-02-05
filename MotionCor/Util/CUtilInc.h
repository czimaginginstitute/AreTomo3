#pragma once
#include <Mrcfile/CMrcFileInc.h>
#include <cufft.h>
#include <pthread.h>

namespace McAreTomo::MotionCor::Util
{
//-------------------------------------------------------------------
// 1. Divide a stack of frames into multiple groups of frames. Each
//    group will be summed in relevant classes to form a reduced
//    stack containing these sums before motion measurement. 
//-------------------------------------------------------------------
class CGroupFrames
{
public:
        CGroupFrames(void);
        ~CGroupFrames(void);
        void DoGroupSize(int iNumFrames, int iGroupSize);
        void DoNumGroups(int iNumFrames, int iNumGroups);
        int GetGroupStart(int iGroup);
        int GetGroupSize(int iGroup);
        int GetNumGroups(void);
        int GetNumFrames(void);
        int GetGroup(int iFrame);
private:
        void mClean(void);
        void mGroup(void);
        int m_iNumFrames;
        int m_iGroupSize;
        int m_iNumGroups;
        int* m_piGroupStarts;
        int* m_piGroupSizes;
};

//-------------------------------------------------------------------
//
//-------------------------------------------------------------------
class CNextItem
{
public:
	CNextItem(void);
	~CNextItem(void);
	void Create(int iNumItems);
	void Reset(void);
	int GetNext(void);
	int GetNumItems(void);
private:
	int m_iNumItems;
	int m_iNextItem;
	pthread_mutex_t m_aMutex;
};

class CSplineFit1D
{
public:
	CSplineFit1D(void);
	~CSplineFit1D(void);
	void Clean(void);
	void SetNumKnots(int iNumKnots);
	bool DoIt(float* pfX, float* pfY, bool* pbBad, int iSize,
	   float* pfKnots, float* pfFit, float fReg);
private:
	float mCalcFit(float fX);
	void mCalcTerms(float fX);
	void mCalcCurTerms(float fX);
	int m_iNumKnots;
	int m_iDim;
	float* m_pfTerms;
	float* m_pfCoeff;
	float* m_pfMatrix;
	float* m_pfKnots;
};

class CRemoveSpikes1D
{
public:
	CRemoveSpikes1D(void);
	~CRemoveSpikes1D(void);
	void Clean(void);
	void SetDataSize(int iSize);
	void DoIt(float* pfShiftX, float* pfShiftY, bool bSingle);
private:
	bool mFindBad(float* pfRawShift, float* pfFitShift, float fTol);
	void mRemoveSingle(float* pfRawShift);
	float* m_pfFitX;
	float* m_pfFitY;
	float* m_pfBuf;
	float* m_pfTime;
	float* m_pfKnots;
	bool* m_pbBad;
	int m_iSize;
	int m_iNumKnots;
	CSplineFit1D m_aSplineFit;
};
}

namespace MMU = McAreTomo::MotionCor::Util;
