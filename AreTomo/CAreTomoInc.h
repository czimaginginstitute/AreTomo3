#pragma once
#include "../CMcAreTomoInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include "../MaUtil/CMaUtilInc.h"
#include "Correct/CCorrectFwd.h"
#include <stdio.h>

namespace McAreTomo::AreTomo
{
class CAreTomoInput
{
public:
	static CAreTomoInput* GetInstance(void);
	static void DeleteInstance(void);
	~CAreTomoInput(void);
	float m_afTiltAxis[2];
	int m_iAlignZ;
	int m_iVolZ;
	float m_fTomoBin;
	float m_afTiltCor[2];
	float m_afReconRange[2];
	float m_fAmpContrast;
	float m_afExtPhase[2];
	int m_iFlipVol;
	int m_iFlipInt;
	int m_aiSartParam[2];
	int m_iWbp;
	int m_aiTomoPatches[2];
	int m_aiCropVol[2];
	int m_iOutXF;
	int m_iAlign;
	int m_iOutImod;
	float m_fDarkTol;
	float m_afBFactor[2];
	int m_iIntpCor;
	//-------------
	char m_acTiltAxisTag[32];
	char m_acAlignZTag[32];
	char m_acVolZTag[32];
	char m_acTomoBinTag[32];
	char m_acTiltCorTag[32];
	char m_acReconRangeTag[32];
	char m_acAmpContrastTag[32];
	char m_acExtPhaseTag[32];
	char m_acFlipVolTag[32];
	char m_acFlipIntTag[32];
	char m_acSartTag[32];
	char m_acWbpTag[32];
	char m_acTomoPatchTag[32];
	char m_acOutXFTag[32];
	char m_acAlignTag[32];
	char m_acCropVolTag[32];
	char m_acOutImodTag[32];
	char m_acDarkTolTag[32];
	char m_acBFactorTag[32];
	char m_acIntpCorTag[32];
private:
        CAreTomoInput(void);
        void mPrint(void);
        int m_argc;
        char** m_argv;
        static CAreTomoInput* m_pInstance;
};

class CAtInstances
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
};

class CAreTomoMain
{
public:
	CAreTomoMain(void);
	~CAreTomoMain(void);
	bool DoIt(int iNthGpu);
private:
	void mRemoveDarkFrames(void);
	void mRemoveSpikes(void);
	void mCreateAlnParams(void);
	void mFindCtf(void);
	void mMassNorm(void);
	//-----------------
	void mAlign(void);
	void mCoarseAlign(void);
	void mStretchAlign(void);
	void mRotAlign(float fAngRange, int iNumSteps);
	void mRotAlign(void);
	void mFindTiltOffset(void);
	void mProjAlign(void);
	void mPatchAlign(void);
	//-----------------
	void mRecon(void);
	void mSetPositivity(void);
	void mSaveForImod(void);
	void mReconSeries(int iSeries);
	void mSartRecon
	( int iVolZ, int iSeries, 
	  MD::CTiltSeries* pSeries
	);
	void mWbpRecon
	( int iVolZ, int iSeries, 
	  MD::CTiltSeries* pSeries
	);
	//-----------------
	void mDoseWeight(void);
	void mSaveAlignment(void);
	//-----------------
	void mFlipVol(MD::CTiltSeries** ppVolSeries);
	//-----------------
	void mLogGlobalShift(void);
	void mLogLocalShift(void);
	//-----------------
	MAC::CCorrTomoStack* m_pCorrTomoStack;
	float m_fRotScore;
	float m_fTiltOffset;
	int m_iNthGpu;

};
}
namespace MA = McAreTomo::AreTomo;
