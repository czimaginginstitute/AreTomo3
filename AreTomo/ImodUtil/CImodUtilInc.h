#pragma once
#include "../CAreTomoInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <stdio.h>

namespace McAreTomo::AreTomo::ImodUtil
{
class CSaveXF
{
public:
	CSaveXF(void);
	~CSaveXF(void);
	void DoIt(int iNthGpu, const char* pcFileName);
private:
	void mSaveForRelion(void);
	void mSaveForWarp(void);
	void mSaveForAligned(void);
	FILE* m_pFile;
	int m_iNthGpu;
};

class CSaveTilts
{
public:
	CSaveTilts(void);
	~CSaveTilts(void);
	void DoIt
	( int iNthGpu,
	  const char* pcFileName
	);
private:
	void mSaveForRelion(void);
	void mSaveForWarp(void);
	void mSaveForAligned(void);
	FILE* m_pFile;
	int m_iNthGpu;
};

class CSaveCsv
{
public:
	CSaveCsv(void);
	~CSaveCsv(void);
	void DoIt(int iNthGpu, const char* pcFileName);
private:
	void mSaveForRelion(void);
        void mSaveForWarp(void);
        void mSaveForAligned(void);
        FILE* m_pFile;
	int m_iNthGpu;
};

class CSaveXtilts
{
public:
	CSaveXtilts(void);
	~CSaveXtilts(void);
	void DoIt(int iNthGpu, const char* pcFileName);
};

class CImodUtil
{
public:
	static void CreateInstances(int iNumGpus);
	static void DeleteInstances(void);
	static CImodUtil* GetInstance(int iNthGpu);
	//-----------------
	~CImodUtil(void);
	void CreateFolder(void);
	void SaveTiltSeries(MD::CTiltSeries* pTiltSeries);
	void SaveVolume
	( MD::CTiltSeries* pVolSeries,
	  float fPixelSize,
	  float* pfStats
	);
	int m_iNthGpu;
private:
	CImodUtil(void);
	void mSaveTiltSeries(void);
	void mSaveNewstComFile(void);
	void mSaveTiltComFile(void);
	void mSaveCtfFile(void);
	void mCreateFileName(const char* pcInFileName, char* pcOutFileName);
	char m_acOutFolder[256];
	char m_acInMrcFile[128];
	char m_acAliFile[128];
	char m_acTltFile[128];
	char m_acCsvFile[128];
	char m_acXfFile[128];
	char m_acXtiltFile[128];
	char m_acRecFile[128];
	char m_acCtfFile[128];
	MD::CTiltSeries* m_pTiltSeries;
	MD::CTiltSeries* m_pVolSeries;
	MAM::CAlignParam* m_pGlobalParam;
	float m_fTiltAxis;
	float m_fPixelSize;
	static CImodUtil* m_pInstances;
	static int m_iNumGpus;
};

}
