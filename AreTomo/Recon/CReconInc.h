#pragma once
#include "../CAreTomoInc.h"
#include "../MrcUtil/CMrcUtilFwd.h"
#include "../Util/CUtilInc.h"

namespace McAreTomo::AreTomo::Recon
{
class GRWeight
{
public:
	GRWeight(void);
	~GRWeight(void);
	void Clean(void);
	void SetSize(int iPadProjX, int iNumProjs);
	void DoIt(float* gfPadSinogram);
private:
	int m_iCmpSizeX;
	int m_iNumProjs;
	MU::GFFT1D* m_pGForward;
	MU::GFFT1D* m_pGInverse;
};

class GBackProj
{
public:
	GBackProj(void);
	~GBackProj(void);
	void SetSize
	( int* piPadProjSize, // iPadProjX, iAllProjs
	  int* piVolSize      // iVolX, iVolZ
	);
	void SetSubset     // subset of projections for back-projection
	( int iStartProj,  // starting index
	  int iEndProj     // ending index, exclusive
	);
	void DoIt
	( float* gfPadSinogram, // y-slice of all projections
	  float* gfCosSin,
	  bool* gbProjs,        // which projections are used
	  bool bSart,
	  float fRelax,
	  float* gfVolXZ,       // y-slice of volume
	  cudaStream_t stream = 0
	);
private:
	dim3 m_aBlockDim;
	dim3 m_aGridDim;
	int m_iStartProj;
	int m_iEndProj;
};

class GForProj 
{
public:
	GForProj(void);
	~GForProj(void);
	void SetVolSize(int iVolX, bool bPadded, int iVolZ);
	void DoIt
	( float* gfVol,
	  float* gfCosSin,
	  int* piProjSize,    // projX and numTilts
	  bool bPadded,       // projX is badded or not
	  float* gfForProjs,  //
	  cudaStream_t stream = 0 
	);
private:
	int m_aiVolSize[3]; // iVolX, iVolXPadded, iVolZ
};

class GDiffProj // Projsections are y-slice
{
public:
	GDiffProj(void);
	~GDiffProj(void);
	void DoIt
	( float* gfRawProjs,
	  float* gfForProjs,
	  float* gfDiffProjs,
	  int* piProjSize, // iProjX, iNumProjs
	  bool bPadded, // iProjX is padded or not
	  cudaStream_t stream = 0
	);
};

class GCalcRFactor  // Projections are y-slice
{
public:
	GCalcRFactor(void);
	~GCalcRFactor(void);
	void Clean(void);
	void Setup(int iProjX, int iNumProjs);
	void DoIt
	( float* gfProjs, float* pfRfSum, int* piRfCount,
	  cudaStream_t stream = 0
	);
private:
	float* m_gfSum;
	int* m_giCount;
	int m_iProjX;
	int m_iNumProjs;
};

class GWeightProjs
{
public:
	GWeightProjs(void);
	~GWeightProjs(void);
	void DoIt
	( float* gfProjs,
	  float* gfCosSin,
	  int* piProjSize, // iProjSizeX & iNumProjs
	  bool bPadded,    // true when iProjSizeX is padded
	  int iVolZ,
	  cudaStream_t stream = 0
	);
private:
};

class CTomoBase
{
public:
	CTomoBase(void);
	virtual ~CTomoBase(void);
	void Clean(void);
	void Setup
	( int iVolX, int iVolZ,
	  MD::CTiltSeries* pTiltSeries,
	  MAM::CAlignParam* pAlignParam
	);
	void ExcludeTilts(float* pfTilts, int iNumTilts);
protected:
	int m_aiVolSize[2];
	int m_iPadProjX;
	int m_iNumProjs;
	//-----------------
	float* m_gfPadSinogram;
	float* m_gfVolXZ;
	//-----------------
	float* m_gfCosSin;
	bool* m_gbNoProjs;
	//-----------------
	MD::CTiltSeries* m_pTiltSeries;
	MAM::CAlignParam* m_pAlignParam;
	//-----------------
	GWeightProjs m_aGWeightProjs;
	GBackProj m_aGBackProj;
	cudaStream_t m_stream;
};

class CTomoWbp : public CTomoBase
{
public:
	CTomoWbp(void);
	~CTomoWbp(void);
	void Clean(void);
	void Setup
	( int iVolX, int iVolZ,
	  MD::CTiltSeries* pTiltSeries,
	  MAM::CAlignParam* pAlignParam
	);
	void DoIt
	( float* gfPadSinogram,
	  float* gfVolXZ,
	  cudaStream_t stream
	);
private:
	GRWeight m_aGRWeight;
};

class CTomoSart : public CTomoBase
{
public:
	CTomoSart(void);
	virtual ~CTomoSart(void);
	void Clean(void);
	void Setup
	( int iVolX,
	  int iVolZ,
	  int iNumSubsets,
	  int iNumIters,
	  MD::CTiltSeries* pTiltSeries,
	  MAM::CAlignParam* pAlignParam,
	  int iStartTilt,
	  int iNumTilts
	);
	void DoIt
	( float* gfPadSinogram,
	  float* gfVolXZ,
	  cudaStream_t stream = 0
	);
	int m_iNumIters;
private:
	void mExtractSinogram(int iY);
	void mForProj(int iStartProj, int iNumProjs);
	void mDiffProj(int iStartProj, int iNumProjs);
	void mBackProj
	( float* gfSinogram, 
	  int iStartProj, 
	  int iEndProj, 
	  float fRelax
	);
	//-----------------
	float m_fRelax;
	int m_iNumSubsets;
	int m_aiTiltRange[2]; // start and num tilts
	//-----------------
	GForProj m_aGForProj;
	GDiffProj m_aGDiffProj;
	//-----------------
	float* m_gfPadForProjs;
};

class CDoBaseRecon 
{
public:
	CDoBaseRecon(void);
	virtual ~CDoBaseRecon(void);
	virtual void Clean(void);
protected:
	float* mCalcProjXY(void);
	float* mCalcProjXZ(void);
	float* m_gfPadSinogram;
	float* m_pfPadSinogram;
	float* m_gfVolXZ;
	float* m_pfVolXZ;
	//-----------------
	MD::CTiltSeries* m_pTiltSeries;
	MD::CTiltSeries* m_pVolSeries;
	MAM::CAlignParam* m_pAlignParam;
	int m_iVolZ;
	cudaStream_t m_stream;
};

class CDoWbpRecon : public CDoBaseRecon
{
public:
	CDoWbpRecon(void);
	virtual ~CDoWbpRecon(void);
	void Clean(void);
	MD::CTiltSeries* DoIt
	( MD::CTiltSeries* pTiltSeries,
          MAM::CAlignParam* pAlignParam,
          int iVolZ
        );
private:
	void mDoIt(void);
	void mExtractSinogram(int iY);
	void mGetReconResult(int iY);
	void mReconstruct(int iY);
	//------------------------
	CTomoWbp m_aTomoWbp;
	cudaStream_t m_stream;
	cudaEvent_t m_eventSino;
};

class CDoSartRecon : public CDoBaseRecon
{
public:
	CDoSartRecon(void);
	virtual ~CDoSartRecon(void);
	void Clean(void);
	MD::CTiltSeries* DoIt
	( MD::CTiltSeries* pTiltSeries,
	  MAM::CAlignParam* pAlignParam,
	  int iStartTilt,
	  int iNumTilts,
	  int iVolZ,
	  int iIterations,
	  int iNumSubsets
	);
private:
	void mDoIt(void);
	void mExtractSinogram(int iY);
	void mGetReconResult(int iLastY);
	void mReconstruct(int iY);
	//-----------------
	int m_iStartTilt;
	int m_iNumTilts;
	int m_iNumIters;
	int m_iNumSubsets;
	//-----------------
	CTomoSart m_aTomoSart;
	cudaStream_t m_stream;
	cudaEvent_t m_eventSino;
};

class CCalcVolThick
{
public:
	CCalcVolThick(void);
	~CCalcVolThick(void);
	void DoIt(int iNthGpu);
	float GetThickness(bool bAngstrom);
	float GetLowEdge(bool bAngstrom);
	float GetHighEdge(bool bAngstrom);
private:
	float mMeasure(int iZ, int* piStart);
	void mSetup(void);
	void mClean(void);
	void mDetectEdges(float* pfCCs, int iSize);
	//-----------------
	MD::CTiltSeries* m_pVolSeries;
	MAU::GLocalCC2D* m_gLocalCC2D;
	float* m_gfImg1;
	float* m_gfImg2;
	int m_aiTileSize[2];
	//-----------------
	int m_aiSampleEdges[2];
	float m_fBinning;
	float m_fPixSize;
};

class CAlignMetric
{
public:
	CAlignMetric(void);
	~CAlignMetric(void);
	void Calculate(int iNthGpu, int iThickness);
	float m_fRms;
private:
	void mReproj(void);
	void mSetup(int* piImgSize);
	void mClean(void);
	//-----------------
	MD::CTiltSeries* m_pVolSeries;
	float* m_gfImg1;
	float* m_gfImg2;
	float* m_gfImg3;
	int m_aiTileSize[2];
	//-----------------
	float m_fBinning;
	float m_fPixSize;
};

}
