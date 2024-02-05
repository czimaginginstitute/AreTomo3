#pragma once
#include "../CAreTomoInc.h"
#include "../MrcUtil/CMrcUtilFwd.h"

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
	void DoIt
	( float* gfPadSinogram, // y-slice of all projections
	  float* gfCosSin,
	  int iStartProj,    // starting projection of subset
	  int iNumProjs,     // number of projections in subset
	  bool bSart,
	  float fRelax,
	  float* gfVolXZ,    // y-slice of volume
	  cudaStream_t stream = 0
	);
private:
	dim3 m_aBlockDim;
	dim3 m_aGridDim;
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

class CTomoWbp
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
	int m_aiVolSize[2];
	int m_iPadProjX;
	int m_iNumProjs;
	float* m_gfCosSin;
	float* m_gfPadSinogram;
	float* m_gfVolXZ;
	GBackProj m_aGBackProj;
	GRWeight m_aGRWeight;
	GWeightProjs m_aGWeightProjs;
	MD::CTiltSeries* m_pTiltSeries;
	MAM::CAlignParam* m_pAlignParam;
	cudaStream_t m_stream;
};

class CTomoSart
{
public:
	CTomoSart(void);
	~CTomoSart(void);
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
	void mBackProj(float* gfSinogram, int iStartProj, 
		int iNumProjs, float fRelax);
	//-----------------
	float m_fRelax;
	int m_iNumSubsets;
	MD::CTiltSeries* m_pTiltSeries;
	MAM::CAlignParam* m_pAlignParam;
	int m_aiTiltRange[2]; // start and num tilts
	//-----------------
	GBackProj m_aGBackProj;
	GForProj m_aGForProj;
	GDiffProj m_aGDiffProj;
	GWeightProjs m_aGWeightProjs;
	//-----------------
	int m_aiVolSize[2];
	int m_iPadProjX;
	int m_iNumProjs;
	//----------------
	float* m_gfCosSin;
	float* m_gfPadForProjs;
	float* m_gfPadSinogram;
	float* m_gfVolXZ;
	cudaStream_t m_stream;
};

class CDoBaseRecon 
{
public:
	CDoBaseRecon(void);
	virtual ~CDoBaseRecon(void);
	virtual void Clean(void);
protected:
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

}
