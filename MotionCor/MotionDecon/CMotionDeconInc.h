#pragma once

#include "../DataUtil/CDataUtilInc.h"
#include <stdio.h>
#include <cufft.h>

namespace MMD = McAreTomo::MotionCor::DataUtil;

namespace McAreTomo::MotionCor::MotionDecon
{

class GDeconFrame
{
public:
	GDeconFrame(void);
	~GDeconFrame(void);
	void CleanGpu(void);
	void CalcSinc2Sum
	(  float* pfInFrmShifts,
	   int iNumFrames,
	   int* piCmpSize
	);
	void DirWeightFrame
	(  int iFrame,
	   cufftComplex* pCmpFrm,
	   bool bGpu
	);
	void WeightFrame
	(  float fShift,
	   cufftComplex* pCmpFrm,
	   bool bGpu,
	   int* piCmpSize
	);
	int m_aiCmpSize[2];
	int m_iNumFrames;
private:
	cufftComplex* mH2D
	(  cufftComplex* pCmpFrm,
	   bool bGpu,
	   int* piCmpSize
	);
	void mD2H
	(  cufftComplex* gCmpFrm,
	   cufftComplex* pCmpFrm,
	   bool bGpu,
	   int* piCmpSize
	);
	float* m_gfSinc2Sum;
	float* m_pfInFrmShifts;
};

class GMotionWeight
{
public:
	GMotionWeight(void);
	~GMotionWeight(void);
	void DirWeight
	( float* pfMotion,
	  cufftComplex* pCmpFrm,
	  int* piCmpSize,
          cudaStream_t stream=0
	);
	void Weight
	( float fMotion,
	  cufftComplex* pCmpFrm,
	  int* piCmpSize,
          cudaStream_t stream=0
	);
};

class CInFrameMotion
{
public:
	CInFrameMotion(void);
	~CInFrameMotion(void);
	void SetFullShift(MMD::CStackShift* pStackShift);
	void SetPatchShifts(MMD::CPatchShifts* pPatchShifts);
	void DoFullMotion
	( int iFrame,
	  cufftComplex* pCmpFrm,
	  int* piCmpSize,
          cudaStream_t stream=0
	);
	void DoLocalMotion
	( int iFrame,
	  cufftComplex* pCmpFrm,
	  int* piCmpSize,
          cudaStream_t stream=0
	);
	void GetFullMotion
	( int iFrame,
	  float* pfMotion // 2 elements
	);
	void GetLocalMotion
	( int iFrame,
	  int iPatch,
	  float* pfMotion // 2 elements
	);
	int m_iNumFrames;
	int m_iNumPatches;
private:
	void mCalculate(MMD::CStackShift* pStackShift);
	MMD::CStackShift* m_pFullShift;
	MMD::CPatchShifts* m_pPatchShifts;
};

}

