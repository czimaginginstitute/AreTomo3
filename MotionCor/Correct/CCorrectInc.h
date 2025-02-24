#pragma once
#include "../CMotionCorInc.h"
#include "../Util/CUtilInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include "../MotionDecon/CMotionDeconInc.h"
#include <cufft.h>

namespace McAreTomo::MotionCor::Correct
{
class GWeightFrame
{
public:
	GWeightFrame(void);
	~GWeightFrame(void);
	void Clean(void);
	//-------------------------------------------------
	// fInitDose: Dose received before 1st frame.
	// fFramDose: Dose received by each frame.
	// piStkSize[0]: Frame width, or size x
	// piStkSize[1]: Frame height, or size y
	// fPixelSize: in angstrom.
	// piStkSize[2]: Number of frames.
	//-------------------------------------------------
	void BuildWeight
	( float fPixelSize,
	  int iKv,
	  float* pfFmDose,
	  int* piStkSize,
	  float* gfWeightSum,
	  cudaStream_t stream = 0
	);
	//-------------------------------------------------
	// Weighting is performed in Fourier space
	//-------------------------------------------------
	void DoIt
	( cufftComplex* gCmpFrame,
	  int iFrame,
	  cudaStream_t stream = 0
	);
private:
	int m_aiCmpSize[2];
	int m_iNumFrames;
	float* m_pfFmDose; // accumulated dose
	float* m_gfWeightSum;
};

//-----------------------------------------------------------------------------
// 1. This class generates both global-motion corrected stack and the sum.
// 2. The corrected stack is saved in pGFFTStack. Therefore, pGFFTStack
//    stores the real and padded global-motion corrected frames. This
//    stack will be further corrected for local motion.
// 3. The returned sum is in Fourier space. The caller will need to
//    bin to the specfied resolution.
//-----------------------------------------------------------------------------
class CGenRealStack
{
public:
	CGenRealStack(void);
	~CGenRealStack(void);
	void Setup
	( int iBuffer,
	  bool bGenReal,
	  int iNthGpu
	);
	void DoIt(MMD::CStackShift* pStackShift);
private:
	void mDoGpuFrames(void);
	void mDoCpuFrames(void);
	void mAlignFrame(cufftComplex* gCmpFrm);
	//-----------------
	bool m_bGenReal;
	int m_iNthGpu;
	MMD::CStackShift* m_pStackShift;
	//-----------------
	MU::CCufft2D* m_pCufft2D;
	cudaStream_t m_streams[2];
	MotionDecon::CInFrameMotion m_aInFrameMotion;
	MD::CStackBuffer* m_pFrmBuffer;
	MD::CStackBuffer* m_pTmpBuffer;
	int m_iFrame;
};

class CCorrectFullShift 
{
public:
	CCorrectFullShift(void);
	virtual ~CCorrectFullShift(void);
	void Setup
	( MMD::CStackShift* pStackShift,
	  int iNthGpu
	);
	void DoIt(void);
protected:
	void mCorrectMag(void);
	void mUnpadSums(void);
	//-----------------
	void mCorrectGpuFrames(void);
	void mCorrectCpuFrames(void);
	void mGenSums(cufftComplex* gCmpFrm);
	virtual void mAlignFrame(cufftComplex* gCmpFrm);
	void mMotionDecon(cufftComplex* gCmpFrm);
	void mSum(cufftComplex* gCmpFrm, int iNthSum);
	//-----------------
	MMD::CStackShift* m_pFullShift;
	MotionDecon::CInFrameMotion m_aInFrameMotion;
	MU::CCufft2D* m_pForwardFFT;
	MU::CCufft2D* m_pInverseFFT;
	MD::CStackBuffer* m_pFrmBuffer;
	MD::CStackBuffer* m_pSumBuffer;
	MD::CStackBuffer* m_pTmpBuffer;
	cudaStream_t m_streams[2];
	int m_iNthGpu;
	int m_aiInCmpSize[2];
	int m_aiInPadSize[2];
	int m_aiOutCmpSize[2];
	int m_aiOutPadSize[2];
	int m_iFrame;
};

class GCorrectPatchShift : public CCorrectFullShift 
{
public:
	GCorrectPatchShift(void);
	virtual ~GCorrectPatchShift(void);
	void DoIt
	( MMD::CPatchShifts* pPatchShifts,
	  int iNthGpu
	);
protected:
	void mClean(void);
	void mDoIt(void);
	void mCorrectCpuFrames(void);
	void mCorrectGpuFrames(void);
	virtual void mAlignFrame(cufftComplex* gCmpFrm);
	void mCalcMeanShift(int iStream);
	void mSetupUpSample(void);
	void mUpSample(cufftComplex* gCmpFrm);
	//-----------------
	MMD::CPatchShifts* m_pPatchShifts;
	float* m_gfPatShifts;
	bool* m_gbBadShifts;
	float* m_gfPatCenters;
	//-----------------
	cufftComplex* m_gCmpUpsampled;
	int m_aiUpCmpSize[2];
	int m_iUpsample;
	//-----------------
	dim3 m_aBlockDim;
	dim3 m_aGridDim;
	int m_iNthGpu;
};	

//-------------------------------------------------------------------
// For correcting anisotropic magnification.
//-------------------------------------------------------------------
class GStretch
{
public:
	GStretch(void);
	~GStretch(void);
	void Setup(float fStretch, float fStretchAxis);
	float* GetMatrix(bool bGpu); // [out]
	void DoIt
	( float* gfInImg,
	  bool bPadded,
	  int* piFrmSize,
	  float* gfOutImg
	);
	void Unstretch(float* pfInShift, float* pfOutShift);
	//--------------------------------------------------
	float m_afMatrix[3];
private:
	float* mCopyMatrixToGpu(float* pfMatrix);
	void mCreateTexture(float* pfImg, bool bGpu, int* piImgSize);
	float m_fDet;
};

class CCorrMagThread
{
public:
        CCorrMagThread(void);
        ~CCorrMagThread(void);
        void Run
        (  MD::CMrcStack* pMrcStack,
           float fStretch,
           float fStretchAxis,
           Util::CNextItem* pNextItem,
           int iGpuId
        );
        void ThreadMain(void);
private:
        MD::CMrcStack* m_pMrcStack;
        float m_fStretch;
        float m_fStretchAxis;
        int m_iGpuId;
};

}
