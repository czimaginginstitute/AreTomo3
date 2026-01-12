#pragma once
#include "../CMotionCorInc.h"
#include <Util/Util_Thread.h>
#include <Mrcfile/CMrcFileInc.h>
#include <cuda.h>

namespace McAreTomo::MotionCor::MrcUtil
{
class CLoadRefs
{
public:
	static CLoadRefs* GetInstance(void);
	static void DeleteInstance(void);
	~CLoadRefs(void);
	void CleanRefs(void);
	bool LoadGain(char* pcMrcFile);
	bool LoadDark(char* pcMrcFile);
	void PostProcess(int iRotFact, int iFlip, int iInverse);
	bool AugmentRefs(int* piFmSize);
	float* m_pfGain;
	float* m_pfDark;
	int m_aiRefSize[2];
private:
	void mClearGain(void);
	void mClearDark(void);
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

class GAugmentRef
{
public:
	GAugmentRef(void);
	~GAugmentRef(void);
	void DoIt
	( float* gfInRef, int* piInSize,
	  float* gfOutRef, int* piOutSize
	);
};

class GApplyRefsToFrame
{
public:
	GApplyRefsToFrame(void);
	~GApplyRefsToFrame(void);
	void SetRefs
	( float* gfGain, // Already cropped if "-Crop" is enabled
	  float* gfDark  // Already cropped if "-Crop" is enabled
	);
	void SetSizes
	( int* piMrcSize,// They are larger than piFrmSize that is
	  int* piFrmSize,// the size after cropping.
	  bool bFrmPadded
	);
	void Unpack
	( unsigned char* gucPkdFrm,
	  unsigned char* gucRawFrm,
	  int* piFrmSize,
	  cudaStream_t stream=0
	);
	void DoIt
	( void* gvFrame, 
	  int iMrcMode, 
	  float* gfFrame,
	  cudaStream_t stream=0
	);
	void DoRaw
	( unsigned char* gucRawFrm, 
	  float* gfFrame,
	  cudaStream_t stream=0
	);
	void DoPkd
	( unsigned char* gucPkdFrm, 
	  float* gfFrame,
	  cudaStream_t stream=0
	);
	void DoShort
	( short* gsFrm, 
	  float* gfFrame,
	  cudaStream_t stream=0
	);
	void DoUShort
	( unsigned short* gusFrm, 
	  float* gfFrame,
	  cudaStream_t stream=0
	);
	void DoFloat
	( float* gfInFrm, 
	  float* gfOutFrm, 
	  cudaStream_t stream=0
	);
private:
	int m_aiMrcSize[2];
	int m_aiFrmSize[2];
	int m_iPadSizeX;
	int m_iMrcOffset;
	float* m_gfGain;
	float* m_gfDark;
};

class CApplyRefs 
{
public:
	CApplyRefs(void);
	~CApplyRefs(void);
	void DoIt(float* pfGain, float* pfDark, int iNthGpu);
private:
	void mCopyRefs(float* pfGain, float* pfDark);
	void mCorrectGpuFrames(void);
	void mCorrectCpuFrames(void); 
	void mApplyRefs(cufftComplex* gCmpFrm, int iStream);
	//-----------------
	int m_iNthGpu;
	//-----------------
	MD::CStackBuffer* m_pFrmBuffer;
	MD::CStackBuffer* m_pTmpBuffer;
	MD::CStackBuffer* m_pSumBuffer;
	MD::CMrcStack* m_pRawStack;
	float* m_gfGain;
	float* m_gfDark;
	int m_iFrame;
	cudaStream_t m_streams[2];
	void* m_pvMrcFrames[2];
	GApplyRefsToFrame m_aGAppRefsToFrame;
};

class G90Rotate2D
{
public:
	G90Rotate2D(void);
	~G90Rotate2D(void);
   void GetRotSize
     (  int* piSize,
        int iRotFactor,
        int* piRotSize
     );
	void Setup
	(  int* piImgSize,
	   int iRotFactor
	);
	void DoIt
	(  float* pfImg,
	   bool bGpu
	);
	float* GetRotImg(bool bClean);
	void GetRotImg(float* pfRotImg, bool bGpu);
	float* m_gfRotImg;
	int m_aiRotSize[2];
private:
	void* mCopyToDevice(void* pvData, int iBytes);
	int m_aiImgSize[2];
	int m_aiCosSin[2];
	int m_iImgBytes;
};	// G90Rotate2D

class GFlip2D
{
public:
	GFlip2D(void);
	~GFlip2D(void);
	void Vertical(float* pfImg, bool bGpu, int* piImgSize);
	void Horizontal(float* pfImg, bool bGpu, int* piImgSize);
private:
	float* mCopyToDevice(float* pfImg, int* piImgSize);
	void mVertical(float* gfImg, int* piImgSize);
	void mHorizontal(float* gfImg, int* piImgSize);
};	//GFlip2D

class GInverse2D
{
public:
	GInverse2D(void);
	~GInverse2D(void);
	void DoIt(float* pfImg, bool bGpu, int* piImgSize);
private:
	void mInverse(float* gfImg, int* piImgSize);
};

//-------------------------------------------------------------------
// 1. Sum all the frames in the stack. These frames are padded
//    if they are in real-space.
//-------------------------------------------------------------------
class CSumFFTStack 
{
public:
	CSumFFTStack(void);
	~CSumFFTStack(void);
	void DoIt(int iBuffer, bool bSplitSum, int iNthGpu);
private:
	void mSumFrames(void);
	void mWait(void);
	void mSumGpuFrames(void);
	void mSumCpuFrames(void);
	void mSplitSums(void);
	//-----------------
	int m_iBuffer;
	bool m_bSplitSum;
	int m_iNthGpu;
	cudaStream_t m_streams[2];
};

}

namespace MMM = McAreTomo::MotionCor::MrcUtil;
