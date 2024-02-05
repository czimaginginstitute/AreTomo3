#pragma once
#include "../CMotionCorInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include <tiffio.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace McAreTomo::MotionCor::EerUtil
{
class GAddRawFrame
{
public:
	GAddRawFrame(void);
	~GAddRawFrame(void);
	void DoIt
	( unsigned char* gucFrm1,
	  unsigned char* gucFrm2,
	  unsigned char* gucSum,
	  unsigned int uiPixels,
	  cudaStream_t stream = 0
	);
};

class CLoadEerHeader
{
public:
        CLoadEerHeader(void);
        ~CLoadEerHeader(void);
	bool DoIt(int iFile, int iEerSampling);
	int m_aiCamSize[2];      // TIFFTAG_IMAGEWIDTH TIFFTAG_IMAGEHEIGHT
	int m_aiFrmSize[2];
	int m_iNumFrames;
	int m_iNumBits;
	int m_iEerSampling;
	bool m_bLoaded;
private:
	bool mCheckError(void);
	void mCleanTiff(void);
	unsigned short m_usCompression;
	int m_iFile;
	TIFF* m_pTiff;
};

class CLoadEerFrames
{
public:
        CLoadEerFrames(void);
        ~CLoadEerFrames(void);
	void Clean(void);
	bool DoIt(int iFile, int iNumFrames);
	unsigned char* GetEerFrame(int iFrame);     // do not free
	int GetEerFrameSize(int iFrame);
	int m_iNumFrames;
private:
	void mReadFrame(int iFrame);
	TIFF* m_pTiff;
	unsigned char* m_pucFrames;
	int* m_piFrmStarts;
	int* m_piFrmSizes;
	int m_iBytesRead;
};

class CDecodeEerFrame
{
public:
	CDecodeEerFrame(void);
	~CDecodeEerFrame(void);
	void Setup(int* piCamSize, int iEerUpSampling);
	void Do7Bits
	( unsigned char* pucEerFrame,
	  int iEerFrameSize,
	  unsigned char* pucRawFrame
	);
	void Do8Bits
	( unsigned char* pucEerFrame,
	  int iEerFrameSize,
	  unsigned char* pucRawFrame
	);
	int m_aiFrmSize[2];
private:
	void mDo7BitsCounted(void);
	void mDo7BitsSuperRes(void);
	void mDo8BitsCounted(void);
	void mDo8BitsSuperRes(void);
	void mFindElectron(void);
	//-----------------------
	unsigned int m_uiCamPixels;
	unsigned char* m_pucEerFrame;
	unsigned char* m_pucRawFrame;
	unsigned int m_uiNumPixels;
	unsigned char m_ucS;
	unsigned int m_uiX;
	unsigned int m_uiY;
	int m_iUpSampling;
	int m_iEerFrameSize;
	int m_aiCamSize[2];
	int m_aiSuperResAnd[2];
	int m_aiSuperResShift[3];
};

class CRenderMrcStack 
{
public:
	CRenderMrcStack(void);
	~CRenderMrcStack(void);
	void DoIt
	( CLoadEerHeader* pLoadHeader,
	  CLoadEerFrames* pLoadFrames,
	  int iNthGpu 
	);
private:
	void mLoadHeader(void);
	void mLoadStack(int iNumFrames);
	void mRender(void);
	void mRenderInt(void);
	void mRenderFrame(int iIntFrm);
	void mDecodeFrame(int iEerFrame, unsigned char* pucDecodedFrm);
	//-----------------
	CDecodeEerFrame m_aDecodeEerFrame;
	CLoadEerHeader* m_pLoadHeader;
	CLoadEerFrames* m_pLoadFrames;
	//-----------------
	MD::CMrcStack* m_pRawStack;
	MMD::CFmIntParam* m_pFmIntParam;

	int m_iNthGpu;
};

class CLoadEerMain
{
public:
	CLoadEerMain(void);
	~CLoadEerMain(void);
	bool DoIt(int iNthGpu);
	bool m_bLoaded;
private:
	void mLoadHeader(void);
	void mLoadStack(void);
	void mClean(void);
	//-----------------
	int m_iNthGpu;
	int m_iFile;
	TIFF* m_pTiff;
	CLoadEerHeader* m_pLoadHeader;
	CLoadEerFrames* m_pLoadFrames;
	int m_aiStkSize[3]; // superRes X, Y, rendered frames
};
} //namespace MotionCor4

