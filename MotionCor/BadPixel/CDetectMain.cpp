#include "CBadPixelInc.h"
#include "../CMotionCorInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Util/CUtilInc.h"
#include <Util/Util_Time.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

using namespace McAreTomo::MotionCor::BadPixel;
namespace MMU = McAreTomo::MotionCor::Util;
namespace MMM = McAreTomo::MotionCor::MrcUtil;

CDetectMain* CDetectMain::m_pInstances = 0L;
int CDetectMain::m_iNumGpus = 0;

void CDetectMain::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	//-----------------
	m_pInstances = new CDetectMain[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CDetectMain::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CDetectMain* CDetectMain::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CDetectMain::CDetectMain(void)
{
	m_fThreshold = 6.0f;
	m_aiDefectSize[0] = 6;
	m_aiDefectSize[1] = 6;
}

CDetectMain::~CDetectMain(void)
{
}

//-------------------------------------------------------------------
// 1. pMrcStack must be gain applied before calling this
//    function.
// 2. We should have a main worker thread where gain is applied,
//    bad pixe is detected, corrected, and drift alignment and
//    correction are performed.
//-------------------------------------------------------------------
void CDetectMain::DoIt(int iNthGpu)
{
	printf("Detect bad and hot pixels.\n");
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pSumBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::sum);
	//-----------------
	m_aiPadSize[0] = pSumBuffer->m_aiCmpSize[0] * 2;
	m_aiPadSize[1] = pSumBuffer->m_aiCmpSize[1];
	//-----------------
	// do not free
	//-----------------
	m_pucBadMap = (unsigned char*)pBufferPool->GetPinnedBuf(0); 
	memset(m_pucBadMap, 0, sizeof(char) * m_aiPadSize[0] * m_aiPadSize[1]);
	//-----------------
	MrcUtil::CSumFFTStack sumFFTStack;
	sumFFTStack.DoIt(MD::EBuffer::frm, false, iNthGpu);
	mDetectHot();
	//-----------------
	mDetectPatch();
	cudaDeviceSynchronize();
	//-----------------
	mLoadDefectFile();
}

unsigned char* CDetectMain::GetDefectMap(bool bClean)
{
	unsigned char* pucBadMap = m_pucBadMap;
	if(bClean) m_pucBadMap = 0L;
	return pucBadMap;
}

void CDetectMain::mDetectPatch(void)
{
	CLocalCCMap localCCMap;
	localCCMap.DoIt(m_aiDefectSize, m_iNthGpu);
	//-----------------
	MD::CBufferPool* pBufferPool = 
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pSumBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::sum);
	cufftComplex* gCmpSum = pSumBuffer->GetFrame(0);
	//-----------------
	MD::CStackBuffer* pTmpBuffer = 
	   pBufferPool->GetBuffer(MD::EBuffer::tmp);
	cufftComplex* gCmpBuf1 = pTmpBuffer->GetFrame(0);
	cufftComplex* gCmpBuf2 = pTmpBuffer->GetFrame(1);
	//-----------------
	float* gfPadSum = reinterpret_cast<float*>(gCmpSum);
	float* gfPadCC = reinterpret_cast<float*>(gCmpBuf1);
	float* gfPadBuf = reinterpret_cast<float*>(gCmpBuf2);
	//-----------------
	GDetectPatch detectPatch;
	detectPatch.DoIt(gfPadSum, gfPadCC, gfPadBuf, m_pucBadMap,
	   m_aiPadSize, m_aiDefectSize, m_fThreshold);
}

void CDetectMain::mDetectHot(void)
{
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pSumBuffer = pBufferPool->GetBuffer(MD::EBuffer::sum);
	cufftComplex* gCmpSum = pSumBuffer->GetFrame(0);
	//-----------------
	MD::CStackBuffer* pTmpBuffer = pBufferPool->GetBuffer(MD::EBuffer::tmp);
	cufftComplex* gCmpBuf = pTmpBuffer->GetFrame(0);
	float* gfPadSum = reinterpret_cast<float*>(gCmpSum);
	float* gfPadBuf = reinterpret_cast<float*>(gCmpBuf);
	//-----------------
	GDetectHot aDetectHot;
        aDetectHot.DoIt(gfPadSum, gfPadBuf, m_aiPadSize, 
	   m_fThreshold, m_pucBadMap);
}

void CDetectMain::mLabelDefects
(  	float* gfPadImg,
   	int* piDefects,
   	int iNumDefects
)
{	/*
	if(iNumDefects == 0) return;
	if(strlen(pInput->m_acTmpFile) == 0) return;
	//------------------------------------------
	GLabelDefect labelDefect;
	labelDefect.SetLabelSize(6);
	labelDefect.DoIt(gfImg, m_aiFrmSize, piDefects, iNumDefects);
	//-----------------------------------------------------------
	printf("Label defects in pre-correction image.\n");
	CInput* pInput = CInput::GetInstance();
	CSaveTempMrc aSTM;
	aSTM.SetFile(pInput->m_acTmpFile, "-Defect");
	aSTM.GDoIt(gfImg, 2, m_aiFrmSize);
	*/
}

void CDetectMain::mLoadDefectFile(void)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(strlen(pMcInput->m_acDefectFile) == 0) return;
	//-----------------
	FILE* pFile = fopen(pMcInput->m_acDefectFile, "rt");
	if(pFile == 0L) return;
	//---------------------
	printf("Load camera defects.\n");
	int iFrmSizeX = (m_aiPadSize[0] / 2 - 1) * 2;
	char acBuf[256] = {0};
	int aiEntry[4];
	//-------------
	while(!feof(pFile))
	{	fgets(acBuf, 256, pFile);
		int iItems = sscanf(acBuf, "%d  %d  %d  %d", aiEntry+0, 
			aiEntry+1, aiEntry+2, aiEntry+3);
		memset(acBuf, 0, sizeof(acBuf));
		if(iItems != 4) continue;
		//-----------------------
		printf("...... Defect: %6d  %6d  %6d  %6d\n",
			aiEntry[0], aiEntry[1], aiEntry[2], aiEntry[3]);
		//------------------------------------------------------
		if(aiEntry[0] < 0) aiEntry[0] = 0;
		if(aiEntry[1] < 0) aiEntry[1] = 0;
		int iEndX = aiEntry[0] + aiEntry[2];
		int iEndY = aiEntry[1] + aiEntry[3];
		if(iEndX > iFrmSizeX) iEndX = iFrmSizeX;
		if(iEndY > m_aiPadSize[1]) iEndY = m_aiPadSize[1];
		//------------------------------------------------
		for(int y=aiEntry[1]; y<iEndY; y++)
		{	int k = y * m_aiPadSize[0];
			for(int x=aiEntry[0]; x<iEndX; x++)
			{	m_pucBadMap[k+x] = 1;
			}
		}
	}
	fclose(pFile);
	printf("Load camera defects: done.\n\n");
}

