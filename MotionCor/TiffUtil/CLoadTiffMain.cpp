#include "CTiffUtilInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include <Util/Util_Time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::TiffUtil;
namespace MMD = McAreTomo::MotionCor::DataUtil;

CLoadTiffMain::CLoadTiffMain(void)
{
}

CLoadTiffMain::~CLoadTiffMain(void)
{
}

bool CLoadTiffMain::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	m_bLoaded = true;
	//-----------------
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	m_iFile = open(pPackage->m_acMoviePath, O_RDONLY);
	if(m_iFile == -1)
	{	fprintf(stderr, "CLoadTiffMain: cannot open Tiff file, "
		   "skip,\n   %s\n\n", pPackage->m_acMoviePath);
		m_bLoaded = false;
		return false;
	}
	//-----------------
	mLoadHeader();
	mLoadStack();
	close(m_iFile);
	return m_bLoaded;
}

void CLoadTiffMain::mLoadHeader(void)
{
	CLoadTiffHeader aLoadHeader;
	aLoadHeader.DoIt(m_iFile);
	aLoadHeader.GetSize(m_aiStkSize, 3);
	m_iMode = aLoadHeader.GetMode();
	printf("TIFF file size & mode: %d  %d  %d  %d\n",
	   m_aiStkSize[0], m_aiStkSize[1], m_aiStkSize[2], m_iMode);
	//---------------------------------------------------------
	// 1. If we have per-frame dose, we should use it since it
	//    is more accurate than the MDOC's result.
	// 2. If not, we use MDOC's result for better than nothing.
	//---------------------------------------------------------
	CInput* pInput = CInput::GetInstance();
	float fImgDose = pInput->m_fFmDose * m_aiStkSize[2];
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	if(fImgDose > 0) pPackage->m_fTotalDose = fImgDose;
	//----------------------------------------------------
	// If s_pPackage does not have m_pFmIntParam, this is
	// a new package amd we create an object
	//----------------------------------------------------
	MMD::CFmIntParam* pFmIntParam = 
	   MMD::CFmIntParam::GetInstance(m_iNthGpu);
	pFmIntParam->Setup(m_aiStkSize[2], m_iMode, pPackage->m_fTotalDose);
	m_aiStkSize[2] = pFmIntParam->m_iNumIntFms;
	//------------------------------------------------------------
	// Calculate group parameters for global and local alignments
	//------------------------------------------------------------
	CMcInput* pMcInput = CMcInput::GetInstance();
	MMD::CFmGroupParam* pFmGroupParam = 0L; 
	pFmGroupParam = MMD::CFmGroupParam::GetInstance(m_iNthGpu, false);
	pFmGroupParam->Setup(pMcInput->m_aiGroup[0]);
	//-----------------
	pFmGroupParam = MMD::CFmGroupParam::GetInstance(m_iNthGpu, true);
	pFmGroupParam->Setup(pMcInput->m_aiGroup[1]);
	//-----------------------------------------------------------
	// Create the rendered stack if s_pPackage does not have one
	//-----------------------------------------------------------
	int iIntMode = m_iMode;
	if(pFmIntParam->bIntegrate()) iIntMode = Mrc::eMrcUChar;
	pPackage->m_pRawStack->Create(iIntMode, m_aiStkSize);
	//-----------------
	printf("Rendered size & mode:  %d  %d  %d  %d\n", 
	   m_aiStkSize[0], m_aiStkSize[1], m_aiStkSize[2], m_iMode);
}

void CLoadTiffMain::mLoadStack(void)
{
	Util_Time aTimer;
	aTimer.Measure();
	nvtxRangePushA("CLoadTiffMain");
	//-----------------
	m_pLoadTiffImage = new CLoadTiffImage;
	m_pLoadTiffImage->SetFile(m_iFile);
	//-----------------
	MMD::CFmIntParam* pFmIntParam = 
	   MMD::CFmIntParam::GetInstance(m_iNthGpu);
	if(pFmIntParam->bIntegrate()) mLoadInt();
	else mLoadSingle();
	//-----------------
	delete m_pLoadTiffImage;
	m_pLoadTiffImage = 0L;
	//-----------------
	nvtxRangePop();
	m_fLoadTime = aTimer.GetElapsedSeconds();
}

void CLoadTiffMain::mLoadSingle(void)
{
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	MMD::CFmIntParam* pFmIntParam = 
	   MMD::CFmIntParam::GetInstance(m_iNthGpu);
	MD::CMrcStack* pRawStack = pPackage->m_pRawStack;
	//-----------------
	for(int i=0; i<pRawStack->m_aiStkSize[2]; i++)
	{	int iIntFmStart = pFmIntParam->GetIntFmStart(i);
		void* pvFrame = pPackage->m_pRawStack->GetFrame(i);
		m_bLoaded = m_pLoadTiffImage->DoIt(iIntFmStart, 
		   (char*)pvFrame);
		if(!m_bLoaded) break;
	}
}

void CLoadTiffMain::mLoadInt(void)
{
	MD::CMcPackage* pPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	MD::CMrcStack* pRawStack = pPackage->m_pRawStack;
	MMD::CFmIntParam* pFmIntParam = 
	   MMD::CFmIntParam::GetInstance(m_iNthGpu);
	//-----------------
	size_t tFmBytes = pRawStack->m_tFmBytes;
	unsigned char *gucRaw = 0L, *gucSum = 0L;
	cudaMalloc(&gucSum, tFmBytes);
	cudaMalloc(&gucRaw, tFmBytes);
	//-----------------
	for(int i=0; i<pRawStack->m_aiStkSize[2]; i++)
	{	void* pvIntFm = pRawStack->GetFrame(i);
		//----------------
		int iIntFmStart = pFmIntParam->GetIntFmStart(i);
		int iIntFmSize = pFmIntParam->GetIntFmSize(i);
		m_bLoaded = m_pLoadTiffImage->DoIt(iIntFmStart, pvIntFm);
		//----------------
		if(iIntFmSize == 1)
		{	if(m_bLoaded) continue;
			else break;
		}
		//----------------
		cudaMemcpy(gucSum, pvIntFm, tFmBytes, cudaMemcpyDefault);
		MU::GAddFrames addFrames;
		//----------------
		for(int i=1; i<iIntFmSize; i++)
		{	m_bLoaded = m_pLoadTiffImage->DoIt(iIntFmStart + i,
			   (char*)pvIntFm);
			if(!m_bLoaded) break;
			cudaMemcpy(gucRaw, pvIntFm, tFmBytes, 
			   cudaMemcpyDefault);
			//---------------
			addFrames.DoIt(gucRaw, gucSum, gucSum, 
			   pRawStack->m_aiStkSize);
		}
		if(!m_bLoaded) break;
		cudaMemcpy(pvIntFm, gucSum, tFmBytes, cudaMemcpyDefault);
	}
	cudaFree(gucRaw);
	cudaFree(gucSum);
}
