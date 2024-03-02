#include "CMcAreTomoInc.h"
#include "MotionCor/CMotionCorInc.h"
#include "AreTomo/CAreTomoInc.h"
#include "MaUtil/CMaUtilInc.h"
#include "DataUtil/CDataUtilInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>

using namespace McAreTomo;

CProcessThread* CProcessThread::m_pInstances = 0L;
int CProcessThread::m_iNumGpus = 0;
std::unordered_map<std::string, int>* CProcessThread::m_pMdocFiles = 0L;

void CProcessThread::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	//-----------------
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CProcessThread[iNumGpus];
	//-----------------
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
	//--------------------------------------------------
	// store reading-failed mdoc files and the number
	// of attempts.
	//--------------------------------------------------
	m_pMdocFiles = new std::unordered_map<std::string, int>;
}

void CProcessThread::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
	//-----------------
	m_pMdocFiles->clear();
	delete m_pMdocFiles;
	m_pMdocFiles = 0L;	
}

CProcessThread* CProcessThread::GetFreeThread(void)
{
	int iWaitTime = 0.0f;
	while(true)
	{	for(int i=0; i<m_iNumGpus; i++)
		{	CProcessThread* pThread = &m_pInstances[i];
			if(!pThread->IsAlive()) return pThread;
		}
		for(int i=0; i<m_iNumGpus; i++)
		{	CProcessThread* pThread = &m_pInstances[i];
			pThread->WaitForExit(0.1f);
			if(!pThread->IsAlive()) return pThread;
			else iWaitTime += 0.1f;
		}
		if(iWaitTime > 600.0f) break;
	}
	return 0L;
}

bool CProcessThread::WaitExitAll(float fSeconds)
{
	for(int i=0; i<m_iNumGpus; i++)
	{	CProcessThread* pThread = &m_pInstances[i];
		if(!pThread->IsAlive()) continue;
		//----------------
		pThread->WaitForExit(fSeconds);
		if(pThread->IsAlive()) return false;
	}
	return true; 
}

CProcessThread::CProcessThread(void)
{
}

CProcessThread::~CProcessThread(void)
{
}

bool CProcessThread::DoIt(void)
{
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CReadMdoc* pReadMdoc = MD::CReadMdoc::GetInstance(m_iNthGpu);
	bool bLoaded = pReadMdoc->DoIt(pTsPackage->m_pcMdocFile);
	if(bLoaded)
	{	this->Start();
		return true;
	}
	//-----------------
	MD::CStackFolder* pStackFolder = MD::CStackFolder::GetInstance();
	auto item = m_pMdocFiles->find(pTsPackage->m_pcMdocFile);
	if(item == m_pMdocFiles->end())
	{	m_pMdocFiles->insert({pTsPackage->m_pcMdocFile, 1});
		pStackFolder->PushFile(pTsPackage->m_pcMdocFile);
		return true;
	}
	//-----------------
	if(item->second < 10)
	{	item->second = item->second + 1;
		pStackFolder->PushFile(pTsPackage->m_pcMdocFile);
		return true;
	}
	//-----------------
	printf("Warning: failed to read mdoc file.\n");
	printf("   mdoc file: %s\n\n", pTsPackage->m_pcMdocFile);
	return false;
}

void CProcessThread::ThreadMain(void)
{
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[m_iNthGpu]);
	//-----------------
	MD::CReadMdoc* pReadMdoc = MD::CReadMdoc::GetInstance(m_iNthGpu);
	MD::CLogFiles* pLogFiles = MD::CLogFiles::GetInstance(m_iNthGpu);
	pLogFiles->Create(pReadMdoc->m_acMdocFile);
	//-----------------	
	mProcessTsPackage();
	printf("GPU %d: process thread exiting.\n\n", m_iNthGpu);
}

void CProcessThread::mProcessTsPackage(void)
{
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CReadMdoc* pReadMdoc = MD::CReadMdoc::GetInstance(m_iNthGpu);
	//-----------------
	bool bSuccess = true;
	for(int i=0; i<pReadMdoc->m_iNumTilts; i++)
	{	mProcessMovie(i);
		mAssembleTiltSeries(i);
	}
	//--------------------------------------------------
	// 1) Tilt series are sorted by tilt angle and then
	// saved into MRC file.
	// 2) The subsequent processing is done on the
	// tilt-angle sorted tilt series.
	//--------------------------------------------------
	pTsPackage->SortTiltSeries(0);
	pTsPackage->SaveTiltSeries();
	//--------------------------------------------------
	// 1) Since the saved tilt series have been sorted
	// by tilt angles, its section indices should be
	// in ascending order as the tilt angles.
	// 2) Note: if not sorted by tilt angles, section
	// indices should be the same as acquisition ones.
	//--------------------------------------------------
	pTsPackage->ResetSectionIndices();
	//----------------------------------------------
	// Start AreTomo processing: to be implemented.
	//----------------------------------------------
	mProcessTiltSeries();
}

void CProcessThread::mProcessMovie(int iTilt)
{
	CInput* pInput = CInput::GetInstance();
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
        MD::CReadMdoc* pReadMdoc = MD::CReadMdoc::GetInstance(m_iNthGpu);
	//-----------------
	char* pcFileName = pReadMdoc->GetFrameFileName(iTilt);
	pMcPackage->SetMovieName(pcFileName);
	pMcPackage->m_iAcqIdx = pReadMdoc->GetAcqIdx(iTilt);
	pMcPackage->m_fTilt = pReadMdoc->GetTilt(iTilt);
	pMcPackage->m_fPixSize = pInput->m_fPixSize;
	//-----------------
	printf("GPU %d: Motion correct %s\n"
	   "------------------\n\n", 
	   m_iNthGpu, pcFileName);
	//-----------------
	MotionCor::CMotionCorMain mcMain;
	mcMain.DoIt(m_iNthGpu);
}

void CProcessThread::mAssembleTiltSeries(int iTilt)
{
	MD::CReadMdoc* pReadMdoc = MD::CReadMdoc::GetInstance(m_iNthGpu);
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	//-----------------
	if(iTilt == 0) pTsPackage->CreateTiltSeries();
	//-----------------
	float fTilt = pReadMdoc->GetTilt(iTilt);
	pTsPackage->SetTiltAngle(iTilt, fTilt);
	//--------------------------------------------------
	// 1) when processing starts with movies, section
	// indices are the same as acquisition  indices. 
	// 2) when starting with MRC files of tilt series,
	// section indices (MRC indices) likely differs
	// from acquisition indices since MRC files usually
	// sort tilt images in terms of tilt angles, not
	// acquisition sequence.
	//--------------------------------------------------
	int iAcqIdx = pReadMdoc->GetAcqIdx(iTilt);
	pTsPackage->SetAcqIdx(iTilt, iAcqIdx);
	pTsPackage->SetSecIdx(iTilt, iAcqIdx);
	//-----------------
	pTsPackage->SetSums(iTilt, pMcPackage->m_pAlnSums);
	pTsPackage->SetImgDose(pMcPackage->m_pRawStack->m_fStkDose);
}

void CProcessThread::mProcessTiltSeries(void)
{
	MA::CAreTomoMain areTomoMain;
	areTomoMain.DoIt(m_iNthGpu);
}
