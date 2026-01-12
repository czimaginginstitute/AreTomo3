#include "CMcAreTomoInc.h"
#include "MotionCor/CMotionCorInc.h"
#include "AreTomo/CAreTomoInc.h"
#include "DataUtil/CDataUtilInc.h"
#include <Util/Util_Time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

using namespace McAreTomo;
namespace MD = McAreTomo::DataUtil;
namespace MM = McAreTomo::MotionCor;

static MD::CStackFolder* s_pStackFolder = 0L;

CMcAreTomoMain::CMcAreTomoMain(void)
{
	CInput* pInput = CInput::GetInstance();
	int iNumGpus = pInput->m_iNumGpus;
	//-----------------
	MD::CDuInstances::CreateInstances(iNumGpus);
	MotionCor::CMcInstances::CreateInstances(iNumGpus);
	AreTomo::CAtInstances::CreateInstances(iNumGpus);
	//-----------------
	CProcessThread::CreateInstances(iNumGpus);
}

CMcAreTomoMain::~CMcAreTomoMain(void)
{
	CProcessThread::DeleteInstances();
	MD::CDuInstances::DeleteInstances();
	MotionCor::CMcInstances::DeleteInstances();
	AreTomo::CAtInstances::DeleteInstances();
}

bool CMcAreTomoMain::DoIt(void)
{
	//---------------------------------------------------------
	// 1) Load MdocDone.txt from the output folder if there 
	// is one. 2) This is used for resuming processing without
	// reppcessing those that have been processed.
	//--------------------------------------------------
	MD::CReadMdocDone* pReadMdocDone = MD::CReadMdocDone::GetInstance();
	pReadMdocDone->DoIt();
	//---------------------------------------------------------
	// 1) If -Resume 1 and -Cmd 0 are both specified,
	// CStackFolder checks a mdoc file name is in the list of 
	// processed mdoc files. If yes, this mdoc will not be 
	// processed.
	//---------------------------------------------------------
	s_pStackFolder = MD::CStackFolder::GetInstance();
	bool bSuccess = s_pStackFolder->ReadFiles();
	if(!bSuccess)
	{	fprintf(stderr, "Error: no input image files "
		   "are found, quit.\n\n");
		return false;
	}
	//--------------------------------------------------
	// Use the first GPU since dark and gain references
	// are allocated in pinned memory.
	//--------------------------------------------------
	CInput* pInput = CInput::GetInstance();
	cudaSetDevice(pInput->m_piGpuIDs[0]);
	//-----------------------------------
	// load gain and/or dark references
	// only when we start from movies.
	//-----------------------------------
	if(pInput->m_iCmd == 0)
	{	MM::CMotionCorMain::LoadRefs();	
	}
	//--------------------------------------------------------
	// wait a new movie for 10 minutes and quit if not found.
	//--------------------------------------------------------
	bool bExit = false;
	while(true)
	{	int iQueueSize = s_pStackFolder->GetQueueSize();
		if(iQueueSize > 0) 
		{	mProcess();
			if(pInput->m_iCmd == 0)
			{	s_pStackFolder->WaitForExit(5.0f);
			}
			continue;
		}
		bExit = s_pStackFolder->WaitForExit(1.0f);
		if(bExit) break;
	}
	printf("All input files have been processed, "
	   "waiting processing to finish.\n\n");
	//-----------------
	while(true)
	{	bExit = CProcessThread::WaitExitAll(1.0f);
		if(bExit) break;
	}
	printf("All threads have finished, program exits.\n\n");
	return true;
}

void CMcAreTomoMain::mProcess(void)
{
	CProcessThread* pProcessThread = CProcessThread::GetFreeThread();
	if(pProcessThread == 0L)
	{	printf("All GPUs are used, wait ......\n\n");
		return;
	}
	int iNthGpu = pProcessThread->m_iNthGpu; 
	//-----------------
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(iNthGpu);
	char* pcInFile = s_pStackFolder->GetFile(true);
	pTsPackage->SetInFile(pcInFile);
	if(pcInFile != 0L) delete[] pcInFile;
	//-----------------
	pProcessThread->DoIt();
}

