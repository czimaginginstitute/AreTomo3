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
	// load gain and/or dark references.
	//-----------------------------------
	MM::CMotionCorMain::LoadRefs();	
	MM::CMotionCorMain::LoadFmIntFile();
	//--------------------------------------------------------
	// wait a new movie for 10 minutes and quit if not found.
	//--------------------------------------------------------
	while(true)
	{	int iQueueSize = s_pStackFolder->GetQueueSize();
		if(iQueueSize > 0) mProcess();
		bool bExit = s_pStackFolder->WaitForExit(1.0f);
		if(bExit) break;
	}
	printf("All mdoc files have been processed, "
	   "waiting processing to finish.\n\n");	
	//-----------------
	while(true)
	{	bool bExit = CProcessThread::WaitExitAll(1.0f);
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
	char* pcMdocFile = s_pStackFolder->GetFile(true);
	pTsPackage->SetMdoc(pcMdocFile);
	if(pcMdocFile != 0L) delete[] pcMdocFile;
	//-----------------
	pProcessThread->DoIt();
}

