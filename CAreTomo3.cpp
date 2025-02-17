#include "CMcAreTomoInc.h"
#include <Util/Util_Time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <nvToolsExt.h>

using namespace McAreTomo;

bool mCheckLoad(void);
bool mCheckSave(char* pcMrcFile);
bool mCheckGPUs(void);
void mCheckPeerAccess(void);

enum ERetCode 
{	eSuccess = 0,
	eNoFreeGPUs = 1,
	eInputOutputSame = 2,
	eFailLoad = 3,
	eFailSave = 4,
	eNoValidGPUs = 5,
	eFailProcess = 6
};
	

int main(int argc, char* argv[])
{
	CInput* pInput = CInput::GetInstance();
	CMcInput* pMcInput = CMcInput::GetInstance();
	CAtInput* pAtInput = CAtInput::GetInstance();
	//-----------------
	char acVersion[64] = {'\0'};
	strcpy(acVersion, "version 2.0.10, built on Feb 17, 2025");
	if(argc == 1)
	{	printf("\nAreTomo3: fully integrated and automated cryoET "
		   "pipeline for both real-time and offline tomographic "
		   "reconstruction starting from the correction of beam "
		   "induced motion on tilt movies.\n");
		printf("AreTomo3 --version: get version information\n");
		printf("AreTomo3 --help: get command line information.\n\n");
		return 0;	
	}
	if(argc == 2)
	{	if(strcasecmp(argv[1], "--version") == 0 ||
		   strcasecmp(argv[1], "-v") == 0)
		{	printf("%s\n", acVersion);
		}
		else if(strcasecmp(argv[1], "--help") == 0)
		{	printf("\nUsage: AreTomo3 Tags\n");
			pInput->ShowTags();
			pMcInput->ShowTags();
			pAtInput->ShowTags();
		}
		return 0;
	}
	//-----------------
	pInput->Parse(argc, argv);
	pMcInput->Parse(argc, argv);
	pAtInput->Parse(argc, argv);
	//-----------------
	CAreTomo3Json areTomo3Json;
	areTomo3Json.Create(acVersion);
	//-----------------
	cuInit(0);
	bool bGpu = mCheckGPUs();
	if(!bGpu) return eNoValidGPUs;
	//-----------------
	Util_Time aTimer;
	aTimer.Measure();
	CMcAreTomoMain mcAreTomoMain;
	bool bSuccess = mcAreTomoMain.DoIt();
	//-----------------
	nvtxRangePop();
	float fSecs = aTimer.GetElapsedSeconds();
	printf("Total time: %f sec\n", fSecs);
	if(!bSuccess) return eFailProcess;
	else return eSuccess;
}

bool mCheckLoad(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iSerial == 1) return true;
	//-------------------------------------
	/*
	fprintf(stderr, "Error: no valid input file name.\n"
	   "   mCheckLoad: \n\n");
	return false;
	*/
	return true;
}

bool mCheckSave(char* pcMrcFile)
{
	CInput* pInput = CInput::GetInstance();
        if(pInput->m_iSerial == 1) return true;
        //-------------------------------------
	Mrc::CSaveMrc aSaveMrc;
	bool bSave = aSaveMrc.OpenFile(pcMrcFile);
	if(bSave)
	{	remove(pcMrcFile);
		return true;
	}
	//------------------
	fprintf(stderr, "Error:cannot open output MRC file.\n"
	   "   mCheckSave: %s\n\n", pcMrcFile);
	return false;
}
	
bool mCheckGPUs(void)
{
	CInput* pInput = CInput::GetInstance();
	int* piGpuIds = new int[pInput->m_iNumGpus];
	int* piGpuMems = new int[pInput->m_iNumGpus];
	//-------------------------------------------
	int iCount = 0;
	cudaDeviceProp aDeviceProp;
	//-------------------------
	for(int i=0; i<pInput->m_iNumGpus; i++)
	{	int iGpuId = pInput->m_piGpuIDs[i];
		cudaError_t tErr = cudaSetDevice(iGpuId);
		if(tErr != cudaSuccess)
		{	printf
			(  "Info: skip device %d, %s\n",
			   pInput->m_piGpuIDs[i],
			   cudaGetErrorString(tErr)
			);
			continue;
		}
		piGpuIds[iCount] = iGpuId;
		cudaGetDeviceProperties(&aDeviceProp, iGpuId);
		piGpuMems[iCount] = (int)(aDeviceProp.totalGlobalMem
			/ (1024 * 1024));
		iCount++;
	}
	//---------------
	for(int i=0; i<iCount; i++)
	{	pInput->m_piGpuIDs[i] = piGpuIds[i];
		printf("GPU %d memory: %d MB\n", 
		   pInput->m_piGpuIDs[i], piGpuMems[i]);
	}
	printf("\n");
	pInput->m_iNumGpus = iCount;
	if(piGpuIds != 0L) delete[] piGpuIds;
	if(piGpuMems != 0L) delete[] piGpuMems;
	if(iCount > 0) return true;
	//-------------------------
	fprintf(stderr, "mCheckGPUs: no valid device detected.\n\n");
	return false;
}

void mCheckPeerAccess(void)
{
	cudaError_t cuErr;
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iNumGpus == 1) return;
	//---------------------------------
	for(int i=0; i<pInput->m_iNumGpus; i++)
	{	cudaSetDevice(pInput->m_piGpuIDs[i]);
		for(int j=0; j<pInput->m_iNumGpus; j++)
		{	if(j == i) continue;
			cuErr = cudaDeviceEnablePeerAccess
			( pInput->m_piGpuIDs[j], 0 );
			if(cuErr == cudaSuccess) continue;
			printf("mCheckGPUs: GPU %d cannot access %d memory\n"
				"   %s.\n\n", pInput->m_piGpuIDs[i], 
				pInput->m_piGpuIDs[j], 
				cudaGetErrorString(cuErr));
		}
	}
	cudaSetDevice(pInput->m_piGpuIDs[0]);
}
