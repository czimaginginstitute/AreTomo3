#include "CInput.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "Align/CAlignInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <Util/Util_Time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

bool mCheckLoad(char* pcMrcFile);
bool mCheckSave(char* pcMrcFile);
bool mCheckGPUs(int iGpuID);

int main(int argc, char* argv[])
{
	cuInit(0);
	printf("\nUsage: CommonLine Tags\n");
	CInput* pInput = CInput::GetInstance();
	pInput->ShowTags();
	pInput->Parse(argc, argv);
	//-----------------------
	bool bLoad = mCheckLoad(pInput->m_acInMrcFile);
	bool bSave = mCheckSave(pInput->m_acOutMrcFile);
	bool bGpu = mCheckGPUs(0);
	if(!bLoad || !bSave || !bGpu) return 1;
	//-------------------------------------
	Util_Time aTimer;
	aTimer.Measure();
	MrcUtil::CLoadStack* pLoadStack
	= MrcUtil::CLoadStack::GetInstance();
	pLoadStack->OpenFile(pInput->m_acInMrcFile);
	pLoadStack->DoIt();
	MrcUtil::CTomoStack* pTomoStack
	= pLoadStack->GetStack(true);
	//---------------------------
	CommonLine::CAlignParam* pAlignParam
	= CommonLine::CAlignParam::GetInstance();
	pAlignParam->m_fAngStep = pInput->m_fAngStep;
	pAlignParam->m_iNumSteps = pInput->m_iNumSteps;
	//---------------------------------------------
	CommonLine::CAlignMain aAlignMain;
	aAlignMain.DoIt(pTomoStack);
	//--------------------------
	if(pTomoStack != 0L) delete pTomoStack;
	float fSecs = aTimer.GetElapsedSeconds();
	printf("Total time: %f sec\n", fSecs);
	return 0;
}

bool mCheckLoad(char* pcMrcFile)
{
	Mrc::CLoadMrc aLoadMrc;
	bool bLoad = aLoadMrc.OpenFile(pcMrcFile);
	if(bLoad) return true;
	printf("Error: Unable to open input MRC file.\n");
	printf("...... %s\n\n", pcMrcFile);
	return false;
}

bool mCheckSave(char* pcMrcFile)
{
	Mrc::CSaveMrc aSaveMrc;
	bool bSave = aSaveMrc.OpenFile(pcMrcFile);
	remove(pcMrcFile);
	if(bSave) return true;
	//--------------------
	printf("Error: Unable to open output MRC file.\n");
	printf("...... %s\n\n", pcMrcFile);
	return false;
}
	
bool mCheckGPUs(int iGpuID)
{
	cudaSetDevice(iGpuID);
	return true;
}
