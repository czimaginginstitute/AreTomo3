#include "CPatchAlignInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::PatchAlign;

static float s_fD2R = 0.0174533f;
CPatchAlignMain* CPatchAlignMain::m_pInstances = 0L;
int CPatchAlignMain::m_iNumGpus = 0;

void CPatchAlignMain::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CPatchAlignMain[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
	//-----------------
	CDetectFeatures::CreateInstances(iNumGpus);
	CPatchTargets::CreateInstances(iNumGpus);
}

void CPatchAlignMain::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
	//-----------------
	CDetectFeatures::DeleteInstances();
	CPatchTargets::DeleteInstances();
}

CPatchAlignMain* CPatchAlignMain::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CPatchAlignMain::CPatchAlignMain(void)
{
}

CPatchAlignMain::~CPatchAlignMain(void)
{
}

void CPatchAlignMain::DoIt(void)
{	
	MD::CTsPackage* pPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	m_pTiltSeries = pPkg->GetSeries(0);
	m_pFullParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	m_pLocalParam = MAM::CLocalAlignParam::GetInstance(m_iNthGpu);
	m_pPatchShifts = MAM::CPatchShifts::GetInstance(m_iNthGpu);
	CPatchTargets* pPatchTargets = CPatchTargets::GetInstance(m_iNthGpu);
	//-----------------
	m_pLocalParam->Setup(m_pTiltSeries->m_aiStkSize[2],
	   pPatchTargets->m_iNumTgts);
	//-----------------
	m_pPatchShifts->Setup(pPatchTargets->m_iNumTgts, 
	   m_pTiltSeries->m_aiStkSize[2]);
	//-----------------
	m_pLocalAlign = new CLocalAlign;
	m_pLocalAlign->Setup(m_iNthGpu);
	for(int i=0; i<pPatchTargets->m_iNumTgts; i++)
	{	mAlignStack(i);
	}
	//-----------------
	CFitPatchShifts aFitPatchShifts;
	aFitPatchShifts.Setup(m_pFullParam, pPatchTargets->m_iNumTgts);
	aFitPatchShifts.DoIt(m_pPatchShifts, m_pLocalParam);
	//-----------------
	delete m_pLocalAlign; 
	m_pLocalAlign = 0L;
}

void CPatchAlignMain::mAlignStack(int iPatch)
{
	CPatchTargets* pPatchTargets = CPatchTargets::GetInstance(m_iNthGpu);
	int iLeft = pPatchTargets->m_iNumTgts - 1 - iPatch;
	//------------------
	int aiCent[2] = {0};
	pPatchTargets->GetTarget(iPatch, aiCent);
	//------------------
	MAM::CAlignParam* pAlignParam = m_pFullParam->GetCopy();
	//------------------
	printf("Align patch at (%d, %d), %d patches left\n", aiCent[0],
	   aiCent[1], iLeft);
	m_pLocalAlign->DoIt(pAlignParam, aiCent);
	m_pPatchShifts->SetRawShift(iPatch, pAlignParam);
	//-----------------
	/*
	CInput* pInput = CInput::GetInstance();
	char* pcLogFile = pInput->GetLogFile("PatchStack.txt", &iPatch);
	pAlignParam->LogShift(pcLogFile);
	if(pcLogFile != 0L) delete[] pcLogFile;
	*/
}

