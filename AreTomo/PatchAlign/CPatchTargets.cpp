#include "CPatchAlignInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::PatchAlign;

static float s_fD2R = 0.0174533f;

CPatchTargets* CPatchTargets::m_pInstances = 0L;
int CPatchTargets::m_iNumGpus = 0;

void CPatchTargets::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CPatchTargets[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CPatchTargets::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CPatchTargets* CPatchTargets::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CPatchTargets::CPatchTargets(void)
{
	m_iNumTgts = 0;
	//-----------------------------------------------------------
	// m_iTgtImg is the image where the targets are selected. It
	//-----------------------------------------------------------
	m_iTgtImg = -1;
	m_piTargets = 0L;
}

CPatchTargets::~CPatchTargets(void)
{
	this->Clean();
}

void CPatchTargets::Clean(void)
{
	if(m_piTargets != 0L) delete[] m_piTargets;
	m_piTargets = 0L;
}

void CPatchTargets::Detect(void)
{	
	bool bClean = true;
	this->Clean();
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
	m_iNumTgts = pInput->m_aiAtPatches[0] * pInput->m_aiAtPatches[1];
	if(m_iNumTgts <= 0) return;
	//-----------------
	MD::CTsPackage* pPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPkg->GetSeries(0);
	MAM::CAlignParam* pAlignParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	int iStkAxis = (pTiltSeries->m_aiStkSize[0] > 
	   pTiltSeries->m_aiStkSize[1]) ? 1 : 2;
	int iPatAxis = (pInput->m_aiAtPatches[0] > 
	   pInput->m_aiAtPatches[1]) ? 1 : 2;
	if(iStkAxis != iPatAxis)
	{	int iPatchesX = pInput->m_aiAtPatches[0];
		pInput->m_aiAtPatches[0] = pInput->m_aiAtPatches[1];
		pInput->m_aiAtPatches[1] = iPatchesX;
	}
	//-----------------
	m_iTgtImg = pAlignParam->GetFrameIdxFromTilt(0.0f);
	m_piTargets = new int[m_iNumTgts * 2];
	//-----------------
	CDetectFeatures* pDetectFeatures = 
	   CDetectFeatures::GetInstance(m_iNthGpu);
	pDetectFeatures->SetSize(pTiltSeries->m_aiStkSize,
	   pInput->m_aiAtPatches);
	//-----------------
	m_iTgtImg = pAlignParam->GetFrameIdxFromTilt(0.0f);
	float* pfZeroImg = (float*)pTiltSeries->GetFrame(m_iTgtImg);
	pDetectFeatures->DoIt(pfZeroImg);
	//-----------------
	printf("# Patch alignment: targets detected automatically\n");
	printf("# Image size: %d  %d\n", pTiltSeries->m_aiStkSize[0],
	   pTiltSeries->m_aiStkSize[1]);
	printf("# Number of patches: %d  %d\n", pInput->m_aiAtPatches[0],
	   pInput->m_aiAtPatches[1]);
	//----------------------------------------------------------------
        int iPatX = pTiltSeries->m_aiStkSize[0] / pInput->m_aiAtPatches[0];
        int iPatY = pTiltSeries->m_aiStkSize[1] / pInput->m_aiAtPatches[1];
        for(int y=0; y<pInput->m_aiAtPatches[1]; y++)
        {       int iCentY = y * iPatY + iPatY / 2;
                for(int x=0; x<pInput->m_aiAtPatches[0]; x++)
                {       int iCentX = x * iPatX + iPatX / 2;
                        int iPatch = y * pInput->m_aiAtPatches[0] + x;
			int* piTgt = m_piTargets + 2 * iPatch;
                        pDetectFeatures->GetCenter(iPatch, piTgt);
                        printf("%3d %6d %6d %6d %6d\n", iPatch + 1,
                           iCentX, iCentY, piTgt[0], piTgt[1]);
                }
        }
        printf("\n");
}

void CPatchTargets::GetTarget(int iTgt, int* piTgt)
{
	piTgt[0] = m_piTargets[iTgt * 2];
	piTgt[1] = m_piTargets[iTgt * 2 + 1];
}
