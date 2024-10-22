#include "CProjAlignInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::ProjAlign;

CParam* CParam::m_pInstances = 0L;
int CParam::m_iNumGpus = 0;

void CParam::CreateInstances(int iNumGpus)
{
	if(iNumGpus == m_iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CParam[iNumGpus];
	//-----------------
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

CParam* CParam::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

void CParam::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CParam::CParam(void)
{
	m_iIterations = 5;
	m_fTol = 0.5f;
	m_iAlignZ = 600;
	m_afMaskSize[0] = 1.0f;
	m_afMaskSize[1] = 1.0f;
	m_fXcfSize = 1024.0f;
	m_iNthGpu = -1;
}

CParam::~CParam(void)
{
}

