#include "CAlignInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::MotionCor::Align;

CPatchCenters* CPatchCenters::m_pInstances = 0L;
int CPatchCenters::m_iNumGpus = 0;

void CPatchCenters::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CPatchCenters[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

CPatchCenters* CPatchCenters::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

void CPatchCenters::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CPatchCenters::CPatchCenters(void)
{
	m_iNumPatches = 0;
	m_piPatStarts = 0L;
}

CPatchCenters::~CPatchCenters(void)
{
	if(m_piPatStarts != 0L) delete[] m_piPatStarts;
	m_piPatStarts = 0L;
}

void CPatchCenters::Calculate(void)
{
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pXcfBuffer = pBufferPool->GetBuffer(MD::EBuffer::xcf);
	MD::CStackBuffer* pPatBuffer = pBufferPool->GetBuffer(MD::EBuffer::pat);
	m_aiXcfSize[0] = (pXcfBuffer->m_aiCmpSize[0] - 1) * 2;
	m_aiXcfSize[1] = pXcfBuffer->m_aiCmpSize[1];
	//-----------------
	CMcInput* pInput = CMcInput::GetInstance();
	m_aiPatSize[0] = m_aiXcfSize[0] / pInput->m_aiNumPatches[0] / 2 * 2;
	m_aiPatSize[1] = m_aiXcfSize[1] / pInput->m_aiNumPatches[1] / 2 * 2;
	//-----------------
	m_iNumPatches = pInput->m_aiNumPatches[0] * pInput->m_aiNumPatches[1];
	if(m_piPatStarts != 0L) delete[] m_piPatStarts;
	m_piPatStarts = new int[2 * m_iNumPatches];
	//-----------------
	CDetectFeatures* pDetectFeatures = 
	   CDetectFeatures::GetInstance(m_iNthGpu);
	float afLoc[2], afNewLoc[2];
	for(int i=0; i<m_iNumPatches; i++)
	{	int x = i % pInput->m_aiNumPatches[0];
		int y = i / pInput->m_aiNumPatches[0];
		afLoc[0] = (x + 0.5f) * m_aiPatSize[0];
		afLoc[1] = (y + 0.5f) * m_aiPatSize[1];
		pDetectFeatures->GetCenter(i, m_aiXcfSize, afNewLoc);
		//----------------
		if(i == 0) 
		{	printf("# xcf img size: %d  %d\n", 
			   m_aiXcfSize[0], m_aiXcfSize[1]);
			printf("# CPatchCenters::Calculate()\n");
		}
		printf("%3d %9.2f  %9.2f  %9.2f  %9.2f\n", i,
		   afLoc[0], afLoc[1], afNewLoc[0], afNewLoc[1]);
		//----------------
		int j =  2 * i;
		m_piPatStarts[j] = mCalcStart(afNewLoc[0], 
		   m_aiPatSize[0], m_aiXcfSize[0]);
		m_piPatStarts[j+1] = mCalcStart(afNewLoc[1], 
		   m_aiPatSize[1], m_aiXcfSize[1]);
	}
}	

void CPatchCenters::GetCenter(int iPatch, int* piCenter)
{
	int i = 2 * iPatch;
	piCenter[0] = m_piPatStarts[i] + m_aiPatSize[0] / 2;
	piCenter[1] = m_piPatStarts[i+1] + m_aiPatSize[1] / 2;
}

void CPatchCenters::GetStart(int iPatch, int* piStart)
{
	int i = 2 * iPatch;
	piStart[0] = m_piPatStarts[i];
	piStart[1] = m_piPatStarts[i+1];
}	

int CPatchCenters::mCalcStart(float fCent, int iPatSize, int iXcfSize)
{
	int iStart = (int)(fCent - iPatSize / 2);
	if(iStart < 0) iStart = 0;
	else if((iStart + iPatSize) > iXcfSize) iStart =  iXcfSize - iPatSize;
	return iStart;
}

