#include "CDataUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>

using namespace McAreTomo::MotionCor::DataUtil;

CFmGroupParam* CFmGroupParam::m_pInstances = 0L;
int CFmGroupParam::m_iNumGpus = 0;

void CFmGroupParam::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	int iSize = 2 * iNumGpus;
	m_pInstances = new CFmGroupParam[iSize];
	//-----------------
	for(int i=0; i<iNumGpus; i++)
	{	int k = 2 * i;
		m_pInstances[k].m_iNthGpu = i;
		m_pInstances[k+1].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CFmGroupParam::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CFmGroupParam* CFmGroupParam::GetInstance(int iNthGpu, bool bLocal)
{
	int k = 2 * iNthGpu;
	if(bLocal) k += 1;
	return &m_pInstances[k];
}

CFmGroupParam::CFmGroupParam(void)
{
	m_iGroupSize = 1;
	m_iNumGroups = 0;
	m_iNumIntFms = 0;
	m_ppiGroupIdxs = 0L;
}


CFmGroupParam::~CFmGroupParam(void)
{
	this->mClean();
}

void CFmGroupParam::mClean(void)
{
	if(m_ppiGroupIdxs != 0L)
	{	for(int i=0; i<m_iNumGroups; i++)
		{	if(m_ppiGroupIdxs[i] != 0L)
			{	delete[] m_ppiGroupIdxs[i];
			}
		}
		delete[] m_ppiGroupIdxs;
		m_ppiGroupIdxs = 0L;
	}
}

void CFmGroupParam::mAlloc(void)
{
	if(m_iNumGroups <= 0) return;
	m_ppiGroupIdxs = new int*[m_iNumGroups];
	for(int i=0; i<m_iNumGroups; i++)
	{	m_ppiGroupIdxs[i] = new int[m_iGroupSize];
	}
}

//--------------------------------------------------------------------
// Make grouping a sliding window and circular average.
//--------------------------------------------------------------------
void CFmGroupParam::Setup(int iGroupSize)
{
	CFmIntParam* pFmIntParam = CFmIntParam::GetInstance(m_iNthGpu);
	if(m_iNumIntFms == pFmIntParam->m_iNumIntFms 
	   && m_iGroupSize == iGroupSize) return;
	//---------------------------
	mClean();
	m_iNumIntFms = pFmIntParam->m_iNumIntFms;
	m_iNumGroups = pFmIntParam->m_iNumIntFms;
	m_iGroupSize = iGroupSize;
	mAlloc();
	//---------------------------
	for(int g=0; g<m_iNumGroups; g++)
	{	int iStart = g - m_iGroupSize / 2;
		if(iStart < 0) iStart = 0;
		if((iStart + m_iGroupSize) > m_iNumIntFms)
		{	iStart = m_iNumIntFms - m_iGroupSize;
		}
		//-------------------
		int* piGroupIdxs = m_ppiGroupIdxs[g];
		for(int i=0; i<m_iGroupSize; i++)
		{	int k = iStart + i;
			//if(k < 0) k += m_iNumIntFms;
			//else if(k >= m_iNumIntFms) k -= m_iNumIntFms;
			piGroupIdxs[i] = k;
		}
	}
}

int* CFmGroupParam::GetGroupIdxs(int iGroup)
{
	int* piGroupIdxs = m_ppiGroupIdxs[iGroup];
	return piGroupIdxs;
}


