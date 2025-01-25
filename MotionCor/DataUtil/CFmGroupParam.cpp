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
	m_piGroupStart = 0L;
	m_piGroupSize = 0L;
	m_pfGroupCenters = 0L;
}


CFmGroupParam::~CFmGroupParam(void)
{
	this->mClean();
}

void CFmGroupParam::Setup(int iGroupSize)
{
	CFmIntParam* pFmIntParam = CFmIntParam::GetInstance(m_iNthGpu);
	if(m_iNumIntFms == pFmIntParam->m_iNumIntFms 
	   && m_iGroupSize == iGroupSize) return;
	//-----------------
	mClean();
	m_iNumIntFms = pFmIntParam->m_iNumIntFms;
	m_iNumGroups = m_iNumIntFms;
	m_iGroupSize = iGroupSize;
	m_bGrouping = (m_iGroupSize > 1) ? true : false;	
	mAllocate();
	//---------------------------------------------------------------
	// Example of group by size. Example of group by dose
	// 20   2   0.5              20  1  0.5
	// 40   4   0.5              40  1  0.7
	// 80   8   0.5              80  1  1.2
	//---------------------------------------------------------------
	mFindMaxGroupRawFms();
	mGroupByRawSize();
	//-----------------
	for(int g=0; g<m_iNumGroups; g++)
	{	float fSumCents = 0.0f;
		for(int i=0; i<m_piGroupSize[g]; i++)
		{	int iIntFm = m_piGroupStart[g] + i;
			fSumCents += pFmIntParam->m_pfIntFmCents[iIntFm];
		}
		m_pfGroupCenters[g] = fSumCents / m_piGroupSize[g];
	}
}

void CFmGroupParam::mGroupByRawSize(void)
{
	CFmIntParam* pFmIntParam = CFmIntParam::GetInstance(m_iNthGpu);
	m_iNumGroups = 0;
	int iIntFm = 0;
	//-----------------
	for(int i=0; i<m_iNumIntFms; i++)
	{	m_piGroupStart[i] = iIntFm;
		int iRawFms = 0;
		while(true)
		{	iRawFms += pFmIntParam->GetIntFmSize(iIntFm);
			m_piGroupSize[i] += 1;
			iIntFm += 1;
			if(iIntFm >= m_iNumIntFms) break;
			if(iRawFms >= m_iMaxGroupRawFms) break;
                }
                m_iNumGroups += 1;
                if(iIntFm >= m_iNumIntFms) break;
        }
}

int CFmGroupParam::GetGroupStart(int iGroup)
{
	return m_piGroupStart[iGroup];
}

int CFmGroupParam::GetGroupSize(int iGroup)
{
	return m_piGroupSize[iGroup];
}

float CFmGroupParam::GetGroupCenter(int iGroup)
{
	return m_pfGroupCenters[iGroup];
}

void CFmGroupParam::mFindMaxGroupRawFms(void)
{
	CFmIntParam* pFmIntParam = CFmIntParam::GetInstance(m_iNthGpu);
	int iMinRawFms = (int)1e20;
	for(int i=0; i<pFmIntParam->m_iNumIntFms; i++)
	{	int iRawFms = pFmIntParam->GetIntFmSize(i);
		if(iMinRawFms > iRawFms) iMinRawFms = iRawFms;
	}
	m_iMaxGroupRawFms = m_iGroupSize * iMinRawFms;
}

void CFmGroupParam::mAllocate(void)
{
	m_piGroupStart = new int[m_iNumIntFms * 2];
	m_piGroupSize = m_piGroupStart + m_iNumIntFms;
	memset(m_piGroupStart, 0, sizeof(int) * m_iNumIntFms * 2);
	//--------------------------------------------------------
	m_pfGroupCenters = new float[m_iNumIntFms];
	memset(m_pfGroupCenters, 0, sizeof(float) * m_iNumIntFms);
}

void CFmGroupParam::mClean(void)
{
	if(m_piGroupStart != 0L) delete[] m_piGroupStart;
	if(m_pfGroupCenters != 0L) delete[] m_pfGroupCenters;
	m_piGroupStart = 0L;
	m_piGroupSize = 0L;
	m_pfGroupCenters = 0L;
	m_iNumIntFms = 0;
	m_iNumGroups = 0;
}
