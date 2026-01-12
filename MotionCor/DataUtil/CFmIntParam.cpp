#include "CDataUtilInc.h"
#include "../CMotionCorInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/types.h>

using namespace McAreTomo::MotionCor::DataUtil;

CFmIntParam* CFmIntParam::m_pInstances = 0L;
int CFmIntParam::m_iNumGpus = 0;

void CFmIntParam::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CFmIntParam[iNumGpus];
	//-----------------
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CFmIntParam::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CFmIntParam* CFmIntParam::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CFmIntParam::CFmIntParam(void)
{
	m_piIntFmStarts = 0L;
	m_piIntFmSizes = 0L;
	m_pfIntFmDoses = 0L;
	m_pfAccFmDoses = 0L;
	m_fTotalDose = 0.0f;
	m_iNumIntFms = 0;
	m_iMrcMode = -1;
}


CFmIntParam::~CFmIntParam(void)
{
	mClean();
}

void CFmIntParam::Setup(int iNumRawFms, int iMrcMode, float fMdocDose)
{
	mClean();
	//-----------------
	CMcInput* pMcInput = CMcInput::GetInstance();
	m_iNumRawFms = iNumRawFms;
	m_iMrcMode = iMrcMode;
	//-----------------
	m_fTotalDose = fMdocDose;
	mCalcIntFms(); 
	mRemoveFrames();
}

int CFmIntParam::GetIntFmStart(int iIntFrame)
{
	return m_piIntFmStarts[iIntFrame];
}

int CFmIntParam::GetIntFmSize(int iIntFrame)
{
	return m_piIntFmSizes[iIntFrame];
}

int CFmIntParam::GetNumIntFrames(void)
{
	return m_iNumIntFms;
}

float CFmIntParam::GetAccDose(int iIntFrame)
{
	return m_pfAccFmDoses[iIntFrame];
}

float CFmIntParam::GetTotalDose(void)
{
	return m_fTotalDose;
}

bool CFmIntParam::bIntegrate(void)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(pMcInput->m_iFmInt > 1) return true;
	else return false;
}

bool CFmIntParam::bHasDose(void)
{
	if(m_fTotalDose <= 0) return false;
	else return true;
}

void CFmIntParam::mCalcIntFms(void)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	m_iNumIntFms = m_iNumRawFms / pMcInput->m_iFmInt;
	if(m_iNumIntFms < 1) m_iNumIntFms = 1;
	//---------------------------
	mAllocate();
	float fRawFmDose = m_fTotalDose / m_iNumRawFms;
	int iLast = m_iNumIntFms - 1;
	//---------------------------
	for(int i=0; i<iLast; i++)
	{	m_piIntFmStarts[i] = i * pMcInput->m_iFmInt;
		m_piIntFmSizes[i] = pMcInput->m_iFmInt;
		m_pfIntFmDoses[i] = m_piIntFmSizes[i] * fRawFmDose;
		int iAccFms = m_piIntFmStarts[i] + m_piIntFmSizes[i];
		m_pfAccFmDoses[i] = iAccFms * fRawFmDose;
	}
	//-----------------
	m_piIntFmStarts[iLast] = iLast * pMcInput->m_iFmInt;
	m_piIntFmSizes[iLast] = m_iNumRawFms - m_piIntFmStarts[iLast];
	m_pfIntFmDoses[iLast] = m_piIntFmSizes[iLast] * fRawFmDose;
	m_pfAccFmDoses[iLast] = m_fTotalDose;
}

void CFmIntParam::mRemoveFrames(void)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	int* piThrow = pMcInput->m_aiThrow;
	int iThrows = piThrow[0] + piThrow[1];
	if(iThrows == 0 || iThrows >= m_iNumIntFms) return;
	//---------------------------
	int iNewFms = m_iNumIntFms - iThrows;
	for(int i=0; i<iNewFms; i++)
	{	int j = i + piThrow[0];
		m_piIntFmStarts[i] = m_piIntFmStarts[j];
		m_piIntFmSizes[i] = m_piIntFmSizes[j];
		m_pfIntFmDoses[i] = m_pfIntFmDoses[j];
		m_pfAccFmDoses[i] = m_pfAccFmDoses[j];
	}
	m_iNumIntFms = iNewFms;
}


void CFmIntParam::mClean(void)
{
	if(m_piIntFmStarts != 0L) 
	{	delete[] m_piIntFmStarts;
		m_piIntFmStarts = 0L;
		m_piIntFmSizes = 0L;
	}
	if(m_pfIntFmDoses != 0L) 
	{	delete[] m_pfIntFmDoses;
		m_pfIntFmDoses = 0L;
		m_pfAccFmDoses = 0L;
	}
}

void CFmIntParam::mAllocate(void)
{
	m_piIntFmStarts = new int[m_iNumIntFms * 2];
	m_piIntFmSizes = &m_piIntFmStarts[m_iNumIntFms];
	m_pfIntFmDoses = new float[m_iNumIntFms * 3];
	m_pfAccFmDoses = &m_pfIntFmDoses[m_iNumIntFms];
}

//--------------------------------------------------------------------
// 1. This is for debugging and usually commented out.
//--------------------------------------------------------------------
void CFmIntParam::mDisplay(void)
{
	printf("\n Calculation of frame integration\n");
	printf(" IntFm  Start  Size    Dose     AccDose\n");
	for(int i=0; i<m_iNumIntFms; i++)
	{	printf("%4d   %4d  %5d  %8.3f  %9.3f\n", i,
		   m_piIntFmStarts[i], m_piIntFmSizes[i],
		   m_pfIntFmDoses[i], m_pfAccFmDoses[i]);
	}
	printf("\n");
}

