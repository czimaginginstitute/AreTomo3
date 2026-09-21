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
	m_piProvStarts = 0L;
	m_piProvSizes = 0L;
	m_iProvIntFms = 0;
	m_iProvKeepFirst = 0;
	m_iProvKeepCount = 0;
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
	//-----------------------------------------------------------
	// Snapshot the pre-throw integration layout before
	// mRemoveFrames compacts the arrays in place (frame
	// provenance for the .mcaln frame table).
	//-----------------------------------------------------------
	m_iProvIntFms = m_iNumIntFms;
	m_piProvStarts = new int[m_iProvIntFms * 2];
	m_piProvSizes = m_piProvStarts + m_iProvIntFms;
	for(int i=0; i<m_iProvIntFms; i++)
	{	m_piProvStarts[i] = m_piIntFmStarts[i];
		m_piProvSizes[i] = m_piIntFmSizes[i];
	}
	//-----------------
	mRemoveFrames();
	//-----------------
	int iThrows = m_iProvIntFms - m_iNumIntFms;
	if(iThrows > 0)
	{	CMcInput* pIn = CMcInput::GetInstance();
		m_iProvKeepFirst = pIn->m_aiThrow[0];
	}
	else m_iProvKeepFirst = 0;
	m_iProvKeepCount = m_iNumIntFms;
}

int CFmIntParam::GetNumRawFrames(void)
{
	return m_iNumRawFms;
}

int CFmIntParam::GetProvNumIntFrames(void)
{
	return m_iProvIntFms;
}

int CFmIntParam::GetProvStart(int iIntFrame)
{
	return m_piProvStarts[iIntFrame];
}

int CFmIntParam::GetProvSize(int iIntFrame)
{
	return m_piProvSizes[iIntFrame];
}

bool CFmIntParam::GetProvIncluded(int iIntFrame)
{
	if(iIntFrame < m_iProvKeepFirst) return false;
	if(iIntFrame >= m_iProvKeepFirst + m_iProvKeepCount) return false;
	return true;
}

int CFmIntParam::GetProvAligned(int iIntFrame)
{
	if(!GetProvIncluded(iIntFrame)) return -1;
	return iIntFrame - m_iProvKeepFirst;
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
	if(m_iNumRawFms == m_iNumIntFms) return false;
	else return true;
}

bool CFmIntParam::bHasDose(void)
{
	if(m_fTotalDose <= 0) return false;
	else return true;
}

void CFmIntParam::mCalcIntFms(void)
{
	float fRawFmDose = m_fTotalDose / m_iNumRawFms;
	//---------------------------
	CMcInput* pMcInput = CMcInput::GetInstance();
	int iFmInt = pMcInput->m_iFmInt;
	if(iFmInt <= 0)
	{	if(fRawFmDose > 0) iFmInt = (int)(0.15f / fRawFmDose);
		else iFmInt = m_iNumRawFms / 12;
	}
	if(iFmInt < 1) iFmInt = 1;
	//---------------------------
	m_iNumIntFms = m_iNumRawFms / iFmInt;
	if(m_iNumIntFms < 1) m_iNumIntFms = 1;
	//---------------------------
	mAllocate();
	int iLast = m_iNumIntFms - 1;
	//---------------------------
	for(int i=0; i<iLast; i++)
	{	m_piIntFmStarts[i] = i * iFmInt;
		m_piIntFmSizes[i] = iFmInt;
		m_pfIntFmDoses[i] = m_piIntFmSizes[i] * fRawFmDose;
		int iAccFms = m_piIntFmStarts[i] + m_piIntFmSizes[i];
		m_pfAccFmDoses[i] = iAccFms * fRawFmDose;
	}
	//-----------------
	m_piIntFmStarts[iLast] = iLast * iFmInt;
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
	if(m_piProvStarts != 0L)
	{	delete[] m_piProvStarts;
		m_piProvStarts = 0L;
		m_piProvSizes = 0L;
		m_iProvIntFms = 0;
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

