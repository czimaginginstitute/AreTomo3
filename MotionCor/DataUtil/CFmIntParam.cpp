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
	m_piIntFmStart = 0L;
	m_piIntFmSize = 0L;
	m_pfIntFmDose = 0L;
	m_pfAccFmDose = 0L;
	m_pfIntFmCents = 0L;
	m_iNumIntFms = 0;
	m_iMrcMode = -1;
}


CFmIntParam::~CFmIntParam(void)
{
	mClean();
}

void CFmIntParam::Setup(int iNumRawFms, int iMrcMode)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(m_iNumRawFms == iNumRawFms && m_iMrcMode == iMrcMode) return; 
	//-----------------
	mClean();
	m_iNumRawFms = iNumRawFms;
	m_iMrcMode = iMrcMode;
	//-----------------
	CReadFmIntFile* pReadFmIntFile = CReadFmIntFile::GetInstance();
	bool bIntegrate = pReadFmIntFile->NeedIntegrate();
	if(bIntegrate) mCalcIntFms(); 
	else mSetup();
	//-----------------
	mCalcIntFmCenters();
	//printf("Raw and rendered frames: %4d  %4d\n\n",
	//   m_iNumRawFms, m_iNumIntFms);

}

int CFmIntParam::GetIntFmStart(int iIntFrame)
{
	return m_piIntFmStart[iIntFrame];
}

int CFmIntParam::GetIntFmSize(int iIntFrame)
{
	return m_piIntFmSize[iIntFrame];
}

int CFmIntParam::GetNumIntFrames(void)
{
	return m_iNumIntFms;
}

float CFmIntParam::GetAccruedDose(int iIntFrame)
{
	return m_pfAccFmDose[iIntFrame];
}

float CFmIntParam::GetTotalDose(void)
{
	return m_pfAccFmDose[m_iNumIntFms - 1];
}

bool CFmIntParam::bIntegrate(void)
{
	CReadFmIntFile* pReadFmIntFile = CReadFmIntFile::GetInstance();
	return pReadFmIntFile->NeedIntegrate();
}

//--------------------------------------------------------------------
// Note: Dose weighting is done in AreTomo, not in MotionCor.
//--------------------------------------------------------------------
bool CFmIntParam::bDoseWeight(void)
{
	return false;
}

//-------------------------------------------------------------------
// For the case where there is no frame integration file is given.
//-------------------------------------------------------------------
void CFmIntParam::mSetup(void)
{
	m_iNumIntFms = m_iNumRawFms; 
	mAllocate();
	//-----------------
	CInput* pInput = CInput::GetInstance();
	float fFmDose = pInput->m_fFmDose;
	//-----------------
	for(int i=0; i<m_iNumIntFms; i++)
	{	m_piIntFmStart[i] = i;
		m_piIntFmSize[i] = 1;
		m_pfIntFmDose[i] = fFmDose;
		//----------------
		int iFmCount = m_piIntFmStart[i] + m_piIntFmSize[i];
		m_pfAccFmDose[i] = fFmDose * (i+1);
	}
}

void CFmIntParam::mCalcIntFms(void)
{
	CReadFmIntFile* pReadFmIntFile = CReadFmIntFile::GetInstance();
        int iNumEntries = pReadFmIntFile->m_iNumEntries;
	//-----------------
	int* piIntFmSizes = new int[m_iNumRawFms];
	float* pfIntFmDoses = new float[m_iNumRawFms];
	//-----------------
	m_iNumIntFms = 0;
	int iCountRaws = 0;
	int iLeftRawFms = 0;
	//-----------------
	for(int i=0; i<iNumEntries; i++)
	{	int iGroupSize = pReadFmIntFile->GetGroupSize(i);
		int iIntSize = pReadFmIntFile->GetIntSize(i);
		float fRawDose = pReadFmIntFile->GetDose(i);
		//----------------
		iLeftRawFms = m_iNumRawFms - iCountRaws;
		if(iLeftRawFms < iGroupSize) iGroupSize = iLeftRawFms;
		if(iGroupSize <= 0) continue;
		//----------------
		int iNumInts = iGroupSize / iIntSize;
		if(iNumInts == 0) continue;
		for(int j=0; j<iNumInts; j++)
		{	piIntFmSizes[m_iNumIntFms] = iIntSize;
			m_iNumIntFms += 1;
		}
		//----------------
		int iLeft = iGroupSize % iIntSize;
		int iSplits = iLeft / iNumInts;		
		for(int j=0; j<iNumInts; j++)
		{	int k = m_iNumIntFms - 1 - j;
			piIntFmSizes[k] += iSplits;
		}
		iLeft = iLeft % iNumInts;
		for(int j=0; j<iLeft; j++)
		{	int k = m_iNumIntFms - 1 - j;
			piIntFmSizes[k] += 1;
		}
		//----------------
		for(int j=0; j<iNumInts; j++)
		{	int k = m_iNumIntFms - 1 - j;
			pfIntFmDoses[k] = fRawDose * piIntFmSizes[k];
			iCountRaws += piIntFmSizes[k];
		}
	}
	//-----------------
	mAllocate();
	//-----------------
	m_piIntFmStart[0] = 0;
	m_piIntFmSize[0] = piIntFmSizes[0];
	m_pfIntFmDose[0] = pfIntFmDoses[0];
	for(int i=1; i<m_iNumIntFms; i++)
	{	int m1 = i - 1;
		m_piIntFmSize[i] = piIntFmSizes[i];
		m_piIntFmStart[i] = m_piIntFmStart[m1] + piIntFmSizes[m1];
		//----------------
		m_pfIntFmDose[i] = pfIntFmDoses[i];
		m_pfAccFmDose[i] = m_pfAccFmDose[m1] + piIntFmSizes[m1];	
	}
	if(piIntFmSizes != 0L) delete[] piIntFmSizes;
	if(pfIntFmDoses != 0L) delete[] pfIntFmDoses;

	for(int i=0; i<m_iNumIntFms; i++)
	{	printf("%4d %4d %4d %8.3f %8.3f\n", i, m_piIntFmStart[i],
		   m_piIntFmSize[i], m_pfIntFmDose[i], m_pfAccFmDose[i]);
	}	
}

void CFmIntParam::mClean(void)
{
	if(m_piIntFmStart != 0L) delete[] m_piIntFmStart;
	if(m_pfIntFmDose != 0L) delete[] m_pfIntFmDose;
	m_piIntFmStart = 0L;
	m_piIntFmSize = 0L;
	m_pfIntFmDose = 0L;
	m_pfAccFmDose = 0L;
	m_pfIntFmCents = 0L;
	m_iNumRawFms = 0;
	m_iNumIntFms = 0;
}

void CFmIntParam::mAllocate(void)
{
	m_piIntFmStart = new int[m_iNumIntFms * 2];
	m_piIntFmSize = &m_piIntFmStart[m_iNumIntFms];
	m_pfIntFmDose = new float[m_iNumIntFms * 3];
	m_pfAccFmDose = &m_pfIntFmDose[m_iNumIntFms];
	m_pfIntFmCents = &m_pfIntFmDose[m_iNumIntFms * 2];
}

void CFmIntParam::mCalcIntFmCenters(void)
{
	for(int i=0; i<m_iNumIntFms; i++)
	{	m_pfIntFmCents[i] = m_piIntFmStart[i] +
		   0.5f * (m_piIntFmSize[i] - 1.0f);
	}
}
