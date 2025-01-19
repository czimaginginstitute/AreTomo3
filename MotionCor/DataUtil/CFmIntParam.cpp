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
	//-----------------------------------------------
	// 1) Check whether the number frames in the
	// FmIntFile matches the m_iNumRawFms.
	// 2) If more, subtract extra from the each
	// entry from last to first.
	// 3) If less, add extra to the last entry
	//-----------------------------------------------
	int* piGroupSizes = new int[iNumEntries * 2];
	int* piNumInts = &piGroupSizes[iNumEntries];
	//-----------------
	int iCount = 0, iLeft = 0;
	for(int i=0; i<iNumEntries; i++)
	{	int iGroupSize = pReadFmIntFile->GetGroupSize(i);
		int iIntSize = pReadFmIntFile->GetIntSize(i);
		iGroupSize = iGroupSize / iIntSize * iIntSize;
		//----------------
		iLeft = m_iNumRawFms - iCount;
		if(iLeft < iGroupSize)
		{	iGroupSize = iLeft / iIntSize * iIntSize;
		}
		piGroupSizes[i] = iGroupSize;
		iCount += iGroupSize;
	}
	iLeft = m_iNumRawFms - iCount;
	//-----------------
        m_iNumIntFms = 0;
        for(int i=0; i<iNumEntries; i++)
        {       int iIntSize = pReadFmIntFile->GetIntSize(i);
		m_iNumIntFms += (piGroupSizes[i] / iIntSize);
        }
        mAllocate();
        //-----------------
        iCount = 0;
        for(int i=0; i<iNumEntries; i++)
        {       int iIntSize = pReadFmIntFile->GetIntSize(i);
		int iGroupSize = piGroupSizes[i];
		int iNumIntFms = iGroupSize / iIntSize;
		//----------------
                for(int k=0; k<iNumIntFms; k++)
                {       m_piIntFmSize[iCount] = iIntSize;
                        iCount += 1;
                }
        }
        if(piGroupSizes != 0L) delete[] piGroupSizes;
	if(m_iNumIntFms == 0) return;
	//-----------------
	int iStep = iLeft / m_iNumIntFms;
	iLeft = iLeft % m_iNumIntFms;
	for(int i=0; i<m_iNumIntFms; i++)
	{	m_piIntFmSize[i] += iStep;
	}
        for(int i=0; i<iLeft; i++)
        {       int j = m_iNumIntFms - 1 - i;
                m_piIntFmSize[j] += 1;
        }
        //-----------------
        m_piIntFmStart[0] = 0;
        for(int i=1; i<m_iNumIntFms; i++)
        {       int k = i - 1;
                m_piIntFmStart[i] = m_piIntFmStart[k] + m_piIntFmSize[k];
        }
        //-----------------
        float fRawFmDose = pReadFmIntFile->GetDose(0);
        for(int i=0; i<m_iNumIntFms; i++)
        {       m_pfIntFmDose[i] = m_piIntFmSize[i] * fRawFmDose;
        }
        m_pfAccFmDose[0] = m_pfIntFmDose[0];
        for(int i=1; i<m_iNumIntFms; i++)
        {       m_pfAccFmDose[i] = m_pfAccFmDose[i-1] + m_pfIntFmDose[i];
        }
        //------------------
        mCalcIntFmCenters();
        //mDisplay();
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

//--------------------------------------------------------------------
// 1. This is for debugging and usually commented out.
//--------------------------------------------------------------------
void CFmIntParam::mDisplay(void)
{
	printf("\n Calculation of frame integration\n");
	printf(" IntFm  Start  Size    Dose     AccDose\n");
	for(int i=0; i<m_iNumIntFms; i++)
	{	printf("%4d   %4d  %5d  %8.3f  %9.3f\n", i,
		   m_piIntFmStart[i], m_piIntFmSize[i],
		   m_pfIntFmDose[i], m_pfAccFmDose[i]);
	}
	printf("\n");
}

