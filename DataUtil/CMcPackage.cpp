#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo;
using namespace McAreTomo::DataUtil;

CMcPackage* CMcPackage::m_pInstances = 0L;
int CMcPackage::m_iNumGpus = 0;

void CMcPackage::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	//-----------------
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CMcPackage[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CMcPackage::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CMcPackage* CMcPackage::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CMcPackage::CMcPackage(void)
{
	m_pRawStack = new CMrcStack;
	m_pAlnSums = new CAlnSums;
	memset(m_acMoviePath, 0, sizeof(m_acMoviePath));
	//-----------------
	m_iAcqIdx = 0;
	m_fTilt = 0.0f;
	m_iNthGpu = 0;
}

CMcPackage::~CMcPackage(void)
{
	delete m_pRawStack;
	delete m_pAlnSums;
}

void CMcPackage::SetMovieName(char* pcMovieName)
{
	CInput* pInput = CInput::GetInstance();
	strcpy(m_acMoviePath, pInput->m_acInDir);
	strcat(m_acMoviePath, pcMovieName);
}

bool CMcPackage::bTiffFile(void)
{
	char* pcDot = strrchr(m_acMoviePath, '.');
	if(pcDot == 0L) return false;
	//-----------------
	char* pcTif = strcasestr(pcDot, ".tif");
	if(pcTif == 0L) return false;
	else return true;
}

bool CMcPackage::bEerFile(void)
{
	char* pcDot = strrchr(m_acMoviePath, '.');
	if(pcDot == 0L) return false;
	//-----------------
	char* pcEer = strcasestr(pcDot, ".eer");
	if(pcEer == 0L) return false;
	else return true;
}

int* CMcPackage::GetMovieSize(void)
{
	if(m_pRawStack == 0L) return 0L;
	return m_pRawStack->m_aiStkSize;
}

int CMcPackage::GetMovieMode(void)
{
	if(m_pRawStack == 0L) return 0;
	return m_pRawStack->m_iMode;
}
