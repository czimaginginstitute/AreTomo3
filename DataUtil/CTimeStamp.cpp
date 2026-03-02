#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <string.h>
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::DataUtil;

CTimeStamp* CTimeStamp::m_pInstances = 0L;
int              CTimeStamp::m_iNumGpus = 0;
FILE*            CTimeStamp::m_pFile = 0L;
pthread_mutex_t* CTimeStamp::m_pMutex = 0L;
Util_Time*       CTimeStamp::m_pTimer = 0L;

void CTimeStamp::CreateInstances(void)
{
	CTimeStamp::DeleteInstances();
	//---------------------------
	CInput* pInput = CInput::GetInstance();
	m_iNumGpus = pInput->m_iNumGpus;
	if(m_iNumGpus == 0) return;
	//---------------------------
	m_pInstances = new CTimeStamp[m_iNumGpus];
	for(int i=0; i<m_iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	//---------------------------
	m_pMutex = new pthread_mutex_t;
	pthread_mutex_init(m_pMutex, 0L);
	//---------------------------
	m_pTimer = new Util_Time;
	m_pTimer->Measure();
	//---------------------------
	mOpenFile();
}

CTimeStamp* CTimeStamp::GetInstance(int iNthGpu)
{
	if(iNthGpu >= m_iNumGpus) return 0L;
	return &m_pInstances[iNthGpu];
}

void CTimeStamp::DeleteInstances(void)
{
	if(m_pInstances != 0L)
	{	delete[] m_pInstances;
		m_pInstances = 0L;
	}
	m_iNumGpus = 0;
	//---------------------------
	if(m_pFile != 0L)
	{	fclose(m_pFile);
		m_pFile = 0L;
	}
	//---------------------------
	if(m_pMutex != 0L)
	{	pthread_mutex_destroy(m_pMutex);
		delete m_pMutex;
		m_pMutex = 0L;
        }
	//---------------------------
	if(m_pTimer != 0L)
	{	delete m_pTimer;
		m_pTimer = 0L;
	}
}	


CTimeStamp::CTimeStamp(void)
{
}

CTimeStamp::~CTimeStamp(void)
{
	while(!m_aTimeStampQ.empty())
	{	char* pcLine = m_aTimeStampQ.front();
		m_aTimeStampQ.pop();
		if(pcLine != 0L) delete[] pcLine;
	}
}

void CTimeStamp::Record(const char* pcAction)
{
	pthread_mutex_lock(m_pMutex);
	float fSeconds = m_pTimer->GetElapsedSeconds();
	pthread_mutex_unlock(m_pMutex);
	//---------------------------
	char* pcLine = new char[256];
	memset(pcLine, 0, sizeof(char) * 256);
	//---------------------------
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	CInput* pInput = CInput::GetInstance();
	int iGpuID = pInput->m_piGpuIDs[m_iNthGpu];
	sprintf(pcLine, "%s, %d, %s, %.1f\n", pTsPackage->m_acMrcMain,
	   iGpuID, pcAction, fSeconds);
	//---------------------------
	m_aTimeStampQ.push(pcLine);
}

void CTimeStamp::Save(void)
{       
	pthread_mutex_lock(m_pMutex);
	if(m_pFile == 0L)
	{	pthread_mutex_unlock(m_pMutex);
		return;
	}
	//---------------------------
	int iSize = m_aTimeStampQ.size();
	while(!m_aTimeStampQ.empty())
	{	char* pcLine = m_aTimeStampQ.front();
		m_aTimeStampQ.pop();
		if(pcLine != 0L)
		{	fprintf(m_pFile, "%s", pcLine);
			delete[] pcLine;
		}
	}
	if(iSize > 0) fflush(m_pFile);
	pthread_mutex_unlock(m_pMutex);
}

void CTimeStamp::mOpenFile(void)
{
	char acFile[256] = {'\0'};
	CInput* pInput = CInput::GetInstance();
	strcpy(acFile, pInput->m_acOutDir);
	strcat(acFile, "TiltSeries_TimeStamp.csv");
	//---------------------------
	bool bFirst = false;
	if(pInput->m_iResume == 0)
	{	m_pFile = fopen(acFile, "w");
		bFirst = true;
	}
	else
	{	m_pFile = fopen(acFile, "r");
		if(m_pFile == 0L)
		{	m_pFile = fopen(acFile, "w");
			bFirst = true;
		}
		else
		{	fclose(m_pFile);
			m_pFile = fopen(acFile, "a");
			if(m_pFile == 0L)
			{       m_pFile = fopen(acFile, "w");
			        bFirst = true;
			}
			else bFirst = false;
		}
	}
	//---------------------------
	if(m_pFile == 0L)
	{	printf("Warning: timestamp file cannot be created, "
		   "proceed without saving metrics.\n\n");
		return;
	}
	//---------------------------
        if(bFirst)
        {       fprintf(m_pFile, "Tilt_Series,GPU,Operation,Time(s)\n");
        }
}

