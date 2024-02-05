#include "CDataUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/types.h>

using namespace McAreTomo::DataUtil;

CReadMdoc* CReadMdoc::m_pInstances = 0L;
int CReadMdoc::m_iNumGpus = 0;

void CReadMdoc::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	//-----------------
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CReadMdoc[iNumGpus];
	for(int i=0; i<m_iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CReadMdoc::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CReadMdoc* CReadMdoc::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CReadMdoc::CReadMdoc(void)
{
	m_iNthGpu = 0;
	m_iBufSize = 1024;
	m_iNumTilts = 0;
	m_ppcFrmPath = new char*[m_iBufSize];
	m_piAcqIdxs = new int[m_iBufSize];
	m_pfTilts = new float[m_iBufSize];
}

CReadMdoc::~CReadMdoc(void)
{
	mClean();
	if(m_ppcFrmPath != 0L) delete[] m_ppcFrmPath;
	if(m_piAcqIdxs != 0L) delete[] m_piAcqIdxs;
	if(m_pfTilts != 0L) delete[] m_pfTilts;
}

char* CReadMdoc::GetFramePath(int iTilt)
{
	return m_ppcFrmPath[iTilt];
}

char* CReadMdoc::GetFrameFileName(int iTilt)
{
	char* pcFrmPath = m_ppcFrmPath[iTilt];
	char* pcSlash = strrchr(pcFrmPath, '\\');
	if(pcSlash != 0L) return &pcSlash[1];
	//-----------------
	pcSlash = strrchr(pcFrmPath, '/');
	if(pcSlash != 0L) return &pcSlash[1];
	//-----------------
	return pcFrmPath;
}

int CReadMdoc::GetAcqIdx(int iTilt)
{	
	return m_piAcqIdxs[iTilt];
}

float CReadMdoc::GetTilt(int iTilt)
{
	return m_pfTilts[iTilt];
}

bool CReadMdoc::DoIt(const char* pcMdocFile)
{
	mClean();
	FILE* pFile = fopen(pcMdocFile, "rt");
	if(pFile == 0L) return false;
	//-----------------
	memset(m_acMdocFile, 0, sizeof(m_acMdocFile));
	strcpy(m_acMdocFile, pcMdocFile);
	//-----------------
	m_iNumTilts = 0;
	char acBuf[256] = {'\0'};
	int iCount = 0;
	//-----------------
	while(!feof(pFile))
	{	memset(acBuf, 0, sizeof(char) * 256);
		char* pcRet = fgets(acBuf, 256, pFile);
		if(pcRet == 0L) continue;
		//----------------
		int iValZ = mExtractValZ(acBuf);
		if(iValZ < 0) continue;
		m_piAcqIdxs[m_iNumTilts] = iValZ;
		//----------------
		iCount = 0;
		while(!feof(pFile))
		{	memset(acBuf, 0, sizeof(char) * 256);
			char* pcRet = fgets(acBuf, 256, pFile);
			if(pcRet == 0L) continue;
			//---------------
			if(mExtractTilt(acBuf, &m_pfTilts[m_iNumTilts]))
			{	iCount += 1;
				if(iCount == 1) continue;
				m_iNumTilts += 1;
				break;
			}
			//---------------
			char* pcFramePath = mExtractFramePath(acBuf);
			if(pcFramePath != 0L)
			{	m_ppcFrmPath[m_iNumTilts] = pcFramePath;
				iCount += 1;
				if(iCount == 1) continue;
				m_iNumTilts += 1;
				break;
			}
		}
	}
	fclose(pFile);
	//-----------------
	if(m_iNumTilts > 0) return true;
	else return false;
}

int CReadMdoc::mExtractValZ(char* pcLine)
{
	char* pcZValue = strstr(pcLine, "ZValue");
	if(pcZValue == 0L) return -99;
	//-----------------
	char* pcEqual = strrchr(pcLine, '=');
	int iValZ = atoi(&pcEqual[1]);
	return iValZ;
}

bool CReadMdoc::mExtractTilt(char* pcLine, float* pfTilt)
{
	char* pcTiltAngle = strstr(pcLine, "TiltAngle");
	if(pcTiltAngle == 0L) return false;
	//-----------------
	char* pcEqual = strrchr(pcLine, '=');
	pfTilt[0] = (float)atof(&pcEqual[1]);
	return true;
}

char* CReadMdoc::mExtractFramePath(char* pcLine)
{
	char* pcPath = strstr(pcLine, "SubFramePath");
	if(pcPath == 0L) return 0L;
	//-----------------
	char* pcEqual = strrchr(pcLine, '=');
	pcPath = new char[128];
	if(pcEqual[1] == ' ') strcpy(pcPath, &pcEqual[2]);
	else strcpy(pcPath, &pcEqual[1]);
	//-----------------
	char* pcRetN = strrchr(pcPath, '\n');
	if(pcRetN != 0L) pcRetN[0] = '\0';
	//-----------------
	char* pcRetR = strrchr(pcPath, '\r');
	if(pcRetR != 0L) pcRetR[0] = '\0';
	//-----------------
	return pcPath;
}

void CReadMdoc::mClean(void)
{
	for(int i=0; i<m_iBufSize; i++)
	{	if(m_ppcFrmPath[i] == 0L) continue;
		delete[] m_ppcFrmPath[i];
		m_ppcFrmPath[i] = 0L;
	}
	m_iNumTilts = 0;
}
