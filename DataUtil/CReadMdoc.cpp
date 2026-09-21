#include "CDataUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <stdint.h>
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
	m_iNumTilts = 1024;
	m_iNumTilts = 0;
	//---------------------------
	m_ppcFrmPath = new char*[m_iNumTilts];
	m_piAcqIdxs = new int[m_iNumTilts];
	m_pfTilts = new float[m_iNumTilts];
	m_pfDoses = new float[m_iNumTilts];
	m_pfExpTimes = new float[m_iNumTilts];
	m_pDateTimes = new time_t[m_iNumTilts];
	//---------------------------
	memset(m_ppcFrmPath, 0, sizeof(char*) * m_iNumTilts);
	memset(m_piAcqIdxs, 0, sizeof(int) * m_iNumTilts);
	memset(m_pfTilts, 0, sizeof(float) * m_iNumTilts);
	memset(m_pfDoses, 0, sizeof(float) * m_iNumTilts);
	memset(m_pfExpTimes, 0, sizeof(float) * m_iNumTilts);
	memset(m_pDateTimes, 0, sizeof(time_t) * m_iNumTilts);
	memset(m_acMdocFile, 0, sizeof(m_acMdocFile));
}

CReadMdoc::~CReadMdoc(void)
{
	mClean();
	if(m_ppcFrmPath != 0L) delete[] m_ppcFrmPath;
	if(m_piAcqIdxs != 0L) delete[] m_piAcqIdxs;
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_pfDoses != 0L) delete[] m_pfDoses;
	if(m_pfExpTimes != 0L) delete[] m_pfExpTimes;
	if(m_pDateTimes != 0L) delete[] m_pDateTimes;
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

float CReadMdoc::GetDose(int iTilt)
{
	return m_pfDoses[iTilt];
}

//--------------------------------------------------------------------
// 1. Exposure times are stored as percentages relative to the total
//    exposure time of the entire tilt series.
// 2. This is done to facilitate the calulation of per-tilt dose
//    given the total dose of the tilt series.
// 3. This implementation takes into account of the variable per
//    per tilt dose.
//-------------------------------------------------------------------- 
float CReadMdoc::GetExpTime(int iTilt)
{
	return m_pfExpTimes[iTilt];
}

bool CReadMdoc::DoIt(const char* pcMdocFile)
{
	mClean();
	FILE* pFile = fopen(pcMdocFile, "rt");
	if(pFile == 0L) return false;
	//---------------------------
	memset(m_acMdocFile, 0, sizeof(m_acMdocFile));
	strcpy(m_acMdocFile, pcMdocFile);
	//---------------------------
	char acBuf[256] = {'\0'};
	std::queue<float> qTilt;
	std::queue<float> qDose;
	std::queue<float> qExpTime;
	std::queue<char*> qFrmPath;
	std::queue<time_t> qDateTime;
	//---------------------------
	while(!feof(pFile))
	{	char* pcRet = fgets(acBuf, 256, pFile);
		//-------------------
		float fTilt = -99.0f;
		if(mExtractTilt(acBuf, &fTilt))
 		{	qTilt.push(fTilt);
			continue;
		}
		//-------------------
		float fDose = 0.0f;
		if(mExtractDose(acBuf, &fDose))
		{	qDose.push(fDose);
			continue;
		}
		//-------------------	
		char* pcFrmPath = mExtractFramePath(acBuf);
		if(pcFrmPath != 0L)
		{	qFrmPath.push(pcFrmPath);
			continue;
		}
		//-------------------
		float fExpTime = 0.0f;	
		if(mExtractExpTime(acBuf, &fExpTime))
		{	qExpTime.push(fExpTime);
			continue;
		}
		//-------------------	
		time_t tDateTime;
		if(mExtractDateTime(acBuf, &tDateTime))
		{	qDateTime.push(tDateTime);
			continue;
		}
	}
	fclose(pFile);
	//---------------------------
	int iNumTilts = qTilt.size();
	bool bComplete = true;
	if(iNumTilts != qDose.size()) bComplete = false;
	else if(iNumTilts != qFrmPath.size()) bComplete = false;
	else if(iNumTilts != qExpTime.size()) bComplete = false;
	else if(iNumTilts != qDateTime.size()) bComplete = false;
	//---------------------------
	if(!bComplete)
	{	while(qFrmPath.size() > 0)
		{	char* pcFrmPath = qFrmPath.front();
			qFrmPath.pop();
			if(pcFrmPath != 0L) delete[] pcFrmPath;
		}
		printf("Warning: faulty mdoc file found, skip it!\n"
		   "   MDOC: %s\n\n", m_acMdocFile);
		return false;
	}
	//---------------------------
	mAllocate(iNumTilts);
	for(int i=0; i<m_iNumTilts; i++)
	{	m_pfTilts[i] = qTilt.front();
		m_pfDoses[i] = qDose.front();
		m_ppcFrmPath[i] = qFrmPath.front();
		m_pfExpTimes[i] = qExpTime.front();
		m_pDateTimes[i] = qDateTime.front();
		//-------------------
		qTilt.pop();
		qDose.pop();
		qFrmPath.pop();
		qExpTime.pop();
		qDateTime.pop();	
	}
	//---------------------------
	mOrderAcquisition();
	mMakeExpTimeRelative();
	if(m_iNumTilts >= 7) return true;
	//---------------------------
	printf("Warning: mdoc file has less than 7 tilts, skip it!\n"
	   "   MDOC: %s\n\n", m_acMdocFile);
	return false;
}

bool CReadMdoc::mExtractValZ(char* pcLine, int* piValZ)
{
	char* pcZValue = strstr(pcLine, "ZValue");
	if(pcZValue == 0L) return false;
	//-----------------
	char* pcEqual = strrchr(pcLine, '=');
	piValZ[0] = atoi(&pcEqual[1]);
	return true;
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

bool CReadMdoc::mExtractDose(char* pcLine, float* pfDose)
{
	pfDose[0] = 0.0f;
	char* pcExpDose = strstr(pcLine, "ExposureDose");
	if(pcExpDose == 0L) return false;
	//-----------------
	char* pcEqual = strrchr(pcLine, '=');
	if(strlen(pcEqual) < 2) return false;
	//-----------------
	pfDose[0] = (float)atof(&pcEqual[1]);
	return true;
}

bool CReadMdoc::mExtractExpTime(char* pcLine, float* pfExpTime)
{
        pfExpTime[0] = 0.0f;
        char* pcExpTime = strstr(pcLine, "ExposureTime");
        if(pcExpTime == 0L) return false;
        //-----------------
        char* pcEqual = strrchr(pcLine, '=');
        if(strlen(pcEqual) < 2) return false;
        //-----------------
        pfExpTime[0] = (float)atof(&pcEqual[1]);
        return true;
}

char* CReadMdoc::mExtractFramePath(char* pcLine)
{
	char* pcPath = strstr(pcLine, "SubFramePath");
	if(pcPath == 0L) return 0L;
	//-----------------
	char* pcEqual = strrchr(pcLine, '=');
	char acPath[256] = {'\0'};
	strcpy(acPath, &pcEqual[1]);
	//-----------------
	char* pcRetN = strrchr(acPath, '\n');
	if(pcRetN != 0L) pcRetN[0] = '\0';
	//-----------------
	char* pcRetR = strrchr(acPath, '\r');
	if(pcRetR != 0L) pcRetR[0] = '\0';
	//-----------------
	int iSize = strlen(acPath);
	int iStart = 0;
	for(int i=0; i<iSize; i++)
	{	if(acPath[i] != ' ') break;
		else iStart = i;
	}
	//-------------------------------------
	// remove leading white space.
	//-------------------------------------
	pcPath = new char[256];
	strcpy(pcPath, &acPath[iStart]);
	//-------------------------------------
	// remove trailing white space
	//-------------------------------------
	iSize = strlen(pcPath);
	for(int i=iSize-1; i>=0; i--)
	{	if(pcPath[i] == ' ') pcPath[i] = '\0';
		else break;
	}
	return pcPath;
}

bool CReadMdoc::mExtractDateTime(char* pcLine, time_t* pDateTime)
{
	char* pcDateTime = strstr(pcLine, "DateTime");
	if(pcDateTime == 0L) return false;
	//---------------------------
	char* pcEqual = strrchr(pcLine, '=');
	char acDateTime[64] = {'\0'};
	int iStartIdx = (pcEqual[1] == ' ') ? 2 : 1;
	strcpy(acDateTime, &pcEqual[iStartIdx]);
	//---------------------------
	char* pcRetN = strrchr(acDateTime, '\n');
	if(pcRetN != 0L) pcRetN[0] = '\0';
	char* pcRetR = strrchr(acDateTime, '\r');
	if(pcRetR != 0L) pcRetR[0] = '\0';
	//---------------------------
	struct tm tmDateTime = {0};
	strptime(acDateTime, "%d-%b-%Y %H:%M:%S", &tmDateTime);
	pDateTime[0] = mktime(&tmDateTime);
	return true;
}

void CReadMdoc::mMakeExpTimeRelative(void)
{
	float fSum = 0.0f;
	for(int i=0; i<m_iNumTilts; i++)
	{	fSum += m_pfExpTimes[i];
	}
	//---------------------------
	if(fSum > 0)
	{	for(int i=0; i<m_iNumTilts; i++)
		{	m_pfExpTimes[i] /= fSum;
		}
	}
	else
	{	for(int i=0; i<m_iNumTilts; i++)
		{	m_pfExpTimes[i] = 1.0f / m_iNumTilts;
		}
	}
}

void CReadMdoc::mOrderAcquisition(void)
{
	for(int i=0; i<m_iNumTilts; i++)
	{	m_piAcqIdxs[i] = i;
	}
	//---------------------------
	for(int i=0; i<m_iNumTilts; i++)
	{	int iCount = 0;
		for(int j=0; j<m_iNumTilts; j++)
		{	if(j == i) continue;
			if(m_pDateTimes[i] < m_pDateTimes[j]) continue;
			iCount += 1;
		}
		m_piAcqIdxs[i] = iCount;
	}	
}

void CReadMdoc::mClean(void)
{
	for(int i=0; i<m_iNumTilts; i++)
	{	if(m_ppcFrmPath[i] == 0L) continue;
		delete[] m_ppcFrmPath[i];
		m_ppcFrmPath[i] = 0L;
	}
	m_iNumTilts = 0;
}

void CReadMdoc::mAllocate(int iNumTilts)
{
	m_iNumTilts = iNumTilts;
	m_ppcFrmPath = new char*[m_iNumTilts];
        m_piAcqIdxs = new int[m_iNumTilts];
        m_pfTilts = new float[m_iNumTilts];
        m_pfDoses = new float[m_iNumTilts];
        m_pfExpTimes = new float[m_iNumTilts];
        m_pDateTimes = new time_t[m_iNumTilts];
        //---------------------------
        memset(m_ppcFrmPath, 0, sizeof(char*) * m_iNumTilts);
        memset(m_piAcqIdxs, 0, sizeof(int) * m_iNumTilts);
        memset(m_pfTilts, 0, sizeof(float) * m_iNumTilts);
        memset(m_pfDoses, 0, sizeof(float) * m_iNumTilts);
        memset(m_pfExpTimes, 0, sizeof(float) * m_iNumTilts);
        memset(m_pDateTimes, 0, sizeof(time_t) * m_iNumTilts);
}
