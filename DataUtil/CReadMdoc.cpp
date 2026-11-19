#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace McAreTomo::DataUtil;

CReadMdoc* CReadMdoc::m_pInstances = 0L;
int CReadMdoc::m_iNumGpus = 0;

void CReadMdoc::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	//-----------------
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CReadMdoc[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
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
	m_iNumTilts = 0;
	m_iBufSize = 0;
	//---------------------------
	m_ppcFrmPath = 0L;
	m_piAcqIdxs = 0L;
	m_pfTilts = 0L;
	m_pfDoses = 0L;
	m_pfExpTimes = 0L;
	m_pDateTimes = 0L;
	//---------------------------
	memset(m_acMdocFile, 0, sizeof(m_acMdocFile));
	memset(m_acErrMsg, 0, sizeof(m_acErrMsg));
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

int CReadMdoc::DoIt(const char* pcMdocFile)
{
	mClean();
	memset(m_acErrMsg, 0, sizeof(m_acErrMsg));
	//-----------------------------------------------
	// When the .openTS exists, the movie files of
	// the tilt series are incomplete. This check
	// applies only when processing starts from
	// movies (-Cmd 0).
	//-----------------------------------------------
	CInput* pInput = CInput::GetInstance();
	bool bOpenTS = false;
	if(pInput->m_iCmd == 0) bOpenTS = mCheckOpenTS(pcMdocFile);
	if(bOpenTS)
	{	strcpy(m_acErrMsg, ".openTS file exists, tilt series "
		   "is still being collected");
		printf("*** Info: %s\n   %s\n\n", pcMdocFile, m_acErrMsg);
		return 0;
	}
	//---------------------------
	FILE* pFile = fopen(pcMdocFile, "rt");
	if(pFile == 0L)
	{	strcpy(m_acErrMsg, "cannot open mdoc file");
		return -1;
	}
	//---------------------------
	memset(m_acMdocFile, 0, sizeof(m_acMdocFile));
	strncpy(m_acMdocFile, pcMdocFile, sizeof(m_acMdocFile) - 1);
	//---------------------------
	char acBuf[1024] = {'\0'};
	std::queue<float> qTilt;
	std::queue<float> qDose;
	std::queue<float> qExpTime;
	std::queue<char*> qFrmPath;
	std::queue<time_t> qDateTime;
	//---------------------------
	while(fgets(acBuf, sizeof(acBuf), pFile) != 0L)
	{	//-------------------
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
		strcpy(m_acErrMsg, "incomplete or faulty mdoc content");
		return 0;
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
	if(m_iNumTilts >= 7) return 1;
	//---------------------------
	strcpy(m_acErrMsg, "mdoc file has less than 7 tilts");
	return 0;
}

//--------------------------------------------------------------------
// 1. pcName1: last extension replaced, e.g. TS_1.mrc.mdoc ->
//    TS_1.mrc.openTS.
// 2. pcName2: all extensions of the base name replaced, e.g.
//    TS_1.mrc.mdoc -> TS_1.openTS. Same as pcName1 when the
//    mdoc file has only one extension.
//--------------------------------------------------------------------
bool CReadMdoc::mGetOpenTSNames(const char* pcMdocFile,
	char* pcName1, char* pcName2, int iSize)
{
	if(pcMdocFile == 0L || strlen(pcMdocFile) == 0) return false;
	if((int)strlen(pcMdocFile) + 8 >= iSize) return false;
	//---------------------------
	strcpy(pcName1, pcMdocFile);
	char* pcDot = strrchr(pcName1, '.');
	char* pcSlash = strrchr(pcName1, '/');
	if(pcDot == 0L || (pcSlash != 0L && pcDot < pcSlash)) return false;
	strcpy(pcDot, ".openTS");
	//---------------------------
	strcpy(pcName2, pcMdocFile);
	char* pcBase = strrchr(pcName2, '/');
	pcBase = (pcBase == 0L) ? pcName2 : pcBase + 1;
	pcDot = strchr(pcBase, '.');
	if(pcDot != 0L) strcpy(pcDot, ".openTS");
	return true;
}

bool CReadMdoc::mCheckOpenTS(const char* pcMdocFile)
{
	char acOpenTS1[512] = {'\0'}, acOpenTS2[512] = {'\0'};
	bool bValid = mGetOpenTSNames(pcMdocFile, acOpenTS1,
	   acOpenTS2, sizeof(acOpenTS1));
	if(!bValid) return false;
	//---------------------------
	struct stat st;
	if(stat(acOpenTS1, &st) == 0) return true;
	if(stat(acOpenTS2, &st) == 0) return true;
	return false;
}

//--------------------------------------------------------------------
// Returns the latest modification time of the mdoc file and its
// .openTS file(s). Returns 0 if none of them exists. The .openTS
// file is considered only for -Cmd 0.
//--------------------------------------------------------------------
time_t CReadMdoc::GetLastModified(const char* pcMdocFile)
{
	time_t tLatest = 0;
	struct stat st;
	if(pcMdocFile == 0L) return tLatest;
	if(stat(pcMdocFile, &st) == 0) tLatest = st.st_mtime;
	//---------------------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iCmd != 0) return tLatest;
	//---------------------------
	char acOpenTS1[512] = {'\0'}, acOpenTS2[512] = {'\0'};
	bool bValid = mGetOpenTSNames(pcMdocFile, acOpenTS1,
	   acOpenTS2, sizeof(acOpenTS1));
	if(!bValid) return tLatest;
	//---------------------------
	if(stat(acOpenTS1, &st) == 0 && st.st_mtime > tLatest)
	{	tLatest = st.st_mtime;
	}
	if(stat(acOpenTS2, &st) == 0 && st.st_mtime > tLatest)
	{	tLatest = st.st_mtime;
	}
	return tLatest;
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
	char acPath[1024] = {'\0'};
	strncpy(acPath, &pcEqual[1], sizeof(acPath) - 1);
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
	pcPath = new char[strlen(&acPath[iStart]) + 1];
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
	char acDateTime[256] = {'\0'};
	int iStartIdx = (pcEqual[1] == ' ') ? 2 : 1;
	strncpy(acDateTime, &pcEqual[iStartIdx], sizeof(acDateTime) - 1);
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
	if(m_ppcFrmPath != 0L) delete[] m_ppcFrmPath;
	if(m_piAcqIdxs != 0L) delete[] m_piAcqIdxs;
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_pfDoses != 0L) delete[] m_pfDoses;
	if(m_pfExpTimes != 0L) delete[] m_pfExpTimes;
	if(m_pDateTimes != 0L) delete[] m_pDateTimes;
	//---------------------------
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
