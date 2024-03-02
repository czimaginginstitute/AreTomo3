#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

using namespace McAreTomo::DataUtil;

CStackFolder* CStackFolder::m_pInstance = 0L;

CStackFolder* CStackFolder::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CStackFolder;
	return m_pInstance;
}

void CStackFolder::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

//------------------------------------------------------------------------------
// m_aMutex: It is initialized in Util_Thread. Do not init here
//------------------------------------------------------------------------------
CStackFolder::CStackFolder(void)
{
	m_iNumChars = 256;
}

//------------------------------------------------------------------------------
// m_aMutex: It is destroyed in Util_Thread. Do not destroy here.
//------------------------------------------------------------------------------
CStackFolder::~CStackFolder(void)
{
	this->mClean();
}

void CStackFolder::PushFile(char* pcMdocFile)
{
	if(pcMdocFile == 0L) return;
	char* pcBuf = new char[m_iNumChars];
	strcpy(pcBuf, pcMdocFile);
	//-----------------
	pthread_mutex_lock(&m_aMutex);
	m_aFileQueue.push(pcBuf);
	pthread_mutex_unlock(&m_aMutex);
}

char* CStackFolder::GetFile(bool bPop)
{
	char* pcMdocFile = 0L;
	pthread_mutex_lock(&m_aMutex);
	if(!m_aFileQueue.empty())
	{	pcMdocFile = m_aFileQueue.front();
		if(bPop) m_aFileQueue.pop();
	}
	pthread_mutex_unlock(&m_aMutex);
	return pcMdocFile;
}

void CStackFolder::DeleteFront(void)
{
	char* pcMdocFile = this->GetFile(true);
	if(pcMdocFile != 0L) delete[] pcMdocFile;
}

int CStackFolder::GetQueueSize(void)
{
	pthread_mutex_lock(&m_aMutex);
	int iSize = m_aFileQueue.size();
	pthread_mutex_unlock(&m_aMutex);
	return iSize;
}

bool CStackFolder::ReadFiles(void)
{
	this->mClean();
	//----------------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iSerial == 0) 
	{	bool bSuccess = mReadSingle();
		return bSuccess;
	}
	//----------------------
	bool bSuccess = mGetDirName();
	if(!bSuccess) return false;
	strcpy(m_acSuffix, pInput->m_acInSuffix);
	strcpy(m_acSkips, pInput->m_acInSkips);
	printf("Directory: %s\n", m_acDirName);
	printf("Prefix:    %s\n", m_acPrefix);
	printf("Suffix:    %s\n", m_acSuffix);
	printf("Skips:     %s\n", m_acSkips);
	//-------------------------------------------------
	// Read all the movies in the specified folder for
	// batch processing.
	//-------------------------------------------------
	if(pInput->m_iSerial == 1) 
	{	int iNumRead = mReadFolder();
		if(iNumRead > 0) return true;
		else return false;
	}
	//-------------------------------------------------------
	// Monitor the specified folder for new movie files that
	// have been saved recently. If yes, read the new movie
	// file name for future loading.
	//------------------------------------------------------- 
	if(pInput->m_iSerial > 1) 
	{	bSuccess = mAsyncReadFolder();
		return bSuccess;
	}
	else return false;
}

bool CStackFolder::mReadSingle(void)
{
	CInput* pInput = CInput::GetInstance();
	m_aFileQueue.push(pInput->m_acInMdoc);
	return true;
}

//--------------------------------------------------------------------
// 1. User passes in a file name that is used as the template to 
//    search for a series stack files containing serial numbers.
// 2. The template file contains the full path that is used to
//    determine the folder containing the series stack files
//--------------------------------------------------------------------
int CStackFolder::mReadFolder(void)
{
	DIR* pDir = opendir(m_acDirName);
	if(pDir == 0L)
	{	fprintf(stderr, "Error: cannot open folder\n   %s"
		   "in CStackFolder::mReadFolder.\n\n", m_acDirName);
		return -1;
	}
	//-----------------
	int iNumRead = 0;
	int iPrefix = strlen(m_acPrefix);
	int iSuffix = strlen(m_acSuffix);
	int iSkips = strlen(m_acSkips);
	struct dirent* pDirent;
	char *pcPrefix = 0L, *pcSuffix = 0L;
	//-----------------
	struct stat statBuf;
	char acFullFile[m_iNumChars] = {'\0'};
	strcpy(acFullFile, m_acDirName);
	char* pcMainFile = acFullFile + strlen(m_acDirName);
	//-----------------
	while(true)
	{	pDirent = readdir(pDir);
		if(pDirent == 0L) break;
		if(pDirent->d_name[0] == '.') continue;
		//----------------
		if(iPrefix > 0)
		{	pcPrefix = strstr(pDirent->d_name, m_acPrefix);
			if(pcPrefix == 0L) continue;
		}
		//----------------
		if(iSuffix > 0)
		{	pcSuffix = strcasestr(pDirent->d_name 
			   + iPrefix, m_acSuffix);
			if(pcSuffix == 0L) continue;
		}
		//----------------
		if(iSkips > 0)
		{	bool bSkip = mCheckSkips(pDirent->d_name);
			if(bSkip) continue;
		}
		//----------------
		if(m_aReadFiles.find(pDirent->d_name) ==
		   m_aReadFiles.end())
		{	int iNumFiles = m_aReadFiles.size();
                	m_aReadFiles[pDirent->d_name] = iNumFiles;
		}
		else continue;
		//----------------
		strcpy(pcMainFile, pDirent->d_name);
		this->PushFile(acFullFile);
		//----------------
		printf("added: %s\n", acFullFile);
		iNumRead += 1;
	}
	closedir(pDir);
	//-----------------
	if(iNumRead <= 0)
	{	fprintf(stderr, "Error: no files are found.");
		fprintf(stderr, "   in %s\n\n", m_acDirName);
	}
	else printf("\n");
	//-----------------
	return iNumRead;
}

bool CStackFolder::mGetDirName(void)
{
	CInput* pInput = CInput::GetInstance();
	char* pcSlash = strrchr(pInput->m_acInMdoc, '/');
	if(pcSlash == 0L)
	{	strcpy(m_acPrefix, pInput->m_acInMdoc);
		char* pcRet = getcwd(m_acDirName, sizeof(m_acDirName));
		strcpy(m_acDirName, "./");
		return true;
	}
	else
	{	strcpy(m_acDirName, pInput->m_acInMdoc);
		pcSlash = strrchr(m_acDirName, '/');
		//----------------
		strcpy(m_acPrefix, &pcSlash[1]);
		pcSlash[1] = '\0';
	}
	return true;
}

bool CStackFolder::mCheckSkips(const char* pcString)
{
	char acBuf[m_iNumChars] = {'\0'};
	strcpy(acBuf, m_acSkips);
	//-----------------
	char* pcToken = strtok(acBuf, ", ");
	while(pcToken != 0L)
	{	if(strstr(pcString, pcToken) != 0L) return true;
		pcToken = strtok(0L, ", ");
	}
	return false;
}

void CStackFolder::mClean(void)
{
	while(!m_aFileQueue.empty())
	{	char* pcFile = m_aFileQueue.front();
		m_aFileQueue.pop();
		if(pcFile != 0L) delete[] pcFile;
	}
	m_aReadFiles.clear();
}

void CStackFolder::mLogFiles(void)
{
	char acFile[256] = {'\0'};
	CInput* pInput = CInput::GetInstance();
	strcpy(acFile, pInput->m_acOutDir);
	strcat(acFile, "MdocList.txt");
	FILE* pFile = fopen(acFile, "wt");
	if(pFile == 0L) return;
	//-----------------
	for(auto x : m_aReadFiles)
	{	fprintf(pFile, "%s  %d\n", x.first.c_str(), x.second);
		printf("%s  %d\n", x.first.c_str(), x.second);
	}
	fclose(pFile);
}

bool CStackFolder::mAsyncReadFolder(void)
{
	this->Start();
	return true;
}

void CStackFolder::ThreadMain(void)
{
	CInput* pInput = CInput::GetInstance();
	int iCount = 0;
	//-----------------
	while(true)
	{	int iQueueSize = GetQueueSize();
		if(iQueueSize > 0)
		{	this->mWait(iQueueSize * 5.0f);
			continue;
		}
		//----------------
		int iNumFiles = mReadFolder();
		if(iNumFiles > 0) 
		{	iCount = 0;
			continue;
		}
		//----------------
		this->mWait(10.0f);
		iCount += 10.0f;
		int iLeftSec = pInput->m_iSerial - iCount;
		printf("No mdoc files have been found, "
		   "wait %d seconds.\n\n", iLeftSec);
		if(iLeftSec <= 0) break;
		else continue;
	}
	mLogFiles();
}
