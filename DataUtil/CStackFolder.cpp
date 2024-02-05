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
	m_dRecentFileTime = 0.0;
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
	char* pcBuf = new char[256];
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
	m_dRecentFileTime = 0.0;
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
	printf("Directory: %s\n", m_acDirName);
	printf("Prefix:    %s\n", m_acPrefix);
	printf("Suffix:    %s\n", m_acSuffix);
	//-------------------------------------------------
	// Read all the movies in the specified folder for
	// batch processing.
	//-------------------------------------------------
	if(pInput->m_iSerial == 1) 
	{	int iNumRead = mReadFolder(true);
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
int CStackFolder::mReadFolder(bool bFirstTime)
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
	struct dirent* pDirent;
	char *pcPrefix = 0L, *pcSuffix = 0L;
	//-----------------
	struct stat statBuf;
	char acFullFile[256] = {'\0'};
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
		//----------------------------------
		// check if this is the latest file
		//----------------------------------
		strcpy(pcMainFile, pDirent->d_name);
		int iStatus = stat(acFullFile, &statBuf);
		double dTime = statBuf.st_mtim.tv_sec +
		   1e-9 * statBuf.st_mtim.tv_nsec;
		double dDeltaT = dTime - m_dRecentFileTime;
		//-----------------------------------------------------
		// But if this is the first time to read the directory,
		// bypass the latest file check, read all instead.
		//-----------------------------------------------------
		if(dDeltaT <= 0 && !bFirstTime) continue;
		if(dDeltaT > 0) m_dRecentFileTime = dTime;
		//----------------
		this->PushFile(acFullFile);
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

char* CStackFolder::mGetSerial(char* pcInputFile)
{
	char acBuf[256] = {'\0'};
	int iPrefixLen = strlen(m_acPrefix);
	strcpy(acBuf, pcInputFile+iPrefixLen);
	//-----------------
	int iSuffix = strlen(m_acSuffix);
	if(iSuffix > 0)
	{	char* pcSuffix = strcasestr(acBuf, m_acSuffix);
		if(pcSuffix != 0L) pcSuffix[0] = '\0';
	}
	else
	{	char* pcExt = strcasestr(acBuf, ".mdoc");
		if(pcExt != 0L) pcExt[0] = '\0';
	}
	//-----------------
	char* pcSerial = new char[256];
	memset(pcSerial, 0, sizeof(char) * 256);
	strcpy(pcSerial, acBuf);
	return pcSerial;
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

void CStackFolder::mClean(void)
{
	while(!m_aFileQueue.empty())
	{	char* pcFile = m_aFileQueue.front();
		m_aFileQueue.pop();
		if(pcFile != 0L) delete[] pcFile;
	}
}

bool CStackFolder::mAsyncReadFolder(void)
{
	this->Start();
	return true;
}

void CStackFolder::ThreadMain(void)
{
	const char* pcMethod = "in CStackFolder::ThreadMain";
	m_ifd = inotify_init1(IN_NONBLOCK);
	m_iwd = inotify_add_watch(m_ifd, m_acDirName, IN_CLOSE_WRITE);
	if(m_ifd == -1)
	{	fprintf(stderr, "Error: unable to init inotify %s.\n\n",
		   pcMethod); return;
	}
	if(m_iwd == -1) 
	{	fprintf(stderr, "Error: add_watch failed %s.\n\n",
		   pcMethod);
		close(m_ifd); return;
	}
	//-----------------
	CInput* pInput = CInput::GetInstance();
	int iEventSize = sizeof(inotify_event);
	char acEventBuf[4096] = {'\0'};
	inotify_event* pEvents = (inotify_event*)acEventBuf;
	//-----------------
	bool bFirstTime = true;
	int iNumEvents = 0, iNumFiles = 0, iCount = 0;
	//-----------------
	while(true)
	{	int iQueueSize = GetQueueSize();
		if(iQueueSize > 0)
		{	this->mWait(iQueueSize * 5.0f);
			continue;
		}
		//----------------
		if(bFirstTime)
		{	iNumFiles = mReadFolder(bFirstTime);
			if(iNumFiles > 0) 
			{	bFirstTime = false;
				iCount = 0;
			}
			else 
			{	this->mWait(10.0f);
				iCount += 10.0f;
				int iLeftSec = pInput->m_iSerial - iCount;
				printf("No mdoc files have been found, "
				   "wait %d seconds.\n\n", iLeftSec);
				if(iLeftSec <= 0) break;
			}
			continue;
		}
		//----------------
		iNumEvents = read(m_ifd, acEventBuf, 4096);
		bool bCloseWrite = false;
		for(int i=0; i<iNumEvents; i++)
		{	inotify_event* pEvent = pEvents + i;
			bCloseWrite = pEvent->mask & IN_CLOSE_WRITE;
			if(bCloseWrite) break;
		}
		//----------------
		float fWaitSec = bCloseWrite ? 0.1f : 2.0f;
		this->mWait(fWaitSec);
		iCount += (int)fWaitSec;
		//----------------
		iNumFiles = mReadFolder(false);
		if(iNumFiles > 0) 
		{	iCount = 0;
			continue;
		}
		else
		{	this->mWait(2.0f);
			iCount += 2;
		}
		//----------------
		int iLeftSec = pInput->m_iSerial - iCount;
		if(iLeftSec <= 0) break;
		//----------------
		if(iCount % 10 != 0) continue; 
		printf("Folder watching thread: no tilt series have been found"
		   " for %d seconds, wait %d seconds before quitting.\n\n", 
		   iCount, iLeftSec);
	}
	inotify_rm_watch(m_ifd, m_iwd);
	close(m_ifd);
}
