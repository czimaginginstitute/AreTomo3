#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

using namespace McAreTomo::DataUtil;

CReadMdocDone* CReadMdocDone::m_pInstance = 0L;
char CReadMdocDone::m_acMdocDone[64] = {'\0'};

CReadMdocDone* CReadMdocDone::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	strcpy(m_acMdocDone, "MdocDone.txt");
	m_pInstance = new CReadMdocDone;
	return m_pInstance;
}

void CReadMdocDone::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

//------------------------------------------------------------------------------
// m_aMutex: It is initialized in Util_Thread. Do not init here
//------------------------------------------------------------------------------
CReadMdocDone::CReadMdocDone(void)
{
	m_pMdocFiles = 0L;
	m_iNumChars = 256;
}

//------------------------------------------------------------------------------
// m_aMutex: It is destroyed in Util_Thread. Do not destroy here.
//------------------------------------------------------------------------------
CReadMdocDone::~CReadMdocDone(void)
{
	this->mClean();
}

void CReadMdocDone::DoIt(void)
{
	this->mClean();
	//-----------------------------------------------
	// 1) Do not read MdocDone.txt for -Serial 0
	// since this processes one mdoc file only.
	// 2) -Cmd 1 is for reprocessing all tilt
	// series. Do not read MdocDone.
	// 3) -Resume 0 is for processing/reprocessing
	// everything from scratch. Do not read.
	//----------------------------------------------- 
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iSerial == 0) return;
	if(pInput->m_iCmd == 1) return;
	if(pInput->m_iResume == 0) return;
	//-----------------
	char acFileName[256] = {'\0'};
	strcpy(acFileName, pInput->m_acOutDir);
	strcat(acFileName, m_acMdocDone);
	FILE* pFile = fopen(acFileName, "rt");
	if(pFile == 0L) return;
	//-----------------
	m_pMdocFiles = new std::unordered_map<std::string, int>;
	char acBuf[m_iNumChars] = {'\0'};
	int iCount = 0;
	while(!feof(pFile))
	{	memset(acBuf, 0, sizeof(char) * m_iNumChars);
		char* pcRet = fgets(acBuf, m_iNumChars, pFile);
		if(pcRet == 0L) continue;
		//----------------
		int iSize1 = strlen(acBuf) - 1;
		if(iSize1 <= 0) continue;
		if(acBuf[iSize1] = '\n') acBuf[iSize1] = '\0';
		//----------------
		iCount += 1;
		m_pMdocFiles->insert({acBuf, iCount});
	}
	fclose(pFile);
}

//--------------------------------------------------------------------
// Check if the file name (without path) has a hit in the unordered
// map. If yes, this mdoc has been processed. Otherwise no.
//-------------------------------------------------------------------- 
bool CReadMdocDone::bExist(const char* pcMdocFile)
{
	if(m_pMdocFiles == 0L) return false;
	const char* pcMainFile = strrchr(pcMdocFile, '/');
	if(pcMainFile == 0L) 
	{	pcMainFile = pcMdocFile;
	}
	else
	{	if(strlen(pcMainFile) < 2) return false;
		pcMainFile = &pcMainFile[1];
	}
	auto item = m_pMdocFiles->find(pcMainFile);
	if(item == m_pMdocFiles->end()) return false;
	else return true;
}

void CReadMdocDone::mClean(void)
{
	if(m_pMdocFiles == 0L) return;
	m_pMdocFiles->clear();
	delete m_pMdocFiles;
	m_pMdocFiles = 0L;
}

