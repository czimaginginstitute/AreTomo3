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

static pthread_mutex_t s_mutex;

CSaveMdocDone* CSaveMdocDone::m_pInstance = 0L;

CSaveMdocDone* CSaveMdocDone::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	//--------------------------------------------------
	// 1) we want CReadMdocDone is created first so
	// that the existing content is read before new
	// mdoc files are saved in MdocDone.txt.
	//--------------------------------------------------
	CReadMdocDone* pReadMdocDone = CReadMdocDone::GetInstance();
	pReadMdocDone->DoIt();
	//-----------------
	m_pInstance = new CSaveMdocDone;
	return m_pInstance;
}

void CSaveMdocDone::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

//------------------------------------------------------------------------------
// m_aMutex: It is initialized in Util_Thread. Do not init here
//------------------------------------------------------------------------------
CSaveMdocDone::CSaveMdocDone(void)
{
	CInput* pInput = CInput::GetInstance();
	char acLogFile[256] = {'\0'};
	strcpy(acLogFile, pInput->m_acOutDir);
	strcat(acLogFile, MD::CReadMdocDone::m_acMdocDone);
	//-----------------
	const char* pcMode = (pInput->m_iResume == 0) ? "wt" : "at";
	m_pLogFile = fopen(acLogFile, pcMode);
	//-----------------
	pthread_mutex_init(&s_mutex, 0L);
}

//------------------------------------------------------------------------------
// m_aMutex: It is destroyed in Util_Thread. Do not destroy here.
//------------------------------------------------------------------------------
CSaveMdocDone::~CSaveMdocDone(void)
{
	pthread_mutex_destroy(&s_mutex);
	if(m_pLogFile != 0L) fclose(m_pLogFile);
}

void CSaveMdocDone::DoIt(const char* pcMdocFile)
{
	//---------------------------------------------------------
	// 1) Saving MdocDone.txt is needed only for -Cmd = 0.
	//---------------------------------------------------------
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iCmd != 0) return;
	//-----------------
	if(m_pLogFile == 0L) return;
	pthread_mutex_lock(&s_mutex);
	//-----------------
	const char* pcMainFile = strrchr(pcMdocFile, '/');
	if(pcMainFile != 0L)
	{	int iSize = strlen(pcMainFile);
		if(iSize > 1) fprintf(m_pLogFile, "%s\n", &pcMainFile[1]);
	}
	else
	{	fprintf(m_pLogFile, "%s\n", pcMdocFile);
	}
	fflush(m_pLogFile);
	//-----------------
	pthread_mutex_unlock(&s_mutex);
}
