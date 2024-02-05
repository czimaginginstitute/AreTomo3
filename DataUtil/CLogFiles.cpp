#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <sys/stat.h>

using namespace McAreTomo;
using namespace McAreTomo::DataUtil;

CLogFiles* CLogFiles::m_pInstances = 0L;
int CLogFiles::m_iNumGpus = 0;

void CLogFiles::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	//-----------------
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CLogFiles[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CLogFiles::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CLogFiles* CLogFiles::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CLogFiles::CLogFiles(void)
{
	m_pMcGlobalLog = 0L;
	m_pMcLocalLog = 0L;
	//-----------------
	m_pAtGlobalLog = 0L;
	m_pAtLocalLog = 0L;
}

CLogFiles::~CLogFiles(void)
{
	mCloseLogs();
}

void CLogFiles::Create(const char* pcMdocFile)
{
	mCloseLogs();
	memset(m_acLogDir, 0, sizeof(m_acLogDir));
	//-----------------
	CInput* pInput = CInput::GetInstance();
	strcpy(m_acLogDir, pInput->m_acOutDir);
	//-------------------------------------------------------------
	// Create a log subfolder in the output folder. The log folder
	// is named after the mdoc file name and appended with _Log.
	//-------------------------------------------------------------
	MU::CFileName fileName(pcMdocFile);
	char* pcPrefix = fileName.m_acFileName;
	//-----------------
	strcat(m_acLogDir, pcPrefix);
	strcat(m_acLogDir, "_Log/");
	//-----------------
	struct stat st = {0};
	if(stat(m_acLogDir, &st) == -1)
	{	mkdir(m_acLogDir, 0700);
	}
	//-----------------
	mCreateMcLogs(pcPrefix);
	mCreateAtLogs(pcPrefix);
}

void CLogFiles::mCreateMcLogs(const char* pcPrefix)
{
	char acLogPath[256] = {'\0'};
	mCreatePath(pcPrefix, "_MC_GL.csv", acLogPath);
	m_pMcGlobalLog = fopen(acLogPath, "wt");
	//-----------------
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(!pMcInput->bLocalAlign()) return;
	//-----------------
	mCreatePath(pcPrefix, "_MC_LO.csv", acLogPath);
	m_pMcLocalLog = fopen(acLogPath, "wt");
}

void CLogFiles::mCreateAtLogs(const char* pcPrefix)
{
	char acLogPath[256] = {'\0'};
	mCreatePath(pcPrefix, "_AT_GL.csv", acLogPath);
	if(m_pAtGlobalLog != 0L) fclose(m_pAtGlobalLog);
	m_pAtGlobalLog = fopen(acLogPath, "wt");
	//-----------------
	CAtInput* pAtInput = CAtInput::GetInstance();
	if(!pAtInput->bLocalAlign()) return;
	//-----------------
	mCreatePath(pcPrefix, "_AT_LT.csv", acLogPath);
	m_pAtLocalLog = fopen(acLogPath, "wt");
}

void CLogFiles::mCreatePath
(	const char* pcPrefix, 
	const char* pcSuffix,
	char* pcPath
)
{	strcpy(pcPath, m_acLogDir);
	strcat(pcPath, pcPrefix);
	strcat(pcPath, pcSuffix);
}

void CLogFiles::mCloseLogs(void)
{
	if(m_pMcGlobalLog != 0L) fclose(m_pMcGlobalLog);
	if(m_pMcLocalLog != 0L) fclose(m_pMcLocalLog);
	m_pMcGlobalLog = 0L;
	m_pMcLocalLog = 0L;
	//-----------------
	if(m_pAtGlobalLog != 0L) fclose(m_pAtGlobalLog);
	if(m_pAtLocalLog != 0L) fclose(m_pAtLocalLog);
	m_pAtGlobalLog = 0L;
	m_pAtLocalLog = 0L;
}
