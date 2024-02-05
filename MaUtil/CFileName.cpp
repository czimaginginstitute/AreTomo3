#include "CMaUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace McAreTomo::MaUtil;

CFileName::CFileName(void)
{
}

CFileName::CFileName(const char* pcFileName)
{
	this->Setup(pcFileName);
}

CFileName::~CFileName(void)
{
}

//------------------------------------------------------------------------------
// 1. m_acFolder stores the path of the folder including the last "/". If
//    pcFileName does not have folder path, use "./" as the folder.
// 2. m_acFileName stores the main file name EXCLUDING the folder and
//    the file extension.
// 3. m_acFileExt stores only the file extension that is the substring
//    after the last dot, such as "mrc" in myfolder/myfile.mrc.
//------------------------------------------------------------------------------
void CFileName::Setup(const char* pcFileName)
{
	memset(m_acFolder, 0, sizeof(m_acFolder));
	memset(m_acFileName, 0, sizeof(m_acFileName));
	memset(m_acFileExt, 0, sizeof(m_acFileExt));
	if(pcFileName == 0L || strlen(pcFileName) == 0) return;
	//-----------------------------------------------------
	char acBuf[256] = {'\0'};
	strcpy(acBuf, pcFileName);
	char* pcSlash = strrchr(acBuf, '/');
	//-------------------------------------------------
	// File name contains the path. The folder is the
	// substring from the beginning to the last "/"
	//-------------------------------------------------
	if(pcSlash != 0L)
	{	int iLen = strlen(pcSlash);
		if(iLen > 1) strcpy(m_acFileName, &pcSlash[1]);
		//---------------------------------------------
		// pcSlash is part of m_acFolder. strcpy puts
		// null at the end. This makes m_acFolder 
		// contains only the folder path.
		//---------------------------------------------
		strcpy(pcSlash, "/");
		strcpy(m_acFolder, acBuf);
	}
	//-------------------------------------------------
	// Since there is no "/", pcFileName is in the
	// working directory.
	//-------------------------------------------------
	else
	{	strcpy(m_acFileName, pcFileName);
		strcpy(m_acFolder, "./");
	}
	//--------------------------------------
	char* pcExt = strrchr(m_acFileName, '.');
	if(pcExt == 0L) return;
	//---------------------
	int iLen = strlen(pcExt);
	if(iLen > 1) strcpy(m_acFileExt, &pcExt[1]);
	//------------------------------------------
	// remove file extension from m_acFileName.
	//------------------------------------------
	memset(pcExt, 0, sizeof(char) * iLen);
}

void CFileName::GetFolder(char* pcFolder)
{
	strcpy(pcFolder, m_acFolder);
}

void CFileName::GetName(char* pcName)
{
	strcpy(pcName, m_acFileName);
}

void CFileName::GetExt(char* pcExt)
{
	strcpy(pcExt, m_acFileExt);
}

