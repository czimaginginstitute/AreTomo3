#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CLoadAlnFile::CLoadAlnFile(void)
{
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
	m_iNumPatches = 0;
}

CLoadAlnFile::~CLoadAlnFile(void)
{
	mClean();
}

void CLoadAlnFile::mClean(void)
{
	while(!m_aHeaderQueue.empty()) 
	{	char* pcLine = m_aHeaderQueue.front();
		m_aHeaderQueue.pop();
		if(pcLine != 0L) delete[] pcLine;
	}
	while(!m_aDataQueue.empty())
	{	char* pcLine = m_aDataQueue.front();
		m_aDataQueue.pop();
		if(pcLine != 0L) delete[] pcLine;
	}
	//---------------------------------------
	if(m_pAlignParam != 0L) delete m_pAlignParam;
	if(m_pLocalParam != 0L) delete m_pLocalParam;
	m_pAlignParam = 0L;
	m_pLocalParam = 0L;
	m_iNumPatches = 0;
}

CAlignParam* CLoadAlnFile::GetAlignParam(bool bClean)
{
	CAlignParam* pAlignParam = m_pAlignParam;
	if(bClean) m_pAlignParam = 0L;
	return pAlignParam;
}

CLocalAlignParam* CLoadAlnFile::GetLocalParam(bool bClean)
{
	CLocalAlignParam* pLocalParam = m_pLocalParam;
	if(bClean) m_pLocalParam = 0L;
	return pLocalParam;
}

bool CLoadAlnFile::DoIt(const char* pcFile)
{
	mClean();
	m_bLoaded = false;
	//-----------------
	if(pcFile == 0L) return false;
	FILE* pFile = fopen(pcFile, "rt");
	if(pFile == 0L) return false;
	//-----------------
	char acBuf[256] = {'\0'};
	while(!feof(pFile))
	{	char* pcRet = fgets(:acBuf, 256, pFile);
		if(pcRet == 0L) break;
		else if(strlen(acBuf) < 4) continue;
		char* pcBuf = new char[256];
		if(acBuf[0] == '#')
		{	strcpy(pcBuf, &acBuf[1]);
			m_aHeaderQueue.push(pcBuf);
		}
		else 
		{	strcpy(pcBuf, acBuf);
			m_aDataQueue.push(pcBuf);
		}
	}
	fclose(pFile);
	//------------
	m_bLoaded = mParseHeader();
	if(!m_bLoaded)
	{	mClean();
		return false;
	}
	mLoadGlobal();
	mLoadLocal();
	return m_bLoaded;
}

bool CLoadAlnFile::mParseHeader(void)
{
	while(!m_aHeaderQueue.empty())
	{	char* pcLine = m_aHeaderQueue.front();
		m_aHeaderQueue.pop();
		//-------------------
		if(mParseRawSize(pcLine)) delete[] pcLine;
		else if(mParseDarkFrame(pcLine)) delete[] pcLine;
		else if(mParseNumPatches(pcLine)) delete[] pcLine;
		else delete[] pcLine;
	}
	//-----------------------------------------------
	// raw tilt series size must exist in .aln file.
	//-----------------------------------------------
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance();
	if(pDarkFrames->m_aiRawStkSize[2] == 0) return false;
	//-----------------
	int iNumTilts = pDarkFrames->m_aiRawStkSize[2] -
	   pDarkFrames->m_iNumDarks;
	if(iNumTilts <= 0) return false;
	//-----------------
	m_pAlignParam = new CAlignParam;
	m_pAlignParam->Create(iNumTilts);
	//-------------------------------
	if(m_iNumPatches > 0)
	{	m_pLocalParam = new CLocalAlignParam;
		m_pLocalParam->Setup(iNumTilts, m_iNumPatches);
	}
	return true;
}

bool CLoadAlnFile::mParseRawSize(char* pcLine)
{
	char* pcRawSize = strstr(pcLine, "RawSize");
	if(pcRawSize == 0L) return false;
	//-----------------
	char* pcTok = strtok(pcLine, "=");
	pcRawSize = strtok(0L, "=");
	//-----------------
	int aiRawSize[3] = {0};
	sscanf(pcRawSize, "%d %d %d", &aiRawSize[0],
	   &aiRawSize[1], &aiRawSize[2]);
	//-----------------
	return true;
}

bool CLoadAlnFile::mParseDarkFrame(char* pcLine)
{
	char* pcDarkFrm = strstr(pcLine, "DarkFrame");
	if(pcDarkFrm == 0L) return false;
	//-----------------
	char* pcTok = strtok(pcLine, "=");
	pcDarkFrm = strtok(0L, "=");
	//-----------------
	int iFrmIdx = 0, iSecIdx = 0; 
	float fTilt = 0.0f;
	sscanf(pcDarkFrm, "%d %d %f", &iFrmIdx, &iSecIdx, &fTilt);
	//-----------------
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance();
	pDarkFrames->AddDark(iFrmIdx);
	return true;
}

bool CLoadAlnFile::mParseNumPatches(char* pcLine)
{
	char* pcNumPatches = strstr(pcLine, "NumPatches");
	if(pcNumPatches == 0L) return true;
	//-----------------
	char* pcTok = strtok(pcLine, "=");
	pcNumPatches = strtok(0L, "=");
	//-----------------
	sscanf(pcNumPatches, "%d", &m_iNumPatches);
	return true;
}

void CLoadAlnFile::mLoadGlobal(void)
{
	if(m_pAlignParam == 0L) return;
	//-----------------------------
	int iSecIndex = 0;
	float fTilt, fTiltAxis, afShift[2];
	float GMAG, SMEAN, SFIT, SCALE, BASE;
	//-----------------------------------
	for(int i=0; i<m_pAlignParam->m_iNumFrames; i++)
	{	char* pcLine = m_aDataQueue.front();
		m_aDataQueue.pop();
		//-----------------
		sscanf(pcLine, "%d  %f  %f  %f  %f  %f  %f  %f  %f  %f",
		   &iSecIndex, &fTiltAxis, &GMAG, afShift+0, afShift+1,
		   &SMEAN, &SFIT, &SCALE, &BASE, &fTilt);
		//---------------------------------------
		m_pAlignParam->SetSecIdx(i, iSecIndex);
		m_pAlignParam->SetTilt(i, fTilt);
		m_pAlignParam->SetTiltAxis(i, fTiltAxis);
		m_pAlignParam->SetShift(i, afShift);
		if(pcLine != 0L) delete[] pcLine;
	}
}

void CLoadAlnFile::mLoadLocal(void)
{
	if(m_pLocalParam == 0L) return;
	//-----------------------------
	int t = 0, p = 0;
	int iSize = m_pAlignParam->m_iNumFrames * m_iNumPatches;
	for(int i=0; i<iSize; i++)
	{	char* pcLine = m_aDataQueue.front();
		m_aDataQueue.pop();
		//-----------------
		sscanf(pcLine, "%d %d %f %f %f %f %f", &t, &p, 
		   m_pLocalParam->m_pfCoordXs+i,
		   m_pLocalParam->m_pfCoordYs+i, 
		   m_pLocalParam->m_pfShiftXs+i,
		   m_pLocalParam->m_pfShiftYs+i,
		   m_pLocalParam->m_pfGoodShifts);
		if(pcLine != 0L) delete[] pcLine;
	}
}	

