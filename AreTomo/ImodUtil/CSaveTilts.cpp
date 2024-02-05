#include "CImodUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace McAreTomo::AreTomo::ImodUtil;
namespace MAM = McAreTomo::AreTomo::MrcUtil;

CSaveTilts::CSaveTilts(void)
{
	m_pFile = 0L;
}

CSaveTilts::~CSaveTilts(void)
{
	if(m_pFile != 0L) fclose(m_pFile);
}

void CSaveTilts::DoIt(int iNthGpu, const char* pcFileName)
{
	m_pFile = fopen(pcFileName, "wt");
	if(m_pFile == 0L) return;
	m_iNthGpu = iNthGpu;
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_iOutImod == 1) mSaveForRelion();
	else if(pInput->m_iOutImod == 2) mSaveForWarp();
	else if(pInput->m_iOutImod == 3) mSaveForAligned();
	//-----------------
	fclose(m_pFile);
	m_pFile = 0L;
}

//-----------------------------------------------------------------------------
// Relion 4 requires line return at the last line per Ge Peng of UCLA
//-----------------------------------------------------------------------------
void CSaveTilts::mSaveForAligned(void)
{
	//------------------------------------
	// aligned & dark-removed tilt series
	//------------------------------------
	MAM::CAlignParam* pAlnParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	int iLast = pAlnParam->m_iNumFrames - 1;
	for(int i=0; i<=iLast; i++)
	{	float fTilt = pAlnParam->GetTilt(i);
		fprintf(m_pFile, "%8.2f\n", fTilt);
	}
}

void CSaveTilts::mSaveForWarp(void)
{
	this->mSaveForAligned();	
}

//-----------------------------------------------------------------------------
// Relion 4 requires the last line have a line return per Ge Peng of UCLA
//-----------------------------------------------------------------------------
void CSaveTilts::mSaveForRelion(void)
{
	//--------------------------------------
	// raw tilt series as input to Relion 4
	//--------------------------------------
	MAM::CAlignParam* pAlnParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	MAM::CDarkFrames* pDarkFrames =
	   MAM::CDarkFrames::GetInstance(m_iNthGpu);
	//-----------------
	int iAllTilts = pDarkFrames->m_aiRawStkSize[2];
	char* pcLines = new char[iAllTilts * 256];
	for(int i=0; i<pAlnParam->m_iNumFrames; i++)
	{	float fTilt = pAlnParam->GetTilt(i);
		int iSecIdx = pAlnParam->GetSecIndex(i);
		char* pcLine = pcLines + iSecIdx * 256;
		sprintf(pcLine, "%8.2f", fTilt);
	}
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
	{	float fTilt = pDarkFrames->GetTilt(i);
		int iSecIdx = pDarkFrames->GetSecIdx(i);
		char* pcLine = pcLines + iSecIdx * 256;
		sprintf(pcLine, "%8.2f", fTilt);
	}
	//-----------------
	int iLast = iAllTilts - 1;
	for(int i=0; i<=iLast; i++)
	{	char* pcLine = pcLines + i * 256;
		fprintf(m_pFile, "%s\n", pcLine);
	}
	if(pcLines != 0L) delete[] pcLines;
}
