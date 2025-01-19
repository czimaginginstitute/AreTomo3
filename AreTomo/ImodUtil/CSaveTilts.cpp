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
	m_iLineSize = 256;
	m_pcOrderedList = 0L;
	m_pbDarkImgs = 0L;
	m_pFile = 0L;

}

CSaveTilts::~CSaveTilts(void)
{
	if(m_pFile != 0L) fclose(m_pFile);
	mClean();
}

//-----------------------------------------------------------------------------
// 1. This generates a single-column text file that lists the tilt angles
//    in the same order as the tilt images in the input MRC file.
// 2. For Relion4 (-OutImod 1), the tilt angles of dark images must be 
//    included since it works on the raw tilt series.
// 3. For -OutImod 2 or 3, dark-image tilt angles need to be excluded since
//    a new tilt series is generated with dark images. Subtomo averaging
//    is expected to work on the dark-image removed tilt series.
//-----------------------------------------------------------------------------
void CSaveTilts::DoIt(int iNthGpu, const char* pcFileName)
{
	mClean();
	//-----------------
	m_pFile = fopen(pcFileName, "wt");
	if(m_pFile == 0L) return;
	//-----------------
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
// 1. -OutImod = 3 used for aligned tilt series where the tilt images are
//    ordered according to the tilt angles.
// 2. This is why we use MAM::CAlignParam to generate tilt angle list. 
//-----------------------------------------------------------------------------
void CSaveTilts::mSaveForAligned(void)
{
	MAM::CAlignParam* pAlnParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	int iLast = pAlnParam->m_iNumFrames - 1;
	for(int i=0; i<=iLast; i++)
	{	float fTilt = pAlnParam->GetTilt(i);
		fprintf(m_pFile, "%8.2f\n", fTilt);
	}
}

//-----------------------------------------------------------------------------
// 1. -OutImod = 2 used for dark-removed tilt series. Since this tilt series
//    gets saved after being sorted by tilt angle. We should use CAlignParam
//    as in mSaveForAligned.
//-----------------------------------------------------------------------------
void CSaveTilts::mSaveForWarp(void)
{
	this->mSaveForAligned();	
}

//-----------------------------------------------------------------------------
// Relion 4 requires the last line have a line return per Ge Peng of UCLA
//-----------------------------------------------------------------------------
void CSaveTilts::mSaveForRelion(void)
{
	mGenList();
	//-----------------
	for(int i=0; i<m_iAllTilts; i++)
	{	char* pcLine = m_pcOrderedList + i * m_iLineSize;
		fprintf(m_pFile, "%s\n", pcLine);
	}
	//-----------------
	mClean();
}

void CSaveTilts::mGenList(void)
{
	MAM::CDarkFrames* pDarkFrames =
	   MAM::CDarkFrames::GetInstance(m_iNthGpu);
	//-----------------
	m_iAllTilts = pDarkFrames->m_aiRawStkSize[2];
	m_pcOrderedList = new char[m_iAllTilts * m_iLineSize];
	m_pbDarkImgs = new bool[m_iAllTilts];
	//-----------------
	for(int i=0; i<m_iAllTilts; i++)
	{	float fTilt = pDarkFrames->GetTilt(i);
		int iSecIdx = pDarkFrames->GetSecIdx(i);
		int iImgIdx = iSecIdx - 1;
		char* pcLine = m_pcOrderedList + iImgIdx * m_iLineSize;
		sprintf(pcLine, "%8.2f", fTilt);
		//----------------
		m_pbDarkImgs[iImgIdx] = pDarkFrames->IsDarkFrame(i);
	}
}

void CSaveTilts::mClean(void)
{
	if(m_pcOrderedList != 0L)
	{	delete[] m_pcOrderedList;
		m_pcOrderedList = 0L;
	}
	if(m_pbDarkImgs != 0L)
	{	delete[] m_pbDarkImgs;
		m_pbDarkImgs = 0L;
	}
}
