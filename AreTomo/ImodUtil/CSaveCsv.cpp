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

CSaveCsv::CSaveCsv(void)
{
	m_pFile = 0L;
	m_pcOrderedList = 0L;
	m_pbDarkImgs = 0L;
}

CSaveCsv::~CSaveCsv(void)
{
	if(m_pFile != 0L) fclose(m_pFile);
	mClean();
}

void CSaveCsv::DoIt
(	int iNthGpu,
	const char* pcFileName
)
{	mClean();
	m_pFile = fopen(pcFileName, "wt");
	if(m_pFile == 0L) return;
	//-----------------
	m_iNthGpu = iNthGpu;
	mGenList();
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_iOutImod == 1) mSaveForRelion();
	else if(pInput->m_iOutImod == 2) mSaveForWarp();
	else if(pInput->m_iOutImod == 3) mSaveForAligned();
	//-----------------
	fclose(m_pFile);
	m_pFile = 0L;
	mClean();
}

//-----------------------------------------------------------------------------
// 1. Ordered list for tilt serieds without including dark images.
// 2. Relion 4 requires line return at the last line per Ge Peng of UCLA
//-----------------------------------------------------------------------------
void CSaveCsv::mSaveForAligned(void)
{
	int iCounter = 0;
	fprintf(m_pFile, "ImageNumber, TiltAngle\n");
	for(int i=0; i<m_iAllTilts; i++) // i is iAcqIdx
	{	if(m_pbDarkImgs[i]) continue; 
		char* pcLine = m_pcOrderedList + iCounter * 256;
		fprintf(m_pFile, "%s\n", pcLine);
		iCounter += 1;
	}
}

void CSaveCsv::mSaveForWarp(void)
{
	this->mSaveForAligned();	
}

//-----------------------------------------------------------------------------
// 1. Relion 4 manual indicates that Ordered List is a chronological order
//    (sorted in ascending order) of the tilt series acquistion. It is a 
//    2-column, comma-separated, and no-space file with the frame-order 
//    list of the tilt series.
// 2. The first column is the frame (image) number (starting at 1) and the
//    second is the tilt angle (in degree).
// 3. Relion 4 works on tilt series including dark images that are specified
//    in tilt.com file. 
// 4. Relion 4 requires the last line have a line return per Ge Peng of UCLA
//-----------------------------------------------------------------------------
void CSaveCsv::mSaveForRelion(void)
{
	fprintf(m_pFile, "ImageNumber, TiltAngle\n");
	for(int i=0; i<m_iAllTilts; i++)
	{	char* pcLine = m_pcOrderedList + i * 256;
		fprintf(m_pFile, "%s\n", pcLine);
	}
}

void CSaveCsv::mGenList(void)
{
	MAM::CDarkFrames* pDarkFrames = 
	   MAM::CDarkFrames::GetInstance(m_iNthGpu);
	m_iAllTilts = pDarkFrames->m_aiRawStkSize[2];
	//-----------------
	m_pcOrderedList = new char[m_iAllTilts * 256];
	m_pbDarkImgs = new bool[m_iAllTilts];
	//-----------------------------------------------
	// pSeries stores bright images. Dark images have
	// been removed from pSeries.
	//-----------------------------------------------
	for(int i=0; i<m_iAllTilts; i++)
	{	float fTilt = pDarkFrames->GetTilt(i);
		int iAcqIdx = pDarkFrames->GetAcqIdx(i);
		//----------------
		char* pcLine = m_pcOrderedList + iAcqIdx * 256;
		sprintf(pcLine, "%4d,%.2f", iAcqIdx+1, fTilt);
		//----------------
		m_pbDarkImgs[iAcqIdx] = pDarkFrames->IsDarkFrame(i);
	}
}

void CSaveCsv::mClean(void)
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
