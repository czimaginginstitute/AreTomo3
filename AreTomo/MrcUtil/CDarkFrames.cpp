#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <string.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CDarkFrames* CDarkFrames::m_pInstances = 0L;
int CDarkFrames::m_iNumGpus = 0;

void CDarkFrames::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CDarkFrames[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CDarkFrames::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CDarkFrames* CDarkFrames::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CDarkFrames::CDarkFrames(void)
{
	memset(m_aiRawStkSize, 0, sizeof(m_aiRawStkSize));
	m_piAcqIdxs = 0L;
	m_piSecIdxs = 0L;
	m_piDarkIdxs = 0L;
	m_pfTilts = 0L;
	m_pbDarkImgs = 0L;
}

CDarkFrames::~CDarkFrames(void)
{
	mClean();
}

void CDarkFrames::mClean(void)
{
	if(m_piAcqIdxs != 0L) delete[] m_piAcqIdxs;
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_pbDarkImgs != 0L) delete[] m_pbDarkImgs;
	m_piAcqIdxs = 0L;
	m_piSecIdxs = 0L;
	m_piDarkIdxs = 0L;
	m_pfTilts = 0L;
	m_pbDarkImgs = 0L;
}

//--------------------------------------------------------------------
// 1. pSeries must contain all tilt images including dark ones.
// 2. pSeries must be sorted in ascending order of tilt angles.
//--------------------------------------------------------------------
void CDarkFrames::Setup(MD::CTiltSeries* pSeries)
{
	mClean();
	m_iNumDarks = 0;
	//-----------------
	memcpy(m_aiRawStkSize, pSeries->m_aiStkSize, sizeof(int) * 3);
	m_piAcqIdxs = new int[3 * m_aiRawStkSize[2]];
	m_piSecIdxs = &m_piAcqIdxs[m_aiRawStkSize[2]];
	m_piDarkIdxs = &m_piAcqIdxs[m_aiRawStkSize[2] * 2];
	//-----------------
	m_pfTilts = new float[m_aiRawStkSize[2]];
	m_pbDarkImgs = new bool[m_aiRawStkSize[2]];
	//-----------------
	size_t tBytes = sizeof(int) * m_aiRawStkSize[2];
	memcpy(m_piAcqIdxs, pSeries->m_piAcqIndices, tBytes);
	memcpy(m_piSecIdxs, pSeries->m_piSecIndices, tBytes);
	//-----------------
	tBytes = sizeof(float) * m_aiRawStkSize[2];
	memcpy(m_pfTilts, pSeries->m_pfTilts, tBytes);
	//-----------------
	memset(m_pbDarkImgs, 0, sizeof(bool) * m_aiRawStkSize[2]);
}

void CDarkFrames::Setup(int iNthGpu)
{
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPackage->GetSeries(0);
	this->Setup(pTiltSeries);
}

void CDarkFrames::AddDark(int iFrmIdx)
{
	m_pbDarkImgs[iFrmIdx] = true;
	m_piDarkIdxs[m_iNumDarks] = iFrmIdx;
	m_iNumDarks += 1;
}

//--------------------------------------------------------------------
// 1. This method is specifically for AreTomo/MrcUtil/CLoadAlignFile
//    so that it places the dark-image information loaded from .aln
//    file in here.
//--------------------------------------------------------------------
void CDarkFrames::AddDark(int iFrmIdx, int iSecIdx, float fTilt)
{
	this->AddDark(iFrmIdx);
	m_piSecIdxs[iFrmIdx] = iSecIdx;
	m_pfTilts[iFrmIdx] = fTilt;
}

void CDarkFrames::AddTiltOffset(float fTiltOffset)
{
	for(int i=0; i<m_aiRawStkSize[2]; i++)
	{	m_pfTilts[i] += fTiltOffset;
	}
}

int CDarkFrames::GetAcqIdx(int iFrame)
{
	return m_piAcqIdxs[iFrame];
}

int CDarkFrames::GetSecIdx(int iFrame)
{
	return m_piSecIdxs[iFrame];
}

float CDarkFrames::GetTilt(int iFrame)
{
	return m_pfTilts[iFrame];
}

int CDarkFrames::GetDarkIdx(int iDark)
{
	return m_piDarkIdxs[iDark];
}

int CDarkFrames::GetNumAlnTilts(void)
{
	int iNumAlnTilts = m_aiRawStkSize[2] - m_iNumDarks;
	if(iNumAlnTilts < 0) iNumAlnTilts = 0;
	return iNumAlnTilts;
}

bool CDarkFrames::IsDarkFrame(int iFrame)
{
	return m_pbDarkImgs[iFrame];
}

void CDarkFrames::GenImodExcludeList(char* pcLine, int iSize)
{
	if(m_iNumDarks <= 0) return;
	//-----------------
	strcpy(pcLine, "EXCLUDELIST ");
	char acBuf[16] = {'\0'};
	int iLast = m_iNumDarks - 1;
	for(int i=0; i<iLast; i++)
	{	int iDarkFm = m_piDarkIdxs[i];
		int iSecIdx = m_piSecIdxs[iDarkFm] + 1;
		sprintf(acBuf, "%d,", iSecIdx); 
		strcat(pcLine, acBuf); // Relion 1-based index
	}
	//-----------------
	int iDarkFm = m_piDarkIdxs[iLast];
	int iSecIdx = m_piSecIdxs[iDarkFm] + 1;
	sprintf(acBuf, "%d", iSecIdx);
	strcat(pcLine, acBuf);
}
