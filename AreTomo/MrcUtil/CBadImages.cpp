#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <string.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CBadImages* CBadImages::m_pInstances = 0L;
int CBadImages::m_iNumGpus = 0;

void CBadImages::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CBadImages[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CBadImages::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CBadImages* CBadImages::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CBadImages::CBadImages(void)
{
	memset(m_aiRawStkSize, 0, sizeof(m_aiRawStkSize));
	m_piAcqIdxs = 0L;
	m_piSecIdxs = 0L;
	m_pfTilts = 0L;
	//---------------------------
	m_piDarkIdxs = 0L;
	m_piDarkSecs = 0L;
	m_pfDarkTilts = 0L;
	m_pbDarkImgs = 0L;
}

CBadImages::~CBadImages(void)
{
	mClean();
}

void CBadImages::mClean(void)
{
	if(m_piAcqIdxs != 0L) delete[] m_piAcqIdxs;
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_piSecIdxs != 0L) delete[] m_piSecIdxs;
	m_piAcqIdxs = 0L;
	m_pfTilts = 0L;
	m_piSecIdxs = 0L;
	//---------------------------
	if(m_pbDarkImgs != 0L) delete[] m_pbDarkImgs;
	if(m_piDarkIdxs != 0L) delete[] m_piDarkIdxs;
	if(m_piDarkSecs != 0L) delete[] m_piDarkSecs;
	if(m_pfDarkTilts != 0L) delete[] m_pfDarkTilts;
	m_pbDarkImgs = 0L;
	m_piDarkIdxs = 0L;
	m_piDarkSecs = 0L;
	m_pfDarkTilts = 0L;
}

//--------------------------------------------------------------------
// 1. pSeries must contain all tilt images including dark ones.
// 2. pSeries must be sorted in ascending order of tilt angles.
//--------------------------------------------------------------------
void CBadImages::Setup(MD::CTiltSeries* pSeries)
{
	mClean();
	m_iNumBads = 0;
	//-----------------
	memcpy(m_aiRawStkSize, pSeries->m_aiStkSize, sizeof(int) * 3);
	m_piAcqIdxs = new int[m_aiRawStkSize[2]];
	m_piSecIdxs = new int[m_aiRawStkSize[2]];
	m_pfTilts = new float[m_aiRawStkSize[2]];
	//-----------------
	m_piDarkIdxs = new int[m_aiRawStkSize[2]];
	m_piDarkSecs = new int[m_aiRawStkSize[2]];
	m_pfDarkTilts = new float[m_aiRawStkSize[2]];
	//-----------------
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

void CBadImages::Setup(int iNthGpu)
{
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPackage->GetSeries(0);
	this->Setup(pTiltSeries);
}

/*
void CBadImages::AddBadImg(int iFrmIdx)
{
	m_pbDarkImgs[iFrmIdx] = true;
	m_piDarkIdxs[m_iNumBads] = iFrmIdx;
	m_iNumBads += 1;
}
*/

//--------------------------------------------------------------------
// 1. This method is specifically for AreTomo/MrcUtil/CLoadAlignFile
//    so that it places the dark-image information loaded from .aln
//    file in here.
// 2. iFrmIdx is zero based, iSecIdx is 1 based. If they are the
//    same, add 1 to iSecIdx.
// 3. Also used by CRemoveDarkFrames and CDetectBadShifts. Entries
//    are kept in ascending order of iFrmIdx and duplicates are
//    ignored, since CRemoveDarkFrames::mRemoveSeries removes frames
//    from the last entry to the first.
//--------------------------------------------------------------------
void CBadImages::AddBadImg(int iFrmIdx, int iSecIdx, float fTilt)
{
	if(m_pbDarkImgs[iFrmIdx]) return;
	m_pbDarkImgs[iFrmIdx] = true;
	//-----------------
	int i = m_iNumBads;
	while(i > 0 && m_piDarkIdxs[i-1] > iFrmIdx)
	{	m_piDarkIdxs[i] = m_piDarkIdxs[i-1];
		m_piDarkSecs[i] = m_piDarkSecs[i-1];
		m_pfDarkTilts[i] = m_pfDarkTilts[i-1];
		i -= 1;
	}
	m_piDarkIdxs[i] = iFrmIdx;
	m_piDarkSecs[i] = iSecIdx;
	m_pfDarkTilts[i] = fTilt;
	m_iNumBads += 1;
}

void CBadImages::AddTiltOffset(float fTiltOffset)
{
	for(int i=0; i<m_iNumBads; i++)
	{	m_pfDarkTilts[i] += fTiltOffset;
	}
	for(int i=0; i<m_aiRawStkSize[2]; i++)
	{	m_pfTilts[i] += fTiltOffset;
	}
}

int CBadImages::GetAcqIdx(int iFrame)
{
	return m_piAcqIdxs[iFrame];
}

int CBadImages::GetSecIdx(int iFrame)
{
	return m_piSecIdxs[iFrame];
}

float CBadImages::GetTilt(int iFrame)
{
	return m_pfTilts[iFrame];
}

int CBadImages::GetDarkIdx(int iDark)
{
	return m_piDarkIdxs[iDark];
}

int CBadImages::GetDarkSec(int iDark)
{
	return m_piDarkSecs[iDark];
}

float CBadImages::GetDarkTilt(int iDark)
{
	return m_pfDarkTilts[iDark];
}

int CBadImages::GetNumAlnTilts(void)
{
	int iNumAlnTilts = m_aiRawStkSize[2] - m_iNumBads;
	if(iNumAlnTilts < 0) iNumAlnTilts = 0;
	return iNumAlnTilts;
}

bool CBadImages::IsDarkImg(int iFrame)
{
	return m_pbDarkImgs[iFrame];
}

void CBadImages::GenImodExcludeList(char* pcLine, int iSize)
{
	if(m_iNumBads <= 0) return;
	//-----------------
	strcpy(pcLine, "EXCLUDELIST ");
	char acBuf[16] = {'\0'};
	int iLast = m_iNumBads - 1;
	for(int i=0; i<iLast; i++)
	{	sprintf(acBuf, "%d,", m_piDarkSecs[i]); 
		strcat(pcLine, acBuf); // Relion 1-based index
	}
	//-----------------
	sprintf(acBuf, "%d", m_piDarkSecs[iLast]);
	strcat(pcLine, acBuf);
}

bool CBadImages::bZeroBased(void)
{
	if(m_iNumBads <= 0) return false;
	//---------------------------
	if(m_piDarkIdxs[0] == m_piDarkSecs[0]) return true;
	//---------------------------
	for(int i=0; i<m_iNumBads; i++)
	{	if(m_piDarkSecs[i] == 0) return true;
	}
	return false;
}

void CBadImages::ToOneBased(void)
{
	for(int i=0; i<m_iNumBads; i++)
	{	m_piDarkSecs[i] = m_piDarkIdxs[i] + 1;
	}
}
