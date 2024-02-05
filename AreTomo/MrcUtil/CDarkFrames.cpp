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
	m_iNumDarks = 0;
	m_piFrmIdxs = 0L;
	m_piSecIdxs = 0L;
	m_pfTilts = 0L;
}

CDarkFrames::~CDarkFrames(void)
{
	mClean();
}

void CDarkFrames::mClean(void)
{
	if(m_piFrmIdxs != 0L) delete[] m_piFrmIdxs;
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	m_piFrmIdxs = 0L; m_piSecIdxs = 0L;
	m_pfTilts = 0L;
}

void CDarkFrames::Setup(int* piRawStkSize)
{
	mClean();
	m_iNumDarks = 0;
	//--------------
	memcpy(m_aiRawStkSize, piRawStkSize, sizeof(m_aiRawStkSize));
	m_piFrmIdxs = new int[2 * m_aiRawStkSize[2]];
	m_piSecIdxs = m_piFrmIdxs + m_aiRawStkSize[2];
	//--------------------------------------------
	m_pfTilts = new float[m_aiRawStkSize[2]];
}

void CDarkFrames::Add(int iFrmIdx, int iSecIdx, float fTilt)
{
	m_piFrmIdxs[m_iNumDarks] = iFrmIdx;
	m_piSecIdxs[m_iNumDarks] = iSecIdx;
	m_pfTilts[m_iNumDarks] = fTilt;
	m_iNumDarks += 1;
}

void CDarkFrames::AddTiltOffset(float fTiltOffset)
{
	for(int i=0; i<m_iNumDarks; i++)
	{	m_pfTilts[i] += fTiltOffset;
	}
}

int CDarkFrames::GetFrmIdx(int iNthDark)
{
	return m_piFrmIdxs[iNthDark];
}

int CDarkFrames::GetSecIdx(int iNthDark)
{
	return m_piSecIdxs[iNthDark];
}

float CDarkFrames::GetTilt(int iNthDark)
{
	return m_pfTilts[iNthDark];
}

bool CDarkFrames::IsDarkSection(int iSection)
{
	for(int i=0; i<m_iNumDarks; i++)
	{	if(iSection == m_piSecIdxs[i]) return true;
	}
	return false;
}

bool CDarkFrames::IsDarkFrame(int iFrame)
{
	for(int i=0; i<m_iNumDarks; i++)
	{	if(iFrame == m_piFrmIdxs[i]) return true;
	}
	return false;
}

void CDarkFrames::GenImodExcludeList(char* pcLine, int iSize)
{
	if(m_iNumDarks <= 0) return;
	strcpy(pcLine, "EXCLUDELIST ");
	//-----------------------------
	char acBuf[16] = {'\0'};
	int iLast = m_iNumDarks - 1;
	for(int i=0; i<iLast; i++)
	{	sprintf(acBuf, "%d, ", m_piSecIdxs[i]+1); 
		strcat(pcLine, acBuf); // Relion 1-based index
	}
	sprintf(acBuf, "%d", m_piSecIdxs[iLast]+1);
	strcat(pcLine, acBuf);
}
