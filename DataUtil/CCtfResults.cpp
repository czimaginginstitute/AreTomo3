#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo;
using namespace McAreTomo::DataUtil;

CCtfResults* CCtfResults::m_pInstances = 0L;
int CCtfResults::m_iNumGpus = 0;

void CCtfResults::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CCtfResults[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CCtfResults::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CCtfResults* CCtfResults::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CCtfResults::CCtfResults(void)
{
	mInit();
}

void CCtfResults::mInit(void)
{
	m_iNumImgs = 0;
	m_pfDfMins = 0L;
	m_ppfSpects = 0L;
}

CCtfResults::~CCtfResults(void)
{
	this->Clean();
}

void CCtfResults::Setup(int iNumImgs, int* piSpectSize)
{
	this->Clean();
	m_iNumImgs = iNumImgs;
	m_aiSpectSize[0] = piSpectSize[0];
	m_aiSpectSize[1] = piSpectSize[1];
	//--------------------------------------------
	int iNumCols = 6;
	m_pfDfMins = new float[m_iNumImgs * iNumCols];
	m_pfDfMaxs = &m_pfDfMins[m_iNumImgs];
	m_pfAzimuths = &m_pfDfMaxs[m_iNumImgs];
	m_pfExtPhases = &m_pfAzimuths[m_iNumImgs];
	m_pfScores = &m_pfExtPhases[m_iNumImgs];
	m_pfTilts = &m_pfScores[m_iNumImgs];
	//-------------------------------------------------
	int iBytes = sizeof(float) * m_iNumImgs * iNumCols;
	memset(m_pfDfMins, 0, iBytes);
	//-----------------------------------
	m_ppfSpects = new float*[m_iNumImgs];
	memset(m_ppfSpects, 0, sizeof(float*) * m_iNumImgs);
}

void CCtfResults::Clean(void)
{
	if(m_iNumImgs == 0) return;
	//-------------------------
	if(m_pfDfMins != 0L) delete[] m_pfDfMins;
	for(int i=0; i<m_iNumImgs; i++)
	{	if(m_ppfSpects[i] == 0L) continue;
		delete[] m_ppfSpects[i];
		m_ppfSpects[i] = 0L;
	}
	if(m_ppfSpects != 0L) delete[] m_ppfSpects;
	mInit();
}

void CCtfResults::SetTilt(int iImage, float fTilt)
{
	m_pfTilts[iImage] = fTilt;
}

void CCtfResults::SetDfMin(int iImage, float fDfMin)
{
	m_pfDfMins[iImage] = fDfMin;
}

void CCtfResults::SetDfMax(int iImage, float fDfMax)
{
	m_pfDfMaxs[iImage] = fDfMax;
}

void CCtfResults::SetAzimuth(int iImage, float fAzimuth)
{
	m_pfAzimuths[iImage] = fAzimuth;
}

void CCtfResults::SetExtPhase(int iImage, float fExtPhase)
{
	m_pfExtPhases[iImage] = fExtPhase;
}

void CCtfResults::SetScore(int iImage, float fScore)
{
	m_pfScores[iImage] = fScore;
}

void CCtfResults::SetSpect(int iImage, float* pfSpect)
{
	if(m_ppfSpects[iImage] != 0L) delete[] m_ppfSpects[iImage];
	m_ppfSpects[iImage] = pfSpect;
}

float CCtfResults::GetTilt(int iImage)
{
	return m_pfTilts[iImage];
}

float CCtfResults::GetDfMin(int iImage)
{
	return m_pfDfMins[iImage];
}

float CCtfResults::GetDfMax(int iImage)
{
	return m_pfDfMaxs[iImage];
}

float CCtfResults::GetAzimuth(int iImage)
{
	return m_pfAzimuths[iImage];
}

float CCtfResults::GetExtPhase(int iImage)
{
	return m_pfExtPhases[iImage];
}

float CCtfResults::GetScore(int iImage)
{
	return m_pfScores[iImage];
}

float* CCtfResults::GetSpect(int iImage, bool bClean)
{
	float* pfSpect = m_ppfSpects[iImage];
	if(bClean) m_ppfSpects[iImage] = 0L;
	return pfSpect;
}

void CCtfResults::SaveImod(const char* pcCtfTxtFile)
{
	FILE* pFile = fopen(pcCtfTxtFile, "w");
	if(pFile == 0L) return;
	//-----------------
	float fExtPhase = this->GetExtPhase(0);
	if(fExtPhase == 0) fprintf(pFile, "1  0  0.0  0.0  0.0  3\n");
	else fprintf(pFile, "5  0  0.0  0.0  0.0  3\n");
	//-----------------
	const char *pcFormat1 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f\n";
	const char *pcFormat2 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f  %8.2f\n";
	float fDfMin, fDfMax;
	if(fExtPhase == 0)
	{	for(int i=0; i<m_iNumImgs; i++)
		{	float fTilt = this->GetTilt(i);
			fDfMin = this->GetDfMin(i) * 0.1f;
			fDfMax = this->GetDfMax(i) * 0.1f;
			fprintf(pFile, pcFormat1, i+1, i+1, fTilt, fTilt,
			   fDfMin, fDfMax, this->GetAzimuth(i));
		}
	}
	else
	{	for(int i=0; i<m_iNumImgs; i++)
		{	float fTilt = this->GetDfMin(i);
			fDfMin = this->GetDfMin(i) * 0.1f;
			fDfMax = this->GetDfMax(i) * 0.1f;
			fprintf(pFile, pcFormat2, i+1, i+1, fTilt, fTilt,
			   fDfMin, fDfMax, this->GetAzimuth(i),
			   this->GetExtPhase(i));
		}
	}
	fclose(pFile);
}

void CCtfResults::Display(int iNthCtf, char* pcLog)
{
	char acBuf[128] = {'\0'};
	sprintf(acBuf, "%4d  %8.2f  %8.2f  %6.2f %6.2f %9.5f\n", 
	   iNthCtf+1, this->GetDfMin(iNthCtf), 
	   this->GetDfMax(iNthCtf), 
	   this->GetAzimuth(iNthCtf), 
	   this->GetExtPhase(iNthCtf),
	   this->GetScore(iNthCtf));
	strcat(pcLog, acBuf);
}

void CCtfResults::DisplayAll(void)
{
	int iSize = (m_iNumImgs + 16) * 128;
	char* pcLog = new char[iSize];
	memset(pcLog, 0, sizeof(char) * iSize);
	//-----------------
	sprintf(pcLog, "GPU %d: Estimated CTFs\n", m_iNthGpu);
        strcat(pcLog, "Index  dfmin     dfmax    azimuth  phase   score\n");
	//-----------------
	for(int i=0; i<m_iNumImgs; i++) 
	{	this->Display(i, pcLog);
	}
	//-----------------
	printf("%s\n", pcLog);
	if(pcLog != 0L) delete[] pcLog;
}
