#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo;
using namespace McAreTomo::DataUtil;

static float s_fD2R = 0.01745329f;

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
	m_pCtfParams = 0L;
	m_pfScores = 0L;
	m_pfTilts = 0L;
	m_ppfSpects = 0L;
}

CCtfResults::~CCtfResults(void)
{
	this->Clean();
}

void CCtfResults::Setup
(	int iNumImgs, 
	int* piSpectSize, 
	CCtfParam* pCtfParam
)
{	this->Clean();
	m_iNumImgs = iNumImgs;
	m_aiSpectSize[0] = piSpectSize[0];
	m_aiSpectSize[1] = piSpectSize[1];
	//-----------------
	m_pfScores = new float[m_iNumImgs * 2];
	m_pfTilts = &m_pfScores[m_iNumImgs];
	//-----------------
	int iBytes = sizeof(float) * m_iNumImgs * 2;
	memset(m_pfScores, 0, iBytes);
	//-----------------
	m_ppfSpects = new float*[m_iNumImgs];
	memset(m_ppfSpects, 0, sizeof(float*) * m_iNumImgs);
	//-----------------
	m_pCtfParams = new CCtfParam[m_iNumImgs];
	for(int i=0; i<m_iNumImgs; i++)
	{	m_pCtfParams[i].SetParam(pCtfParam);
	}
}

void CCtfResults::Clean(void)
{
	if(m_iNumImgs == 0) return;
	//-------------------------
	if(m_pCtfParams != 0L) delete[] m_pCtfParams;
	if(m_ppfSpects != 0L) delete[] m_ppfSpects;
	if(m_pfScores != 0L) delete[] m_pfScores;
	mInit();
}

void CCtfResults::SetTilt(int iImage, float fTilt)
{
	m_pfTilts[iImage] = fTilt;
}

void CCtfResults::SetDfMin(int iImage, float fDfMin)
{
	m_pCtfParams[iImage].m_fDefocusMin = 
	   fDfMin / m_pCtfParams[iImage].m_fPixelSize;
}

void CCtfResults::SetDfMax(int iImage, float fDfMax)
{
	m_pCtfParams[iImage].m_fDefocusMax = 
	   fDfMax / m_pCtfParams[iImage].m_fPixelSize;
}

void CCtfResults::SetAzimuth(int iImage, float fAzimuth)
{
	m_pCtfParams[iImage].m_fAstAzimuth = fAzimuth * s_fD2R;
}

void CCtfResults::SetExtPhase(int iImage, float fExtPhase)
{
	m_pCtfParams[iImage].m_fExtPhase = fExtPhase * s_fD2R;
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
	return m_pCtfParams[iImage].m_fDefocusMin *
	   m_pCtfParams[iImage].m_fPixelSize;
}

float CCtfResults::GetDfMax(int iImage)
{
	return m_pCtfParams[iImage].m_fDefocusMax *
	   m_pCtfParams[iImage].m_fPixelSize;
}

float CCtfResults::GetAzimuth(int iImage)
{
	return m_pCtfParams[iImage].m_fAstAzimuth / s_fD2R;
}

float CCtfResults::GetExtPhase(int iImage)
{
	return m_pCtfParams[iImage].m_fExtPhase / s_fD2R;
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

int CCtfResults::GetImgIdxFromTilt(float fTilt)
{
	int iMin = -1;
	float fMin = (float)1e30;
	for(int i=0; i<m_iNumImgs; i++)
	{	float fDif = fabsf(fTilt - m_pfTilts[i]);
		if(fDif < fMin)
		{	fMin = fDif;
			iMin = i;
		}
	}
	return iMin;
}

CCtfParam* CCtfResults::GetCtfParam(int iImage)
{
	return &m_pCtfParams[iImage];
}

CCtfParam* CCtfResults::GetCtfParamFromTilt(float fTilt)
{
	int iImage = this->GetImgIdxFromTilt(fTilt);
	return &m_pCtfParams[iImage];
}
