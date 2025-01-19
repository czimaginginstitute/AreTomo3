#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include "../AreTomo/MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo;
using namespace McAreTomo::DataUtil;

static float s_fD2R = 0.01745329f;

CCtfResults** CCtfResults::m_ppInstances = 0L;
int CCtfResults::m_iNumGpus = 0;

void CCtfResults::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	//-----------------
	CCtfResults::DeleteInstances();
	m_ppInstances = new CCtfResults*[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	CCtfResults* pCtfResults = new CCtfResults;
		pCtfResults->m_iNthGpu = i;
		m_ppInstances[i] = pCtfResults;
	}
	m_iNumGpus = iNumGpus;
}

void CCtfResults::DeleteInstances(void)
{
	if(m_ppInstances == 0L) return;
	//-----------------
	for(int i=0; i<m_iNumGpus; i++)
	{	if(m_ppInstances[i] == 0L) continue;
		else delete m_ppInstances[i];
	}
	delete[] m_ppInstances;
	m_ppInstances = 0L;
	m_iNumGpus = 0;
}

CCtfResults* CCtfResults::GetInstance(int iNthGpu)
{
	return m_ppInstances[iNthGpu];
}

void CCtfResults::Replace(int iNthGpu, CCtfResults* pInstance)
{
	if(m_ppInstances[iNthGpu] != 0L) delete m_ppInstances[iNthGpu];
	m_ppInstances[iNthGpu] = pInstance;	
}

CCtfResults::CCtfResults(void)
{
	m_iDfHand = 1;
	m_iNumImgs = 0;
	m_ppCtfParams = 0L;
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
	//-----------------
	m_iNumImgs = iNumImgs;
	m_aiSpectSize[0] = piSpectSize[0];
	m_aiSpectSize[1] = piSpectSize[1];
	//-----------------
	m_ppfSpects = new float*[m_iNumImgs];
	int iSpectSize = m_aiSpectSize[0] * m_aiSpectSize[1];
	//-----------------
	m_ppCtfParams = new CCtfParam*[m_iNumImgs];
	for(int i=0; i<m_iNumImgs; i++)
	{	m_ppCtfParams[i] = new CCtfParam;
		m_ppCtfParams[i]->SetParam(pCtfParam);
		//----------------
		m_ppfSpects[i] = new float[iSpectSize];
	}
}

void CCtfResults::Clean(void)
{
	if(m_iNumImgs == 0) return;
	//-----------------
	for(int i=0; i<m_iNumImgs; i++)
	{	if(m_ppfSpects[i] != 0L) delete[] m_ppfSpects[i];
		if(m_ppCtfParams[i] != 0L) delete m_ppCtfParams[i];
	}
	if(m_ppfSpects != 0L) delete[] m_ppfSpects;
	if(m_ppCtfParams != 0L) delete[] m_ppCtfParams;
	//-----------------
	m_iNumImgs = 0;
	m_ppCtfParams = 0L;
	m_ppfSpects = 0L;
}

bool CCtfResults::bHasCTF(void)
{
	if(m_iNumImgs == 0) return false;
	if(m_ppCtfParams == 0) return false;
	//-----------------
	for(int i=0; i<m_iNumImgs; i++)
	{	if(m_ppCtfParams[i]->m_fDefocusMax < 1.0f) return false;
	}
	return true;
}

CCtfResults* CCtfResults::GetCopy(void)
{
	CCtfResults* pCopy = new CCtfResults;
	pCopy->Setup(m_iNumImgs, m_aiSpectSize, m_ppCtfParams[0]);
	//-----------------
	for(int i=0; i<m_iNumImgs; i++)
	{	pCopy->SetCtfParam(i, m_ppCtfParams[i]);
		pCopy->SetSpect(i, m_ppfSpects[i]);
	}
	pCopy->m_iNthGpu = m_iNthGpu;
	pCopy->m_iDfHand = m_iDfHand;
	return pCopy;
}

void CCtfResults::SetTilt(int iImage, float fTilt)
{
	m_ppCtfParams[iImage]->m_fTilt = fTilt;
}

void CCtfResults::SetDfMin(int iImage, float fDfMin)
{
	m_ppCtfParams[iImage]->m_fDefocusMin = 
	   fDfMin / m_ppCtfParams[iImage]->m_fPixelSize;
}

void CCtfResults::SetDfMax(int iImage, float fDfMax)
{
	m_ppCtfParams[iImage]->m_fDefocusMax = 
	   fDfMax / m_ppCtfParams[iImage]->m_fPixelSize;
}

void CCtfResults::SetAzimuth(int iImage, float fAzimuth)
{
	m_ppCtfParams[iImage]->m_fAstAzimuth = fAzimuth * s_fD2R;
}

void CCtfResults::SetExtPhase(int iImage, float fExtPhase)
{
	m_ppCtfParams[iImage]->m_fExtPhase = fExtPhase * s_fD2R;
}

void CCtfResults::SetScore(int iImage, float fScore)
{
	m_ppCtfParams[iImage]->m_fScore = fScore;
}

void CCtfResults::SetCtfRes(int iImage, float fRes)
{
	m_ppCtfParams[iImage]->m_fCtfRes = fRes;
}

void CCtfResults::SetCtfParam(int iImage, CCtfParam* pCtfParam)
{
	m_ppCtfParams[iImage]->SetParam(pCtfParam);
}

void CCtfResults::SetSpect(int iImage, float* pfSpect)
{
	float* pfSrcSpect = m_ppfSpects[iImage];
	int iPixels = m_aiSpectSize[0] * m_aiSpectSize[1];
	if(pfSrcSpect == 0L) pfSrcSpect = new float[iPixels];
	memcpy(pfSrcSpect, pfSpect, iPixels * sizeof(float));
	m_ppfSpects[iImage] = pfSrcSpect;
}

float CCtfResults::GetTilt(int iImage)
{
	return m_ppCtfParams[iImage]->m_fTilt;
}

float CCtfResults::GetDfMin(int iImage)
{
	return m_ppCtfParams[iImage]->m_fDefocusMin *
	   m_ppCtfParams[iImage]->m_fPixelSize;
}

float CCtfResults::GetDfMax(int iImage)
{
	return m_ppCtfParams[iImage]->m_fDefocusMax *
	   m_ppCtfParams[iImage]->m_fPixelSize;
}

float CCtfResults::GetDfMean(int iImage)
{
	float fDfMean = (GetDfMin(iImage) + GetDfMax(iImage)) * 0.5f;
	return fDfMean;
}

float CCtfResults::GetAstMag(int iImage)
{
	float fDfMin = this->GetDfMin(iImage);
	float fDfMax = this->GetDfMax(iImage);
	float fAst = (float)((fDfMax - fDfMin) / (fDfMax + fDfMin + 1e-30));
	return fAst;
}

float CCtfResults::GetAzimuth(int iImage)
{
	return m_ppCtfParams[iImage]->m_fAstAzimuth / s_fD2R;
}

float CCtfResults::GetExtPhase(int iImage)
{
	return m_ppCtfParams[iImage]->m_fExtPhase / s_fD2R;
}

float CCtfResults::GetPixSize(int iImage)
{
	return m_ppCtfParams[iImage]->m_fPixelSize;
}

float CCtfResults::GetScore(int iImage)
{
	return m_ppCtfParams[iImage]->m_fScore;
}

float CCtfResults::GetTsScore(void)
{
	float fSum = 0.0f;
	for(int i=0; i<m_iNumImgs; i++)
	{	fSum += this->GetScore(i);
	}
	float fScore = fSum / (m_iNumImgs + 0.0001f);
	return fScore;
}

float CCtfResults::GetLowTiltScore(float fLowTilt)
{	
	float fSum = 0.0f;
	int iCount = 0;
	for(int i=0; i<m_iNumImgs; i++)
	{	if(fabs(this->GetTilt(i)) > fLowTilt) continue;
		fSum += this->GetScore(i);
		iCount += 1;
	}
	if(iCount == 0) return 0.0f;
	float fScore = fSum / iCount;
	return fScore;
}

float CCtfResults::GetCtfRes(int iImage)
{
	return m_ppCtfParams[iImage]->m_fCtfRes;
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
	sprintf(acBuf, "%4d  %7.2f  %8.2f  %8.2f  %6.2f "
	   "%9.4f %6.2f %9.5f %3d\n", 
	   iNthCtf+1, this->GetTilt(iNthCtf),
	   this->GetDfMin(iNthCtf),   this->GetDfMax(iNthCtf), 
	   this->GetAzimuth(iNthCtf), this->GetExtPhase(iNthCtf),
	   this->GetCtfRes(iNthCtf),  this->GetScore(iNthCtf), 
	   m_iDfHand);
	strcat(pcLog, acBuf);
}

void CCtfResults::DisplayAll(void)
{
	int iSize = (m_iNumImgs + 16) * 128;
	char* pcLog = new char[iSize];
	memset(pcLog, 0, sizeof(char) * iSize);
	//-----------------
	sprintf(pcLog, "GPU %d: Estimated CTFs\n", m_iNthGpu);
	strcat(pcLog, "#index  tilt  dfmin     dfmax    azimuth  phase  "
	   "res   score   DfHand\n");
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
	{	float fDif = fabsf(fTilt - m_ppCtfParams[i]->m_fTilt);
		if(fDif < fMin)
		{	fMin = fDif;
			iMin = i;
		}
	}
	return iMin;
}

CCtfParam* CCtfResults::GetCtfParam(int iImage)
{
	return m_ppCtfParams[iImage];
}

CCtfParam* CCtfResults::GetCtfParamFromTilt(float fTilt)
{
	int iImage = this->GetImgIdxFromTilt(fTilt);
	return m_ppCtfParams[iImage];
}

//--------------------------------------------------------------------
// 1. Dark CTFs are those estimated on dark images.
//--------------------------------------------------------------------
void CCtfResults::RemoveDarkCTFs(void)
{
	if(!bHasCTF()) return;
	//-----------------
	MAM::CDarkFrames* pDarkFrames = 
	   MAM::CDarkFrames::GetInstance(m_iNthGpu);
	if(pDarkFrames->m_iNumDarks <= 0) return;
	//-----------------
	for(int i=pDarkFrames->m_iNumDarks-1; i>=0; i--)
	{	int iDark = pDarkFrames->GetDarkIdx(i);
		mRemoveEntry(i);
	}
}

void CCtfResults::mRemoveEntry(int iEntry)
{
	if(m_ppCtfParams[iEntry] != 0L) delete m_ppCtfParams[iEntry];
	for(int i=iEntry+1; i<m_iNumImgs; i++)
	{	int j = i - 1;
		m_ppCtfParams[j] = m_ppCtfParams[i];
	}
	m_iNumImgs -= 1;
}
