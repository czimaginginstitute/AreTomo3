#include "CAreTomoInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>

using namespace McAreTomo::AreTomo;

CTsMetrics* CTsMetrics::m_pInstance = 0L;

CTsMetrics* CTsMetrics::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CTsMetrics;
	return m_pInstance;
}

void CTsMetrics::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CTsMetrics::CTsMetrics(void)
{
	m_pFile = 0L;
	m_bFirstTime = true;
	pthread_mutex_init(&m_aMutex, NULL);
}

CTsMetrics::~CTsMetrics(void)
{
	if(m_pFile != 0L) fclose(m_pFile);
	pthread_mutex_destroy(&m_aMutex);
}

void CTsMetrics::Save(int iNthGpu)
{
	pthread_mutex_lock(&m_aMutex);
	//-----------------
	if(m_pFile == 0L && m_bFirstTime)
	{	mOpenFile();
		m_bFirstTime = false;
	}
	//-----------------
	if(m_pFile == 0L)
	{	pthread_mutex_unlock(&m_aMutex);
		return;
	}
	//-----------------
	m_iNthGpu = iNthGpu;
	mGetMrcName();
	mGetPixelSize();
	mGetThickness();
	mGetGlobalShift();
	mGetBadPatches();
	mGetTiltAxis();
	mGetCTF();
	//-----------------
	mSave();
	//-----------------
	pthread_mutex_unlock(&m_aMutex);
}

void CTsMetrics::mGetMrcName(void)
{
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	strcpy(m_acMrcName, pTsPackage->m_acMrcMain);
	strcat(m_acMrcName, ".mrc");
}

void CTsMetrics::mGetPixelSize(void)
{
        MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPackage->GetSeries(0);
	m_fPixSize = pTiltSeries->m_fPixSize;
}

void CTsMetrics::mGetThickness(void)
{
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	m_iThickness = pAlnParam->m_iThickness;
}

void CTsMetrics::mGetGlobalShift(void)
{
	MAM::CAlignParam* pAlnParam; 
	pAlnParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	float afS[] = {0.0f};
	m_fGlobalShift = 0.0f;
	//-----------------
	for(int i=0; i<pAlnParam->m_iNumFrames; i++)
	{	pAlnParam->GetShift(i, afS);
		float fS = (float)sqrt(afS[0] * afS[0] + afS[1] * afS[1]);
		if(fS < m_fGlobalShift) continue;
		else m_fGlobalShift = fS;
	}
}

void CTsMetrics::mGetBadPatches(void)
{
	MAM::CLocalAlignParam* pLocalParam;
	pLocalParam = MAM::CLocalAlignParam::GetInstance(m_iNthGpu);
	m_fBadPatchLow = pLocalParam->GetBadPercentage(30.9);
	m_fBadPatchAll = pLocalParam->GetBadPercentage(90.0);
}

void CTsMetrics::mGetTiltAxis(void)
{
	MAM::CAlignParam* pAlnParam;
        pAlnParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	m_fTiltAxis = pAlnParam->GetTiltAxis(0);
}

void CTsMetrics::mGetCTF(void)
{
	MD::CCtfResults* pCtfRes = MD::CCtfResults::GetInstance(m_iNthGpu);
	int iZeroTilt = pCtfRes->GetImgIdxFromTilt(0.0f);
	m_fCtfScore = pCtfRes->GetScore(iZeroTilt);
	m_fCtfRes = pCtfRes->GetCtfRes(iZeroTilt);
	m_iDfHand = pCtfRes->m_iDfHand;
	m_fAlphaOffset = pCtfRes->m_fAlphaOffset;
	m_fBetaOffset = pCtfRes->m_fBetaOffset;
}

void CTsMetrics::mOpenFile(void)
{
	char acFile[256] = {'\0'};
	CInput* pInput = CInput::GetInstance();
	strcpy(acFile, pInput->m_acOutDir);
	strcat(acFile, "TiltSeries_Metrics.csv");
	//-----------------
	if(pInput->m_iResume == 0) m_pFile = fopen(acFile, "w");
	else m_pFile = fopen(acFile, "wa");
	//-----------------
	if(m_pFile == 0L)
	{	printf("Warning: metrics file cannot be created, "
		   "proceed without saving metrics.\n\n");
		return;
	}
	//-----------------
	fprintf(m_pFile, "Tilt_Series,Thickness(Pix),Tilt_Axis,"
	   "Global_Shift(Pix),Bad_Patch_Low,Bad_Patch_All,"
   	   "CTF_Res(A),CTF_Score,DF_Hand,Pix_Size(A),"
	   "Cs(nm),Kv,Alpha0,Beta0\n");   
}

void CTsMetrics::mSave(void)
{
	CInput* pInput = CInput::GetInstance();
	fprintf(m_pFile, "%s, %6d, %6.2f, %8.2f, %5.2f, %5.2f, "
	   "%5.2f, %7.4f, %2d, %5.2f, %.1f, %3d, %.1f, %.1f\n", 
	   m_acMrcName, m_iThickness, m_fTiltAxis, m_fGlobalShift, 
	   m_fBadPatchLow, m_fBadPatchAll, 
	   m_fCtfRes, m_fCtfScore, m_iDfHand,
	   m_fPixSize, pInput->m_fCs, pInput->m_iKv, 
	   m_fAlphaOffset, m_fBetaOffset);
      	fflush(m_pFile);
}	
