#include "CAreTomoInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>

using namespace McAreTomo::AreTomo;

CTsMetrics* CTsMetrics::m_pInstances = 0L;
int CTsMetrics::m_iNumGpus = 0;
FILE* CTsMetrics::m_pFile = 0L;
pthread_mutex_t* CTsMetrics::m_pMutex = 0L;

void CTsMetrics::CreateInstances(void)
{
	CTsMetrics::DeleteInstances();
	//-----------------
	CInput* pInput = CInput::GetInstance();
	m_iNumGpus = pInput->m_iNumGpus;
	if(m_iNumGpus == 0) return;
	//-----------------
	m_pInstances = new CTsMetrics[m_iNumGpus];
	for(int i=0; i<m_iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	//-----------------
	m_pMutex = new pthread_mutex_t;
	pthread_mutex_init(m_pMutex, 0L);
}

CTsMetrics* CTsMetrics::GetInstance(int iNthGpu)
{
	if(iNthGpu >= m_iNumGpus) return 0L;
	return &m_pInstances[iNthGpu];
}

void CTsMetrics::DeleteInstances(void)
{
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
	//-----------------
	if(m_pFile != 0L) fclose(m_pFile);
	m_pFile = 0L;
	//-----------------
	if(m_pMutex != 0L)
	{	pthread_mutex_destroy(m_pMutex);
		delete m_pMutex;
		m_pMutex = 0L;
	}
}

CTsMetrics::CTsMetrics(void)
{
}

CTsMetrics::~CTsMetrics(void)
{
}

void CTsMetrics::BuildMetrics(void)
{
	mGetMrcName();
        mGetPixelSize();
        mGetThickness();
        mGetGlobalShift();
        mGetBadPatches();
        mGetTiltAxis();
        mGetCTF();
}

void CTsMetrics::Save(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iCmd == 2) return;
	if(pInput->m_iCmd == 3) return;
	if(pInput->m_iCmd == 4) return;
	//-----------------
	pthread_mutex_lock(m_pMutex);
	if(m_pFile != 0L) 
	{	mSave();
	}
	else if(m_pFile == 0L)
	{	CTsMetrics::mOpenFile();
		if(m_pFile != 0L) mSave();
	}
	pthread_mutex_unlock(m_pMutex);
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
	float afS[2] = {0.0f};
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
	m_fDfMean = pCtfRes->GetDfMean(iZeroTilt);
	m_fExtPhase = pCtfRes->GetExtPhase(iZeroTilt);
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
	bool bFirst = false;
	if(pInput->m_iResume == 0) 
	{	m_pFile = fopen(acFile, "w");
		bFirst = true;
	}
	else
        {       m_pFile = fopen(acFile, "r");
                if(m_pFile == 0L)
                {       m_pFile = fopen(acFile, "w");
                        bFirst = true;
                }
                else
                {       fclose(m_pFile);
                        m_pFile = fopen(acFile, "a");
                        if(m_pFile == 0L)
                        {       m_pFile = fopen(acFile, "w");
                                bFirst = true;
                        }
                        else bFirst = false;
		}
        }
	//-----------------
	if(m_pFile == 0L)
	{	printf("Warning: metrics file cannot be created, "
		   "proceed without saving metrics.\n\n");
		return;
	}
	//-----------------
	if(bFirst)
	{	fprintf(m_pFile, "Tilt_Series,Thickness(Pix),Tilt_Axis,"
	   	   "Global_Shift(Pix),Bad_Patch_Low,Bad_Patch_All,"
   	   	   "Defocus(A),ExtPhase(Deg),CTF_Res(A),CTF_Score,"
		   "DF_Hand,Pix_Size(A),"
	   	   "Cs(nm),Kv,Alpha0,Beta0\n");
	}
}

void CTsMetrics::mSave(void)
{
	CInput* pInput = CInput::GetInstance();
	fprintf(m_pFile, "%s, %6d, %6.2f, %8.2f, %5.2f," 
	   "%5.2f, %8.1f, %5.1f, %5.2f, %7.4f," 
	   "%2d, %5.2f, %.1f, %3d, %.1f, %.1f\n", 
	   m_acMrcName, m_iThickness, m_fTiltAxis, m_fGlobalShift, 
	   m_fBadPatchLow, m_fBadPatchAll, m_fDfMean, m_fExtPhase, 
	   m_fCtfRes, m_fCtfScore, m_iDfHand, m_fPixSize, 
	   pInput->m_fCs, pInput->m_iKv, m_fAlphaOffset, m_fBetaOffset);
      	fflush(m_pFile);
}	
