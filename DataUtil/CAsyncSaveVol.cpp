#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>

using namespace McAreTomo::DataUtil;

CAsyncSaveVol* CAsyncSaveVol::m_pInstances = 0L;
int CAsyncSaveVol::m_iNumGpus = 0;

void CAsyncSaveVol::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	//-----------------
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CAsyncSaveVol[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

CAsyncSaveVol* CAsyncSaveVol::GetInstance(int iNthGpu)
{
	if(m_pInstances == 0L) return 0L;
	return &m_pInstances[iNthGpu];
}

void CAsyncSaveVol::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
}

//------------------------------------------------------------------------------
// m_aMutex: It is initialized in Util_Thread. Do not init here
//------------------------------------------------------------------------------
CAsyncSaveVol::CAsyncSaveVol(void)
{
	m_iNthGpu = 0;
}

//------------------------------------------------------------------------------
// m_aMutex: It is destroyed in Util_Thread. Do not destroy here.
//------------------------------------------------------------------------------
CAsyncSaveVol::~CAsyncSaveVol(void)
{
}

bool CAsyncSaveVol::DoIt
(	CTiltSeries* pVolSeries, 
	int iNthVol,
	bool bAsync,
	bool bClean
)
{	m_pVolSeries = pVolSeries;
	m_iNthVol = iNthVol;
	m_bClean = bClean;
	//-----------------
	if(bAsync) this->Start();
	else mSaveVol();
	return true;
}

void CAsyncSaveVol::ThreadMain(void)
{
	mSaveVol();
}

void CAsyncSaveVol::mSaveVol(void)
{
	char acExt[32] = {'\0'}, acMrcFile[256] = {'\0'};
	if(m_iNthVol == 0) strcpy(acExt, "_Vol.mrc");
	else if(m_iNthVol == 1) strcpy(acExt, "_EVN_Vol.mrc");
	else if(m_iNthVol == 2) strcpy(acExt, "_ODD_Vol.mrc");
	else if(m_iNthVol == 3) strcpy(acExt, "_2ND_Vol.mrc");
	else if(m_iNthVol == 4) strcpy(acExt, "_3RD_Vol.mrc");
	mGenFullPath(acExt, acMrcFile);
	//-----------------
	Mrc::CSaveMrc saveMrc;
	saveMrc.OpenFile(acMrcFile);
	saveMrc.SetMode(Mrc::eMrcFloat);
	saveMrc.SetExtHeader(0, 32, 0);
	saveMrc.SetImgSize(m_pVolSeries->m_aiStkSize,
	   m_pVolSeries->m_aiStkSize[2], 1,
	   m_pVolSeries->m_fPixSize);
	saveMrc.m_pSaveMain->DoIt();
        //-----------------
	float** ppfImages = m_pVolSeries->GetImages();
	for(int i=0; i<m_pVolSeries->m_aiStkSize[2]; i++)
	{	float fTilt = m_pVolSeries->m_pfTilts[i];
		saveMrc.m_pSaveExt->SetTilt(i, &fTilt, 1);
		saveMrc.m_pSaveExt->DoIt();
		saveMrc.m_pSaveImg->DoIt(i, ppfImages[i]);
	}
	saveMrc.CloseFile();
	//-----------------
	if(m_bClean && m_pVolSeries != 0L) delete m_pVolSeries;
	m_pVolSeries = 0L;
	//-----------------
	printf("GPU %d: MRC file saved: %s\n\n", m_iNthGpu, acMrcFile);
}

void CAsyncSaveVol::mGenFullPath(const char* pcSuffix, char* pcFullPath)
{          
        CInput* pInput = CInput::GetInstance();
        strcpy(pcFullPath, pInput->m_acOutDir);
	//-----------------
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
        strcat(pcFullPath, pTsPackage->m_acMrcMain);
	//-----------------
        if(pcSuffix != 0L && strlen(pcSuffix) > 0)
        {       strcat(pcFullPath, pcSuffix);
        }
}       
