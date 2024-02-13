#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo::DataUtil;

CTsPackage* CTsPackage::m_pInstances = 0L;
int CTsPackage::m_iNumGpus = 0;

void CTsPackage::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	//-----------------
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CTsPackage[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CTsPackage::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CTsPackage* CTsPackage::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CTsPackage::CTsPackage(void)
{
	m_pcMdocFile = 0L;
	m_ppTsStacks = new CTiltSeries*[CAlnSums::m_iNumSums];
	m_ppVolStacks = new CTiltSeries*[CAlnSums::m_iNumSums];
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	m_ppTsStacks[i] = new CTiltSeries;
		m_ppVolStacks[i] = 0L;
	}
}

CTsPackage::~CTsPackage(void)
{
	if(m_pcMdocFile != 0L) delete[] m_pcMdocFile;
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	if(m_ppTsStacks[i] != 0L) delete m_ppTsStacks[i];
		if(m_ppVolStacks[i] != 0L) delete m_ppVolStacks[i];
	}
	delete[] m_ppTsStacks;
	delete[] m_ppVolStacks;
}

void CTsPackage::SetMdoc(char* pcMdocFile)
{
	if(m_pcMdocFile != 0L) delete[] m_pcMdocFile;
	m_pcMdocFile = 0L;
	memset(m_acMrcMain, 0, sizeof(m_acMrcMain));
	//-----------------
	if(pcMdocFile == 0L || strlen(pcMdocFile) == 0) return;
	m_pcMdocFile = new char[256];
	strcpy(m_pcMdocFile, pcMdocFile);
	//-----------------
	MU::CFileName fileName(m_pcMdocFile);
	fileName.GetName(m_acMrcMain);
}

void CTsPackage::CreateTiltSeries(void)
{
	CReadMdoc* pReadMdoc = CReadMdoc::GetInstance(m_iNthGpu);
	CMcPackage* pMcPackage = CMcPackage::GetInstance(m_iNthGpu);
	CMrcStack* pAlnSums = pMcPackage->m_pAlnSums;
	int iNumTilts = pReadMdoc->m_iNumTilts;
	//-----------------
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->Create(pAlnSums->m_aiStkSize, iNumTilts);
		pSeries->m_fPixSize = pAlnSums->m_fPixSize;
	}
}

void CTsPackage::SetTiltAngle(int iTilt, float fTiltAngle)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->m_pfTilts[iTilt] = fTiltAngle;
	}
}

void CTsPackage::SetAcqIdx(int iTilt, int iAcqIdx)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->m_piAcqIndices[iTilt] = iAcqIdx;
	}
}

void CTsPackage::SetSecIdx(int iTilt, int iSecIdx)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->m_piSecIndices[iTilt] = iSecIdx;
	}
}

void CTsPackage::SetSums(int iTilt, CAlnSums* pAlnSums)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	void* pvImg = pAlnSums->GetFrame(i);
		CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->SetImage(iTilt, pvImg);
	}
}

void CTsPackage::SetImgDose(float fImgDose)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		if(i == 0) pSeries->m_fImgDose = fImgDose;
		else pSeries->m_fImgDose = fImgDose * 0.5f;
	}
}

CTiltSeries* CTsPackage::GetSeries(int iSeries)
{
	return m_ppTsStacks[iSeries];
}

void CTsPackage::SortTiltSeries(int iOrder)
{
	for(int i=0; i<3; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		if(iOrder == 0) pSeries->SortByTilt();
		else pSeries->SortByAcq();
	}
}

void CTsPackage::ResetSectionIndices(void)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->ResetSecIndices();
        }
}

void CTsPackage::SaveVol(CTiltSeries* pVol, int iVol)
{
	CInput* pInput = CInput::GetInstance();
	char acMrcFile[256] = {'\0'};
	strcpy(acMrcFile, pInput->m_acOutDir);
	strcat(acMrcFile, m_acMrcMain);
	//-----------------
	char acExt[32] = {'\0'};
	if(iVol == 0) strcpy(acExt, "_Vol.mrc");
	else if(iVol == 1) strcpy(acExt, "_EVN_Vol.mrc");
	else if(iVol == 2) strcpy(acExt, "_ODD_Vol.mrc");
	//-----------------
	mSaveMrc(acMrcFile, acExt, pVol);	
}

void CTsPackage::SaveTiltSeries(void)
{
	CInput* pInput = CInput::GetInstance();
	char acMrcFile[256] = {'\0'};
	strcpy(acMrcFile, pInput->m_acOutDir);
	strcat(acMrcFile, m_acMrcMain);
	//-----------------
	mSaveTiltFile(acMrcFile, m_ppTsStacks[0]);
	mSaveMrc(acMrcFile, ".mrc", m_ppTsStacks[0]);
	mSaveMrc(acMrcFile, "_EVN.mrc", m_ppTsStacks[1]);
	mSaveMrc(acMrcFile, "_ODD.mrc", m_ppTsStacks[2]);
}

void CTsPackage::mSaveMrc
(	const char* pcMainName, 
	const char* pcExt, 
	CTiltSeries* pTiltSeries
)
{	char acMrcFile[256] = {'0'};
	strcpy(acMrcFile, pcMainName);
	strcat(acMrcFile, pcExt);
	//-----------------
	Mrc::CSaveMrc saveMrc;
	saveMrc.OpenFile(acMrcFile);
	saveMrc.SetMode(Mrc::eMrcFloat);
	saveMrc.SetExtHeader(0, 32, 0);
	saveMrc.SetImgSize(pTiltSeries->m_aiStkSize,
	   pTiltSeries->m_aiStkSize[2], 1,
	   pTiltSeries->m_fPixSize);
	saveMrc.m_pSaveMain->DoIt();
	//-----------------
	float** ppfImages = pTiltSeries->GetImages();
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float fTilt = pTiltSeries->m_pfTilts[i];
		saveMrc.m_pSaveExt->SetTilt(i, &fTilt, 1);
		saveMrc.m_pSaveExt->DoIt();
		saveMrc.m_pSaveImg->DoIt(i, ppfImages[i]);
	}
	saveMrc.CloseFile();
}

void CTsPackage::mSaveTiltFile
(	const char* pcMainName,
	CTiltSeries* pTiltSeries
)
{	char acTiltFile[256] = {'0'};
	strcpy(acTiltFile, pcMainName);
	strcat(acTiltFile, "_TLT.txt");
	//-----------------
	FILE* pFile = fopen(acTiltFile, "wt");
	if(pFile == 0L)
	{	printf("GPU %d warning: Unable to save tilt angles\n\n",
		   m_iNthGpu, acTiltFile);
		return;
	}
	//-----------------
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float fTilt = pTiltSeries->m_pfTilts[i];
		int iAcqIdx = pTiltSeries->m_piAcqIndices[i];
		fprintf(pFile, "%8.2f  %4d\n", fTilt, iAcqIdx);
	}
	fclose(pFile);
}
