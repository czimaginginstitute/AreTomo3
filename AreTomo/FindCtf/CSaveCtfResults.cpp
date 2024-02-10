#include "CFindCtfInc.h"
#include <stdio.h>
#include <memory.h>

using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.017453f;

CSaveCtfResults::CSaveCtfResults(void)
{
}

CSaveCtfResults::~CSaveCtfResults(void)
{
}

void CSaveCtfResults::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	CInput* pInput = CInput::GetInstance();
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	//-----------------
	strcpy(m_acOutFolder, pInput->m_acOutDir);
	strcpy(m_acInMrcFile, pTsPackage->m_acMrcMain);	
	//-----------------
	char acCtfFile[256] = {'\0'};
        strcpy(acCtfFile, m_acOutFolder);
        strcat(acCtfFile, m_acInMrcFile);
        strcat(acCtfFile, "_CTF");
	//-----------------
	mSaveImages(acCtfFile);
	mSaveFittings(acCtfFile);
}

void CSaveCtfResults::mSaveImages(const char* pcCtfFile)
{
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	Mrc::CSaveMrc aSaveMrc;
	bool bClean = true;
	//-----------------
	char acCtfImgFile[256] = {'\0'};
	strcpy(acCtfImgFile, pcCtfFile);
	strcat(acCtfImgFile, ".mrc");
	//-----------------
	aSaveMrc.OpenFile(acCtfImgFile);
	aSaveMrc.SetMode(Mrc::eMrcFloat);
	aSaveMrc.SetImgSize(pCtfResults->m_aiSpectSize,
	   pCtfResults->m_iNumImgs, 1, 1.0f);
	//-----------------
	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
	{	float* pfSpect = pCtfResults->GetSpect(i, bClean);
		aSaveMrc.DoIt(i, pfSpect);
		if(pfSpect != 0L) delete[] pfSpect;
	}
	aSaveMrc.CloseFile();
}

void CSaveCtfResults::mSaveFittings(const char* pcCtfFile)
{
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	char acCtfTxtFile[256] = {'\0'};
	strcpy(acCtfTxtFile, pcCtfFile);
        strcat(acCtfTxtFile, ".txt");
	//---------------------
	FILE* pFile = fopen(acCtfTxtFile, "w");
	if(pFile == 0L) return;
	//---------------------
	fprintf(pFile, "# Columns: #1 micrograph number; "
	   "#2 - defocus 1 [A]; #3 - defocus 2; "
	   "#4 - azimuth of astigmatism; "
	   "#5 - additional phase shift [radian]; "
	   "#6 - cross correlation; "
	   "#7 - spacing (in Angstroms) up to which CTF rings were "
	   "fit successfully\n");
	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
	{	fprintf(pFile, "%4d   %8.2f  %8.2f  %8.2f  %8.2f  "
		   "%8.4f  %8.4f\n", i+1,
		   pCtfResults->GetDfMax(i),
		   pCtfResults->GetDfMin(i),
		   pCtfResults->GetAzimuth(i),
		   pCtfResults->GetExtPhase(i) * s_fD2R,
		   pCtfResults->GetScore(i),
		   10.0f); // 10.0f temporary
	}
	fclose(pFile);
	//-----------------
	strcpy(acCtfTxtFile, pcCtfFile);
	strcat(acCtfTxtFile, "_Imod.txt");
	pCtfResults->SaveImod(acCtfTxtFile);
}