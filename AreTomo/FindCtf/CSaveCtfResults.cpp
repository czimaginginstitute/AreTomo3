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

void CSaveCtfResults::GenFileName(int iNthGpu, char* pcCtfFile)
{
	CInput* pInput = CInput::GetInstance();
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(iNthGpu);
	strcpy(pcCtfFile, pInput->m_acOutDir);
	strcat(pcCtfFile, pTsPackage->m_acMrcMain);
	strcat(pcCtfFile, "_CTF");
}
		
void CSaveCtfResults::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------
	char acCtfFile[256] = {'\0'};
	CSaveCtfResults::GenFileName(m_iNthGpu, acCtfFile);
	//-----------------
	mSaveImages(acCtfFile);
	mSaveFittings(acCtfFile);
	mSaveImod(acCtfFile);
}

void CSaveCtfResults::DoFittings(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	char acCtfFile[256] = {'\0'};
	CSaveCtfResults::GenFileName(m_iNthGpu, acCtfFile);
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
	{	float* pfSpect = pCtfResults->GetSpect(i, !bClean);
		aSaveMrc.DoIt(i, pfSpect);
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
	   "fit successfully; "
	   "#8 - dfHand\n");
	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
	{	fprintf(pFile, "%4d   %8.2f  %8.2f  %8.2f  %9.4f  "
		   "%8.4f  %8.4f  %3d\n", i+1,
		   pCtfResults->GetDfMax(i),
		   pCtfResults->GetDfMin(i),
		   pCtfResults->GetAzimuth(i),
		   pCtfResults->GetExtPhase(i) * s_fD2R,
		   pCtfResults->GetScore(i),
		   pCtfResults->GetCtfRes(i),
		   pCtfResults->m_iDfHand);
	}
	fclose(pFile);
}

void CSaveCtfResults::mSaveImod(const char* pcCtfFile)
{
	MD::CCtfResults* pCtfResults = 
	   MD::CCtfResults::GetInstance(m_iNthGpu);
	char acCtfTxtFile[256] = {'\0'};
	strcpy(acCtfTxtFile, pcCtfFile);
	strcat(acCtfTxtFile, "_Imod.txt");
	//-----------------
	FILE* pFile = fopen(acCtfTxtFile, "w");
	if(pFile == 0L) return;
	//-----------------
	float fExtPhase = pCtfResults->GetExtPhase(0);
	if(fExtPhase == 0) fprintf(pFile, "1  0  0.0  0.0  0.0  3\n");
	else fprintf(pFile, "5  0  0.0  0.0  0.0  3\n");
	//-----------------
	const char *pcFormat1 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f\n";
	const char *pcFormat2 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f  %8.2f\n";
	float fDfMin, fDfMax;
	if(fExtPhase == 0)
	{	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
		{	float fTilt = pCtfResults->GetTilt(i);
			fDfMin = pCtfResults->GetDfMin(i) * 0.1f;
			fDfMax = pCtfResults->GetDfMax(i) * 0.1f;
			fprintf(pFile, pcFormat1, i+1, i+1, fTilt, fTilt,
			   fDfMin, fDfMax, pCtfResults->GetAzimuth(i));
		}
	}
	else
	{	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
		{	float fTilt = pCtfResults->GetDfMin(i);
			fDfMin = pCtfResults->GetDfMin(i) * 0.1f;
			fDfMax = pCtfResults->GetDfMax(i) * 0.1f;
			fprintf(pFile, pcFormat2, i+1, i+1, fTilt, fTilt,
			   fDfMin, fDfMax, pCtfResults->GetAzimuth(i),
			   pCtfResults->GetExtPhase(i));
		}
	}
	fclose(pFile);
}
