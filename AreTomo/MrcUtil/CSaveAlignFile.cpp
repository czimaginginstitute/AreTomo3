#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CSaveAlignFile::CSaveAlignFile(void)
{
	m_pFile = 0L;
}

CSaveAlignFile::~CSaveAlignFile(void)
{
	mCloseFile();
}

void CSaveAlignFile::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------	
	McAreTomo::CInput* pInput = McAreTomo::CInput::GetInstance();
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(iNthGpu);
	m_pAlignParam = CAlignParam::GetInstance(iNthGpu);
	m_pLocalParam = CLocalAlignParam::GetInstance(iNthGpu);
	//-----------------
	char acAlnFile[256] = {'\0'};
	strcpy(acAlnFile, pInput->m_acOutDir);
	strcat(acAlnFile, pPackage->m_acMrcMain);
	strcat(acAlnFile, ".aln");
	//-----------------
	m_pFile = fopen(acAlnFile, "wt");
	if(m_pFile == 0L)
	{	printf("Unable to open %s.\n", acAlnFile);
		printf("Alignment data will not be saved\n\n");
		return;
	}
	//-----------------
	mSaveHeader();
	mSaveGlobal();
	mSaveLocal();
	//-----------
	mCloseFile();
}

void CSaveAlignFile::mSaveHeader(void)
{
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance(m_iNthGpu);
	m_iNumTilts = m_pAlignParam->m_iNumFrames;
	m_iNumPatches = m_pLocalParam->m_iNumPatches;
	//-----------------
	fprintf(m_pFile, "# AreTomo Alignment / Priims bprmMn \n");
	fprintf(m_pFile, "# RawSize = %d %d %d\n", 
	   pDarkFrames->m_aiRawStkSize[0],
	   pDarkFrames->m_aiRawStkSize[1],
	   pDarkFrames->m_aiRawStkSize[2]);
	fprintf(m_pFile, "# NumPatches = %d\n", m_iNumPatches);
	//-----------------
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
	{	int iFrmIdx = pDarkFrames->GetFrmIdx(i);
		int iSecIdx = pDarkFrames->GetSecIdx(i);
		float fTilt = pDarkFrames->GetTilt(i);
		fprintf(m_pFile, "# DarkFrame =  %4d %4d %8.2f\n",
		   iFrmIdx, iSecIdx, fTilt);
	}
}

void CSaveAlignFile::mSaveGlobal(void)
{
	fprintf(m_pFile, "# SEC     ROT         GMAG       "
	   "TX          TY      SMEAN     SFIT    SCALE     BASE     TILT\n");
	//-----------------
	float afShift[] = {0.0f, 0.0f};
	for(int i=0; i<m_iNumTilts; i++)
	{	int iSecIdx = m_pAlignParam->GetSecIndex(i);
		float fTilt = m_pAlignParam->GetTilt(i);
		float fTiltAxis = m_pAlignParam->GetTiltAxis(i);
		m_pAlignParam->GetShift(i, afShift);
		fprintf(m_pFile, "%5d  %9.4f  %9.5f  %9.3f  %9.3f  %7.2f  "
		   "%7.2f  %7.2f  %7.2f  %8.2f\n", iSecIdx, fTiltAxis, 
		   1.0f, afShift[0], afShift[1], 1.0f, 1.0f, 1.0f, 
		   0.0f, fTilt);
	}
}

void CSaveAlignFile::mSaveLocal(void)
{
	if(m_iNumPatches <= 0) return;
	//-----------------
	fprintf(m_pFile, "# Local Alignment\n");
	int iSize = m_iNumPatches * m_iNumTilts;
	for(int i=0; i<iSize; i++)
	{	int t = i / m_iNumPatches;
		int p = i % m_iNumPatches;
		fprintf(m_pFile, "%4d %3d %8.2f  %8.2f  %8.2f  %8.2f  %4.1f\n", 
		   t, p, m_pLocalParam->m_pfCoordXs[i], 
		   m_pLocalParam->m_pfCoordYs[i],
		   m_pLocalParam->m_pfShiftXs[i],
		   m_pLocalParam->m_pfShiftYs[i],
		   m_pLocalParam->m_pfGoodShifts[i]);
	}
}

void CSaveAlignFile::mCloseFile(void)
{
	if(m_pFile == 0L) return;
	fclose(m_pFile);
	m_pFile = 0L;
}
