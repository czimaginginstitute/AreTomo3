#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace McAreTomo::AreTomo::MrcUtil;

char CSaveAlignFile::m_acRawSizeTag[] = "RawSize";
char CSaveAlignFile::m_acNumPatchesTag[] = "NumPatches";
char CSaveAlignFile::m_acDarkFrameTag[] = "DarkFrame";
char CSaveAlignFile::m_acAlphaOffsetTag[] = "AlphaOffset";
char CSaveAlignFile::m_acBetaOffsetTag[] = "BetaOffset";
char CSaveAlignFile::m_acLocalAlignTag[] = "Local Alignment";
char CSaveAlignFile::m_acThicknessTag[] = "Thickness";

CSaveAlignFile::CSaveAlignFile(void)
{
	m_pFile = 0L;
}

CSaveAlignFile::~CSaveAlignFile(void)
{
	mCloseFile();
}

void CSaveAlignFile::GenFileName
(	int iNthGpu, 
	bool bSave,
	char* pcAlnFile
)
{	McAreTomo::CInput* pInput = McAreTomo::CInput::GetInstance();
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(iNthGpu);
	//-----------------
	if(bSave) strcpy(pcAlnFile, pInput->m_acOutDir);
	else strcpy(pcAlnFile, pInput->m_acInDir);
	strcat(pcAlnFile, pPackage->m_acMrcMain);
	strcat(pcAlnFile, ".aln");
}

void CSaveAlignFile::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------	
	char acAlnFile[256] = {'\0'};
	bool bSave = true;
	CSaveAlignFile::GenFileName(iNthGpu, bSave, acAlnFile);
	//-----------------
	m_pFile = fopen(acAlnFile, "wt");
	if(m_pFile == 0L)
	{	printf("GPU %d: Alignment data will not be saved\n"
		   "   Unable to open aln file %s\n\n", m_iNthGpu, acAlnFile);
		return;
	}
	//-----------------
	McAreTomo::CInput* pInput = McAreTomo::CInput::GetInstance();
        MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(iNthGpu);
        m_pAlignParam = CAlignParam::GetInstance(iNthGpu);
        m_pLocalParam = CLocalAlignParam::GetInstance(iNthGpu);
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
	fprintf(m_pFile, "%s\n", "# AreTomo Alignment / Priims bprmMn");
	fprintf(m_pFile, "# %s = %d %d %d\n", m_acRawSizeTag, 
	   pDarkFrames->m_aiRawStkSize[0],
	   pDarkFrames->m_aiRawStkSize[1],
	   pDarkFrames->m_aiRawStkSize[2]);
	fprintf(m_pFile, "# %s = %d\n", m_acNumPatchesTag, m_iNumPatches);
	//-----------------------------------------------
	// 1) Track section IDs of dark images here so
	// that we know which dark images are discarded
	// in the raw tilt series. 
	// 2) When tilt images are sorted by tilt angles,
	// iDarkFm shows which tilt image is dark.
	// 3) This info is needed when a tilt series 
	// needs to be reconstructed again without 
	// repeating the alignment process.
	//-----------------------------------------------
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
	{	int iDarkFm = pDarkFrames->GetDarkIdx(i);
		int iSecIdx = pDarkFrames->GetDarkSec(i);
		float fTilt = pDarkFrames->GetDarkTilt(i);
		fprintf(m_pFile, "# %s =  %4d %4d %8.2f\n", m_acDarkFrameTag,
		   iDarkFm, iSecIdx, fTilt);
	}
	//-----------------------------------------------
	// 1) Tracking whether or not the alpha and beta
	// tilt offsets are applied to tilt angle.
	// 2) This information is needed in the future
	// for determine per-particle defocus.
	//-----------------------------------------------
	fprintf(m_pFile, "# %s = %8.2f\n", m_acAlphaOffsetTag,
	   m_pAlignParam->m_fAlphaOffset);
	fprintf(m_pFile, "# %s = %8.2f\n", m_acBetaOffsetTag,
	   m_pAlignParam->m_fBetaOffset);
	//-----------------------------------------------
	fprintf(m_pFile, "# %s = %d\n", m_acThicknessTag,
	   m_pAlignParam->m_iThickness);
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
	fprintf(m_pFile, "# %s\n", m_acLocalAlignTag);
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
