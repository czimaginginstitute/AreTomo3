//-------------------------------------------------------------------
// CLoadAlign: reads a .mcaln v1 motion-alignment file (-InMotion 1)
// from <OutDir>/<movie basename>.mcaln and rebuilds CStackShift /
// CPatchShifts so the correction stages can run without measurement.
// MakeRelative is NOT applied - file shifts are final (frozen format,
// see the arewarpo docs/mcaln_format.md).
//-------------------------------------------------------------------
#include "CAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace McAreTomo::MotionCor::Align;

CLoadAlign::CLoadAlign(void)
{
	m_pFullShift = 0L;
	m_pPatchShifts = 0L;
}

CLoadAlign::~CLoadAlign(void)
{
	mClean();
}

void CLoadAlign::mClean(void)
{
	if(m_pFullShift != 0L) delete m_pFullShift;
	if(m_pPatchShifts != 0L) delete m_pPatchShifts;
	m_pFullShift = 0L;
	m_pPatchShifts = 0L;
}

MMD::CStackShift* CLoadAlign::TakeFullShift(void)
{
	MMD::CStackShift* p = m_pFullShift;
	m_pFullShift = 0L;
	return p;
}

MMD::CPatchShifts* CLoadAlign::TakePatchShifts(void)
{
	MMD::CPatchShifts* p = m_pPatchShifts;
	m_pPatchShifts = 0L;
	return p;
}

bool CLoadAlign::DoIt(int iNthGpu)
{
	mClean();
	//-----------------
	CInput* pInput = CInput::GetInstance();
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(iNthGpu);
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(iNthGpu);
	//-----------------
	char acPath[512] = {'\0'};
	strcpy(acPath, pInput->m_acOutDir);
	int iSize = strlen(acPath);
	if(iSize > 0 && acPath[iSize-1] != '/') strcat(acPath, "/");
	char* pcSlash = strrchr(pMcPackage->m_acMoviePath, '/');
	char* pcName = (pcSlash == 0L) ? pMcPackage->m_acMoviePath
	   : pcSlash + 1;
	strcat(acPath, pcName);
	char* pcDot = strrchr(acPath, '.');
	if(pcDot == 0L) strcat(acPath, ".mcaln");
	else strcpy(pcDot, ".mcaln");
	//-----------------
	bool bLoaded = mParse(acPath, pBufferPool->m_aiStkSize);
	if(!bLoaded)
	{	fprintf(stderr, "Error: -InMotion could not load %s\n\n",
		   acPath);
		mClean();
	}
	else printf("Motion alignment loaded: %s\n", acPath);
	return bLoaded;
}

bool CLoadAlign::mParse(const char* pcPath, int* piStkSize)
{
	FILE* pFile = fopen(pcPath, "rt");
	if(pFile == 0L) return false;
	//-----------------
	char acLine[512] = {'\0'};
	if(fgets(acLine, sizeof(acLine), pFile) == 0L ||
	   strncmp(acLine, "# AreTomo3 MotionAlign 1.0", 26) != 0)
	{	fprintf(stderr, "Error: %s is not a "
		   ".mcaln v1 file.\n", pcPath);
		fclose(pFile);
		return false;
	}
	//-----------------
	int iSizeX = 0, iSizeY = 0, iAligned = 0;
	int iPatX = 0, iPatY = 0;
	int iSection = 0; // 1 setting, 2 frameTable, 3 global, 4 local
	int iPatchSlot = -1;
	//-----------------
	while(fgets(acLine, sizeof(acLine), pFile) != 0L)
	{	char* pcTrim = acLine;
		while(*pcTrim == ' ' || *pcTrim == '\t') pcTrim++;
		if(*pcTrim == '\n' || *pcTrim == '\0') continue;
		//----------------
		if(strncmp(pcTrim, "setting", 7) == 0)
		{	iSection = 1; continue;
		}
		else if(strncmp(pcTrim, "frameTable", 10) == 0)
		{	iSection = 2; continue;
		}
		else if(strncmp(pcTrim, "globalShift", 11) == 0)
		{	iSection = 3;
			if(iAligned <= 0 || iSizeX <= 0) break;
			m_pFullShift = new MMD::CStackShift;
			m_pFullShift->Setup(iAligned);
			if(iPatX * iPatY > 0)
			{	int aiFull[] = {iSizeX, iSizeY, iAligned};
				m_pPatchShifts = new MMD::CPatchShifts;
				m_pPatchShifts->Setup(iPatX * iPatY, aiFull);
			}
			continue;
		}
		else if(strncmp(pcTrim, "localShift", 10) == 0)
		{	iSection = 4; iPatchSlot = -1; continue;
		}
		//----------------
		if(iSection == 1)
		{	if(strncmp(pcTrim, "aligned_frame_count:", 20) == 0)
			{	iAligned = atoi(pcTrim + 20);
			}
			else if(strncmp(pcTrim,
			   "alignment_image_size_px:", 24) == 0)
			{	sscanf(pcTrim + 24, "%d %d",
				   &iSizeX, &iSizeY);
			}
			else if(strncmp(pcTrim, "patches:", 8) == 0)
			{	sscanf(pcTrim + 8, "%d %d", &iPatX, &iPatY);
			}
			else if(strncmp(pcTrim, "frame_index_base:", 17) == 0)
			{	if(atoi(pcTrim + 17) != 0) break;
			}
		}
		else if(iSection == 3)
		{	int iFrame = -1;
			float afShift[2] = {0.0f};
			if(sscanf(pcTrim, "%d %f %f", &iFrame,
			   afShift, afShift + 1) != 3) break;
			if(iFrame < 0 || iFrame >= iAligned) break;
			m_pFullShift->SetShift(iFrame, afShift);
		}
		else if(iSection == 4 && m_pPatchShifts != 0L)
		{	if(strncmp(pcTrim, "patchID:", 8) == 0)
			{	iPatchSlot = atoi(pcTrim + 8);
				continue;
			}
			int iFrame = -1, iValid = 1;
			float afCent[2] = {0.0f}, afShift[2] = {0.0f};
			if(sscanf(pcTrim, "%d %f %f %f %f %d", &iFrame,
			   afCent, afCent + 1, afShift, afShift + 1,
			   &iValid) != 6) break;
			if(iPatchSlot < 0 ||
			   iPatchSlot >= iPatX * iPatY) break;
			if(iFrame < 0 || iFrame >= iAligned) break;
			//----------------
			if(iFrame == 0)
			{	// per-patch center: buffer via a stack
				// shift so SetRawShift stores it.
				MMD::CStackShift aShift;
				aShift.Setup(iAligned);
				aShift.SetCenter(afCent[0], afCent[1]);
				m_pPatchShifts->SetRawShift(&aShift,
				   iPatchSlot);
			}
			float afLocal[] = {afShift[0], afShift[1]};
			mSetPatchShift(iFrame, iPatchSlot, afLocal,
			   iValid != 0);
		}
	}
	fclose(pFile);
	//-----------------
	if(m_pFullShift == 0L) return false;
	if(iSizeX != piStkSize[0] || iSizeY != piStkSize[1] ||
	   iAligned != piStkSize[2])
	{	fprintf(stderr, "Error: .mcaln geometry (%d x %d x %d) "
		   "does not match the loaded stack (%d x %d x %d).\n",
		   iSizeX, iSizeY, iAligned, piStkSize[0],
		   piStkSize[1], piStkSize[2]);
		return false;
	}
	if(m_pPatchShifts != 0L)
	{	m_pPatchShifts->SetFullShift(m_pFullShift->GetCopy());
	}
	return true;
}

void CLoadAlign::mSetPatchShift
(	int iFrame,
	int iPatch,
	float* pfShift,
	bool bValid
)
{	m_pPatchShifts->SetLocalShift(iFrame, iPatch, pfShift);
	m_pPatchShifts->SetBadFlag(iFrame, iPatch, !bValid);
}
