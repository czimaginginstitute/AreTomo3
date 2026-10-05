//-------------------------------------------------------------------
// CSaveAlign: writes the .mcaln v1 motion-alignment file when
// -OutMotion 1 is given. Format frozen 2026-08-28; see the arewarpo
// docs/mcaln_format.md for the pinned semantics:
//   raw(x, f) = x - S_local(x, f) - glob_f
// with globalShift = FmRef-relative full shifts and localShift = the
// post-MakeRelative residual patch shifts, exactly the state the
// correction kernels consumed. Coordinates are alignment-image pixel
// indices, corner origin.
//-------------------------------------------------------------------
#include "CAlignInc.h"
#include <memory.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

using namespace McAreTomo::MotionCor::Align;

CSaveAlign* CSaveAlign::m_pInstances = 0L;
int CSaveAlign::m_iNumGpus = 0;

void CSaveAlign::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CSaveAlign[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CSaveAlign::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CSaveAlign* CSaveAlign::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CSaveAlign::CSaveAlign(void)
{
}

CSaveAlign::~CSaveAlign(void)
{
}

//-------------------------------------------------------------------
// <OutDir>/<movie basename>.mcaln
//-------------------------------------------------------------------
FILE* CSaveAlign::mOpen(void)
{
	CInput* pInput = CInput::GetInstance();
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	//-----------------
	char acPath[512] = {'\0'};
	strcpy(acPath, pInput->m_acOutDir);
	int iSize = strlen(acPath);
	if(iSize > 0 && acPath[iSize-1] != '/') strcat(acPath, "/");
	//-----------------
	char* pcSlash = strrchr(pMcPackage->m_acMoviePath, '/');
	char* pcName = (pcSlash == 0L) ? pMcPackage->m_acMoviePath
	   : pcSlash + 1;
	strcat(acPath, pcName);
	char* pcDot = strrchr(acPath, '.');
	if(pcDot == 0L) strcat(acPath, ".mcaln");
	else strcpy(pcDot, ".mcaln");
	//-----------------
	FILE* pFile = fopen(acPath, "wt");
	if(pFile == 0L)
	{	fprintf(stderr, "Warning: cannot open %s for writing, "
		   "motion alignment not saved.\n\n", acPath);
	}
	else printf("Motion alignment saved: %s\n", acPath);
	return pFile;
}

void CSaveAlign::mSaveHeader
(	FILE* pFile,
	int* piStkSize,
	int iPatX,
	int iPatY,
	int iNumFrames
)
{	CMcInput* pMcInput = CMcInput::GetInstance();
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	MMD::CFmIntParam* pFmIntParam =
	   MMD::CFmIntParam::GetInstance(m_iNthGpu);
	CAlignParam* pAlignParam = CAlignParam::GetInstance();
	//-----------------
	int iFmRef = pAlignParam->GetFrameRef(iNumFrames);
	//-----------------------------------------------------------
	// Pixel size of the ALIGNMENT grid (the frame buffer the
	// shifts live on): the final sum pixel scaled back by the
	// binning applied after alignment. E.g. EER sampling 2 +
	// McBin 2: alignment at 8192 px / 0.77 A, sums at 4096 / 1.54.
	//-----------------------------------------------------------
	int aiBinned[2] = {0};
	pMcInput->GetBinnedSize(piStkSize, aiBinned);
	float fPixSize = pMcInput->GetFinalPixelSize();
	if(piStkSize[0] > 0)
	{	fPixSize = fPixSize * aiBinned[0] / (float)piStkSize[0];
	}
	int* piMovieSize = pMcPackage->GetMovieSize();
	float fScaleX = (piMovieSize[0] > 0) ?
	   piStkSize[0] / (float)piMovieSize[0] : 1.0f;
	float fScaleY = (piMovieSize[1] > 0) ?
	   piStkSize[1] / (float)piMovieSize[1] : 1.0f;
	//-----------------
	fprintf(pFile, "# AreTomo3 MotionAlign 1.0\n");
	fprintf(pFile, "setting\n");
	fprintf(pFile, "   raw_frame_count: %d\n",
	   pFmIntParam->GetNumRawFrames());
	fprintf(pFile, "   integrated_frame_count: %d\n",
	   pFmIntParam->GetProvNumIntFrames());
	fprintf(pFile, "   aligned_frame_count: %d\n", iNumFrames);
	fprintf(pFile, "   alignment_image_size_px: %d %d\n",
	   piStkSize[0], piStkSize[1]);
	fprintf(pFile, "   alignment_pixel_size_A: %.6g\n", fPixSize);
	fprintf(pFile, "   input_to_alignment_scale_xy: %.6g %.6g\n",
	   fScaleX, fScaleY);
	fprintf(pFile, "   patches: %d %d\n", iPatX, iPatY);
	fprintf(pFile, "   fmRef: %d\n", iFmRef);
	fprintf(pFile, "   frame_index_base: 0\n");
	//-----------------
	fprintf(pFile, "frameTable\n");
	for(int i=0; i<pFmIntParam->GetProvNumIntFrames(); i++)
	{	fprintf(pFile, "   %4d %6d %5d %d %4d\n", i,
		   pFmIntParam->GetProvStart(i),
		   pFmIntParam->GetProvSize(i),
		   pFmIntParam->GetProvIncluded(i) ? 1 : 0,
		   pFmIntParam->GetProvAligned(i));
	}
}

void CSaveAlign::mSaveGlobal(FILE* pFile, MMD::CStackShift* pFullShift)
{
	fprintf(pFile, "globalShift\n");
	float afShift[2] = {0.0f};
	for(int i=0; i<pFullShift->m_iNumFrames; i++)
	{	pFullShift->GetShift(i, afShift);
		fprintf(pFile, "   %4d %10.4f %10.4f\n",
		   i, afShift[0], afShift[1]);
	}
}

void CSaveAlign::DoGlobal(MMD::CStackShift* pFullShift)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(pMcInput->m_iOutMotion == 0) return;
	if(pFullShift == 0L) return;
	//-----------------
	FILE* pFile = mOpen();
	if(pFile == 0L) return;
	//-----------------
	MD::CBufferPool* pBufferPool =
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	mSaveHeader(pFile, pBufferPool->m_aiStkSize, 0, 0,
	   pFullShift->m_iNumFrames);
	mSaveGlobal(pFile, pFullShift);
	fclose(pFile);
}

void CSaveAlign::DoLocal(MMD::CPatchShifts* pPatchShifts)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(pMcInput->m_iOutMotion == 0) return;
	if(pPatchShifts == 0L) return;
	//-----------------
	FILE* pFile = mOpen();
	if(pFile == 0L) return;
	//-----------------
	int iNumFrames = pPatchShifts->m_aiFullSize[2];
	mSaveHeader(pFile, pPatchShifts->m_aiFullSize,
	   pMcInput->m_aiNumPatches[0], pMcInput->m_aiNumPatches[1],
	   iNumFrames);
	mSaveGlobal(pFile, pPatchShifts->m_pFullShift);
	//-----------------
	float afCenter[2] = {0.0f}, afShift[2] = {0.0f};
	for(int p=0; p<pPatchShifts->m_iNumPatches; p++)
	{	fprintf(pFile, "localShift\n");
		fprintf(pFile, "   patchID: %d\n", p);
		pPatchShifts->GetPatchCenter(p, afCenter);
		for(int f=0; f<iNumFrames; f++)
		{	pPatchShifts->GetLocalShift(f, p, afShift);
			bool bBad = pPatchShifts->GetBadFlag(f, p);
			fprintf(pFile, "   %4d %9.2f %9.2f %9.3f %9.3f %d\n",
			   f, afCenter[0], afCenter[1],
			   afShift[0], afShift[1], bBad ? 0 : 1);
		}
	}
	fclose(pFile);
}
