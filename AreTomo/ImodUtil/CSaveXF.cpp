#include "CImodUtilInc.h"
#include "../CAreTomoInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace McAreTomo::AreTomo::ImodUtil;

CSaveXF::CSaveXF(void)
{
	m_pFile = 0L;
	m_iLineSize = 256;
}

CSaveXF::~CSaveXF(void)
{
}

void CSaveXF::DoIt(int iNthGpu, const char* pcFileName)
{	
	m_pFile = fopen(pcFileName, "wt");
	if(m_pFile == 0L) return;
	m_iNthGpu = iNthGpu;
	//-----------------
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_iOutImod == 1) mSaveForRelion();
	else if(pInput->m_iOutImod == 2) mSaveForWarp();
	else if(pInput->m_iOutImod == 3) mSaveForAligned();
	//-----------------
	fclose(m_pFile);
	m_pFile = 0L;
}

void CSaveXF::mSaveForAligned(void)
{
	//--------------------------------------
	// for dark removed aligned tilt series
	//--------------------------------------
	MAM::CAlignParam* pAlnParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	for(int i=0; i<pAlnParam->m_iNumFrames; i++)
	{	fprintf(m_pFile, "%9.3f %9.3f %9.3f %9.3f ",
		   1.0f, 0.0f, 0.0f, 1.0f);
		fprintf(m_pFile, "%9.2f  %9.2f\n", 0.0f, 0.0f);
	}
	fprintf(m_pFile, "\n");
}

void CSaveXF::mSaveForWarp(void)
{
	MAM::CAlignParam* pAlnParam =
           MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	float xshift_imod = 0.0f;
	float yshift_imod = 0.0f;
	float afShift[2] = {0.0f};
	float fD2R = 0.01745329f;
	//-----------------
	for(int i=0; i<pAlnParam->m_iNumFrames; i++)
	{	float fTiltAxis = pAlnParam->GetTiltAxis(i) * fD2R;
		float a11 = (float)cos(-fTiltAxis);
                float a12 = -(float)sin(-fTiltAxis);
                float a21 = (float)sin(-fTiltAxis);
                float a22 = (float)cos(-fTiltAxis);
		//----------------
		pAlnParam->GetShift(i, afShift);
                xshift_imod = a11 * (-afShift[0]) + a12 * (-afShift[1]);
                yshift_imod = a21 * (-afShift[0]) + a22 * (-afShift[1]);
		//----------------
		fprintf(m_pFile, "%9.3f %9.3f %9.3f %9.3f ", a11, a12, a21, a22);
		fprintf(m_pFile, "%9.2f  %9.2f\n", xshift_imod, yshift_imod);
        }
	fprintf(m_pFile, "\n");
}

//--------------------------------------------------------------------
// 1. This is for -OutImod 1 that generates Imod files for Relion4.
// 2. Since Relion4 works on raw tilt series, lines in the .xf file
//    need to be in the same order as the raw tilt series.
// 3. Hence, lines in .xf file are sorted by iSecIdx to keep the same 
//    order as the input MRC file.
//--------------------------------------------------------------------
void CSaveXF::mSaveForRelion(void)
{
	MAM::CDarkFrames* pDarkFrames = 
	   MAM::CDarkFrames::GetInstance(m_iNthGpu);
	MAM::CAlignParam* pAlnParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	int iAllTilts = pDarkFrames->m_aiRawStkSize[2];
	int iSize = iAllTilts * m_iLineSize;
	char* pcOrderedList = new char[iSize];
	memset(pcOrderedList, 0, sizeof(char) * iSize);
	//-----------------
	float xshift_imod = 0.0f;
        float yshift_imod = 0.0f;
        float afShift[2] = {0.0f};
        float fD2R = 0.01745329f;
	//-----------------
	for(int i=0; i<pAlnParam->m_iNumFrames; i++)
	{	float fTiltAxis = pAlnParam->GetTiltAxis(i) * fD2R;
		float a11 = (float)cos(-fTiltAxis);
		float a12 = -(float)sin(-fTiltAxis);
		float a21 = (float)sin(-fTiltAxis);
		float a22 = (float)cos(-fTiltAxis);
		//----------------
		pAlnParam->GetShift(i, afShift);
		xshift_imod = a11 * (-afShift[0]) + a12 * (-afShift[1]);
		yshift_imod = a21 * (-afShift[0]) + a22 * (-afShift[1]);
		//----------------
		int iSecIdx = pAlnParam->GetSecIndex(i);
		char* pcLine = pcOrderedList + iSecIdx * m_iLineSize;
		//----------------
		sprintf(pcLine, "%9.3f %9.3f %9.3f %9.3f %9.2f %9.2f",
		   a11, a12, a21, a22, xshift_imod, yshift_imod);
	}
	//---------------------------------------------------------
	// 1) For dark images their tilt axes are set to 0 degree
	// and their shifts are set to 0.
	//---------------------------------------------------------
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
	{	int iDarkFrm = pDarkFrames->GetDarkIdx(i);
		int iSecIdx = pDarkFrames->GetSecIdx(iDarkFrm);
		//----------------
		char* pcLine = pcOrderedList + iSecIdx * m_iLineSize;
		sprintf(pcLine, "%9.3f %9.3f %9.3f %9.3f %9.2f %9.2f",
		   1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
	}
	//-----------------
	for(int i=0; i<iAllTilts; i++)		
	{	char* pcLine = pcOrderedList + i * m_iLineSize;
		fprintf(m_pFile, "%s\n", pcLine);
	}
	//-----------------
	if(pcOrderedList != 0L) delete[] pcOrderedList;
}

