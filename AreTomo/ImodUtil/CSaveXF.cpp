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
	int iLast = pAlnParam->m_iNumFrames - 1;
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
	float fD2R = 3.141592654f / 180.0f;
	float afShift[2], fTiltAxis, fCos, fSin;
	float a11, a12, a21, a22, xshift_imod, yshift_imod;
	int iLast = pAlnParam->m_iNumFrames - 1;
	for(int i=0; i<pAlnParam->m_iNumFrames; i++)
	{	fTiltAxis = pAlnParam->GetTiltAxis(i) * fD2R;
                a11 = (float)cos(-fTiltAxis);
                a12 = -(float)sin(-fTiltAxis);
                a21 = (float)sin(-fTiltAxis);
                a22 = (float)cos(-fTiltAxis);
		//---------------------------
		pAlnParam->GetShift(i, afShift);
                xshift_imod = a11 * (-afShift[0]) + a12 * (-afShift[1]);
                yshift_imod = a21 * (-afShift[0]) + a22 * (-afShift[1]);
		//------------------------------------------------------
		fprintf(m_pFile, "%9.3f %9.3f %9.3f %9.3f ", a11, a12, a21, a22);
		fprintf(m_pFile, "%9.2f  %9.2f\n", xshift_imod, yshift_imod);
        }
	fprintf(m_pFile, "\n");
}

void CSaveXF::mSaveForRelion(void)
{
	MAM::CDarkFrames* pDarkFrames = 
	   MAM::CDarkFrames::GetInstance(m_iNthGpu);
	MAM::CAlignParam* pAlnParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	int iAllTilts = pDarkFrames->m_aiRawStkSize[2];
	int iLineSize = 256;
	char* pcLines = new char[iAllTilts * iLineSize];
	memset(pcLines, 0, sizeof(char) * iAllTilts * iLineSize);
	//-----------------
	float fD2R = 3.141592654f / 180.0f;
	float afShift[2], fTiltAxis, fCos, fSin;
	float a11, a12, a21, a22;
	float xshift_imod, yshift_imod;
	for(int i=0; i<pAlnParam->m_iNumFrames; i++)
	{	fTiltAxis = pAlnParam->GetTiltAxis(i) * fD2R;
		a11 = (float)cos(-fTiltAxis);
		a12 = -(float)sin(-fTiltAxis);
		a21 = (float)sin(-fTiltAxis);
		a22 = (float)cos(-fTiltAxis);
		//---------------------------
		pAlnParam->GetShift(i, afShift);
		xshift_imod = a11 * (-afShift[0]) + a12 * (-afShift[1]);
		yshift_imod = a21 * (-afShift[0]) + a22 * (-afShift[1]);
		//------------------------------------------------------
		int iSecIdx = pAlnParam->GetSecIndex(i);
		char* pcLine = pcLines + iSecIdx * iLineSize;
		//-------------------------------------------
		sprintf(pcLine, "%9.3f %9.3f %9.3f %9.3f %9.2f %9.2f",
		   a11, a12, a21, a22, xshift_imod, yshift_imod);
	}
	//-------------------------------------------------------
	a11 = 1.0f; a12 = 0.0f; a21 = 0.0f; a22 = 1.0f;
	for(int i=0; i<pDarkFrames->m_iNumDarks; i++)
	{	int iSecIdx = pDarkFrames->GetSecIdx(i);
		char* pcLine = pcLines + iSecIdx * iLineSize;
		sprintf(pcLine, "%9.3f %9.3f %9.3f %9.3f %9.2f %9.2f",
		   1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);
	}
	//---------------------------------------------
	for(int i=0; i<iAllTilts; i++)		
	{	char* pcLine = pcLines + i * iLineSize;
		fprintf(m_pFile, "%s\n", pcLine);
	}
	fprintf(m_pFile, "\n");
	delete[] pcLines; 
}

