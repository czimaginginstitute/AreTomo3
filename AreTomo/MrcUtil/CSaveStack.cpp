#include "CMrcUtilInc.h"
#include <Util/Util_Time.h>
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include <math.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CSaveStack::CSaveStack(void)
{
	memset(m_acMrcFile, 0, sizeof(m_acMrcFile));
}

CSaveStack::~CSaveStack(void)
{
}

bool CSaveStack::OpenFile(char* pcMrcFile)
{
	strcpy(m_acMrcFile, pcMrcFile);
	bool bOpen = m_aSaveMrc.OpenFile(pcMrcFile);
	return bOpen;
}
	
void CSaveStack::DoIt
(	MD::CTiltSeries* pTiltSeries,
	CAlignParam* pAlignParam,
	float fPixelSize,
	float* pfStats,
	bool bVolume
)
{	Util_Time utilTime;
	utilTime.Measure();
	//-----------------
	m_aSaveMrc.OpenFile(m_acMrcFile);
	m_aSaveMrc.SetMode(Mrc::eMrcFloat);
	m_aSaveMrc.SetImgSize(pTiltSeries->m_aiStkSize, 
	   pTiltSeries->m_aiStkSize[2], 1, fPixelSize);
	m_aSaveMrc.SetExtHeader(0, 32, 0);
	if(pfStats != 0L)
	{	m_aSaveMrc.SaveMinMaxMean(pfStats[0], pfStats[1], pfStats[2]);
	}
	m_aSaveMrc.m_pSaveMain->DoIt();
	//-----------------------------
	printf("Saving %s\n", m_acMrcFile);
	if(!bVolume)
	{	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
		{	float fTilt = pAlignParam->GetTilt(i);
			m_aSaveMrc.m_pSaveExt->SetTilt(i, &fTilt, 1);
			m_aSaveMrc.m_pSaveExt->DoIt();
		}
	}
	//---------------------------------------------
	int iLast = pTiltSeries->m_aiStkSize[2] - 1;
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float* pfFrame = (float*)pTiltSeries->GetFrame(i);
		m_aSaveMrc.m_pSaveImg->DoIt(i, pfFrame);
		if(i % 100 != 0 && i != iLast) continue;
		printf("...... %4d volume sliices saved, %4d left\n",
		   i+1, iLast-i);
	}
	//-----------------------
	m_aSaveMrc.CloseFile();
	printf("Saving time: %.2f\n", utilTime.GetElapsedSeconds());
}

void CSaveStack::mDrawTiltAxis(float* pfImg, int* piSize, float fTiltAxis)
{
	float fMax = pfImg[0];
	int iPixels = piSize[0] * piSize[1];
	for(int i=0; i<iPixels; i++)
	{	if(fMax < pfImg[i]) fMax = pfImg[i];
	}
	fMax = fMax + 1.0f;
	//-----------------
	float fCos = (float)cos(fTiltAxis * 3.14 / 180);
	float fSin = (float)sin(fTiltAxis * 3.14 / 180);
	//-----------------------------------------------
	float fCentX = piSize[0] * 0.5f;
	float fCentY = piSize[1] * 0.5f;
	int iWidth = 10;
	int iLength = piSize[1] / 4;
	//--------------------------
	for(int l=0; l<iLength; l++)
	{	for(int w=0; w<iWidth; w++)
		{	int x = (int)(w * fCos - l * fSin) + piSize[0] / 8;
			int y = (int)(w * fSin + l * fCos) + piSize[1] / 4;
			pfImg[y * piSize[0] + x] = fMax;
		}
	}
}
