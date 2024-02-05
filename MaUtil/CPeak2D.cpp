#include "CMaUtilInc.h"
#include <stdio.h>
#include <memory.h>

using namespace McAreTomo::MaUtil;

CPeak2D::CPeak2D(void)
{
	memset(m_afShift, 0, sizeof(m_afShift));
	memset(m_aiSeaSize, 0, sizeof(m_aiSeaSize));
}

CPeak2D::~CPeak2D(void)
{
}

void CPeak2D::GetShift(float fXcfBin, float* pfShift)
{
	pfShift[0] = m_afShift[0] * fXcfBin;
	pfShift[1] = m_afShift[1] * fXcfBin;
}

void CPeak2D::GetShift(float* pfXcfBin, float* pfShift)
{
	pfShift[0] = m_afShift[0] * pfXcfBin[0];
	pfShift[1] = m_afShift[1] * pfXcfBin[1];
}

void CPeak2D::DoIt
(	float* pfImg, int* piImgSize, bool bPadded,
	int* piSeaSize
)
{	m_pfImg = pfImg;
	m_iPadX = piImgSize[0];
	if(!bPadded) m_aiImgSize[0] = piImgSize[0];
	else m_aiImgSize[0] = (piImgSize[0] / 2 - 1) * 2;
	m_aiImgSize[1] = piImgSize[1];
	//----------------------------
	int iBytes = sizeof(m_aiSeaSize);
	if(piSeaSize == 0L) memcpy(m_aiSeaSize, m_aiImgSize, iBytes);
	else memcpy(m_aiSeaSize, piSeaSize, iBytes);
	//------------------------------------------
	mSearchIntPeak();
	mSearchFloatPeak();
	m_afShift[0] = m_afPeak[0] - m_aiImgSize[0] / 2;
	m_afShift[1] = m_afPeak[1] - m_aiImgSize[1] / 2;
}

void CPeak2D::mSearchIntPeak(void)
{
	m_fPeakInt = (float)-1e30;
	//------------------------
	int iStartX = (m_aiImgSize[0] - m_aiSeaSize[0]) / 2;
	int iStartY = (m_aiImgSize[1] - m_aiSeaSize[1]) / 2;
	int iEndX = iStartX + m_aiSeaSize[0];
	int iEndY = iStartY + m_aiSeaSize[1];
	for(int y=iStartY; y<iEndY; y++)
	{	int i = y * m_iPadX;
		for(int x=iStartX; x<iEndX; x++)
		{	if(m_fPeakInt >= m_pfImg[i+x]) continue;
			m_aiPeak[0] = x;
			m_aiPeak[1] = y;
			m_fPeakInt = m_pfImg[i+x];
		}
	}
	//-------------------------------------
	if(m_aiPeak[0] < 1) m_aiPeak[0] = 1;
	else if(m_aiPeak[0] > (m_aiImgSize[0] - 2))
		m_aiPeak[0] = m_aiImgSize[0] - 2;
	if(m_aiPeak[1] < 1) m_aiPeak[1] = 1;
	else if(m_aiPeak[1] > (m_aiImgSize[1] - 2))
		m_aiPeak[1] = m_aiImgSize[1] - 2; 
}

void CPeak2D::mSearchFloatPeak(void)
{
	int ic = m_aiPeak[1] * m_iPadX + m_aiPeak[0];
	int xp = ic + 1;
	int xm = ic - 1;
	int yp = ic + m_iPadX;
	int ym = ic - m_iPadX;
	//--------------------
	double a = (m_pfImg[xp] + m_pfImg[xm]) * 0.5 - m_pfImg[ic];
	double b = (m_pfImg[xp] - m_pfImg[xm]) * 0.5;
	double c = (m_pfImg[yp] + m_pfImg[ym]) * 0.5f - m_pfImg[ic];
	double d = (m_pfImg[yp] - m_pfImg[ym]) * 0.5;
	double dCentX = -b / (2 * a + 1e-30);
	double dCentY = -d / (2 * c + 1e-30);
	//-----------------------------------
	if(fabs(dCentX) > 1) dCentX = 0;
	if(fabs(dCentY) > 1) dCentY = 0;
	m_afPeak[0] = (float)(m_aiPeak[0] + dCentX);
	m_afPeak[1] = (float)(m_aiPeak[1] + dCentY);
	/*
	m_fPeakInt =  (float)(a * dCentX * dCentX + b * dCentX
		+ c * dCentY * dCentY + d * dCentY
		+ m_pfImg[ic]);
	*/
}
