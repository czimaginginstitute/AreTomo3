#include "CMassNormInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::MassNorm;

CPositivity::CPositivity(void)
{
	m_fMissingVal = (float)-1e10;
}

CPositivity::~CPositivity(void)
{
}

void CPositivity::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	m_fMin = mCalcMin(0);
	for(int i=1; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float fMin = mCalcMin(i);
		if(fMin < m_fMin) m_fMin = fMin;
	}
	if(m_fMin >= 0) return;
	//---------------------
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	printf("...... Set positivity %4d, %4d left\n",
		   i+1, pTiltSeries->m_aiStkSize[2]-1-i);
		mSetPositivity(i); 
	}
	printf("\n");
}

float CPositivity::mCalcMin(int iFrame)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	float fMin = 10000000.0f;
	float* pfFrame = (float*)pTiltSeries->GetFrame(iFrame);
	int iPixels = pTiltSeries->m_aiStkSize[0] *
	   pTiltSeries->m_aiStkSize[1];
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= m_fMissingVal) continue;
		if(fMin > pfFrame[i]) fMin = pfFrame[i];
	}
	return fMin;
}

void CPositivity::mSetPositivity(int iFrame)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	float* pfFrame = (float*)pTiltSeries->GetFrame(iFrame);
	int iPixels = pTiltSeries->m_aiStkSize[0] *
	   pTiltSeries->m_aiStkSize[1];
	for(int i=0; i<iPixels; i++)
	{	if(pfFrame[i] <= m_fMissingVal) continue;
		else pfFrame[i] -= m_fMin;
	}
}

