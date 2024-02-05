#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;

CPossibleLines::CPossibleLines(void)
{
	m_ppCmpPlanes = 0L;
	m_iNumProjs = 0;
	m_iNumLines = 0;
	m_iLineSize = 0;
	m_iCmpSize = 0;
}

CPossibleLines::~CPossibleLines(void)
{
	this->Clean();
}

void CPossibleLines::Clean(void)
{
	if(m_ppCmpPlanes == 0L) return;
	for(int i=0; i<m_iNumProjs; i++)
	{	if(m_ppCmpPlanes[i] == 0L) continue;
		delete[] m_ppCmpPlanes[i];
	}
	delete[] m_ppCmpPlanes;
	m_ppCmpPlanes = 0L;
}

void CPossibleLines::Setup(int iNthGpu)
{	
	this->Clean();
	//-----------------
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance(iNthGpu);
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	m_iNumLines = pClParam->m_iNumLines;
	m_iNumProjs = pTiltSeries->m_aiStkSize[2];
	m_iLineSize = pClParam->m_iLineSize;
	m_iCmpSize = pClParam->m_iCmpLineSize;
	//-----------------
	bool bCopy = true;
	m_pfTiltAngles = pTiltSeries->m_pfTilts;
	m_pfRotAngles = pClParam->m_pfRotAngles;
	//-----------------
	m_ppCmpPlanes = new cufftComplex*[m_iNumProjs];
	int iPixels = m_iNumLines * m_iCmpSize;
	for(int i=0; i<m_iNumProjs; i++)
	{	m_ppCmpPlanes[i] = new cufftComplex[iPixels];
	}
}

void CPossibleLines::SetPlane
(	int iProj,
	cufftComplex* gCmpPlane
)
{	size_t tBytes = m_iNumLines * m_iCmpSize * sizeof(cufftComplex);
	cufftComplex* pDstPlane = m_ppCmpPlanes[iProj];
	cudaMemcpy(pDstPlane, gCmpPlane, tBytes, cudaMemcpyDefault);
}

void CPossibleLines::SetLine
(	int iProj, 
	int iLine,
	cufftComplex* gCmpLine
)
{	cufftComplex* pCmpLine = mGetLine(iProj, iLine);
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMemcpy(pCmpLine, gCmpLine, tBytes, cudaMemcpyDefault);
}

void CPossibleLines::GetLine
(	int iProj, 
	int iLine,
	cufftComplex* gCmpLine
)
{	cufftComplex* pCmpLine = mGetLine(iProj, iLine);
        size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMemcpy(gCmpLine, pCmpLine, tBytes, cudaMemcpyDefault);
}

float CPossibleLines::CalcLinePos(float fRotAngle)
{
	int iEnd = m_iNumLines - 1;
	float fLinePos = iEnd * (fRotAngle - m_pfRotAngles[0])
		/ (m_pfRotAngles[iEnd] - m_pfRotAngles[0]);
	return fLinePos;
}

float CPossibleLines::GetLineAngle(int iLine)
{
	return m_pfRotAngles[iLine];
}

cufftComplex* CPossibleLines::mGetLine(int iProj, int iLine)
{
	cufftComplex* pCmpPlane = m_ppCmpPlanes[iProj];
	cufftComplex* pCmpLine = pCmpPlane + iLine * m_iCmpSize;
	return pCmpLine;
}

