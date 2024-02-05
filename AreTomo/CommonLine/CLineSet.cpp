#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;
namespace MD = McAreTomo::DataUtil;

CLineSet::CLineSet(void)
{
	m_gCmpLines = 0L;
	m_iNumProjs = 0;
	m_iCmpSize = 0;
}

CLineSet::~CLineSet(void)
{
	this->Clean();
}

void CLineSet::Clean(void)
{
	if(m_gCmpLines == 0L) return;
	cudaFree(m_gCmpLines);
	m_gCmpLines = 0L;
}

void CLineSet::Setup(int iNthGpu)
{
	this->Clean();
	m_iNthGpu = iNthGpu;
	//-----------------
	CCommonLineParam* pClParam = CCommonLineParam::GetInstance(iNthGpu);
	m_iCmpSize = pClParam->m_iCmpLineSize;
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	m_iNumProjs = pTiltSeries->m_aiStkSize[2];
	//-----------------
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize * m_iNumProjs;
	cudaMalloc(&m_gCmpLines, tBytes);
}

cufftComplex* CLineSet::GetLine(int iLine)
{
	int iOffset = iLine * m_iCmpSize;
	return &m_gCmpLines[iOffset];
}

