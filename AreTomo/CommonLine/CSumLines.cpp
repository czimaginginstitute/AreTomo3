#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;

CSumLines::CSumLines(void)
{
	m_gCmpSum = 0L;
	m_iCmpSize = 0;
}

CSumLines::~CSumLines(void)
{
	this->Clean();
}

void CSumLines::Clean(void)
{
	if(m_gCmpSum != 0L) cudaFree(m_gCmpSum);
	m_gCmpSum = 0L;
	m_iCmpSize = 0;
}

cufftComplex* CSumLines::GetSum(bool bClean)
{
	cufftComplex* gCmpSum = m_gCmpSum;
	if(bClean) m_gCmpSum = 0L;
	return gCmpSum;
}

void CSumLines::DoIt(CLineSet* pLineSet)
{
	if(m_iCmpSize < pLineSet->m_iCmpSize)
	{	if(m_gCmpSum != 0L) cudaFree(m_gCmpSum);
		m_gCmpSum = 0L;
	}
	//-----------------
	m_iCmpSize = pLineSet->m_iCmpSize;
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	if(m_gCmpSum == 0L) cudaMalloc(&m_gCmpSum, tBytes);
	cudaMemset(m_gCmpSum, 0, tBytes);
	//-----------------
	GFunctions aGFunctions;
	for(int i=0; i<pLineSet->m_iNumProjs; i++)
	{	cufftComplex* gCmpLine = pLineSet->GetLine(i);
		aGFunctions.Sum(m_gCmpSum, gCmpLine, 1.0f, 1.0f,
		   m_gCmpSum, m_iCmpSize);
	}
}

