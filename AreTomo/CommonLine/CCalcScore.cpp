#include "CCommonLineInc.h"
#include <Util/Util_LinEqs.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;
namespace MU = McAreTomo::MaUtil;

CCalcScore::CCalcScore(void)
{
	m_gCmpSum = 0L;
	m_gCmpRef = 0L;
}

CCalcScore::~CCalcScore(void)
{
}

float CCalcScore::DoIt
(	CLineSet* pLineSet,
	cufftComplex* gCmpSum
)
{	m_pLineSet = pLineSet;
	m_gCmpSum = gCmpSum;
	m_iCmpSize = pLineSet->m_iCmpSize;
	//-----------------
	m_gCmpRef = mCudaMallocLine(false);
	//-----------------
	float fCCSum = 0.0f;
	for(int i=0; i<m_pLineSet->m_iNumProjs; i++)
	{	float fCC = mCorrelate(i);
		fCCSum += fCC;
	}
	float fScore = fCCSum / m_pLineSet->m_iNumProjs;
	//----------------------
	if(m_gCmpRef != 0L) cudaFree(m_gCmpRef);
	m_gCmpRef = 0L;
	m_gCmpSum = 0L;
	return fScore;
}

float CCalcScore::mCorrelate(int iLine)
{
	cufftComplex* gCmpLine = m_pLineSet->GetLine(iLine);
	//--------------------------------------------------
	GFunctions aGFunctions;
	aGFunctions.Sum
	( m_gCmpSum, gCmpLine, 1.0f, -1.0f,
	  m_gCmpRef, m_iCmpSize
	);
	//---------------------
	MAU::GCC1D aGCC1D;
	aGCC1D.SetBFactor(10);
	float fCC = aGCC1D.DoIt(m_gCmpRef, gCmpLine, m_iCmpSize);
	return fCC;
}

cufftComplex* CCalcScore::mCudaMallocLine(bool bZero)
{
	cufftComplex* gCmpLine = 0L;
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMalloc(&gCmpLine, tBytes);
	if(bZero) cudaMemset(gCmpLine, 0, tBytes);
	return gCmpLine;
}
