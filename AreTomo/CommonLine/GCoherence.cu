//--------------------------------------------------------------------
// 1. This class determines the coherence within a set of lines
//    obtained from each tilted image.
// 2. The coherence is the sum of all the correlation coefficients
//    of all the pairs of lines.
//--------------------------------------------------------------------
#include "CCommonLineInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;

static __global__ void mGAdd
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	cufftComplex* gCmpSum,
	int iCmpSize
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iCmpSize) return;
	gCmpSum[i].x = gCmp1[i].x + gCmp2[i].x;
	gCmpSum[i].y = gCmp1[i].y + gCmp2[i].y;
}

static __global__ void mGMinus
(       cufftComplex* gCmp1,
        cufftComplex* gCmp2,
        cufftComplex* gCmpDif,
        int iCmpSize
)
{       int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= iCmpSize) return;
        gCmpDif[i].x = gCmp1[i].x - gCmp2[i].x;
        gCmpDif[i].y = gCmp1[i].y - gCmp2[i].y;
}

GCoherence::GCoherence(void)
{
	m_gCmpRef = 0L;
	m_gCmpSum = 0L;
}

GCoherence::~GCoherence(void)
{
}

float GCoherence::DoIt
(	cufftComplex* gCmpLines,
	int iCmpSize,
	int iNumLines
)
{	
	m_iCmpSize = iCmpSize;
	m_iNumLines = iNumLines;
	m_gCmpLines = gCmpLines;
	//----------------------
	if(m_gCmpRef != 0L) cudaFree(m_gCmpRef);
	if(m_gCmpSum != 0L) cudaFree(m_gCmpSum);
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMalloc(&m_gCmpRef, tBytes);
	cudaMalloc(&m_gCmpSum, tBytes);
	//-----------------------------
	mCalcSum();
	m_fCC = 0.0f;
	//-----------
	for(int i=1; i<m_iNumLines; i++)
	{	float fCC = mMeasure(i);
		m_fCC += fCC;
	}
	m_fCC = m_fCC / (m_iNumLines - 1);
	//--------------------------------
	if(m_gCmpRef != 0L) cudaFree(m_gCmpRef);
	if(m_gCmpSum != 0L) cudaFree(m_gCmpSum);
	m_gCmpRef = 0L;
	m_gCmpSum = 0L;
	return m_fCC;	
}

void GCoherence::mCalcSum(void)
{
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMemset(m_gCmpSum, 0, tBytes);
	//-------------------------------
	dim3 aBlockDim(512, 1), aGridDim(1, 1);
        aGridDim.x = m_iCmpSize / aBlockDim.x + 1;
	for(int i=0; i<m_iNumLines; i++)
	{	cufftComplex* gCmpLine = m_gCmpLines + i * m_iCmpSize;
		mGAdd<<<aGridDim, aBlockDim>>>
		(  gCmpLine, m_gCmpSum, m_gCmpSum, m_iCmpSize
		); 
	}
}

float GCoherence::mMeasure(int iLine)
{
	dim3 aBlockDim(512, 1), aGridDim(1, 1);
	aGridDim.x = m_iCmpSize / aBlockDim.x + 1;
	//----------------------------------------
	cufftComplex* gCmpLine = m_gCmpLines + iLine * m_iCmpSize;
	mGMinus<<<aGridDim, aBlockDim>>>
	(  m_gCmpSum, gCmpLine, m_gCmpRef, m_iCmpSize
	);
	//-------------------------------------------	
	MAU::GCC1D aGCC1D;
	float fCC = aGCC1D.DoIt(gCmpLine, m_gCmpRef, m_iCmpSize);
	return fCC;
}

