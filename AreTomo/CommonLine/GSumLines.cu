//--------------------------------------------------------------------
// 1. This class is used to calculate the exclusive sum of a set
//    lines from each tilted image.
// 2. The exclusive sum does not include the line of given tilt
//    image.
//--------------------------------------------------------------------
#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZEY_04222016_0915 256

using namespace McAreTomo::AreTomo::CommonLine;

//--------------------------------------------------------------------
// 1. Each block holds a set of points, one  from each line.
// 2. gridDim.x is equal to iCmpSize
// 3. blockDim.y is defined bigger than iNumLines.
//--------------------------------------------------------------------
static __global__ void mGCalcSum
(	cufftComplex* gCmpLines,
	cufftComplex* gCmpSum,
	int iNumLines
)
{	__shared__ float sfReal[BLOCK_SIZEY_04222016_0915];
	__shared__ float sfImag[BLOCK_SIZEY_04222016_0915];
	sfReal[threadIdx.y] = 0.0f;
	sfImag[threadIdx.y] = 0.0f;
	__syncthreads();
	//--------------
	if(threadIdx.y >= iNumLines) return;
	int i = threadIdx.y * gridDim.x + blockIdx.x;
	sfReal[threadIdx.y] = gCmpLines[i].x;
	sfImag[threadIdx.y] = gCmpLines[i].y;
	__syncthreads();
	//--------------
	i = blockDim.y / 2;
	while(i > 0)
	{	if(threadIdx.y < i)
		{	int j = i + threadIdx.y;
			sfReal[threadIdx.y] += sfReal[j];
			sfImag[threadIdx.y] += sfImag[j];
		}
		__syncthreads();
		i /= 2;
	}
	if(threadIdx.y == 0)
	{	gCmpSum[blockIdx.x].x = sfReal[0];
		gCmpSum[blockIdx.x].y = sfImag[0];
	}
}

static __global__ void mGSubtract
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	cufftComplex* gDiff,
	int iCmpSize
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iCmpSize) return;
	gDiff[i].x = gCmp1[i].x - gCmp2[i].x;
	gDiff[i].y = gCmp1[i].y - gCmp2[i].y;
}

GSumLines::GSumLines(void)
{
	m_pCmpSum = 0L;
	m_pCmpLines = 0L;
}

GSumLines::~GSumLines(void)
{
	if(m_pCmpSum != 0L) delete[] m_pCmpSum;
	if(m_pCmpLines != 0L) delete[] m_pCmpLines;
}

void GSumLines::SetLines
(	cufftComplex* gCmpLines,
	int iNumLines,
	int iCmpSize
)
{	m_iNumLines = iNumLines;
	m_iCmpSize = iCmpSize;
	//--------------------
	if(m_pCmpLines != 0L) delete[] m_pCmpLines;
	int iSize = iNumLines * iCmpSize;
	size_t tBytes = sizeof(float) * iSize;
	m_pCmpLines = new cufftComplex[iSize];
	cudaMemcpy
	(  m_pCmpLines, gCmpLines, tBytes, 
	   cudaMemcpyDeviceToHost
	);
	//-----------------------
	cufftComplex* gCmpSum = 0L;
	tBytes = sizeof(cufftComplex) * iCmpSize;
	cudaMalloc(&gCmpSum, tBytes);
	//---------------------------
	dim3 aBlockDim(1, BLOCK_SIZEY_04222016_0915);
	dim3 aGridDim(m_iCmpSize, 1);
	mGCalcSum<<<aGridDim, aBlockDim>>>
	(  gCmpLines, gCmpSum, m_iNumLines
	); 
	//--------------------------------
	if(m_pCmpSum != 0L) delete[] m_pCmpSum;
	m_pCmpSum = new cufftComplex[m_iCmpSize];
	tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMemcpy(m_pCmpSum, gCmpSum, tBytes, cudaMemcpyDeviceToHost);
	if(gCmpSum != 0L) cudaFree(gCmpSum);	
}

cufftComplex* GSumLines::DoIt
(	int iExcludedLine
)
{	cudaMemcpyKind aH2D = cudaMemcpyHostToDevice; 
	cufftComplex* gCmpSum = 0L;
	size_t tBytes = sizeof(cufftComplex) * m_iCmpSize;
	cudaMalloc(&gCmpSum, tBytes);
	cudaMemcpy(gCmpSum, m_pCmpSum, tBytes, aH2D);
	//-------------------------------------------
	cufftComplex* pExcludedLine = m_pCmpLines 
		+ iExcludedLine * m_iCmpSize;
	cufftComplex* gExcludedLine = 0L;
	cudaMalloc(&gExcludedLine, tBytes);
	cudaMemcpy(gExcludedLine, pExcludedLine, tBytes, aH2D);
	//-----------------------------------------------------
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = m_iCmpSize / aBlockDim.x + 1;
	mGSubtract<<<aGridDim, aBlockDim>>>
	(  gCmpSum, gExcludedLine, gCmpSum, m_iCmpSize
	);
	if(gExcludedLine != 0L) cudaFree(gExcludedLine);
	return gCmpSum;
}

