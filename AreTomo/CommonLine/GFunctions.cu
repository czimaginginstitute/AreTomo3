#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::CommonLine;

static __global__ void mGSumFloat
(	float* gfData1,
	float* gfData2,
	float fFact1,
	float fFact2,
	float* gfSum,
	int iSize
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= iSize || gfSum == 0L) return;
	//-----------------------------------
	if(gfData1 != 0L && gfData2 != 0L)
	{	gfSum[i] =  gfData1[i] * fFact1 + gfData2[i] * fFact2;
	}
	else if(gfData1 != 0L)
	{	gfSum[i] = gfData1[i] * fFact1;
	}
	else if(gfData2 != 0L)
	{	gfSum[i] = gfData2[i] * fFact2;
	}
}

static __global__ void mGSumCmp
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	float fFact1,
	float fFact2,
	cufftComplex* gSum,
	int iCmpSize
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iCmpSize || gSum == 0L) return;
	//-------------------------------------
	if(gCmp1 != 0L && gCmp2 != 0L)
	{	gSum[i].x = gCmp1[i].x * fFact1 + gCmp2[i].x * fFact2;
		gSum[i].y = gCmp1[i].y * fFact1 + gCmp2[i].y * fFact2;
	}
	else if(gCmp1 != 0L)
	{	gSum[i].x = gCmp1[i].x * fFact1;
		gSum[i].y = gCmp1[i].y * fFact1;
	}
	else if(gCmp2 != 0L)
	{	gSum[i].x = gCmp2[i].x * fFact2;
		gSum[i].y = gCmp2[i].y * fFact2;
	}
}

GFunctions::GFunctions(void)
{
}

GFunctions::~GFunctions(void)
{
}

void GFunctions::Sum
(	float* gfData1,
	float* gfData2,
	float fFact1,
	float fFact2,
	float* gfSum,
	int iSize
)
{	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = iSize / aBlockDim.x + 1;
	mGSumFloat<<<aGridDim, aBlockDim>>>
	(  gfData1, gfData2, 
	   fFact1, fFact2,
	   gfSum, iSize
	);
}

void GFunctions::Sum
(	cufftComplex* gCmp1,
	cufftComplex* gCmp2,
	float fFact1,
	float fFact2,
	cufftComplex* gSum,
	int iCmpSize	
)
{	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = iCmpSize / aBlockDim.x + 1;
	mGSumCmp<<<aGridDim, aBlockDim>>>
	(  gCmp1, gCmp2, 
	   fFact1, fFact2,
	   gSum, iCmpSize
	);
}

