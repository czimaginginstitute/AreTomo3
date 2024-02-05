#include "CMaUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::MaUtil;

static __global__ void mGRoundEdge
(	float* gfData, 
	int iSize,
	float fMaskCenter,
	int iMaskSize
)	
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iSize) return;
	//--------------------
	float fR = fabsf(i - fMaskCenter) / iMaskSize;
	if(fR >= 1.0f)
	{	gfData[i] = 0.0f;
	}
	else
	{	fR = 0.5f * (1 - cosf(3.1415926f * fR));
		fR = 1.0f - powf(fR, 3.0f);
		gfData[i] *= fR;
	}
}


GRoundEdge1D::GRoundEdge1D(void)
{
}

GRoundEdge1D::~GRoundEdge1D(void)
{
}

void GRoundEdge1D::DoIt(float* gfData, int iSize)
{
	float fMaskCenter = iSize * 0.5f;
	int iMaskSize = iSize / 2;
	//------------------------
	dim3 aBlockDim(512, 1);
	int iGridX = iSize / aBlockDim.x + 1;
	dim3 aGridDim(iGridX, 1);
	//-----------------------
	mGRoundEdge<<<aGridDim, aBlockDim>>>
	(  gfData, iSize,
	   fMaskCenter, iMaskSize
	);
}

void GRoundEdge1D::DoPad(float* gfPadData, int iPadSize)
{
	int iSize = (iPadSize / 2 - 1) * 2;
	this->DoIt(gfPadData, iSize);
}
