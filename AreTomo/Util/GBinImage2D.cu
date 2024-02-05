#include "CUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::Util;

static __device__ __constant__ int giInSize[2];
static __device__ __constant__ int giOutSize[2];

static __global__ void mGBinImage
(	float* gfInImg,
	int iBinX,
	int iBinY,
	float* gfOutImg
)
{	int y =  blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= giOutSize[1]) return;
	int i = y * giOutSize[0] + blockIdx.x;
	gfOutImg[i] = (float)-1e30;
	//-------------------------
	int x =  blockIdx.x * iBinX;
	y = y * iBinY;
	float fSum = 0.0f;
	for(int iy=0; iy<iBinY; iy++)
	{	float* pfPtr = gfInImg + (y + iy) * giInSize[0];
		for(int ix=0; ix<iBinX; ix++)
		{	float fVal = pfPtr[x + ix];
			if(fVal < (float)-1e10) return;
			else fSum += fVal;
		}
	}
	gfOutImg[i] = fSum;
}

void GBinImage2D::GetBinSize
(	int* piInSize, 
	bool bInPadded,
	int* piBinning,
	int* piOutSize,
	bool bOutPadded
)
{	int iImgX = piInSize[0];
	if(bInPadded) iImgX = (piInSize[0] / 2 - 1) * 2;
	//----------------------------------------------
	piOutSize[0] = iImgX / piBinning[0] / 2 * 2;
	piOutSize[1] = piInSize[1] / piBinning[1] / 2 * 2;
	if(bOutPadded) piOutSize[0] += 2;
}

void GBinImage2D::GetBinSize
(	int* piInSize, bool bInPadded, int iBinning,
	int* piOutSize, bool bOutPadded
)
{	int aiBinning[] = {iBinning, iBinning};
	GBinImage2D::GetBinSize(piInSize, bInPadded, aiBinning,
		piOutSize, bOutPadded);
}

GBinImage2D::GBinImage2D(void)
{
}

GBinImage2D::~GBinImage2D(void)
{
}

void GBinImage2D::SetupBinnings
(	int* piInSize,
	bool bInPadded,
	int* piBinning,
	bool bOutPadded
)
{	cudaMemcpyToSymbol(giInSize, piInSize, sizeof(giInSize));
	memcpy(m_aiBinning, piBinning, sizeof(m_aiBinning));
	//--------------------------------------------------
	GBinImage2D::GetBinSize(piInSize, bInPadded, piBinning, 
		m_aiOutSize, bOutPadded);
	cudaMemcpyToSymbol(giOutSize, m_aiOutSize, sizeof(giOutSize));
	//------------------------------------------------------------
	m_iOutImgX = m_aiOutSize[0];
	if(bOutPadded) m_iOutImgX = (m_aiOutSize[0] / 2 - 1) * 2;
}

void GBinImage2D::SetupBinning
(	int* piInSize,
	bool bInPadded,
	int iBinning,
	bool bOutPadded
)
{	int aiBinning[] = {iBinning, iBinning};
	this->SetupBinnings(piInSize, bInPadded, aiBinning, bOutPadded);
}

void GBinImage2D::SetupSizes
(	int* piInSize, bool bInPadded,
	int* piOutSize, bool bOutPadded
)
{	int iBytes = sizeof(int) * 2;
	cudaMemcpyToSymbol(giInSize, piInSize, iBytes);
	cudaMemcpyToSymbol(giOutSize, piOutSize, iBytes);
	memcpy(m_aiOutSize, piOutSize, iBytes);
	//-------------------------------------
	int iInImgX = piInSize[0];
	m_iOutImgX = piOutSize[0];
	if(bInPadded) iInImgX = (piInSize[0] / 2 - 1) * 2;
	if(bOutPadded) m_iOutImgX = (piOutSize[0] / 2 - 1) * 2;
	//-----------------------------------------------------
	m_aiBinning[0] = iInImgX / m_iOutImgX;
	m_aiBinning[1] = piInSize[1] / piOutSize[1];
}

void GBinImage2D::DoIt
(	float* gfInImg, 
	float* gfOutImg,
	cudaStream_t stream
)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_iOutImgX, 1);
	aGridDim.y = (m_aiOutSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//----------------------------------------------------------
	mGBinImage<<<aGridDim, aBlockDim, 0, stream>>>(gfInImg, 
	   m_aiBinning[0], m_aiBinning[1], gfOutImg);
}
