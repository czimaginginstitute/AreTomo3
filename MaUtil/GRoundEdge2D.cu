#include "CMaUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::MaUtil;

static __global__ void mGRoundEdge2D
(	float* gfImg, int iPadX, int iSizeY,
	float fMaskCentX, float fMaskCentY,
	float fMaskSizeX, float fMaskSizeY,
	float fPower
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * iPadX + blockIdx.x;
	if(gfImg[i] < (float)-1e10)
	{	gfImg[i] = 0.0f;
		return;
	}
	//-------------
	float fX = 2 * fabsf(blockIdx.x - fMaskCentX) / fMaskSizeX;
	float fY = 2 * fabsf(y - fMaskCentY) / fMaskSizeY;
	float fR = sqrtf(fX * fX + fY * fY);
	if(fR >= 1.0f)
	{	gfImg[i] = 0.0f;
		return;
	}
	//-------------
	fR = 0.5f * (1 - cosf(3.1415926f * fR));
	fR = 1.0f - powf(fR, fPower);
	gfImg[i] = gfImg[i] * fR;
}

static __global__ void mGKeepCenter
(	float* gfImg, int iPadX, int iSizeY,
	float fMaskCentX, float fMaskCentY,
	float fMaskSizeX, float fMaskSizeY,
	float fPower
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int i = y * iPadX + blockIdx.x;
	if(gfImg[i] < (float)-1e10)
	{	gfImg[i] = 0.0f;
		return;
	}
	//-----------------
	float fX = 2 * fabsf(blockIdx.x - fMaskCentX) / fMaskSizeX;
	float fY = 2 * fabsf(y - fMaskCentY) / fMaskSizeY;
	if(fX <= 1.05f || fY <= 1.05f) return;
	//-----------------
        float fR = sqrtf((fX - 1.05f) * (fY - 1.05f));
	fR = 0.5f * (1 - cosf(3.1415926f * fR));
	fR = 1.0f - powf(fR, fPower);
	gfImg[i] *= fR;
}

GRoundEdge2D::GRoundEdge2D(void)
{
	memset(m_afMaskCent, 0, sizeof(m_afMaskCent));
	memset(m_afMaskSize, 0, sizeof(m_afMaskSize));
}

GRoundEdge2D::~GRoundEdge2D(void)
{
}

void GRoundEdge2D::SetMask(float* pfCent, float* pfSize)
{
	m_afMaskCent[0] = pfCent[0];
	m_afMaskCent[1] = pfCent[1];
	m_afMaskSize[0] = pfSize[0];
	m_afMaskSize[1] = pfSize[1];
}

void GRoundEdge2D::DoIt
(	float* gfImg, int* piSize, bool bPadded,
	float fPower, cudaStream_t stream
)
{	int iImgX = bPadded ? (piSize[0]/2 - 1) * 2 : piSize[0];
	if(m_afMaskCent[0] == 0 || m_afMaskCent[1] == 0)
	{	m_afMaskCent[0] = 0.5f * iImgX;
		m_afMaskCent[1] = 0.5f * piSize[1];
	}
	if(m_afMaskSize[0] == 0 || m_afMaskSize[1] == 0)
	{	m_afMaskSize[0] = 1.0f * iImgX;
		m_afMaskSize[1] = 1.0f * piSize[1];
	}
	//-----------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iImgX, (piSize[1] + aBlockDim.y - 1) / aBlockDim.y);
	//-----------------
	mGRoundEdge2D<<<aGridDim, aBlockDim, 0, stream>>>(gfImg, 
	   piSize[0], piSize[1], m_afMaskCent[0], m_afMaskCent[1],
	   m_afMaskSize[0], m_afMaskSize[1], fPower);
}

void GRoundEdge2D::DoKeepCenter
(	float* gfImg, int* piSize, bool bPadded,
	float fPower, cudaStream_t stream
)
{	int iImgX = bPadded ? (piSize[0]/2 - 1) * 2 : piSize[0];
	if(m_afMaskCent[0] == 0 || m_afMaskCent[1] == 0)
	{	m_afMaskCent[0] = 0.5f * iImgX;
		m_afMaskCent[1] = 0.5f * piSize[1];
	}
	if(m_afMaskSize[0] == 0 || m_afMaskSize[1] == 0)
	{	m_afMaskSize[0] = 1.0f * iImgX;
		m_afMaskSize[1] = 1.0f * piSize[1];
	}
	//-----------------
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(iImgX, (piSize[1] + aBlockDim.y - 1) / aBlockDim.y);
	//-----------------
	mGKeepCenter<<<aGridDim, aBlockDim, 0, stream>>>(gfImg,
	   piSize[0], piSize[1], m_afMaskCent[0], m_afMaskCent[1],
	   m_afMaskSize[0], m_afMaskSize[1], fPower);
}

