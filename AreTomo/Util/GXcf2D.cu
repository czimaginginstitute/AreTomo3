#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Util;

static __global__ void mGConv
(	cufftComplex* gComp1, 
	cufftComplex* gComp2, 
	int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	float fRe, fIm;
	fRe = gComp1[i].x * gComp2[i].x + gComp1[i].y * gComp2[i].y;
	fIm = gComp1[i].x * gComp2[i].y - gComp1[i].y * gComp2[i].x;
	//-----------------------------------------------------------
	gComp2[i].x = fRe;
	gComp2[i].y = fIm;
}

static __global__ void mGWiener
(	cufftComplex* gComp,
	float fBFactor,
	int iCmpY
)
{       int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
        //---------------------------------
	if(y > (iCmpY / 2)) y -= iCmpY;
	float fNx = 2.0f * (gridDim.x - 1);
        float fFilt = -2.0f * fBFactor /(fNx * fNx + iCmpY * iCmpY);
        fFilt = expf(fFilt * (blockIdx.x * blockIdx.x + y * y));
	//------------------------------------------------------
	float fAmp = sqrtf(gComp[i].x * gComp[i].x 
		+ gComp[i].y * gComp[i].y);
	fFilt /= sqrtf(fAmp + 0.01f);
	gComp[i].x *= fFilt;
	gComp[i].y *= fFilt;
}


static __global__ void mGCenterOrigin(cufftComplex* gComp, int iCmpY)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	int iSign = ((blockIdx.x + y) % 2 == 0) ? 1 : -1;
	gComp[i].x *= iSign;
	gComp[i].y *= iSign;
}


GXcf2D::GXcf2D(void)
{
	m_fBFactor = 100.0f;
	m_pfXcfImg = 0L;
	memset(m_aiXcfSize, 0, sizeof(m_aiXcfSize));
}

GXcf2D::~GXcf2D(void)
{
	this->Clean();
}

void GXcf2D::Clean(void)
{
	if(m_pfXcfImg != 0L) delete[] m_pfXcfImg;
	m_pfXcfImg = 0L;
}

void GXcf2D::Setup(MU::CCufft2D* pCufft2D, int* piCmpSize)
{
	this->Clean();
	//-----------------
	m_aiXcfSize[0] = (piCmpSize[0] - 1) * 2;
	m_aiXcfSize[1] = piCmpSize[1];
	//-----------------
	m_pCufft2D = pCufft2D;
	m_pCufft2D->CreateInversePlan(piCmpSize, true);
	//-----------------
	int iPixels = m_aiXcfSize[0] * m_aiXcfSize[1];
	m_pfXcfImg = new float[iPixels];
}

void GXcf2D::DoIt
(	cufftComplex* gCmp1, 
	cufftComplex* gCmp2,
	float fBFactor
)
{	m_fBFactor = fBFactor;
	//-----------------
	int aiCmpSize[] = {m_aiXcfSize[0]/2 + 1, m_aiXcfSize[1]};
	dim3 aBlockDim(1, 64);
	dim3 aGridDim(aiCmpSize[0], aiCmpSize[1]/aBlockDim.y + 1);
        mGConv<<<aGridDim, aBlockDim>>>(gCmp1, gCmp2, aiCmpSize[1]);
	//-----------------
	mGWiener<<<aGridDim, aBlockDim>>>
	( gCmp2, m_fBFactor, aiCmpSize[1]
	);
	//-----------------
	mGCenterOrigin<<<aGridDim, aBlockDim>>>
	( gCmp2, aiCmpSize[1]
	);
        //-------------------
        m_pCufft2D->Inverse(gCmp2);
	//---------------------
	size_t tBytes = sizeof(float) * m_aiXcfSize[0];
	for(int y=0; y<m_aiXcfSize[1]; y++)
	{	float* pfDst = m_pfXcfImg + y * m_aiXcfSize[0];
		float* gfSrc = (float*)(gCmp2 + y * aiCmpSize[0]);
		cudaMemcpy(pfDst, gfSrc, tBytes, cudaMemcpyDefault);
	}
}

float GXcf2D::SearchPeak(void)
{
	int aiSeaSize[2] = {0};
	aiSeaSize[0] = m_aiXcfSize[0] * 8 / 20 * 2;
	aiSeaSize[1] = m_aiXcfSize[1] * 8 / 20 * 2;
	bool bPadded = true;
	m_peak2D.DoIt(m_pfXcfImg, m_aiXcfSize, !bPadded, aiSeaSize);
	m_afShift[0] = m_peak2D.m_afShift[0];
	m_afShift[1] = m_peak2D.m_afShift[1];
	return m_peak2D.m_fPeakInt;
}

float* GXcf2D::GetXcfImg(bool bClean)
{
	float* pfXcfImg = m_pfXcfImg;
	if(bClean) m_pfXcfImg = 0L;
	return pfXcfImg;
}

void GXcf2D::GetShift(float* pfShift, float fXcfBin)
{
	pfShift[0] = m_afShift[0] * fXcfBin;
	pfShift[1] = m_afShift[1] * fXcfBin;
}
