#include "CCorrectInc.h"
#include "../CAreTomoInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::Correct;

CFourierCropImage::CFourierCropImage(void)
{
}

CFourierCropImage::~CFourierCropImage(void)
{
}

void CFourierCropImage::Setup(int iNthGpu, int* piImgSize, float fBin)
{
	memcpy(m_aiImgSizeIn, piImgSize, sizeof(int) * 2);
	MU::GFourierResize2D::GetBinnedImgSize(piImgSize, 
	   fBin, m_aiImgSizeOut);
	//-----------------
	if(m_aiImgSizeIn[0] == m_aiImgSizeOut[0] &&
	   m_aiImgSizeIn[1] == m_aiImgSizeOut[1])
	{	m_bSameSize = true;
		return;
	}
	//-----------------
	m_bSameSize = false;
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(iNthGpu);
	m_pForward2D = pBufferPool->GetCufft2D(true);
	m_pInverse2D = pBufferPool->GetCufft2D(false);
}

void CFourierCropImage::DoPad(float* gfPadImgIn, float* gfPadImgOut)
{
	if(m_bSameSize) mCopy(gfPadImgIn, gfPadImgOut);
	else mCropFFT(gfPadImgIn, gfPadImgOut);
}

void CFourierCropImage::mCropFFT(float* gfPadImgIn, float* gfPadImgOut)
{
	m_pForward2D->CreateForwardPlan(m_aiImgSizeIn, false);
        m_pInverse2D->CreateInversePlan(m_aiImgSizeOut, false);
	//-----------------
	bool bNorm = true;
	m_pForward2D->Forward(gfPadImgIn, bNorm);
	//-----------------
	bool bNormalized = bNorm;
	int aiCmpSizeIn[] = {m_aiImgSizeIn[0] / 2 + 1, m_aiImgSizeIn[1]};
	int aiCmpSizeOut[] = {m_aiImgSizeOut[0] / 2 + 1, m_aiImgSizeOut[1]};
	//-----------------
	MU::GFourierResize2D fourierResize;
	fourierResize.DoIt((cufftComplex*)gfPadImgIn, aiCmpSizeIn, 
	   (cufftComplex*)gfPadImgOut, aiCmpSizeOut, false);
	//-----------------
	m_pInverse2D->Inverse((cufftComplex*)gfPadImgOut);
}

void CFourierCropImage::mCopy(float* gfPadImgIn, float* gfPadImgOut)
{
	int iPadX = (m_aiImgSizeIn[0] / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * iPadX;
	for(int y=0; y<m_aiImgSizeIn[1]; y++)
	{	float* gfSrc = gfPadImgIn + y * iPadX;
		float* gfDst = gfPadImgOut + y * iPadX;
		cudaMemcpy(gfDst, gfSrc, tBytes, cudaMemcpyDefault);
	}
}

