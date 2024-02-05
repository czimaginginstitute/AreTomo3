#include "CCorrectInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::Correct;
namespace MAU = McAreTomo::AreTomo::Util;

CBinStack::CBinStack(void)
{
}

CBinStack::~CBinStack(void)
{
}

MD::CTiltSeries* CBinStack::DoReal
(	MD::CTiltSeries* pTiltSeries, 
	int iBin, int iNthGpu
)
{	bool bPadded = true;
	int aiOutSize[] = {0, 0, pTiltSeries->m_aiStkSize[2]};
	MAU::GBinImage2D::GetBinSize(pTiltSeries->m_aiStkSize, !bPadded,
	   iBin, aiOutSize, !bPadded); 
	//-----------------
	MD::CTiltSeries* pBinSeries = new MD::CTiltSeries;
	pBinSeries->Create(aiOutSize);
	//-----------------
	MAU::GBinImage2D aGBinImg2D;
	aGBinImg2D.SetupBinning(pTiltSeries->m_aiStkSize, !bPadded, 
	   iBin, !bPadded); 
	//-----------------
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(iNthGpu);
	MD::CStackBuffer* pTmpBuf = pBufPool->GetBuffer(MD::EBuffer::tmp);
	float* gfInImg = (float*)pTmpBuf->GetFrame(0);
	float* gfOutImg = (float*)pTmpBuf->GetFrame(1);
	//-----------------
	size_t tInBytes = sizeof(float) * pTiltSeries->GetPixels();
	size_t tOutBytes = sizeof(float) * pBinSeries->GetPixels();
	//-----------------
	for(int i=0; i<pBinSeries->m_aiStkSize[2]; i++)
	{	float* pfProj = (float*)pTiltSeries->GetFrame(i);
		cudaMemcpy(gfInImg, pfProj, tInBytes, cudaMemcpyDefault);
		//----------------
		aGBinImg2D.DoIt(gfInImg, gfOutImg);
		float* pfBinProj = (float*)pBinSeries->GetFrame(i);
		cudaMemcpy(pfBinProj, gfOutImg, tOutBytes, cudaMemcpyDefault);
	}
	return pBinSeries;
}

MD::CTiltSeries* CBinStack::DoFFT
(	MD::CTiltSeries* pTiltSeries, 
	float fBin,
	int iNthGpu
)
{	int aiOutSize[] = {0, 0, pTiltSeries->m_aiStkSize[2]};
	MU::GFourierResize2D::GetBinnedImgSize(pTiltSeries->m_aiStkSize,
	   fBin, aiOutSize);
	//-----------------
	MD::CTiltSeries* pBinSeries = new MD::CTiltSeries;
	pBinSeries->Create(aiOutSize);
	//-----------------
	int aiPadSizeIn[] = {0, pTiltSeries->m_aiStkSize[1]};
	int aiCmpSizeIn[] = {0, pTiltSeries->m_aiStkSize[1]};
	aiPadSizeIn[0] = (pTiltSeries->m_aiStkSize[0] / 2 + 1) * 2;
	aiCmpSizeIn[0] = aiPadSizeIn[0] / 2;
	//-----------------
	int aiPadSizeOut[] = {0, pBinSeries->m_aiStkSize[1]};
	int aiCmpSizeOut[] = {0, pBinSeries->m_aiStkSize[1]};
	aiPadSizeOut[0] = (pBinSeries->m_aiStkSize[0] / 2 + 1) * 2;
	aiCmpSizeOut[0] = aiPadSizeOut[0] / 2;
	//-----------------
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(iNthGpu);
	MD::CStackBuffer* pTmpBuf = pBufPool->GetBuffer(MD::EBuffer::tmp);
	cufftComplex* gCmpImgIn = pTmpBuf->GetFrame(0);
	cufftComplex* gCmpImgOut = pTmpBuf->GetFrame(1);
	//-----------------
	MU::CCufft2D* pForward2D = pBufPool->GetCufft2D(true);
	MU::CCufft2D* pInverse2D = pBufPool->GetCufft2D(false);
	pForward2D->CreateForwardPlan(pTiltSeries->m_aiStkSize, false);
	pInverse2D->CreateInversePlan(pBinSeries->m_aiStkSize, false);
	//-----------------
	MU::CPad2D pad2D; 
	MU::GFourierResize2D fftResize2D;
	//-----------------
	for(int i=0; i<pBinSeries->m_aiStkSize[2]; i++)
	{	float* pfImgIn = (float*)pTiltSeries->GetFrame(i);
		pad2D.Pad(pfImgIn, pTiltSeries->m_aiStkSize, 
		   (float*)gCmpImgIn);
		pForward2D->Forward((float*)gCmpImgIn, true);
		//----------------
		fftResize2D.DoIt(gCmpImgIn, aiCmpSizeIn, 
		   gCmpImgOut, aiCmpSizeOut, false);
		pInverse2D->Inverse(gCmpImgOut);
		//----------------
		float* pfImgOut = (float*)pBinSeries->GetFrame(i);
		pad2D.Unpad((float*)gCmpImgOut, aiPadSizeOut, pfImgOut);
	}
	return pBinSeries;
}
