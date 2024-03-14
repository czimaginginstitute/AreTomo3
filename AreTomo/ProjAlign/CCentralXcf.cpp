#include "CProjAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::ProjAlign;

CCentralXcf::CCentralXcf(void)
{
	m_gfPadRef = 0L;
	m_gfPadImg = 0L;
	m_gfPadBuf = 0L;	
	m_iVolZ = 0;
	m_fTilt = 0.0f;
	m_fPower = 0.5f;
	m_fBFactor = 300.0f;
	m_pCufft2D = new MU::CCufft2D;
}

CCentralXcf::~CCentralXcf(void)
{
	this->Clean();
	if(m_pCufft2D != 0L) delete m_pCufft2D;
}

void CCentralXcf::Clean(void)
{
	if(m_gfPadRef != 0L) cudaFree(m_gfPadRef);
	if(m_gfPadImg != 0L) cudaFree(m_gfPadImg);
	if(m_gfPadBuf != 0L) cudaFree(m_gfPadBuf);
	m_gfPadRef = 0L;
	m_gfPadImg = 0L;
	m_gfPadBuf = 0L;
	//--------------
	m_pCufft2D->DestroyPlan();
	m_projXcf.Clean();
}

void CCentralXcf::SetupXcf(float fPower, float fBFactor)
{
	m_fPower = fPower;
	m_fBFactor = fBFactor;
}

void CCentralXcf::Setup(int* piImgSize, int iVolZ, int iNthGpu)
{
	this->Clean();
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_iVolZ = iVolZ;
	m_iNthGpu = iNthGpu;
	//-----------------
	m_aiCentSize[0] = m_aiImgSize[0];
        m_aiCentSize[1] = m_aiImgSize[1];
	m_aiPadSize[0] = (m_aiCentSize[0] / 2 + 1) * 2;
	m_aiPadSize[1] = m_aiCentSize[1];
	//-----------------
        size_t tBytes = m_aiPadSize[0] * m_aiPadSize[1] * sizeof(float);
        cudaMalloc(&m_gfPadRef, tBytes);
	cudaMalloc(&m_gfPadImg, tBytes);
	cudaMalloc(&m_gfPadBuf, tBytes);
	//-----------------
	bool bPad = true;
	int aiCmpSize[] = {m_aiPadSize[0]/2, m_aiPadSize[1]};
	m_projXcf.Setup(aiCmpSize);
	m_pCufft2D->CreateForwardPlan(m_aiCentSize, !bPad);
}

void CCentralXcf::DoIt
(	float* pfRef, 
	float* pfImg, 
	float fTilt
)
{	m_fTilt = fTilt;
	//--------------
	mGetCentral(pfRef, m_gfPadRef);	
	mGetCentral(pfImg, m_gfPadImg);
	//-----------------------------
	mNormalize(m_gfPadRef);
	mNormalize(m_gfPadImg);
	//---------------------
	mCorrelate();
}

void CCentralXcf::GetShift(float* pfShift)
{
	pfShift[0] = m_afShift[0];
	pfShift[1] = m_afShift[1];
}

void CCentralXcf::mGetCentral(float* pfImg, float* gfPadImg)
{
	size_t tBytes = sizeof(float) * m_aiCentSize[0];
	int iX = (m_aiImgSize[0] - m_aiCentSize[0]) / 2;
	int iY = (m_aiImgSize[1] - m_aiCentSize[1]) / 2;
	int iOffset = iY * m_aiImgSize[0] + iX;
	//-------------------------------------
	for(int y=0; y<m_aiCentSize[1]; y++)
	{	float* pfSrc = pfImg + y * m_aiImgSize[0] + iOffset;
		float* gfDst = gfPadImg + y * m_aiPadSize[0];
		cudaMemcpy(gfDst, pfSrc, tBytes, cudaMemcpyDefault);
	}
}

void CCentralXcf::mNormalize(float* gfPadImg)
{
	bool bPadded = true;
	MU::GCalcMoment2D aGCalcMoment2D;
	aGCalcMoment2D.SetSize(m_aiPadSize, bPadded);
	float fMean = aGCalcMoment2D.DoIt(gfPadImg, 1, true);
	//-----------------
	MU::GNormalize2D aGNorm2D;
	aGNorm2D.DoIt(gfPadImg, m_aiPadSize, bPadded, fMean, 1.0f);
	//-----------------
	CParam* pParam = CParam::GetInstance(m_iNthGpu);
	float afCent[] = {0.0f, 0.0f};
	float afMaskSize[] = {0.0f, 0.0f}; 
	afCent[0] = m_aiCentSize[0] * 0.5f;
	afCent[1] = m_aiCentSize[1] * 0.5f;
	afMaskSize[0] = m_aiCentSize[0] * pParam->m_afMaskSize[0];
	afMaskSize[1] = m_aiCentSize[1] * pParam->m_afMaskSize[1];
	//-----------------
	MU::GRoundEdge2D roundEdge;
	float fPower = 4.0f;
	roundEdge.DoIt(gfPadImg, m_aiPadSize, bPadded, fPower);
}

void CCentralXcf::mCorrelate(void)
{
	bool bNorm = true;
	m_pCufft2D->Forward(m_gfPadRef, !bNorm);
	m_pCufft2D->Forward(m_gfPadImg, !bNorm);
	//----------------------------------
	cufftComplex* gRefCmp = (cufftComplex*)m_gfPadRef;
	cufftComplex* gImgCmp = (cufftComplex*)m_gfPadImg;
	m_projXcf.DoIt(gRefCmp, gImgCmp, m_fBFactor, m_fPower);
	//-----------------------------------------------------
	bool bClean = true;
	m_projXcf.SearchPeak1();
	m_projXcf.GetShift(m_afShift, 1.0f);
}
