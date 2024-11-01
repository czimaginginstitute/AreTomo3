#include "CCorrectInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::Correct;

CCorrProj::CCorrProj(void)
{
	m_gfRawProj = 0L;
	m_gfCorProj = 0L;
	m_gfRetProj = 0L;
}

CCorrProj::~CCorrProj(void)
{
	this->Clean();
}

void CCorrProj::Clean(void)
{
	if(m_gfRawProj != 0L) cudaFree(m_gfRawProj);
	if(m_gfCorProj != 0L) cudaFree(m_gfCorProj);
	if(m_gfRetProj != 0L) cudaFree(m_gfRetProj);
	m_gfRawProj = 0L;
	m_gfCorProj = 0L;
	m_gfRetProj = 0L;
}

void CCorrProj::Setup
(	int* piInSize,
	bool bInPadded,
	bool bRandomFill,
	bool bFourierCrop,
	float fTiltAxis,
	float fBinning,
	int iNthGpu
)
{	this->Clean();
	memcpy(m_aiInSize, piInSize, sizeof(int) * 2);
	m_bInPadded = bInPadded;
	m_bRandomFill = bRandomFill;
	m_bFourierCrop = bFourierCrop;
	m_fBinning = fBinning;
	m_iNthGpu = iNthGpu;
	//-----------------
	if(m_fBinning == 1) m_bFourierCrop = false;
	if(m_bFourierCrop) m_bRandomFill = true;	
	//-----------------
	m_iInImgX = bInPadded ? (piInSize[0]/2 - 1) * 2 : piInSize[0];
	size_t tBytes = m_aiInSize[0] * m_aiInSize[1] * sizeof(float);
	cudaMalloc(&m_gfRawProj, tBytes);
	//-------------------------------
	int aiInSize[] = {m_iInImgX, m_aiInSize[1]};
	CCorrectUtil::CalcAlignedSize(aiInSize, fTiltAxis, m_aiCorSize);
	m_aiCorSize[0] = (m_aiCorSize[0] / 2 + 1) * 2;
	//--------------------------------------------
	tBytes = sizeof(float) * m_aiCorSize[0] * m_aiCorSize[1];
	cudaMalloc(&m_gfCorProj, tBytes);
	//-------------------------------
	m_aGCorrPatchShift.SetSizes(m_aiInSize, m_bInPadded, 
	   m_aiCorSize, true, 0);
	if(m_fBinning <= 1) return;
	//-------------------------
	if(m_bFourierCrop)
	{	int aiImgSize[] = {1, m_aiCorSize[1]};
		aiImgSize[0] = (m_aiCorSize[0] / 2 - 1) * 2;
		m_aFFTCropImg.Setup(iNthGpu, aiImgSize, m_fBinning);
		MU::GFourierResize2D::GetBinnedImgSize(m_aiCorSize, 
		   m_fBinning, m_aiRetSize);
		m_aiRetSize[0] = (m_aiRetSize[0] / 2 + 1) * 2;
	}
	else if(m_fBinning >= 2)
	{	bool bPadded = true;
		int iBin = (int)(m_fBinning + 0.5f);
		m_aGBinImg2D.SetupBinning(m_aiCorSize, bPadded, 
		   iBin, bPadded);
		MAU::GBinImage2D::GetBinSize(m_aiCorSize, bPadded, 
		   iBin, m_aiRetSize, bPadded);
	}
	tBytes = sizeof(float) * m_aiRetSize[0] * m_aiRetSize[1];
	cudaMalloc(&m_gfRetProj, tBytes);
}

void CCorrProj::SetProj(float* pfInProj)
{
	size_t tBytes = sizeof(float) * m_aiInSize[0] * m_aiInSize[1];
	cudaMemcpy(m_gfRawProj, pfInProj, tBytes, cudaMemcpyDefault);
}

void CCorrProj::DoIt
(	float* pfGlobalShift,
	float fTiltAxis
)
{	m_aGCorrPatchShift.DoIt(m_gfRawProj, pfGlobalShift, 
	   fTiltAxis, 0L, m_bRandomFill, m_gfCorProj);
	//--------------------------------------------
	if(m_fBinning == 1) return;
	if(m_bFourierCrop) m_aFFTCropImg.DoPad(m_gfCorProj, m_gfRetProj);
	else m_aGBinImg2D.DoIt(m_gfCorProj, m_gfRetProj);
}

void CCorrProj::GetProj(float* pfCorProj, int* piSize, bool bPadded)
{
	int iSizeX = piSize[0];
	if(bPadded) iSizeX = (piSize[0] / 2 - 1) * 2;
	size_t tBytes = sizeof(float) * iSizeX;
	//-------------------------------------
	float* gfSrc = m_gfRetProj;
	int iSrcX = m_aiRetSize[0];
	if(m_fBinning == 1)
	{	gfSrc = m_gfCorProj;
		iSrcX = m_aiCorSize[0];
	}
	//-----------------------------
	for(int y=0; y<piSize[1]; y++)
	{	float* pfDst = pfCorProj + y * piSize[0];
		cudaMemcpy(pfDst, gfSrc + y * iSrcX, tBytes,
			cudaMemcpyDefault);
	}
}

