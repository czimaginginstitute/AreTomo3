#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;
namespace MU = McAreTomo::MaUtil;

CFindCtfBase::CFindCtfBase(void)
{
	m_fDfMin = 0.0f;
        m_fDfMax = 0.0f;
        m_fAstAng = 0.0f;
        m_fExtPhase = 0.0f;
        m_fScore = 0.0f;
        //-----------------
        m_pCtfTheory = 0L;
        m_gfFullSpect = 0L;
        //-----------------
	m_iNthGpu = 0;
}

CFindCtfBase::~CFindCtfBase(void)
{
	this->Clean();
}

void CFindCtfBase::Clean(void)
{
	if(m_pCtfTheory != 0L) delete m_pCtfTheory;
	if(m_gfFullSpect != 0L) cudaFree(m_gfFullSpect);
	m_pCtfTheory = 0L;
	m_gfFullSpect = 0L;
}

void CFindCtfBase::Setup1(CCtfTheory* pCtfTheory)
{
	this->Clean();
	m_pCtfTheory = pCtfTheory->GetCopy();
	//-----------------
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	int iTileSize = pTsTiles->GetTileSize();
	m_aiCmpSize[1] = iTileSize;
	m_aiCmpSize[0] = m_aiCmpSize[1] / 2 + 1;
	//-----------------
	float fPixSize = m_pCtfTheory->GetPixelSize();
        m_afResRange[0] = fmin(20.0f * fPixSize, 30.0f);
	m_afResRange[1] = (2.0f * fPixSize) / 0.75f;
	//-----------------
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gfFullSpect, sizeof(float) * iCmpSize * 4);
	m_gfRawSpect = m_gfFullSpect + iCmpSize * 2;
	m_gfCtfSpect = m_gfFullSpect + iCmpSize * 3;
}

void CFindCtfBase::SetHalfSpect(float* pfCtfSpect)
{
	int iBytes = sizeof(float) * m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMemcpy(m_gfCtfSpect, pfCtfSpect, iBytes, cudaMemcpyDefault);
}

float* CFindCtfBase::GetHalfSpect(bool bRaw, bool bToHost)
{
	float* gfSpect = bRaw ? m_gfRawSpect : m_gfCtfSpect;
	if(!bToHost) return gfSpect;
	//--------------------------
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	float* pfSpect = new float[iCmpSize];
	cudaMemcpy(pfSpect, gfSpect, sizeof(float) * iCmpSize,
	   cudaMemcpyDefault);
	return pfSpect;
}

void CFindCtfBase::GetSpectSize(int* piSize, bool bHalf)
{
	piSize[0] = m_aiCmpSize[0];
	piSize[1] = m_aiCmpSize[1];
	if(!bHalf) piSize[0] = (piSize[0] - 1) * 2;
}

void CFindCtfBase::GenHalfSpectrum
(	int iTilt, 
	float fTiltOffset, 
	float fBetaOffset
)
{	CGenAvgSpectrum genAvgSpect;
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	MAM::CAlignParam* pAlnParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	float fInitDF = 0.0f, fTiltAxis = 0.0f;
	if(pCtfResults->m_iNumImgs > 0) fInitDF = pCtfResults->GetDfMean(iTilt);
	if(pAlnParam->m_iNumFrames > 0) fTiltAxis = pAlnParam->GetTiltAxis(0);
	//-----------------
	genAvgSpect.SetTiltOffsets(fTiltOffset, fBetaOffset);
	genAvgSpect.DoIt(iTilt, fTiltAxis, fInitDF, 
	   pCtfResults->m_iDfHand,
	   m_gfRawSpect, m_iNthGpu);
	//-----------------
	mRemoveBackground();
}

void CFindCtfBase::GenFullSpectrum(float* pfFullSpect)
{
	float fAstRad = m_fAstAng * 0.017453f;
	m_pCtfTheory->SetDefocus(m_fDfMin, m_fDfMax, fAstRad);
	m_pCtfTheory->SetExtPhase(m_fExtPhase, true);
	//-------------------------------------------
	CSpectrumImage spectrumImage;
	spectrumImage.DoIt(m_gfCtfSpect, m_gfRawSpect, m_aiCmpSize,
	   m_pCtfTheory, m_afResRange, m_gfFullSpect);
	//--------------------------------------------
	int iPixels = (m_aiCmpSize[0] - 1) * 2 * m_aiCmpSize[1];
	cudaMemcpy(pfFullSpect, m_gfFullSpect, iPixels * sizeof(float),
	   cudaMemcpyDefault);
}


void CFindCtfBase::ShowResult(void)
{
	char acResult[256] = {'\0'};
	sprintf(acResult, "%9.2f  %9.2f  %6.2f  %6.2f  %8.5f\n",
	   m_fDfMin, m_fDfMax, m_fAstAng, m_fExtPhase, m_fScore);
	printf("%s\n", acResult);
}

void CFindCtfBase::mRemoveBackground(void)
{
	//-----------------------------------------------
	// The box size corresponds to 30 A.
	//-----------------------------------------------
	MD::CCtfParam* pCtfParam = m_pCtfTheory->GetParam(false);
	float fMinFreq = pCtfParam->m_fPixelSize / 40.0f;
	int iBoxSize = (int)(fMinFreq * m_aiCmpSize[1]);
	iBoxSize = iBoxSize / 2 * 2 + 1;
	if(iBoxSize < 11) iBoxSize = 11;
	//---------------------------
	GRmBackground2D rmBackground;
	rmBackground.DoIt(
	   m_gfRawSpect, m_gfCtfSpect, 
	   m_aiCmpSize, iBoxSize);
}

