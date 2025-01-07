#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.0174532f;
//------------------------------
// Debugging code
//------------------------------
//static CSaveImages s_aSaveImages;
//static int s_iCount = 0;

//--------------------------------------------------------------------
// 1. making core size smaller than tilt size helps reducing the
//    checkerboard effect when we keep particles dark by flipping
//    the negative phase.
// 2. However, when core size is smaller than tile size, thin dark
//    cicles are seen at CTF zeroes.
// 3. When core and tile has the same size, there is no dark circles
//    in CTF deconvolved image.
//--------------------------------------------------------------------   
CCorrImgCtf::CCorrImgCtf(void)
{
	m_iTileSize = 512;
	m_iCoreSize = 256;
	m_pExtractTiles = new CExtractTiles;
	m_pGCorrCTF2D = new GCorrCTF2D;
}

CCorrImgCtf::~CCorrImgCtf(void)
{
	if(m_pExtractTiles != 0L) delete m_pExtractTiles;
	if(m_pGCorrCTF2D != 0L) delete m_pGCorrCTF2D;
}

void CCorrImgCtf::SetLowpass(int iBFactor)
{
	m_iBFactor = iBFactor;
}

void CCorrImgCtf::Setup(int* piImgSize, int iNthGpu)
{
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_iNthGpu = iNthGpu;
	//-----------------
	m_pExtractTiles->Setup(m_iTileSize, m_iCoreSize,  m_aiImgSize);
	//-----------------
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(iNthGpu);
	MD::CStackBuffer* pTmpBuffer = pBufPool->GetBuffer(MD::EBuffer::tmp);
	//-----------------
	int iPadSize = (m_iTileSize / 2 + 1) * 2 * m_iTileSize;
	float* gfFrame = (float*)pTmpBuffer->GetFrame(0);
	m_ggfTiles[0] = gfFrame;
	m_ggfTiles[1] = &gfFrame[iPadSize];
	//-----------------
	m_streams[0] = pBufPool->GetCudaStream(0);
	m_streams[1] = pBufPool->GetCudaStream(1);
	//-----------------
	bool bForward = true;
	m_pForwardFFT = pBufPool->GetCufft2D(bForward);
	m_pInverseFFT = pBufPool->GetCufft2D(!bForward);
	//-----------------
	int aiFFTSize[] = {m_iTileSize, m_iTileSize};
	m_pForwardFFT->CreateForwardPlan(aiFFTSize, false);
	m_pInverseFFT->CreateInversePlan(aiFFTSize, false);
}

void CCorrImgCtf::DoIt
(	float* pfImage,	
	float fTilt, float fTiltAxis,
	float fAlpha0, float fBeta0,
	bool bPhaseFlip
)
{	m_pfImage = pfImage;
	m_fTilt = fTilt;
	m_fTiltAxis = fTiltAxis;
	m_fAlpha0 = fAlpha0;
	m_fBeta0 = fBeta0;
	//-----------------
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	m_pImgCtfParam = pCtfResults->GetCtfParamFromTilt(m_fTilt);
	m_iDfHand = pCtfResults->m_iDfHand;
	//-----------------
	m_pGCorrCTF2D->SetParam(m_pImgCtfParam);
	m_pGCorrCTF2D->SetPhaseFlip(bPhaseFlip);
	m_pGCorrCTF2D->SetLowpass(m_iBFactor);
	//-----------------
	m_pExtractTiles->DoIt(m_pfImage);
	//-----------------
	bool bNorm = true;
	for(int i=0; i<m_pExtractTiles->m_iNumTiles; i++)
	{	float* gfTile = m_ggfTiles[i % 2];
		cufftComplex* gCmpTile = (cufftComplex*)gfTile;
		//----------------
		mTileToGpu(i);
		mRoundEdge(i);
		m_pForwardFFT->Forward(gfTile, bNorm, m_streams[0]);
		mCorrectCTF(i);
		m_pInverseFFT->Inverse(gCmpTile, m_streams[0]);
		mGpuToTile(i);
	}
	cudaStreamSynchronize(m_streams[0]);
	//-----------------
	/* Debugging code
	MU::CSaveTempMrc saveMrc;
        saveMrc.SetFile("/home/shawn.zheng/szheng/Temp/TestTile", ".mrc");
        CCoreTile* pTile = m_pExtractTiles->GetTile(0);
        saveMrc.DoIt(pTile->GetTile(), 2, pTile->GetSize());
        printf("Tile saved.\n");
	*/
	//-----------------
	for(int i=0; i<m_pExtractTiles->m_iNumTiles; i++)
	{	CCoreTile* pTile = m_pExtractTiles->GetTile(i);
		pTile->PasteCore(m_pfImage, m_aiImgSize);
	}
}

void CCorrImgCtf::mTileToGpu(int iTile)
{
	int iStream = iTile % 2;
	CCoreTile* pTile = m_pExtractTiles->GetTile(iTile);
	size_t tBytes = pTile->GetTileBytes();
	//-----------------
	if(iStream == 1) cudaStreamSynchronize(m_streams[0]);
	cudaMemcpyAsync(m_ggfTiles[iStream], pTile->GetTile(), tBytes,
	   cudaMemcpyDefault, m_streams[iStream]);
	if(iStream == 1) cudaStreamSynchronize(m_streams[1]);
}

void CCorrImgCtf::mRoundEdge(int iTile)
{
	GRoundEdge roundEdge;
	CCoreTile* pTile = m_pExtractTiles->GetTile(iTile);
	float fCoreSize = (float)pTile->GetCoreSize();
	float afMaskSize[] = {fCoreSize, fCoreSize};
	float afMaskCenter[2] = {0.0f};
	pTile->GetCoreCenterInTile(afMaskCenter);
	roundEdge.SetMask(afMaskCenter, afMaskSize);
	//-----------------
	int aiTileSize[] = {(m_iTileSize / 2 + 1) * 2, m_iTileSize};
	float* gfTile = m_ggfTiles[iTile % 2];
	bool bKeepCenter = true;
	roundEdge.DoIt(gfTile, aiTileSize, bKeepCenter);
}

void CCorrImgCtf::mGpuToTile(int iTile)
{
	CCoreTile* pTile = m_pExtractTiles->GetTile(iTile);
	size_t tBytes = pTile->GetTileBytes();
	//-----------------
	cudaMemcpyAsync(pTile->GetTile(), m_ggfTiles[iTile % 2],
	   tBytes, cudaMemcpyDefault, m_streams[0]);
}

float CCorrImgCtf::mCalcDeltaZ(int iTile)
{
	float afCent[2] = {0.0f};
	CCoreTile* pTile = m_pExtractTiles->GetTile(iTile);
	pTile->GetCoreCenter(afCent);
	//-----------------
	afCent[0] -= (m_aiImgSize[0] * 0.5f);
	afCent[1] -= (m_aiImgSize[1] * 0.5f);
	//-----------------
	CTiltInducedZ tiltInducedZ;
	tiltInducedZ.Setup(m_fTilt, m_fTiltAxis, m_fAlpha0, m_fBeta0);
	float fDeltaZ = tiltInducedZ.DoIt(afCent[0], afCent[1]);
	return fDeltaZ;
}

//--------------------------------------------------------------------
// Defocus handedness has been determined and encoded into tilt axis.
// No need to take it into accound again.
//--------------------------------------------------------------------
void CCorrImgCtf::mCorrectCTF(int iTile)
{
	float fDeltaZ = mCalcDeltaZ(iTile);
	//-----------------
	bool bAngstrom = true;
	float fDfMean = m_pImgCtfParam->GetDfMean(!bAngstrom);
	float fDfSigma = m_pImgCtfParam->GetDfSigma(!bAngstrom);
	float fRatio = fDfSigma / fDfMean;
	//-----------------
	fDfMean = fDfMean + fDeltaZ; 
	fDfSigma = fDfMean * fRatio;
	float fDfMin = fDfMean - fDfSigma;
	float fDfMax = fDfMean + fDfSigma;
	//-----------------
	int aiCmpSize[] = {m_iTileSize / 2 + 1, m_iTileSize};
	cufftComplex* gCmp = (cufftComplex*)m_ggfTiles[iTile % 2];	
	//-----------------
	m_pGCorrCTF2D->DoIt(fDfMin, fDfMax, m_pImgCtfParam->m_fAstAzimuth,
	   m_pImgCtfParam->m_fExtPhase, m_fTilt, 
	   gCmp, aiCmpSize, m_streams[0]);
}
