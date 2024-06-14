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
	m_iCoreSize = 512;
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
	float fTilt,
	float fTiltAxis,
	bool bPhaseFlip
)
{	m_pfImage = pfImage;
	m_fTilt = fTilt;
	m_fTiltAxis = fTiltAxis;
	//-----------------
	MD::CCtfResults* pCtfResults = 
	   MD::CCtfResults::GetInstance(m_iNthGpu);
	m_pImgCtfParam = pCtfResults->GetCtfParamFromTilt(m_fTilt);
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
		m_pForwardFFT->Forward(gfTile, bNorm, m_streams[0]);
		mCorrectCTF(i);
		m_pInverseFFT->Inverse(gCmpTile, m_streams[0]);
		mGpuToTile(i);
	}
	cudaStreamSynchronize(m_streams[0]);

	/* This is debugging code		
	MU::CSaveTempMrc saveMrc;
	saveMrc.SetFile("/home/shawn.zheng/szheng/Temp/TestTile", ".mrc");
	CTile* pTile = m_pExtractTiles->GetTile(0);
	int aiSize[] = {pTile->m_iPadSize, pTile->m_iTileSize};
	saveMrc.DoIt(pTile->m_pfTile, 2, aiSize);
	printf("Tile saved.\n");
	*/

	//-----------------
	for(int i=0; i<m_pExtractTiles->m_iNumTiles; i++)
	{	CTile* pTile = m_pExtractTiles->GetTile(i);
		pTile->PasteCore(m_pfImage);
	}
}

void CCorrImgCtf::mTileToGpu(int iTile)
{
	int iStream = iTile % 2;
	CTile* pTile = m_pExtractTiles->GetTile(iTile);
	size_t tBytes = pTile->GetTileBytes();
	//-----------------
	if(iStream == 1) cudaStreamSynchronize(m_streams[0]);
	cudaMemcpyAsync(m_ggfTiles[iStream], pTile->m_pfTile, tBytes,
	   cudaMemcpyDefault, m_streams[iStream]);
	if(iStream == 1) cudaStreamSynchronize(m_streams[1]);
}

void CCorrImgCtf::mGpuToTile(int iTile)
{
	CTile* pTile = m_pExtractTiles->GetTile(iTile);
	size_t tBytes = pTile->GetTileBytes();
	//-----------------
	cudaMemcpyAsync(pTile->m_pfTile, m_ggfTiles[iTile % 2],
	   tBytes, cudaMemcpyDefault, m_streams[0]);
}

float CCorrImgCtf::mCalcDeltaZ(int iTile)
{
	float afCent[2] = {0.0f};
	CTile* pTile = m_pExtractTiles->GetTile(iTile);
	pTile->GetCoreCenter(afCent);
	//-----------------
	afCent[0] -= (m_aiImgSize[0] * 0.5f);
	afCent[1] -= (m_aiImgSize[1] * 0.5f);
	//-----------------
	float fCosTx = (float)cos(m_fTiltAxis * s_fD2R);
	float fSinTx = (float)sin(m_fTiltAxis * s_fD2R);
	float fX = afCent[0] * fCosTx + afCent[1] * fSinTx;
	float fY = -afCent[0] * fSinTx + afCent[1] * fCosTx;
	//-----------------
	float fZ = fX * (float)tan(m_fTilt * s_fD2R);
	return fZ;
}

//--------------------------------------------------------------------
// 1. According to Wim Hagen, -Z moves the sample down, so you get
//    more under-focus.
// 2. In CTFFind4, CTF is -sin[...(...Cs - f) + extPhase + ampPhase].
//    Therefore, defocus (f) should be positive.
// 4. This is why we make fDeltaZ = -fDeltaZ.
//--------------------------------------------------------------------
void CCorrImgCtf::mCorrectCTF(int iTile)
{
	float fDeltaZ = mCalcDeltaZ(iTile);
	float fDeltaF = -fDeltaZ;
	//-----------------
	bool bAngstrom = true;
	float fDfMean = m_pImgCtfParam->GetDfMean(!bAngstrom);
	float fDfSigma = m_pImgCtfParam->GetDfSigma(!bAngstrom);
	float fRatio = fDfSigma / fDfMean;
	//-----------------
	fDfMean += fDeltaF;
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
