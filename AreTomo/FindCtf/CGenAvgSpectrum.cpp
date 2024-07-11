#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.0174532f;

CGenAvgSpectrum::CGenAvgSpectrum(void)
{
	m_fTiltOffset = 0.0f;
	m_fBetaOffset = 0.0f;
	m_fTiltAxis = 0.0f;
}

CGenAvgSpectrum::~CGenAvgSpectrum(void)
{
	this->Clean();
}

void CGenAvgSpectrum::Clean(void)
{
}

void CGenAvgSpectrum::SetTiltOffsets(float fTiltOffset, float fBetaOffset)
{
	m_fTiltOffset = fTiltOffset;
	m_fBetaOffset = fBetaOffset;
}

void CGenAvgSpectrum::DoIt
(	int iTilt, 
	float fTiltAxis,
	float fCentDF,     // in Angstrom
	float* gfAvgSpect,
	int iNthGpu
)
{	m_iTilt = iTilt;
	m_fTiltAxis = fTiltAxis;
	m_fCentDF = fCentDF;
	m_gfAvgSpect = gfAvgSpect;
	m_iNthGpu = iNthGpu;
	//-----------------
	if(m_fCentDF < 0.001f) mDoNoScaling();
	else mDoScaling();
}

void CGenAvgSpectrum::mDoNoScaling(void)
{
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	int iImgTiles = pTsTiles->GetImgTiles();
	//-----------------
	CTile* pTile = pTsTiles->GetTile(m_iTilt, 0);
	int* piTileSize = pTile->GetTileSize();
	//-----------------
	MU::GAddFrames addFrames;
	float fFactor2 = 1.0f / iImgTiles;
	//-----------------
	int iTileSize = piTileSize[0] * piTileSize[1];
	cudaMemset(m_gfAvgSpect, 0, sizeof(float) * iTileSize); 
	//-----------------
	for(int i=0; i<iImgTiles; i++)
	{	pTile = pTsTiles->GetTile(m_iTilt, i);
		float* qfTile = pTile->GetTile();
		addFrames.DoIt(m_gfAvgSpect, 1.0f, qfTile,
		   fFactor2, m_gfAvgSpect, piTileSize);
	}
}

void CGenAvgSpectrum::mDoScaling(void)
{
	mCalcTileCentZs();
	//-----------------
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	int iImgTiles = pTsTiles->GetImgTiles();
	//-----------------
	CTile* pTile = pTsTiles->GetTile(m_iTilt, 0);
	int* piTileSize = pTile->GetTileSize();
	//-----------------
	MU::GAddFrames addFrames;
	float fFactor2 = 1.0f / iImgTiles;
	//-----------------
	int iTileSize = piTileSize[0] * piTileSize[1];
	cudaMemset(m_gfAvgSpect, 0, sizeof(float) * iTileSize); 
	//-----------------
	float* gfScaled = 0L;
	int iBytes = piTileSize[0] * piTileSize[1] * sizeof(float);
	cudaMalloc(&gfScaled, iBytes);
	//-----------------
	for(int i=0; i<iImgTiles; i++)
	{	pTile = pTsTiles->GetTile(m_iTilt, i);
		if(!pTile->IsGood()) continue;
		mScaleTile(i, gfScaled);
		addFrames.DoIt(m_gfAvgSpect, 1.0f, gfScaled,
		   fFactor2, m_gfAvgSpect, piTileSize);
	}
	//-----------------
	if(gfScaled != 0L) cudaFree(gfScaled);
}

void CGenAvgSpectrum::mScaleTile(int iTile, float* gfScaled)
{
	GScaleSpect2D scaleSpect2D;
	//-----------------
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	CTile* pTile = pTsTiles->GetTile(m_iTilt, iTile);
	float fCentZ = pTile->GetCentZ();
	//-------------------------------------
	// This should be sum, not subtract.
	//-------------------------------------
	float fTileDF = m_fCentDF + fCentZ * pTile->GetPixSize();
	//-----------------
	float fScale = sqrtf(m_fCentDF / fTileDF);
	//-----------------
	int* piTileSize = pTile->GetTileSize();
	float* qfTile = pTile->GetTile();
	//-----------------
	scaleSpect2D.DoIt(qfTile, gfScaled, fScale, piTileSize);
}

void CGenAvgSpectrum::mCalcTileCentZs(void)
{
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	CTile* pTile = pTsTiles->GetTile(m_iTilt, 0);
	float fTilt = (pTile->GetTilt() + m_fTiltOffset) * s_fD2R;
	float fCosTilt = (float)cos(fTilt);
	float fTanTilt = (float)tan(fTilt);
	//-----------------
	float fTanBeta = (float)tan(m_fBetaOffset * s_fD2R);
	//-----------------
	float fCosTx = (float)cos(m_fTiltAxis * s_fD2R);
	float fSinTx = (float)sin(m_fTiltAxis * s_fD2R);
	//-----------------
	int iImgTiles = pTsTiles->GetImgTiles();
	for(int i=0; i<iImgTiles; i++)
	{	pTile = pTsTiles->GetTile(m_iTilt, i);
		float fCentX = pTile->GetCentX();
		float fCentY = pTile->GetCentY();
		//-----------------
		float fX = fCentX * fCosTx + fCentY * fSinTx;
		float fY = -fCentX * fSinTx + fCentY * fCosTx;
		float fZ = fX * fTanTilt + fY * fTanBeta * fCosTilt;
		pTile->SetCentZ(fZ);
	}
}

