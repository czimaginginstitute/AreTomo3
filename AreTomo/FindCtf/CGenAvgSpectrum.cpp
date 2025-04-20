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
	int iHandedness,   // 1 or -1
	float* gfAvgSpect,
	int iNthGpu
)
{	m_iTilt = iTilt;
	m_fTiltAxis = fTiltAxis;
	m_fCentDF = fCentDF;
	m_iHandedness = iHandedness;
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
	int* piTileSize = pTile->GetSize();
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
	int* piTileSize = pTile->GetSize();
	//-----------------
	MU::GAddFrames addFrames;
	float fFactor2 = 1.0f / iImgTiles;
	//-----------------
	int iTileSize = piTileSize[0] * piTileSize[1];
	cudaMemset(m_gfAvgSpect, 0, sizeof(float) * iTileSize); 
	//-----------------
	float* gfScaled = 0L;
	int iBytes = iTileSize * sizeof(float);
	cudaMalloc(&gfScaled, iBytes * 3);
	//---------------------------
	cudaStreamCreate(&m_aStreams[0]);
	cudaStreamCreate(&m_aStreams[1]);
	//---------------------------
	for(int i=0; i<iImgTiles; i++)
	{	pTile = pTsTiles->GetTile(m_iTilt, i);
		if(!pTile->IsGood()) continue;
		//-------------------
		int j = i % 2;
		mScaleTile(i, gfScaled);
		addFrames.DoIt(m_gfAvgSpect, 1.0f, gfScaled, fFactor2,
		   m_gfAvgSpect, piTileSize, m_aStreams[0]);
	}
	cudaStreamSynchronize(m_aStreams[0]);
	//---------------------------
	if(gfScaled != 0L) cudaFree(gfScaled);
	cudaStreamDestroy(m_aStreams[0]);
	cudaStreamDestroy(m_aStreams[1]);
}

void CGenAvgSpectrum::mScaleTile(int iTile, float* gfScaled)
{
	GScaleSpect2D scaleSpect2D;
	int iStream = iTile % 2;
	//---------------------------
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	CTile* pTile = pTsTiles->GetTile(m_iTilt, iTile);
	float fCentZ = pTile->GetCentZ();
	float fPixSize = pTile->GetPixSize();
	//-------------------------------------
	// This depends on defocus handedness
	//-------------------------------------
	float fTileDF = m_fCentDF + fCentZ * fPixSize * m_iHandedness;
	float fScale = sqrtf(m_fCentDF / fTileDF);
	//---------------------------
	int* piTileSize = pTile->GetSize();
	int iTileSize = piTileSize[0] * piTileSize[1];
	float* qfTile = pTile->GetTile();
	float* gfTile = &gfScaled[(iStream + 1) * iTileSize];
	cudaMemcpyAsync(gfTile, qfTile, iTileSize * sizeof(float),
	   cudaMemcpyDefault, m_aStreams[iStream]);
	if(iStream == 1) cudaStreamSynchronize(m_aStreams[1]);
	//---------------------------
	scaleSpect2D.DoIt(gfTile, gfScaled, fScale, 
	   piTileSize, m_aStreams[0]);
}

void CGenAvgSpectrum::mCalcTileCentZs(void)
{
	CTsTiles* pTsTiles = CTsTiles::GetInstance(m_iNthGpu);
	CTile* pTile = pTsTiles->GetTile(m_iTilt, 0);
	float fTilt = pTile->GetTilt();
	//-----------------
	CTiltInducedZ tiltInducedZ;
	tiltInducedZ.Setup(fTilt, m_fTiltAxis, m_fTiltOffset, m_fBetaOffset);
	//-----------------
	int iImgTiles = pTsTiles->GetImgTiles();
	for(int i=0; i<iImgTiles; i++)
	{	pTile = pTsTiles->GetTile(m_iTilt, i);
		float fCentX = pTile->GetCentX();
		float fCentY = pTile->GetCentY();
		float fDeltaZ = tiltInducedZ.DoIt(fCentX, fCentY);
		pTile->SetCentZ(fDeltaZ);
	}
}

