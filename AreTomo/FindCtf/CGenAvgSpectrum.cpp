#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

//------------------------------
// Debugging code
//------------------------------
//static CSaveImages s_aSaveImages;
//static int s_iCount = 0;

CGenAvgSpectrum::CGenAvgSpectrum(void)
{
	m_fOverlap = 0.50f;
	m_gfTileSpect = 0L;
	m_pGCalcMoment2D = new MU::GCalcMoment2D;
	m_pGCalcSpectrum = new GCalcSpectrum;
}

CGenAvgSpectrum::~CGenAvgSpectrum(void)
{
	this->Clean();
	delete m_pGCalcMoment2D;
	delete m_pGCalcSpectrum;
}

void CGenAvgSpectrum::Clean(void)
{
	if(m_gfTileSpect != 0L) 
	{	cudaFree(m_gfTileSpect);
		m_gfTileSpect = 0L;
	}
	m_pGCalcMoment2D->Clean();
	m_pGCalcSpectrum->Clean();
}

void CGenAvgSpectrum::SetSizes(int* piImgSize, int iTileSize)
{
	this->Clean();
	//-----------------
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_iTileSize = iTileSize;
	//-----------------
	m_aiCmpSize[0] = iTileSize / 2 + 1;
	m_aiCmpSize[1] = iTileSize;
	m_aiPadSize[0] = m_aiCmpSize[0] * 2;
	m_aiPadSize[1] = m_aiCmpSize[1];
	//-----------------
	m_iOverlap = (int)(m_iTileSize * m_fOverlap);
	m_iOverlap = m_iOverlap / 2 * 2;
	//-----------------
	int iSize = m_iTileSize - m_iOverlap;
	m_aiNumTiles[0] = (m_aiImgSize[0] - m_iOverlap) / iSize;
	m_aiNumTiles[1] = (m_aiImgSize[1] - m_iOverlap) / iSize;
	//-----------------
	m_aiOffset[0] = (m_aiImgSize[0] - m_aiNumTiles[0] * m_iTileSize
	   + (m_aiNumTiles[0] - 1) * m_iOverlap) / 2;
	m_aiOffset[1] = (m_aiImgSize[1] - m_aiNumTiles[1] * m_iTileSize
	   + (m_aiNumTiles[1] - 1) * m_iOverlap) / 2;
	//-----------------
	if(m_gfTileSpect != 0L) cudaFree(m_gfTileSpect);
	int iCmpSize = m_aiCmpSize[0] * m_aiCmpSize[1];
	size_t tBytes = sizeof(float) * iCmpSize * 3;
        cudaMalloc(&m_gfTileSpect, tBytes);
	m_gfPadTile = m_gfTileSpect + iCmpSize;
	//-----------------
	m_pGCalcMoment2D->SetSize(m_aiPadSize, true);
	m_pGCalcSpectrum->SetSize(m_aiPadSize, true);

	//s_aSaveImages.OpenFile("/home/szheng/home1/Temp/TestTiles.mrc");
	//s_aSaveImages.Setup(m_aiPadSize, m_aiNumTiles[0] * m_aiNumTiles[1]);
}

void CGenAvgSpectrum::DoIt
(	float* pfImage,
	float* gfAvgSpect
)
{	m_pfImage = pfImage;
	m_gfAvgSpect = gfAvgSpect;
	//s_iCount += 1;
	//------------------------
	MU::GAddFrames addFrames;
	int iNumTiles = m_aiNumTiles[0] * m_aiNumTiles[1];
	float fFactor2 = 1.0f / iNumTiles;
	//--------------------------------
	cudaMemset(m_gfAvgSpect, 0, sizeof(float) * 
	   m_aiCmpSize[0] * m_aiCmpSize[1]);
	//-----------------------------------------a
	for(int i=0; i<iNumTiles; i++)
	{	mCalcTileSpectrum(i);
		addFrames.DoIt(m_gfAvgSpect, 1.0f, m_gfTileSpect, 
		   fFactor2, m_gfAvgSpect, m_aiCmpSize);
	}
}

void CGenAvgSpectrum::mCalcTileSpectrum(int iTile)
{	
	mExtractPadTile(iTile);
	//---------------------
	float fMean = m_pGCalcMoment2D->DoIt(m_gfPadTile, 1, true);
	float fStd = m_pGCalcMoment2D->DoIt(m_gfPadTile, 2, true);
	fStd = fStd - fMean * fMean;
	if(fStd < 1.0f) fStd = 1.0f;
	else fStd = sqrtf(fStd);
	//-----------------
	MU::GNormalize2D aGNormalize;
	aGNormalize.DoIt(m_gfPadTile, m_aiPadSize, true, fMean, fStd);
	//------------------------------------------------------------
	GRoundEdge aGRoundEdge;
	float afCent[] = {m_iTileSize * 0.5f, m_iTileSize * 0.5f};
	float afSize[] = {m_iTileSize * 1.0f, m_iTileSize * 1.0f};
	aGRoundEdge.SetMask(afCent, afSize);
	aGRoundEdge.DoIt(m_gfPadTile, m_aiPadSize);
	//---------------------------------------
	bool bLogrith = true;
	m_pGCalcSpectrum->DoPad(m_gfPadTile, m_gfTileSpect, !bLogrith);
}

void CGenAvgSpectrum::mExtractPadTile(int iTile)
{	
	int iTileX = iTile % m_aiNumTiles[0];
	int iTileY = iTile / m_aiNumTiles[0];
	int iStartX = m_aiOffset[0] + iTileX * (m_iTileSize - m_iOverlap);
	int iStartY = m_aiOffset[1] + iTileY * (m_iTileSize - m_iOverlap);
	//----------------------------------------------------------------
	size_t tBytes = sizeof(float) * m_iTileSize;
	int iOffset = iStartY * m_aiImgSize[0] + iStartX;
	for(int y=0; y<m_iTileSize; y++)
	{	float* pfSrc = m_pfImage + y * m_aiImgSize[0] + iOffset;
		float* gfDst = m_gfPadTile + y * m_aiPadSize[0];
		cudaMemcpy(gfDst, pfSrc, tBytes, cudaMemcpyDefault);
	}

	/*
	if(s_iCount != 32) return;
	printf("Tile: %4d  %4d  %4d  %4d  %4d\n", 
	   iTile, iStartX, iStartY, m_aiImgSize[0], m_aiImgSize[1]);
	s_aSaveImages.DoIt(iTile, m_gfPadTile, true);	
	*/
}	
