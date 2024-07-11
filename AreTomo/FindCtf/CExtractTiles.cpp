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

CExtractTiles::CExtractTiles(void)
{
	m_iTileSize = 512;
	m_iCoreSize = 64;
	m_pTiles = 0L;
}

CExtractTiles::~CExtractTiles(void)
{
	mClean();
}

void CExtractTiles::Setup(int iTileSize, int iCoreSize, int* piImgSize)
{
	mClean();
	//-----------------
	m_iTileSize = iTileSize;
	m_iCoreSize = iCoreSize;
	if(m_iTileSize < m_iCoreSize) m_iTileSize = m_iCoreSize;
	//-----------------
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	//---------------------------------------------------------
	// 1) Two neighboring cores are adjacent without gap and
	// overlap. 2) Two neighboring tiles overlap each other.
	// 3) A core is a square subarea of a tile. 4) The number
	// of tiles are calculated based on the image size and
	// the core size since they are not overlapped. 
	//---------------------------------------------------------
	m_aiNumTiles[0] = m_aiImgSize[0] / m_iCoreSize;
	m_aiNumTiles[1] = m_aiImgSize[1] / m_iCoreSize;
	//-----------------
	m_iNumTiles = m_aiNumTiles[0] * m_aiNumTiles[1];
	m_pTiles = new CCoreTile[m_iNumTiles];
	for(int i=0; i<m_iNumTiles; i++)
	{	m_pTiles[i].SetTileSize(m_iTileSize);
		m_pTiles[i].SetCoreSize(m_iCoreSize);
	}
	//-----------------
	mCalcTileLocations();
}

CCoreTile* CExtractTiles::GetTile(int iTile)
{
	return &m_pTiles[iTile];
}

bool CExtractTiles::bEdgeTile(int iTile)
{
	int iX = iTile % m_aiNumTiles[0];
	int iY = iTile / m_aiNumTiles[0];
	if(iX == 0 || iY == 0) return true;
	if(iX == (m_aiNumTiles[0] - 1)) return true;
	if(iY == (m_aiNumTiles[1] - 1)) return true;
	return false;
}

void CExtractTiles::DoIt(float* pfImage)
{
	for(int i=0; i<m_iNumTiles; i++)
	{	m_pTiles[i].SetImgSize(m_aiImgSize);
		m_pTiles[i].Extract(pfImage);
	}
}

void CExtractTiles::mCalcTileLocations(void)
{
	int iOffsetX = (m_aiImgSize[0] % m_iCoreSize) / 2;
	int iOffsetY = (m_aiImgSize[1] % m_iCoreSize) / 2;
	int iMargin = (m_iTileSize - m_iCoreSize) / 2;
	//-----------------
	int iNumTiles = m_aiNumTiles[0] * m_aiNumTiles[1];
	for(int i=0; i<iNumTiles; i++)
	{	int iX = i % m_aiNumTiles[0];
		int iY = i / m_aiNumTiles[0];
		//----------------
		int iCoreStartX = iOffsetX + iX * m_iCoreSize;
		int iCoreStartY = iOffsetY + iY * m_iCoreSize;
		//----------------
		int iTileStartX = iCoreStartX - iMargin;
		int iTileStartY = iCoreStartY - iMargin;
		//----------------
		mCheckTileBound(&iTileStartX, m_aiImgSize[0]);
		mCheckTileBound(&iTileStartY, m_aiImgSize[1]);
		//----------------
		m_pTiles[i].SetTileStart(iTileStartX, iTileStartY);
		m_pTiles[i].SetCoreStart(iCoreStartX, iCoreStartY);
	}
}

void CExtractTiles::mCheckTileBound(int* piVal, int iImgSize)
{
	if(piVal[0] < 0) 
	{	piVal[0] = 0;
	}
	else
	{	int iEnd = piVal[0] + m_iTileSize;
		if(iEnd > iImgSize) piVal[0] = iImgSize - m_iTileSize;
	}
}

void CExtractTiles::mClean(void)
{
	if(m_pTiles == 0L) return;
	delete[] m_pTiles;
	m_pTiles = 0L;
	m_iNumTiles = 0;
	memset(m_aiNumTiles, 0, sizeof(m_aiNumTiles));
}
