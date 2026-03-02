#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

//--------------------------------------------------------------------
// 1. This class abstracts in a 2D image a square subarea used for
//    both CTF estimation and correction.
// 2. A tile overlaps with its adjacent tiles by a given amount.
// 3. A core is a square subarea of a tile. A core connects its
//    adjacent cores without overlapping. 1) Cores are used for
//    CTF correction. 2) a core may not be at the center of the
//    tile when it is an edge tile.
// 4. m_iPadSize = (m_iTileSize / 2 + 1) * 2, padded size of tile
// 5. m_qfTile is padded tile in its x dimension.
//--------------------------------------------------------------------
CCoreTile::CCoreTile(void)
{
	memset(m_aiTileStart, 0, sizeof(m_aiTileStart));
	//-----------------
	m_iCoreSize = 0;
	memset(m_aiCoreStart, 0, sizeof(m_aiCoreStart));
}

CCoreTile::~CCoreTile(void)
{
}

void CCoreTile::SetSize(int iTileSize, int iCoreSize)
{
	int aiTileSize[] = {(iTileSize / 2 + 1) * 2, iTileSize};
	CTile::SetSize(aiTileSize);
	//-----------------
	m_iCoreSize = iCoreSize;
}

void CCoreTile::SetCoreStart(int iCoreStartX, int iCoreStartY)
{
	m_aiCoreStart[0] = iCoreStartX;
	m_aiCoreStart[1] = iCoreStartY;
	//-----------------------------------------------
	// 1) Tile center is the same as the core center
	//-----------------------------------------------
	this->GetCoreCenter(m_afCenter);
	//-----------------------------------------------
	// 1) Tile starting coordinates can be negative.
	// This means the tile can be partially outside
	// the belonging image.
	// 2) Tile is padded and square. m_aiTileSize[1]
	// is the square size.
	//-----------------------------------------------
	int iTileSize = m_aiTileSize[1];
	m_aiTileStart[0] = m_afCenter[0] - iTileSize * 0.5f;
	m_aiTileStart[1] = m_afCenter[1] - iTileSize * 0.5f;
	//-----------------------------------------------
	// 1) Core center overlaps with tilt center.
	//-----------------------------------------------
	m_afCenter[0] = m_aiTileStart[0] + iTileSize * 0.5f;
	m_afCenter[1] = m_aiTileStart[1] + iTileSize * 0.5f;

}

void CCoreTile::PasteCore(float* pfImage, int* piImgSize)
{
	int iCoreStartX = m_aiCoreStart[0] - m_aiTileStart[0];
	int iCoreStartY = m_aiCoreStart[1] - m_aiTileStart[1];
	int iOffset = iCoreStartY * m_aiTileSize[0] + iCoreStartX;
	float* qfSrc = &m_qfTile[iOffset];
	//-----------------
	iOffset = m_aiCoreStart[1] * piImgSize[0] + m_aiCoreStart[0];
	float* pfDst = &pfImage[iOffset];
	//-----------------
	int iBytes = m_iCoreSize * sizeof(float);
	for(int y=0; y<m_iCoreSize; y++)
	{	cudaMemcpy(&pfDst[y * piImgSize[0]], 
		   &qfSrc[y * m_aiTileSize[0]], 
		   iBytes, cudaMemcpyDefault);
	}
}

void CCoreTile::GetTileStart(int* piStart)
{
	piStart[0] = m_aiTileStart[0];
	piStart[1] = m_aiTileStart[1];
}

void CCoreTile::GetCoreStart(int* piStart)
{
	piStart[0] = m_aiCoreStart[0];
	piStart[1] = m_aiCoreStart[1];
}

void CCoreTile::GetCoreCenter(float* pfCent)
{
	pfCent[0] = m_aiCoreStart[0] + m_iCoreSize * 0.5f;
	pfCent[1] = m_aiCoreStart[1] + m_iCoreSize * 0.5f;
}

void CCoreTile::GetCoreCenterInTile(float* pfCent)
{
	this->GetCoreCenter(pfCent);
	//-----------------
	int iTileSize = m_aiTileSize[1];
	pfCent[0] = pfCent[0] - m_afCenter[0] + iTileSize * 0.5f;
	pfCent[1] = pfCent[1] - m_afCenter[1] + iTileSize * 0.5f;
}

int CCoreTile::GetTileBytes(void)
{
	int iBytes = CTile::GetPixels() * sizeof(float);
	return iBytes;
}
