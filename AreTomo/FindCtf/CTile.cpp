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
// 5. m_pfTile is padded tile in its x dimension.
//--------------------------------------------------------------------
CTile::CTile(void)
{
	m_pfTile = 0L;
	m_iTileSize = 0;
	m_iPadSize = 0;
	memset(m_aiTileStart, 0, sizeof(m_aiTileStart));
	//-----------------
	m_iCoreSize = 0;
	memset(m_aiCoreStart, 0, sizeof(m_aiCoreStart));
}

CTile::~CTile(void)
{
	this->Clean();
}

void CTile::Clean(void)
{
	if(m_pfTile == 0L) return;
	cudaFreeHost(m_pfTile);
	m_pfTile = 0L;
}

void CTile::SetTileSize(int iTileSize)
{
	if(m_iTileSize < iTileSize) this->Clean();
	m_iTileSize = iTileSize;
	m_iPadSize = (m_iTileSize / 2 + 1) * 2;
	//-----------------
	if(m_pfTile != 0L) return;
	size_t tBytes = m_iPadSize * m_iTileSize * sizeof(float);
	cudaMallocHost(&m_pfTile, tBytes);
}

void CTile::SetCoreSize(int iCoreSize)
{
	m_iCoreSize = iCoreSize;
}

void CTile::SetTileStart(int iTileStartX, int iTileStartY)
{
	m_aiTileStart[0] = iTileStartX;
	m_aiTileStart[1] = iTileStartY;
}

void CTile::SetCoreStart(int iCoreStartX, int iCoreStartY)
{
	m_aiCoreStart[0] = iCoreStartX;
	m_aiCoreStart[1] = iCoreStartY;
}

void CTile::SetImgSize(int* piImgSize)
{
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
}

void CTile::Extract(float* pfImage)
{
	int iBytes = m_iTileSize * sizeof(float);
	int iOffset = m_aiTileStart[1] * m_aiImgSize[0] + m_aiTileStart[0];
	float* pfImg = &pfImage[iOffset];
	//-----------------
	for(int y=0; y<m_iTileSize; y++)
	{	float* pfTgt = &m_pfTile[y * m_iPadSize];
		float* pfSrc = &pfImg[y * m_aiImgSize[0]];
		memcpy(pfTgt, pfSrc, iBytes);
	}	
}

void CTile::PasteCore(float* pfImage)
{
	this->PasteCore(m_pfTile, pfImage);
}

void CTile::PasteCore(float* gfTile, float* pfImage)
{
	int iCoreStartX = m_aiCoreStart[0] - m_aiTileStart[0];
	int iCoreStartY = m_aiCoreStart[1] - m_aiTileStart[1];
	float* gfSrc = &gfTile[iCoreStartY * m_iPadSize + iCoreStartX];
	//-----------------
	float* pfDst = &pfImage[m_aiCoreStart[1] * 
	   m_aiImgSize[0] + m_aiCoreStart[0]];
	//-----------------
	int iBytes = m_iCoreSize * sizeof(float);
	for(int y=0; y<m_iCoreSize; y++)
	{	cudaMemcpy(&pfDst[y * m_aiImgSize[0]], 
		   &gfSrc[y * m_iPadSize], iBytes, cudaMemcpyDefault);
	}
}

void CTile::GetTileStart(int* piStart)
{
	piStart[0] = m_aiTileStart[0];
	piStart[1] = m_aiTileStart[1];
}

void CTile::GetCoreStart(int* piStart)
{
	piStart[0] = m_aiCoreStart[0];
	piStart[1] = m_aiCoreStart[1];
}

void CTile::GetTileCenter(float* pfCent)
{
	pfCent[0] = m_aiTileStart[0] + m_iTileSize * 0.5f;
	pfCent[1] = m_aiTileStart[1] + m_iTileSize * 0.5f;
}

void CTile::GetCoreCenter(float* pfCent)
{
	pfCent[0] = m_aiCoreStart[0] + m_iCoreSize * 0.5f;
	pfCent[1] = m_aiCoreStart[1] + m_iCoreSize * 0.5f;
}

int CTile::GetTileSize(void)
{
	return m_iTileSize * m_iPadSize;
}

int CTile::GetTileBytes(void)
{
	int iBytes = sizeof(float) * m_iTileSize * m_iPadSize;
	return iBytes;
}
