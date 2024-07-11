#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

CTile::CTile(void)
{
	m_qfTile = 0L;
	memset(m_aiTileSize, 0, sizeof(m_aiTileSize));
	memset(m_afCenter, 0, sizeof(m_afCenter));
	m_bGood = true;
}

CTile::~CTile(void)
{
	this->Clean();
}

void CTile::Clean(void)
{
	if(m_qfTile == 0L) return;
	cudaFreeHost(m_qfTile);
	m_qfTile = 0L;
}

void CTile::SetSize(int* piTileSize)
{
	int iOldSize = m_aiTileSize[0] * m_aiTileSize[1];
	int iNewSize = piTileSize[0] * piTileSize[1];
	memcpy(m_aiTileSize, piTileSize, sizeof(int) * 2);
	if(iOldSize >= iNewSize) return;
	//-----------------
	if(m_qfTile != 0L) cudaFreeHost(m_qfTile);
	cudaMallocHost(&m_qfTile, sizeof(float) * iNewSize);
}
