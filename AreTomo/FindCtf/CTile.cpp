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
	m_bGpuMem = false;
}

CTile::~CTile(void)
{
	this->Clean();
}

void CTile::Clean(void)
{
	if(m_qfTile == 0L) return;
	//---------------------------
	if(m_bGpuMem) cudaFree(m_qfTile);
	else cudaFreeHost(m_qfTile);
	m_qfTile = 0L;
}

void CTile::SetSize(int* piTileSize)
{
	size_t tGpuFreeBytes = MU::GetGpuFreeMemory();
	size_t t5GB = (size_t)(5 * 1024) * 1024 * 1024;
	//---------------------------
	int iOldSize = m_aiTileSize[0] * m_aiTileSize[1];
	int iNewSize = piTileSize[0] * piTileSize[1];
	memcpy(m_aiTileSize, piTileSize, sizeof(int) * 2);
	if(iOldSize >= iNewSize) return;
	//---------------------------
	size_t tTileBytes = sizeof(float) * iNewSize;
	if(tGpuFreeBytes > (tTileBytes + t5GB)) m_bGpuMem = true;
	else m_bGpuMem = false;
	//---------------------------
	this->Clean();
	if(m_bGpuMem) cudaMalloc(&m_qfTile, tTileBytes);
	else cudaMallocHost(&m_qfTile, tTileBytes);
	/*
	if(m_bGpuMem) printf("*** Tile allocated on GPU\n");
	else printf("*** Tile allocated on CPU\n");
	*/
}
