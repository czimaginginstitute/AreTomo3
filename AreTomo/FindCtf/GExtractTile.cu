#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

static __global__ void mGExtractTile
(	float* gfImg, int iImgX, int iImgPadX, int iImgY, 
	float* gfTile, int iTilePadX, int iTileY,
	int iStartX, int iStartY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iTileY) return;
	int t = y * iTilePadX + blockIdx.x;
	//-----------------
	int x = blockIdx.x + iStartX;
	if(x < 0 || x >= iImgX) 
	{	gfTile[t] = (float)-1e30;
		return;
	}
	//-----------------
	y = y + iStartY;
	if(y < 0 || y >= iImgY)
	{	gfTile[t] = (float)-1e30;
		return;
	}
	//-----------------
	gfTile[t] = gfImg[y * iImgPadX + x];
}

static __global__ void mGRandomFill
(	float* gfTile,
	int iTilePadX, int iTileY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iTileY) return;
	unsigned int i = y * iTilePadX + blockIdx.x;
	if(gfTile[i] > (float)-1e25) return;
	//-----------------
	unsigned int iTileSize = gridDim.x * iTileY; 
	unsigned int next = (i * 509 + 283) % iTileSize;
	for(int j=0; j<21; j++)
	{	float fVal = gfTile[next];
		if(fVal > (float)-1e25)
		{	gfTile[i] = fVal;
			return;
		}
		next = (next * 509 + 283) % iTileSize;	
	}
	gfTile[i] = gfTile[iTileSize / 2];
}

GExtractTile::GExtractTile(void)
{
}

GExtractTile::~GExtractTile(void)
{
}

void GExtractTile::SetImg(float* gfImg, int* piImgSize, bool bPadded)
{
	m_gfImg = gfImg;
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_iImgPadX = piImgSize[0];
	if(!bPadded) return;
	//-----------------
	m_aiImgSize[0] = (m_iImgPadX / 2 - 1) * 2;
}

void GExtractTile::SetTileSize(int* piTileSize, bool bPadded)
{
	m_aiTileSize[0] = piTileSize[0];
	m_aiTileSize[1] = piTileSize[1];
	m_iTilePadX = piTileSize[0];
	if(!bPadded) return;
	//-----------------
	m_aiTileSize[0] = (m_iTilePadX / 2 - 1) * 2;
}

void GExtractTile::DoIt(float* gfTile, int* piStart, cudaStream_t stream)
{
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(m_aiTileSize[0], 1);
	aGridDim.y = (m_aiTileSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//-----------------
	mGExtractTile<<<aGridDim, aBlockDim, 0, stream>>>(m_gfImg, 
	   m_aiImgSize[0], m_iImgPadX, m_aiImgSize[1], gfTile, 
	   m_iTilePadX, m_aiTileSize[1], piStart[0], piStart[1]);	
	//-----------------
	mGRandomFill<<<aGridDim, aBlockDim, 0, stream>>>(gfTile, 
	   m_iTilePadX, m_aiTileSize[1]);
}
