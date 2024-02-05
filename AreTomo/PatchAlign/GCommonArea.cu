#include "CPatchAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::PatchAlign;

static __global__ void mGFindArea
(	float* gfImg1,
	float* gfImg2,
	unsigned int iPadX,
	unsigned int iSizeY,
	bool* gbCommonArea
)
{	unsigned int i;
	i =  blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= iSizeY) return;
	else i = i * iPadX + blockIdx.x;
	//------------------------------
	if(gfImg1[i] <= (float)-1e10) 
	{	gfImg2[i] = (float)-1e30;
		gbCommonArea[i] = false;
	}
	else if(gfImg2[i] <= (float)-1e10) 
	{	gfImg1[i] = (float)-1e30;
		gbCommonArea[i] = false;
	}
	else gbCommonArea[i] = true;
}

static __global__ void mGFindCenter
(	bool* gbCommonArea,
	int iSizeX,
	int iSizeY,
	int iPadX,
	float* gfRes
)
{	extern __shared__ float shared[];
	//-------------------------------
	int i, j, x, y;
	int iSumX = 0, iSumY = 0, iCount = 0;
	int iPixels = iSizeX * iSizeY;
	for(i=0; i<iPixels; i+=blockDim.x)
	{	j = i + threadIdx.x;
		if(j >= iPixels) continue;
		//------------------------
		x = j % iSizeX;
		y = j / iSizeX;
		j = y * iPadX + x;
		if(gbCommonArea[j])
		{	iSumX += x;
			iSumY += y;
			iCount += 1;
		}
	}
	//-------------------------- 
	shared[threadIdx.x] = iSumX;
	shared[blockDim.x + threadIdx.x] = iSumY;
	shared[blockDim.x * 2 + threadIdx.x] = iCount;
	__syncthreads();
	//--------------
	float* pfSumY = &shared[blockDim.x];
	float* pfCount = &shared[blockDim.x * 2];
	for(int i=blockDim.x/2; i>0; i=i/2)
	{	if(threadIdx.x < i)
		{	j = threadIdx.x + i;
			shared[threadIdx.x] += shared[j];
			pfSumY[threadIdx.x] += pfSumY[j];
			pfCount[threadIdx.x] += pfCount[j];
		}
		__syncthreads();
	}
	//----------------------
	if(threadIdx.x != 0) return;
	if(pfCount[0] <= 0)
	{	gfRes[0] = -1.0f;
		gfRes[1] = -1.0f;
	}
	else
	{	gfRes[0] = shared[0] / pfCount[0];
		gfRes[1] = pfSumY[0] / pfCount[0];
		gfRes[2] = pfCount[0];
	}
}

static __global__ void mGCenterCommArea
(	float* gfInImg,
	float* gfCenter,
	int iPadX,
	int iSizeY,
	float* gfOutImg
)
{	int x, y;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSizeY) return;
	int iNew = y * iPadX + blockIdx.x;
	gfOutImg[iNew] = gfInImg[iNew];
	if(gfCenter[0] < 0 || gfCenter[1] < 0) return;
	//--------------------------------------
	int iOffsetX = (int)gfCenter[0] - gridDim.x / 2;
	int iOffsetY = (int)gfCenter[1] - iSizeY / 2;
	x = blockIdx.x + iOffsetX;
	y = y + iOffsetY;
	if(x < 0 || y < 0 || x >= gridDim.x || y >= iSizeY)
	{	gfOutImg[iNew] = (float)-1e30;
		return;
	}
	gfOutImg[iNew] = gfInImg[y * iPadX + x];
}

GCommonArea::GCommonArea(void)
{
}

GCommonArea::~GCommonArea(void)
{
}

void GCommonArea::DoIt
(	float* gfImg1,
	float* gfImg2,
	float* gf2Bufs,
	int* piImgSize,
	bool bPadded,
	float* gfCommArea,
	cudaStream_t stream
) 
{	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	if(bPadded) m_aiImgSize[0] = (piImgSize[0] / 2 - 1) * 2;
	m_iPadX = piImgSize[0];
	//---------------------
	m_gfImg1 = gfImg1;
	m_gfImg2 = gfImg2;
	m_gfBuf1 = gf2Bufs;
	m_gfBuf2 = m_gfBuf1 + m_iPadX * m_aiImgSize[1];
	m_stream = stream;
	//----------------
	mFindCommonArea();
	mCenterCommonArea(gfCommArea);
}

void GCommonArea::mFindCommonArea(void)
{
	dim3 aBlockDim(1, 64);
	dim3 aGridDim(m_aiImgSize[0], 1);
	aGridDim.y = (m_aiImgSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//------------------------------------------------------------
	bool* gbCommonArea = reinterpret_cast<bool*>(m_gfBuf1);
	mGFindArea<<<aGridDim, aBlockDim, 0, m_stream>>>(m_gfImg1, m_gfImg2,
		m_iPadX, m_aiImgSize[1], gbCommonArea);
}

void GCommonArea::mCenterCommonArea(float* gfCommArea)
{
	dim3 aBlockDim(512, 1);
	dim3 aGridDim(1, 1, 1);
	int iShmBytes = sizeof(float) * aBlockDim.x * 3;
	//----------------------------------------------
	bool* gbCommonArea = reinterpret_cast<bool*>(m_gfBuf1);
	float* gfCommCent = m_gfBuf2;
	mGFindCenter<<<aGridDim, aBlockDim, iShmBytes, m_stream>>>
	( gbCommonArea, m_aiImgSize[0], m_aiImgSize[1], m_iPadX, gfCommCent
	);
	cudaMemcpyAsync(gfCommArea, gfCommCent+2, sizeof(float),
		cudaMemcpyDefault, m_stream);
	//-----------------------------------
	size_t tBytes = sizeof(float) * m_iPadX * m_aiImgSize[1];
	aBlockDim.x = 1;
	aBlockDim.y = 64;
	aGridDim.x = m_aiImgSize[0];
	aGridDim.y = (m_aiImgSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//------------------------------------------------------------
	mGCenterCommArea<<<aGridDim, aBlockDim, 0, m_stream>>>
	( m_gfImg1, gfCommCent, m_iPadX, m_aiImgSize[1], m_gfBuf1
	);
	cudaMemcpyAsync
	( m_gfImg1, m_gfBuf1, tBytes, cudaMemcpyDefault, m_stream
	);
	mGCenterCommArea<<<aGridDim, aBlockDim, 0, m_stream>>>
	( m_gfImg2, gfCommCent, m_iPadX, m_aiImgSize[1], m_gfBuf1
	);
	cudaMemcpyAsync
	( m_gfImg2, m_gfBuf1, tBytes, cudaMemcpyDefault, m_stream
	);
}
