#include "CMaUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::MaUtil;

static __global__ void mGFindEmpty
(	float* gfImg,
	int iPadX,
	int iImgY,
	bool* gbEmpty
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iImgY) return;
	//---------------------------
	int iPad = y * iPadX + blockIdx.x;
	int iImg = y * gridDim.x + blockIdx.x;
	//---------------------------
	if(gfImg[iPad] < -1e10) gbEmpty[iImg] = true;
	else gbEmpty[iImg] = false;
}

static __global__ void mGCalcMean
(	float* gfImg,
	bool* gbEmpty,
	int iImgX,
	int iPadX,
	int iSize,
	float* gfMean
)
{	extern __shared__ float s_afSum[];
	float* s_afCount = &s_afSum[blockDim.x];
	//---------------------------
	float fSum = 0.0f, fCount = 0.0f;
	//---------------------------
	for(int i=threadIdx.x; i<iSize; i+=blockDim.x)
	{	if(gbEmpty[i]) continue;
		int y = i / iImgX;
		fSum += gfImg[y * iPadX + i % iImgX];
		fCount += 1.0f;
	}
	s_afSum[threadIdx.x] = fSum;
	s_afCount[threadIdx.x] = fCount;
	__syncthreads();
	//---------------------------
	for(int offset=blockDim.x/2; offset>0; offset=offset/2)
	{	if(threadIdx.x < offset)
		{	int j = threadIdx.x + offset;
			s_afSum[threadIdx.x] += s_afSum[j];
			s_afCount[threadIdx.x] += s_afCount[j];
		}
		__syncthreads();
	}
	//---------------------------
	if(threadIdx.x != 0) return;
	if(s_afCount[0] <= 0) gfMean[0] = 0.0f;
	else gfMean[0] = s_afSum[0] / s_afCount[0];
}
	
static __global__ void mGFillEmpty2D
(	float* gfImg,
	bool* gbEmpty,
	int* giRandoms,
	float* gfMean,
	int iPadX,
	int iImgY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iImgY) return;
	//---------------------------
	int i = y * gridDim.x + blockIdx.x;
	if(!gbEmpty[i]) return;
	//---------------------------
	float fNewVal = gfMean[0];
	int next = giRandoms[i];
	//---------------------------
	for(int j=0; j<100; j++)
	{	if(gbEmpty[next]) 
		{	next = giRandoms[next];
			continue;
		}
		//-------------------
		int k = (next / gridDim.x) * iPadX + (next % gridDim.x);
		fNewVal = gfImg[k];
		break;
	}
	gfImg[y * iPadX + blockIdx.x] = fNewVal;
}

GFillEmpty2D::GFillEmpty2D(void)
{
	m_gbEmpty = 0L;
	m_gfMean = 0L;
	m_pGenRandoms = new GGenRandoms;
}

GFillEmpty2D::~GFillEmpty2D(void)
{
	mClean();
}

void GFillEmpty2D::SetSize(int* piSize, bool bPadded)
{
	mClean();
	//---------------------------
	memcpy(m_aiImgSize, piSize, sizeof(int) * 2);
	if(bPadded) m_aiImgSize[0] = (piSize[0] / 2 - 1) * 2;	
	m_iPadX = piSize[0];
	//---------------------------
	int iImgSize = m_aiImgSize[0] * m_aiImgSize[1];
	m_pGenRandoms = new GGenRandoms;
	m_pGenRandoms->DoIt(iImgSize);
	//---------------------------
	cudaMalloc(&m_gbEmpty, iImgSize * sizeof(bool));
	cudaMalloc(&m_gfMean, sizeof(float));
}


void GFillEmpty2D::DoIt
(	float* gfImg,
	cudaStream_t stream
)
{	int iBlockDimY = 512;
	int iGridDimY = (m_aiImgSize[1] + iBlockDimY - 1) / iBlockDimY;
	//---------------------------
	dim3 aBlockDim(1, iBlockDimY);
	dim3 aGridDim(m_aiImgSize[0], iGridDimY);
	mGFindEmpty<<<aGridDim, aBlockDim>>>(gfImg,
	   m_iPadX, m_aiImgSize[1], m_gbEmpty);
	//---------------------------
	int iImgSize = m_aiImgSize[0] * m_aiImgSize[1];
	aBlockDim = dim3(512, 1);
	aGridDim = dim3(1, 1);
	int iSMbytes = sizeof(float) * aBlockDim.x * 2;
	mGCalcMean<<<aGridDim, aBlockDim, iSMbytes>>>(gfImg, m_gbEmpty,
	   m_aiImgSize[0], m_iPadX, iImgSize, m_gfMean);
	//---------------------------
	aBlockDim = dim3(1, iBlockDimY);
	aGridDim = dim3(m_aiImgSize[0], iGridDimY);
	mGFillEmpty2D<<<aGridDim, aBlockDim>>>(gfImg, m_gbEmpty, 
	   m_pGenRandoms->m_giRandoms,
	   m_gfMean, m_iPadX, m_aiImgSize[1]);
}

void GFillEmpty2D::mClean(void)
{
	if(m_gbEmpty != 0L) cudaFree(m_gbEmpty);
	if(m_gfMean != 0L) cudaFree(m_gfMean);
	if(m_pGenRandoms != 0L) delete m_pGenRandoms;
	m_gbEmpty = 0L;
	m_gfMean = 0L;
	m_pGenRandoms = 0L;
}

