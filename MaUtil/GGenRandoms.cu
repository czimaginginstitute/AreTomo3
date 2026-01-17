#include "CMaUtilInc.h"
#include <time.h>
#include <curand_kernel.h>

using namespace McAreTomo::MaUtil;

static __global__ void mGGenRandNumbers
(	int* giOutput,
	int iSize,
	unsigned long long seed
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iSize) return;
	//---------------------------
	curandState_t localState;
	curand_init(seed, i, 0, &localState);
	float fVal = curand_uniform(&localState);
	//---------------------------
	int iVal = (int)(fVal * iSize + 0.5f);
	giOutput[i] = iVal % iSize;
}

GGenRandoms::GGenRandoms(void)
{
	m_giRandoms = 0L;
	m_iSize = 0;
}

GGenRandoms::~GGenRandoms(void)
{
	if(m_giRandoms != 0L) cudaFree(m_giRandoms);
}

void GGenRandoms::DoIt(int iSize)
{
	if(iSize > m_iSize)
	{	if(m_giRandoms != 0L) cudaFree(m_giRandoms);
		cudaMalloc(&m_giRandoms, sizeof(int) * iSize);
	}
	m_iSize = iSize;
	//---------------------------
	unsigned long long seed = time(0L);
	dim3 aBlockDim(1024, 1);
	dim3 aGridDim(1, 1);
	aGridDim.x = (m_iSize + aBlockDim.x - 1) / aBlockDim.x;
	//---------------------------
	mGGenRandNumbers<<<aGridDim, aBlockDim>>>(m_giRandoms,
	   m_iSize, seed);
}
