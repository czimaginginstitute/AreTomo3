#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::FindCtf;

//----------------------------------------------------------
// Scale the input spectrum by a factor of fScale.
// 1. Both input and output are half spectrums with their
//    centers at (0, iSpectY / 2).
// 2. When fScale > 1.0, expand the input spectrum, Thon
//    ring spacing increases like less defocused.
// 3. When fScale < 1.0, shrink the inpit spectrum, Thon
//    ring spacing decreases like more defocused. 
//----------------------------------------------------------
static __global__ void mGScale2D
(       float* gfInSpect,
        float* gfOutSpect,
        float fScale,
        int iSpectY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iSpectY) return;
	int i = y * gridDim.x + blockIdx.x;
	int iHalfY = iSpectY / 2;
	//------------------
	int xIn = (int)(blockIdx.x / fScale);
	int yIn = (int)((y - 0.5f * iSpectY) / fScale);
	//-----------------
	if(xIn >= gridDim.x) xIn = gridDim.x - 1;
	if(yIn >= iHalfY) yIn = iHalfY - 1;
	else if(yIn < (-iHalfY)) yIn = (-iHalfY);
	yIn += iHalfY;
	//-----------------
	gfOutSpect[i] = gfInSpect[yIn * gridDim.x + xIn];
}

GScaleSpect2D::GScaleSpect2D(void)
{
	m_aiSpectSize[0] = 0;
	m_aiSpectSize[1] = 0;
	m_gfInSpect = 0L;
}

GScaleSpect2D::~GScaleSpect2D(void)
{
	this->Clean();
}

void GScaleSpect2D::Clean(void)
{
	if(m_gfInSpect == 0L) return;
	cudaFree(m_gfInSpect);
	m_gfInSpect = 0L;
}

void GScaleSpect2D::DoIt
(	float* gfInSpect,
	float* gfOutSpect,
	float fScale,
	int* piSpectSize,
	cudaStream_t stream
)
{	dim3 aBlockDim(1, 512);
	dim3 aGridDim(piSpectSize[0], 1);
	aGridDim.y = (piSpectSize[1] + aBlockDim.y - 1) / aBlockDim.y;
	//------------------
	mGScale2D<<<aGridDim, aBlockDim, 0, stream>>>(gfInSpect,
	   gfOutSpect, fScale, piSpectSize[1]);
}

void GScaleSpect2D::Setup(int* piSpectSize)
{
	int iOldSize = m_aiSpectSize[0] * m_aiSpectSize[1];
	int iNewSize = piSpectSize[0] * piSpectSize[1];
	if(iOldSize != iNewSize) this->Clean();
	//-----------------
	m_aiSpectSize[0] = piSpectSize[0];
	m_aiSpectSize[1] = piSpectSize[1];
	//-----------------
	if(m_gfInSpect != 0L) return;
	cudaMalloc(&m_gfInSpect, sizeof(float) * iNewSize);
}

void GScaleSpect2D::DoIt
(	float* pfInSpect,
	float* gfOutSpect,
	float fScale
)
{	int iBytes = m_aiSpectSize[0] * m_aiSpectSize[1] * sizeof(float);
	cudaMemcpy(m_gfInSpect, pfInSpect, iBytes, cudaMemcpyDefault);
	this->DoIt(m_gfInSpect, gfOutSpect, fScale, m_aiSpectSize);
}


