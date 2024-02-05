#include "CPatchAlignInc.h"
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::PatchAlign;

static __global__ void mGConv
(	cufftComplex* gCmp1, 
	cufftComplex* gCmp2,
	int iCmpY,
	float fBFactor,
	cufftComplex* gCmpRes
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	int iSign = ((blockIdx.x + y) % 2 == 0) ? 1 : -1;
	//-----------------------------------------------
	float fRe, fIm;
	fRe = gCmp1[i].x * gCmp2[i].x + gCmp1[i].y * gCmp2[i].y;
	fIm = gCmp1[i].x * gCmp2[i].y - gCmp1[i].y * gCmp2[i].x;
	float fAmp = sqrtf(fRe * fRe + fIm * fIm) + 0.001f;
	//-------------------------------------------------
	if(y > (iCmpY / 2)) y -= iCmpY;
	float fFilt = 2.0f * (gridDim.x - 1);
        fFilt = -2.0f * fBFactor /(fFilt * fFilt + iCmpY * iCmpY);
        fFilt = expf(fFilt * (blockIdx.x * blockIdx.x + y * y));
	//------------------------------------------------------
	fFilt = iSign * fFilt / fAmp;
	//---------------------------
	gCmpRes[i].x = fFilt * fRe;
	gCmpRes[i].y = fFilt * fIm;
}


GGenXcfImage::GGenXcfImage(void)
{
	m_pInverseFFT2D = new MU::CCufft2D;
}

GGenXcfImage::~GGenXcfImage(void)
{
	if(m_pInverseFFT2D != 0L) delete m_pInverseFFT2D;
}

void GGenXcfImage::Clean(void)
{
	m_pInverseFFT2D->DestroyPlan();
}

void GGenXcfImage::Setup(int* piCmpSize)
{
	this->Clean();
	//-----------------
	m_aiXcfSize[0] = (piCmpSize[0] - 1) * 2;
	m_aiXcfSize[1] = piCmpSize[1];
	//-----------------
	bool bCmp = true;
	m_pInverseFFT2D->CreateInversePlan(piCmpSize, bCmp);
}

void GGenXcfImage::DoIt
(	cufftComplex* gCmp1, 
	cufftComplex* gCmp2,
	float* gfXcfImg,
	float fBFactor,
	cudaStream_t stream
)
{	int aiCmpSize[] = {m_aiXcfSize[0]/2 + 1, m_aiXcfSize[1]};
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(aiCmpSize[0], aiCmpSize[1]/aBlockDim.y + 1);
        mGConv<<<aGridDim, aBlockDim, 0, stream>>>(gCmp1, gCmp2, 
		aiCmpSize[1], fBFactor, (cufftComplex*)gfXcfImg);
	//-----------------
        m_pInverseFFT2D->Inverse((cufftComplex*)gfXcfImg);
}

