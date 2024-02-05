#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::Recon;

static __global__ void mGRWeight
(	cufftComplex* gCmpSinogram,
	int iCmpSize
)
{	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= iCmpSize) return;
	int i = blockIdx.y * iCmpSize + x;
	//--------------------------------
	float fN = 2 * (iCmpSize - 1.0f);
	gCmpSinogram[i].x /= fN;
	gCmpSinogram[i].y /= fN;
	//----------------------
	float fR = x / fN;
	fR = 2 * fR * (0.55f + 0.45f * cosf(6.2831852f * fR));
	gCmpSinogram[i].x *= fR;
	gCmpSinogram[i].y *= fR;
}

GRWeight::GRWeight(void)
{
	m_pGForward = 0L;
	m_pGInverse = 0L;
}

GRWeight::~GRWeight(void)
{
}

void GRWeight::Clean(void)
{
	if(m_pGForward != 0L) delete m_pGForward;
	if(m_pGInverse != 0L) delete m_pGInverse;
	m_pGForward = 0L;
	m_pGInverse = 0L;
}

void GRWeight::SetSize(int iPadProjX, int iNumProjs) 
{
	this->Clean();
	int iFFTSize = (iPadProjX / 2 - 1) * 2;
	m_iCmpSizeX = iFFTSize / 2 + 1;
	m_iNumProjs = iNumProjs;
	//----------------------
	m_pGForward = new MU::GFFT1D;
	m_pGInverse = new MU::GFFT1D;
	bool bForward = true;
	m_pGForward->CreatePlan(iFFTSize, m_iNumProjs, bForward);
	m_pGInverse->CreatePlan(iFFTSize, m_iNumProjs, !bForward);
}

void GRWeight::DoIt(float* gfPadSinogram)
{	
	bool bNorm = true;
	m_pGForward->Forward(gfPadSinogram, !bNorm);
	cufftComplex* gCmpSinogram = (cufftComplex*)gfPadSinogram;
	//-------------------------------------------------------
	dim3 aBlockDim(64, 1);
	dim3 aGridDim(1, m_iNumProjs);
	aGridDim.x = (m_iCmpSizeX + aBlockDim.x - 1) / aBlockDim.x;
	mGRWeight<<<aGridDim, aBlockDim>>>(gCmpSinogram, m_iCmpSizeX);
	//------------------------------------------------------------
	m_pGInverse->Inverse(gCmpSinogram);
}
