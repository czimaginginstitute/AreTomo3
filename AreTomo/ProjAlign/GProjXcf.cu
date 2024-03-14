#include "CProjAlignInc.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::ProjAlign;

static __global__ void mGConv
(	cufftComplex* gComp1, 
	cufftComplex* gComp2, 
	int iCmpY,
	float fPower
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	float fRe, fIm;
	fRe = gComp1[i].x * gComp2[i].x + gComp1[i].y * gComp2[i].y;
	fIm = gComp1[i].x * gComp2[i].y - gComp1[i].y * gComp2[i].x;
	//----------------------------------------------------------
	float fAmp1 = sqrtf(gComp1[i].x * gComp1[i].x
		+ gComp1[i].y * gComp1[i].y);
	float fAmp2 = sqrtf(gComp2[i].x * gComp2[i].x
		+ gComp2[i].y * gComp2[i].y);
	float fAmp = sqrtf(fAmp1 * fAmp2) + 0.0001f;
	gComp2[i].x = fRe / fAmp;
	gComp2[i].y = fIm / fAmp;
}

static __global__ void mGWiener
(	cufftComplex* gComp,
	float fBFactor,
	int iCmpY
)
{       int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= iCmpY) return;
        int i = y * gridDim.x + blockIdx.x;
        //---------------------------------
	if(y > (iCmpY / 2)) y -= iCmpY;
	float fNx = 2.0f * (gridDim.x - 1);
	int ir2 = blockIdx.x * blockIdx.x + y * y;
        float fFilt = -2.0f * fBFactor /(fNx * fNx + iCmpY * iCmpY);
        fFilt = expf(fFilt * ir2);
	//------------------------
	gComp[i].x *= fFilt;
	gComp[i].y *= fFilt;
}

static __global__ void mGCenterOrigin(cufftComplex* gComp, int iCmpY)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//---------------------------------
	int iSign = ((blockIdx.x + y) % 2 == 0) ? 1 : -1;
	gComp[i].x *= iSign;
	gComp[i].y *= iSign;
}


GProjXcf::GProjXcf(void)
{
	m_fBFactor = 300.0f;
	m_fPower = 0.5f;
	m_pfXcfImg = 0L;
	memset(m_aiXcfSize, 0, sizeof(m_aiXcfSize));
	m_pInverseFFT2D = new MU::CCufft2D;
}

GProjXcf::~GProjXcf(void)
{
	this->Clean();
	if(m_pInverseFFT2D != 0L) delete m_pInverseFFT2D;
}

void GProjXcf::Clean(void)
{
	if(m_pfXcfImg != 0L) cudaFreeHost(m_pfXcfImg);
	m_pfXcfImg = 0L;
	//-----------------
	m_pInverseFFT2D->DestroyPlan();
}

void GProjXcf::Setup(int* piCmpSize)
{
	this->Clean();
	//------------
	m_aiXcfSize[0] = (piCmpSize[0] - 1) * 2;
	m_aiXcfSize[1] = piCmpSize[1];
	//------------------
	bool bCmpSize = true;
	m_pInverseFFT2D->CreateInversePlan(piCmpSize, bCmpSize);
	//------------------
	size_t tBytes = sizeof(cufftComplex) * piCmpSize[0] * piCmpSize[1];
	cudaMallocHost(&m_pfXcfImg, tBytes);
}

void GProjXcf::DoIt
(	cufftComplex* gCmp1, 
	cufftComplex* gCmp2,
	float fBFactor,
	float fPower
)
{	m_fBFactor = fBFactor;
	m_fPower = fPower;
	//-----------------
	int aiCmpSize[] = {m_aiXcfSize[0]/2 + 1, m_aiXcfSize[1]};
	dim3 aBlockDim(1, 512);
	dim3 aGridDim(aiCmpSize[0], aiCmpSize[1]/aBlockDim.y + 1);
        mGConv<<<aGridDim, aBlockDim>>>
	( gCmp1, gCmp2, aiCmpSize[1], m_fPower
	);
	//-----------------
	mGWiener<<<aGridDim, aBlockDim>>>
	( gCmp2, m_fBFactor, aiCmpSize[1]
	);
	//-----------------
	mGCenterOrigin<<<aGridDim, aBlockDim>>>
	( gCmp2, aiCmpSize[1]
	);
	//-----------------
        m_pInverseFFT2D->Inverse(gCmp2);
	//-----------------
	size_t tBytes = sizeof(cufftComplex) * aiCmpSize[0] * aiCmpSize[1];
	cudaMemcpy(m_pfXcfImg, gCmp2, tBytes, cudaMemcpyDefault);
}

float GProjXcf::SearchPeak1(void)
{
	float fPeak = (float)-1e20;
	int aiPeak[] = {0, 0};
	int iStartX = m_aiXcfSize[0] / 20;
	int iStartY = m_aiXcfSize[1] / 20;
	if(iStartX < 3) iStartX = 3;
	if(iStartY < 3) iStartY = 3;
	int iEndX = m_aiXcfSize[0] - iStartX;
	int iEndY = m_aiXcfSize[1] - iStartY;
	//-----------------------------------
	int iPadX = (m_aiXcfSize[0] / 2 + 1) * 2;
	for(int y=iStartY; y<iEndY; y++)
	{	int i = y * iPadX;
		for(int x=iStartX; x<iEndX; x++)
		{	int j = i + x;
			if(fPeak >= m_pfXcfImg[j]) continue;
			aiPeak[0] = x;
			aiPeak[1] = y;
			fPeak = m_pfXcfImg[j];
		}
	}
	//----------------------------
	int ic = aiPeak[1] * iPadX + aiPeak[0];
        int xp = ic + 1;
        int xm = ic - 1;
        int yp = ic + iPadX;
        int ym = ic - iPadX;
	//------------------
	double a, b, c, d;
	a = (m_pfXcfImg[xp] + m_pfXcfImg[xm]) * 0.5 - m_pfXcfImg[ic];
        b = (m_pfXcfImg[xp] - m_pfXcfImg[xm]) * 0.5;
        c = (m_pfXcfImg[yp] + m_pfXcfImg[ym]) * 0.5 - m_pfXcfImg[ic];
        d = (m_pfXcfImg[yp] - m_pfXcfImg[ym]) * 0.5;
        double dCentX = -b / (2 * a + 1e-30);
        double dCentY = -d / (2 * c + 1e-30);
	//-----------------------------------
	if(fabs(dCentX) > 1) dCentX = 0;
        if(fabs(dCentY) > 1) dCentY = 0;
        m_afPeak[0] = (float)(aiPeak[0] + dCentX);
        m_afPeak[1] = (float)(aiPeak[1] + dCentY);
        m_fPeak =  (float)(a * dCentX * dCentX + b * dCentX
                + c * dCentY * dCentY + d * dCentY
                + m_pfXcfImg[ic]);
	return m_fPeak;
}
/*
float* GProjXcf::GetXcfImg(bool bClean)
{
	float* pfXcfImg = m_pfXcfImg;
	if(bClean) m_pfXcfImg = 0L;
	return pfXcfImg;
}
*/
void GProjXcf::GetShift(float* pfShift, float fXcfBin)
{
	pfShift[0] = m_afPeak[0] - m_aiXcfSize[0] / 2;
	pfShift[1] = m_afPeak[1] - m_aiXcfSize[1] / 2;
	pfShift[0] *= fXcfBin;
	pfShift[1] *= fXcfBin;
}
