#include "CMrcUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo::MrcUtil;

//--------------------------------------------------------------------
// Gaussian bandpass. fLow2 and fHigh2 are 2 * sigma^2 in units of
// (cycles / pixel)^2.
//--------------------------------------------------------------------
static __global__ void mGBandpass
(	cufftComplex* gCmp,
	int iCmpY,
	int iImgX,
	float fLow2,
	float fHigh2
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//-----------------
	if(y > (iCmpY / 2)) y -= iCmpY;
	float fX = blockIdx.x / (float)iImgX;
	float fY = y / (float)iCmpY;
	float fR2 = fX * fX + fY * fY;
	float fFilt = expf(-fR2 / fLow2) * (1.0f - expf(-fR2 / fHigh2));
	gCmp[i].x *= fFilt;
	gCmp[i].y *= fFilt;
}

//--------------------------------------------------------------------
// gOut = conj(gA) * gB. The inverse FFT of gOut is the correlation
// c(d) = sum_x a(x) b(x + d). gOut can be the same as gA or gB.
//--------------------------------------------------------------------
static __global__ void mGConjMul
(	cufftComplex* gA,
	cufftComplex* gB,
	cufftComplex* gOut,
	int iCmpY
)
{	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y >= iCmpY) return;
	int i = y * gridDim.x + blockIdx.x;
	//-----------------
	cufftComplex a = gA[i], b = gB[i];
	gOut[i].x = a.x * b.x + a.y * b.y;
	gOut[i].y = a.x * b.y - a.y * b.x;
}

static __global__ void mGSquare(float* gfIn, float* gfOut, int iSize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iSize) return;
	gfOut[i] = gfIn[i] * gfIn[i];
}

//--------------------------------------------------------------------
// Overlap-normalized cross correlation. All inputs are correlation
// maps that carry the same FFT scale factor, which cancels out.
// Shifts with too little overlap are marked -2. gfNCC can be gfP.
//--------------------------------------------------------------------
static __global__ void mGNCC
(	float* gfP, float* gfS1, float* gfQ1,
	float* gfS2, float* gfQ2, float* gfO,
	int iSize, float fMinO, float* gfNCC
)
{	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= iSize) return;
	//-----------------
	float fO = gfO[i];
	if(fO < fMinO)
	{	gfNCC[i] = -2.0f;
		return;
	}
	float fS1 = gfS1[i], fS2 = gfS2[i];
	float fNum = gfP[i] - fS1 * fS2 / fO;
	float fV1 = gfQ1[i] - fS1 * fS1 / fO;
	float fV2 = gfQ2[i] - fS2 * fS2 / fO;
	float fEps = 1e-4f * fO;
	float fNCC = -2.0f;
	if(fV1 > fEps && fV2 > fEps) fNCC = fNum / sqrtf(fV1 * fV2);
	if(!isfinite(fNCC)) fNCC = -2.0f;  // NaN must not win the peak
	gfNCC[i] = fNCC;
}

//====================================================================
// GBinBandpass2D
//====================================================================
GBinBandpass2D::GBinBandpass2D(void)
{
	m_gfRawPad = 0L;
	m_gBinCmp = 0L;
	m_pForwardFFT = new MU::CCufft2D;
	m_pInverseFFT = new MU::CCufft2D;
}

GBinBandpass2D::~GBinBandpass2D(void)
{
	this->Clean();
	if(m_pForwardFFT != 0L) delete m_pForwardFFT;
	if(m_pInverseFFT != 0L) delete m_pInverseFFT;
}

void GBinBandpass2D::Clean(void)
{
	m_pForwardFFT->DestroyPlan();
	m_pInverseFFT->DestroyPlan();
	if(m_gfRawPad != 0L) cudaFree(m_gfRawPad);
	if(m_gBinCmp != 0L) cudaFree(m_gBinCmp);
	m_gfRawPad = 0L;
	m_gBinCmp = 0L;
}

void GBinBandpass2D::Setup
(	int* piRawSize,
	int* piBinSize,
	float fLowSigma,
	float fHighSigma
)
{	this->Clean();
	memcpy(m_aiRawSize, piRawSize, sizeof(int) * 2);
	memcpy(m_aiBinSize, piBinSize, sizeof(int) * 2);
	m_fLowSigma = fLowSigma;
	m_fHighSigma = fHighSigma;
	//-----------------
	int aiRawPad[] = {(m_aiRawSize[0] / 2 + 1) * 2, m_aiRawSize[1]};
	int aiBinCmp[] = {m_aiBinSize[0] / 2 + 1, m_aiBinSize[1]};
	size_t tBytes = sizeof(float) * aiRawPad[0] * aiRawPad[1];
	cudaMalloc(&m_gfRawPad, tBytes);
	tBytes = sizeof(cufftComplex) * aiBinCmp[0] * aiBinCmp[1];
	cudaMalloc(&m_gBinCmp, tBytes);
	//-----------------
	m_pForwardFFT->CreateForwardPlan(aiRawPad, true);
	m_pInverseFFT->CreateInversePlan(aiBinCmp, true);
}

void GBinBandpass2D::DoIt(float* pfRawImg, float* pfBinImg)
{
	int iRawPadX = (m_aiRawSize[0] / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * m_aiRawSize[0];
	cudaMemcpy2D(m_gfRawPad, sizeof(float) * iRawPadX,
	   pfRawImg, tBytes, tBytes, m_aiRawSize[1],
	   cudaMemcpyDefault);
	m_pForwardFFT->Forward(m_gfRawPad, false);
	//-----------------
	int aiRawCmp[] = {m_aiRawSize[0] / 2 + 1, m_aiRawSize[1]};
	int aiBinCmp[] = {m_aiBinSize[0] / 2 + 1, m_aiBinSize[1]};
	MU::GFourierResize2D fourierResize;
	fourierResize.DoIt((cufftComplex*)m_gfRawPad, aiRawCmp,
	   m_gBinCmp, aiBinCmp, false);
	//-----------------
	float fLow2 = 2.0f * m_fLowSigma * m_fLowSigma;
	float fHigh2 = 2.0f * m_fHighSigma * m_fHighSigma;
	dim3 aBlockDim(1, 64);
	dim3 aGridDim(aiBinCmp[0], (aiBinCmp[1] + 63) / 64);
	mGBandpass<<<aGridDim, aBlockDim>>>(m_gBinCmp, aiBinCmp[1],
	   m_aiBinSize[0], fLow2, fHigh2);
	m_pInverseFFT->Inverse(m_gBinCmp);
	//-----------------
	tBytes = sizeof(float) * m_aiBinSize[0];
	cudaMemcpy2D(pfBinImg, tBytes, m_gBinCmp,
	   sizeof(cufftComplex) * aiBinCmp[0], tBytes,
	   m_aiBinSize[1], cudaMemcpyDefault);
}

//====================================================================
// GMaskedNCC2D
//====================================================================
GMaskedNCC2D::GMaskedNCC2D(void)
{
	m_gBuf = 0L;
	m_pfNCC = 0L;
	m_pForwardFFT = new MU::CCufft2D;
	m_pInverseFFT = new MU::CCufft2D;
}

GMaskedNCC2D::~GMaskedNCC2D(void)
{
	this->Clean();
	if(m_pForwardFFT != 0L) delete m_pForwardFFT;
	if(m_pInverseFFT != 0L) delete m_pInverseFFT;
}

void GMaskedNCC2D::Clean(void)
{
	m_pForwardFFT->DestroyPlan();
	m_pInverseFFT->DestroyPlan();
	if(m_gBuf != 0L) cudaFree(m_gBuf);
	if(m_pfNCC != 0L) delete[] m_pfNCC;
	m_gBuf = 0L;
	m_pfNCC = 0L;
}

void GMaskedNCC2D::Setup
(	int* piImgSize,
	int* piMaskStart,
	int* piMaskSize,
	float fMinOverlap
)
{	this->Clean();
	memcpy(m_aiImgSize, piImgSize, sizeof(int) * 2);
	m_fMinOverlap = fMinOverlap;
	//-----------------
	m_aiFftSize[0] = m_aiImgSize[0] * 2;
	m_aiFftSize[1] = m_aiImgSize[1] * 2;
	m_aiCmpSize[0] = m_aiFftSize[0] / 2 + 1;
	m_aiCmpSize[1] = m_aiFftSize[1];
	m_iPadX = m_aiCmpSize[0] * 2;
	//-----------------
	size_t tCmpSize = (size_t)m_aiCmpSize[0] * m_aiCmpSize[1];
	cudaMalloc(&m_gBuf, sizeof(cufftComplex) * tCmpSize * 9);
	m_gFm = m_gBuf;
	m_gFr = m_gFm + tCmpSize;
	m_gFr2 = m_gFr + tCmpSize;
	m_gS1 = m_gFr2 + tCmpSize;
	m_gQ1 = m_gS1 + tCmpSize;
	m_gO = m_gQ1 + tCmpSize;
	m_gP = m_gO + tCmpSize;
	m_gS2 = m_gP + tCmpSize;
	m_gQ2 = m_gS2 + tCmpSize;
	m_pfNCC = new float[tCmpSize * 2];
	//-----------------
	int aiPadSize[] = {m_iPadX, m_aiFftSize[1]};
	m_pForwardFFT->CreateForwardPlan(aiPadSize, true);
	m_pInverseFFT->CreateInversePlan(m_aiCmpSize, true);
	//-----------------
	int iPixels = m_aiImgSize[0] * m_aiImgSize[1];
	float* pfMask = new float[iPixels];
	memset(pfMask, 0, sizeof(float) * iPixels);
	for(int y=0; y<piMaskSize[1]; y++)
	{	float* pfRow = pfMask + (y + piMaskStart[1]) * m_aiImgSize[0]
		   + piMaskStart[0];
		for(int x=0; x<piMaskSize[0]; x++) pfRow[x] = 1.0f;
	}
	mLoad(pfMask, (float*)m_gFm);
	delete[] pfMask;
	m_pForwardFFT->Forward((float*)m_gFm, false);
	//-----------------------------------------------
	// Overlap map O(d). Its value at d = 0 is the
	// mask area times the FFT scale factor.
	//-----------------------------------------------
	mConjMul(m_gFm, m_gFm, m_gO);
	m_pInverseFFT->Inverse(m_gO);
	cudaMemcpy(&m_fMaxO, m_gO, sizeof(float), cudaMemcpyDefault);
}

void GMaskedNCC2D::SetRef(float* pfRefImg)
{
	mLoad(pfRefImg, (float*)m_gFr);
	mSquare((float*)m_gFr, (float*)m_gFr2);
	m_pForwardFFT->Forward((float*)m_gFr, false);
	m_pForwardFFT->Forward((float*)m_gFr2, false);
	//-----------------
	mConjMul(m_gFr, m_gFm, m_gS1);
	mConjMul(m_gFr2, m_gFm, m_gQ1);
	m_pInverseFFT->Inverse(m_gS1);
	m_pInverseFFT->Inverse(m_gQ1);
}

float GMaskedNCC2D::DoIt(float* pfImg, float* pfShift)
{
	mLoad(pfImg, (float*)m_gS2);
	mSquare((float*)m_gS2, (float*)m_gQ2);
	m_pForwardFFT->Forward((float*)m_gS2, false);
	m_pForwardFFT->Forward((float*)m_gQ2, false);
	//-----------------
	mConjMul(m_gFr, m_gS2, m_gP);
	mConjMul(m_gFm, m_gS2, m_gS2);
	mConjMul(m_gFm, m_gQ2, m_gQ2);
	m_pInverseFFT->Inverse(m_gP);
	m_pInverseFFT->Inverse(m_gS2);
	m_pInverseFFT->Inverse(m_gQ2);
	//-----------------
	int iSize = m_iPadX * m_aiFftSize[1];
	float fMinO = m_fMinOverlap * m_fMaxO;
	int iBlock = 512;
	int iGrid = (iSize + iBlock - 1) / iBlock;
	mGNCC<<<iGrid, iBlock>>>((float*)m_gP, (float*)m_gS1,
	   (float*)m_gQ1, (float*)m_gS2, (float*)m_gQ2,
	   (float*)m_gO, iSize, fMinO, (float*)m_gP);
	cudaMemcpy(m_pfNCC, m_gP, sizeof(float) * iSize,
	   cudaMemcpyDefault);
	//-----------------
	float fPeak = mSearchPeak(pfShift);
	return fPeak;
}

void GMaskedNCC2D::mLoad(float* pfImg, float* gfPad)
{
	size_t tBytes = sizeof(float) * m_iPadX * m_aiFftSize[1];
	cudaMemset(gfPad, 0, tBytes);
	tBytes = sizeof(float) * m_aiImgSize[0];
	cudaMemcpy2D(gfPad, sizeof(float) * m_iPadX, pfImg, tBytes,
	   tBytes, m_aiImgSize[1], cudaMemcpyDefault);
}

void GMaskedNCC2D::mSquare(float* gfIn, float* gfOut)
{
	int iSize = m_iPadX * m_aiFftSize[1];
	int iBlock = 512;
	int iGrid = (iSize + iBlock - 1) / iBlock;
	mGSquare<<<iGrid, iBlock>>>(gfIn, gfOut, iSize);
}

void GMaskedNCC2D::mConjMul
(	cufftComplex* gA,
	cufftComplex* gB,
	cufftComplex* gOut
)
{	dim3 aBlockDim(1, 64);
	dim3 aGridDim(m_aiCmpSize[0], (m_aiCmpSize[1] + 63) / 64);
	mGConjMul<<<aGridDim, aBlockDim>>>(gA, gB, gOut,
	   m_aiCmpSize[1]);
}

//--------------------------------------------------------------------
// The NCC map has its origin at (0, 0) with wrap-around. The peak
// is refined by a parabolic fit when all its neighbors are valid.
//--------------------------------------------------------------------
float GMaskedNCC2D::mSearchPeak(float* pfShift)
{
	int iNx = m_aiFftSize[0], iNy = m_aiFftSize[1];
	float fPeak = -2.0f;
	int iPeakX = 0, iPeakY = 0;
	for(int y=0; y<iNy; y++)
	{	float* pfRow = m_pfNCC + y * m_iPadX;
		for(int x=0; x<iNx; x++)
		{	if(pfRow[x] <= fPeak) continue;
			fPeak = pfRow[x];
			iPeakX = x;
			iPeakY = y;
		}
	}
	pfShift[0] = 0.0f;
	pfShift[1] = 0.0f;
	if(fPeak <= -1.0f) return -2.0f;
	//-----------------
	int iXm = (iPeakX - 1 + iNx) % iNx, iXp = (iPeakX + 1) % iNx;
	int iYm = (iPeakY - 1 + iNy) % iNy, iYp = (iPeakY + 1) % iNy;
	float fC = fPeak;
	float fXm = m_pfNCC[iPeakY * m_iPadX + iXm];
	float fXp = m_pfNCC[iPeakY * m_iPadX + iXp];
	float fYm = m_pfNCC[iYm * m_iPadX + iPeakX];
	float fYp = m_pfNCC[iYp * m_iPadX + iPeakX];
	float fDx = 0.0f, fDy = 0.0f;
	if(fXm > -1.0f && fXp > -1.0f)
	{	float fDen = fXm + fXp - 2.0f * fC;
		if(fDen < 0) fDx = 0.5f * (fXm - fXp) / fDen;
	}
	if(fYm > -1.0f && fYp > -1.0f)
	{	float fDen = fYm + fYp - 2.0f * fC;
		if(fDen < 0) fDy = 0.5f * (fYm - fYp) / fDen;
	}
	if(fabsf(fDx) > 1.0f) fDx = 0.0f;
	if(fabsf(fDy) > 1.0f) fDy = 0.0f;
	//-----------------
	float fX = iPeakX + fDx, fY = iPeakY + fDy;
	if(fX > iNx / 2) fX -= iNx;
	if(fY > iNy / 2) fY -= iNy;
	pfShift[0] = fX;
	pfShift[1] = fY;
	return fPeak;
}
