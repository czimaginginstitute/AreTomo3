#include "CAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::MotionCor::Align;

//--------------------------------------------------------------------
// CDetectFeatures2D: For each patch, find a prominent feature near
// its geometric center. A feature is prominent if it is much darker
// than the background of its patch and also has a significantly
// higher local standard deviation than the background.
// 1. The aligned XCF sum is Fourier-cropped to a binned image with
//    square pixels and about s_iPixPerPatch pixels per patch.
// 2. At each scale s, the local mean M = G_s(I) and the local std
//    S = sqrt(G_w(I^2) - G_w(I)^2), w = s_fStdWinRatio * s, are
//    computed. For each patch, the background is the median Mb and
//    MAD of M and the median Sb of S. A pixel is prominent when
//       D = (Mb - M) / (1.4826 * MAD) >= s_fMinDark  and
//       R = S / Sb >= s_fMinStdRatio.
//    Its score is D * R. The score map keeps the max over scales.
// 3. Within a central window of each patch, the local maximum of
//    the score map with the highest score is the feature. If there
//    is no prominent pixel, the geometric center is kept.
//--------------------------------------------------------------------
static const int s_iPixPerPatch = 64;
static const int s_iMinBinnedSize = 256;
static const int s_iMaxBinnedSize = 768;
static const float s_afSigmas[] = {1.0f, 2.0f, 4.0f};
static const int s_iNumScales = 3;
static const float s_fStdWinRatio = 2.0f; // std window / mean sigma
static const float s_fMinDark = 3.0f;     // min darkness z-score
static const float s_fMinStdRatio = 1.5f; // min local / background std
static const float s_fSearchFrac = 0.3f;  // half window / patch size

CDetectFeatures2D* CDetectFeatures2D::m_pInstances = 0L;
int CDetectFeatures2D::m_iNumGpus = 0;

void CDetectFeatures2D::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CDetectFeatures2D[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CDetectFeatures2D::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CDetectFeatures2D* CDetectFeatures2D::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CDetectFeatures2D::CDetectFeatures2D(void)
{
	m_aiBinnedSize[0] = 0;
	m_aiBinnedSize[1] = 0;
	m_aiNumPatches[0] = 0;
	m_aiNumPatches[1] = 0;
	m_pfBinnedSum = 0L;
	m_pfScore = 0L;
	m_pfDark = 0L;
	m_pfStdRatio = 0L;
	m_pfCenters = 0L;
}

CDetectFeatures2D::~CDetectFeatures2D(void)
{
	mClean();
}

void CDetectFeatures2D::mClean(void)
{
	if(m_pfBinnedSum != 0L) delete[] m_pfBinnedSum;
	if(m_pfScore != 0L) delete[] m_pfScore;
	if(m_pfDark != 0L) delete[] m_pfDark;
	if(m_pfStdRatio != 0L) delete[] m_pfStdRatio;
	if(m_pfCenters != 0L) delete[] m_pfCenters;
	m_pfBinnedSum = 0L;
	m_pfScore = 0L;
	m_pfDark = 0L;
	m_pfStdRatio = 0L;
	m_pfCenters = 0L;
}

void CDetectFeatures2D::DoIt
(	MMD::CStackShift* pXcfStackShift,
	int* piNumPatches
)
{	mClean();
	m_aiNumPatches[0] = piNumPatches[0];
	m_aiNumPatches[1] = piNumPatches[1];
	int iNumPatches = piNumPatches[0] * piNumPatches[1];
	m_pfCenters = new float[iNumPatches * 5];
	//-----------------
	mBinSum(pXcfStackShift);
	m_afPatSize[0] = m_aiBinnedSize[0] * 1.0f / m_aiNumPatches[0];
	m_afPatSize[1] = m_aiBinnedSize[1] * 1.0f / m_aiNumPatches[1];
	mCalcMargin(pXcfStackShift);
	mBuildScoreMap();
	//-----------------
	for(int i=0; i<iNumPatches; i++)
	{	mFindPatchFeature(i);
	}
	delete[] m_pfBinnedSum;
	delete[] m_pfScore;
	delete[] m_pfDark;
	delete[] m_pfStdRatio;
	m_pfBinnedSum = 0L;
	m_pfScore = 0L;
	m_pfDark = 0L;
	m_pfStdRatio = 0L;
	//-----------------
	printf("# CDetectFeatures2D: binned size %d  %d\n",
	   m_aiBinnedSize[0], m_aiBinnedSize[1]);
	printf("# patch  geom-x  geom-y  feat-x  feat-y    score"
	   "     dark  std-ratio\n");
	int iFound = 0;
	for(int i=0; i<iNumPatches; i++)
	{	float* pfCent = m_pfCenters + 5 * i;
		float fGeomX = (i % m_aiNumPatches[0] + 0.5f) * m_afPatSize[0];
		float fGeomY = (i / m_aiNumPatches[0] + 0.5f) * m_afPatSize[1];
		if(pfCent[2] > 0) iFound += 1;
		printf("%5d  %7.2f %7.2f %7.2f %7.2f  %7.2f  %7.2f  %7.2f\n", i,
		   fGeomX, fGeomY, pfCent[0], pfCent[1], pfCent[2],
		   pfCent[3], pfCent[4]);
	}
	printf("# %d of %d patches have prominent features.\n\n",
	   iFound, iNumPatches);
}

void CDetectFeatures2D::GetCenter
(	int iPatch,
	int* piImgSize,
	float* pfCent
)
{	float fBinX = piImgSize[0] * 1.0f / m_aiBinnedSize[0];
	float fBinY = piImgSize[1] * 1.0f / m_aiBinnedSize[1];
	int i = 5 * iPatch;
	pfCent[0] = m_pfCenters[i] * fBinX;
	pfCent[1] = m_pfCenters[i+1] * fBinY;
}

float CDetectFeatures2D::GetScore(int iPatch)
{
	return m_pfCenters[5 * iPatch + 2];
}

bool CDetectFeatures2D::HasFeature(int iPatch)
{
	return (m_pfCenters[5 * iPatch + 2] > 0);
}

//--------------------------------------------------------------------
// Sum the aligned XCF frames, Fourier-crop the sum to the binned
// size and copy the real-space binned sum to host memory.
//--------------------------------------------------------------------
void CDetectFeatures2D::mBinSum(MMD::CStackShift* pXcfStackShift)
{
	CAlignedSum alignedSum;
	int aiSumRange[2] = {0, pXcfStackShift->m_iNumFrames - 1};
	alignedSum.DoIt(MD::EBuffer::xcf, pXcfStackShift,
	   aiSumRange, m_iNthGpu);
	//-----------------
	MD::CBufferPool* pBufferPool =
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pSumBuffer =
	   pBufferPool->GetBuffer(MD::EBuffer::sum);
	MD::CStackBuffer* pTmpBuffer =
	   pBufferPool->GetBuffer(MD::EBuffer::tmp);
	MD::CStackBuffer* pXcfBuffer =
	   pBufferPool->GetBuffer(MD::EBuffer::xcf);
	int aiXcfSize[2] = { (pXcfBuffer->m_aiCmpSize[0] - 1) * 2,
	   pXcfBuffer->m_aiCmpSize[1] };
	//-----------------------------------------------
	// Same binning along x and y (square pixels),
	// about s_iPixPerPatch pixels per patch.
	//-----------------------------------------------
	float fPatX = aiXcfSize[0] * 1.0f / m_aiNumPatches[0];
	float fPatY = aiXcfSize[1] * 1.0f / m_aiNumPatches[1];
	float fBin = std::max(fPatX, fPatY) / s_iPixPerPatch;
	int iMaxXcf = std::max(aiXcfSize[0], aiXcfSize[1]);
	float fLong = iMaxXcf / fBin;
	if(fLong > s_iMaxBinnedSize) fBin = iMaxXcf * 1.0f / s_iMaxBinnedSize;
	else if(fLong < s_iMinBinnedSize)
	{	fBin = iMaxXcf * 1.0f / s_iMinBinnedSize;
	}
	if(fBin < 1.0f) fBin = 1.0f;
	for(int i=0; i<2; i++)
	{	int iSize = (int)(aiXcfSize[i] / fBin) / 2 * 2;
		if(iSize > aiXcfSize[i]) iSize = aiXcfSize[i] / 2 * 2;
		m_aiBinnedSize[i] = iSize;
	}
	//-----------------
	cufftComplex* gCmpSum = pSumBuffer->GetFrame(0);
	cufftComplex* gCmpBinnedSum = pTmpBuffer->GetFrame(0);
	int aiOutCmpSize[] = {m_aiBinnedSize[0] / 2 + 1, m_aiBinnedSize[1]};
	MU::GFourierResize2D aGFourierResize;
	aGFourierResize.DoIt(gCmpSum, pXcfBuffer->m_aiCmpSize,
	   gCmpBinnedSum, aiOutCmpSize, false, 0);
	//-----------------
	MU::CCufft2D* pInverseFFT = pBufferPool->GetCufft2D(false);
	pInverseFFT->CreateInversePlan(aiOutCmpSize, true);
	pInverseFFT->Inverse(gCmpBinnedSum, (cudaStream_t)0);
	float* gfBinnedSum = reinterpret_cast<float*>(gCmpBinnedSum);
	//-----------------
	int iBinnedSize = m_aiBinnedSize[0] * m_aiBinnedSize[1];
	m_pfBinnedSum = new float[iBinnedSize];
	size_t tPadBytes = sizeof(float) * aiOutCmpSize[0] * 2;
	size_t tRowBytes = sizeof(float) * m_aiBinnedSize[0];
	cudaMemcpy2D(m_pfBinnedSum, tRowBytes, gfBinnedSum, tPadBytes,
	   tRowBytes, m_aiBinnedSize[1], cudaMemcpyDefault);
}

//--------------------------------------------------------------------
// Frame edges in the aligned sum are incomplete. Exclude a margin
// equal to the range of frame shifts, in binned pixels.
//--------------------------------------------------------------------
void CDetectFeatures2D::mCalcMargin(MMD::CStackShift* pXcfStackShift)
{
	MD::CBufferPool* pBufferPool =
	   MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pXcfBuffer =
	   pBufferPool->GetBuffer(MD::EBuffer::xcf);
	float fBinX = (pXcfBuffer->m_aiCmpSize[0] - 1) * 2.0f
	   / m_aiBinnedSize[0];
	float fBinY = pXcfBuffer->m_aiCmpSize[1] * 1.0f
	   / m_aiBinnedSize[1];
	//-----------------
	float afMin[2] = {0.0f}, afMax[2] = {0.0f}, afShift[2] = {0.0f};
	for(int i=0; i<pXcfStackShift->m_iNumFrames; i++)
	{	pXcfStackShift->GetShift(i, afShift);
		for(int k=0; k<2; k++)
		{	if(i == 0 || afShift[k] < afMin[k]) afMin[k] = afShift[k];
			if(i == 0 || afShift[k] > afMax[k]) afMax[k] = afShift[k];
		}
	}
	m_aiMargin[0] = (int)ceilf((afMax[0] - afMin[0]) / fBinX) + 2;
	m_aiMargin[1] = (int)ceilf((afMax[1] - afMin[1]) / fBinY) + 2;
}

//--------------------------------------------------------------------
// Score(x) = max over scales of D * R at pixels that are both much
// darker (D >= s_fMinDark) and much more textured (R >=
// s_fMinStdRatio) than the background of their patch, 0 elsewhere.
//--------------------------------------------------------------------
void CDetectFeatures2D::mBuildScoreMap(void)
{
	int iSizeX = m_aiBinnedSize[0];
	int iSizeY = m_aiBinnedSize[1];
	int iBinnedSize = iSizeX * iSizeY;
	m_pfScore = new float[iBinnedSize];
	m_pfDark = new float[iBinnedSize];
	m_pfStdRatio = new float[iBinnedSize];
	memset(m_pfScore, 0, sizeof(float) * iBinnedSize);
	memset(m_pfDark, 0, sizeof(float) * iBinnedSize);
	memset(m_pfStdRatio, 0, sizeof(float) * iBinnedSize);
	//-----------------------------------------------
	// Remove the global mean so that the variance
	// G(I^2) - G(I)^2 does not lose float precision.
	// D and R are invariant to this offset.
	//-----------------------------------------------
	double dMean = 0.0;
	for(int i=0; i<iBinnedSize; i++) dMean += m_pfBinnedSum[i];
	float fMean = (float)(dMean / iBinnedSize);
	float* pfSqr = new float[iBinnedSize];
	for(int i=0; i<iBinnedSize; i++)
	{	m_pfBinnedSum[i] -= fMean;
		pfSqr[i] = m_pfBinnedSum[i] * m_pfBinnedSum[i];
	}
	float* pfMean = new float[iBinnedSize];
	float* pfStd = new float[iBinnedSize];
	float* pfWinMean = new float[iBinnedSize];
	float* pfBuf = new float[iBinnedSize];
	int iMaxPatPix = ((int)m_afPatSize[0] + 2) * ((int)m_afPatSize[1] + 2);
	float* pfPatch = new float[iMaxPatPix];
	int iNumPatches = m_aiNumPatches[0] * m_aiNumPatches[1];
	//-----------------
	for(int s=0; s<s_iNumScales; s++)
	{	float fWin = s_afSigmas[s] * s_fStdWinRatio;
		mGaussBlur(m_pfBinnedSum, pfMean, pfBuf, s_afSigmas[s]);
		mGaussBlur(m_pfBinnedSum, pfWinMean, pfBuf, fWin);
		mGaussBlur(pfSqr, pfStd, pfBuf, fWin);
		for(int i=0; i<iBinnedSize; i++)
		{	float fVar = pfStd[i] - pfWinMean[i] * pfWinMean[i];
			pfStd[i] = (fVar > 0.0f) ? sqrtf(fVar) : 0.0f;
		}
		//----------------
		for(int p=0; p<iNumPatches; p++)
		{	int aiStart[2], aiEnd[2];
			mGetPatchRect(p, aiStart, aiEnd);
			//---------------
			int iCount = 0;
			for(int y=aiStart[1]; y<aiEnd[1]; y++)
			{	float* pfRow = pfMean + y * iSizeX;
				for(int x=aiStart[0]; x<aiEnd[0]; x++)
				{	pfPatch[iCount++] = pfRow[x];
				}
			}
			float fMeanBg = 0.0f, fMeanMad = 0.0f;
			mPatchStats(pfPatch, iCount, &fMeanBg, &fMeanMad);
			//---------------
			iCount = 0;
			for(int y=aiStart[1]; y<aiEnd[1]; y++)
			{	float* pfRow = pfStd + y * iSizeX;
				for(int x=aiStart[0]; x<aiEnd[0]; x++)
				{	pfPatch[iCount++] = pfRow[x];
				}
			}
			float fStdBg = 0.0f, fStdMad = 0.0f;
			mPatchStats(pfPatch, iCount, &fStdBg, &fStdMad);
			if(fMeanMad <= 0.0f || fStdBg <= 0.0f) continue; // flat
			float fDarkScale = 1.0f / (1.4826f * fMeanMad);
			float fStdScale = 1.0f / fStdBg;
			//---------------
			for(int y=aiStart[1]; y<aiEnd[1]; y++)
			{	int i = y * iSizeX;
				for(int x=aiStart[0]; x<aiEnd[0]; x++)
				{	float fD = (fMeanBg - pfMean[i+x]) * fDarkScale;
					if(fD < s_fMinDark) continue;
					float fR = pfStd[i+x] * fStdScale;
					if(fR < s_fMinStdRatio) continue;
					float fScore = fD * fR;
					if(fScore <= m_pfScore[i+x]) continue;
					m_pfScore[i+x] = fScore;
					m_pfDark[i+x] = fD;
					m_pfStdRatio[i+x] = fR;
				}
			}
		}
	}
	delete[] pfSqr;
	delete[] pfMean;
	delete[] pfStd;
	delete[] pfWinMean;
	delete[] pfBuf;
	delete[] pfPatch;
}

//--------------------------------------------------------------------
// Separable Gaussian blur with clamped borders. pfBuf is scratch
// of the binned image size.
//--------------------------------------------------------------------
void CDetectFeatures2D::mGaussBlur
(	float* pfIn,
	float* pfOut,
	float* pfBuf,
	float fSigma
)
{	int iRadius = (int)ceilf(3.0f * fSigma);
	int iKernel = 2 * iRadius + 1;
	float* pfKernel = new float[iKernel];
	double dSum = 0.0;
	for(int i=0; i<iKernel; i++)
	{	float fX = (float)(i - iRadius);
		pfKernel[i] = expf(-0.5f * fX * fX / (fSigma * fSigma));
		dSum += pfKernel[i];
	}
	for(int i=0; i<iKernel; i++) pfKernel[i] /= (float)dSum;
	//-----------------
	int iSizeX = m_aiBinnedSize[0];
	int iSizeY = m_aiBinnedSize[1];
	for(int y=0; y<iSizeY; y++)
	{	float* pfSrc = pfIn + y * iSizeX;
		float* pfDst = pfBuf + y * iSizeX;
		for(int x=0; x<iSizeX; x++)
		{	float fVal = 0.0f;
			for(int k=0; k<iKernel; k++)
			{	int xx = x + k - iRadius;
				if(xx < 0) xx = 0;
				else if(xx >= iSizeX) xx = iSizeX - 1;
				fVal += pfKernel[k] * pfSrc[xx];
			}
			pfDst[x] = fVal;
		}
	}
	for(int y=0; y<iSizeY; y++)
	{	float* pfDst = pfOut + y * iSizeX;
		for(int x=0; x<iSizeX; x++)
		{	float fVal = 0.0f;
			for(int k=0; k<iKernel; k++)
			{	int yy = y + k - iRadius;
				if(yy < 0) yy = 0;
				else if(yy >= iSizeY) yy = iSizeY - 1;
				fVal += pfKernel[k] * pfBuf[yy * iSizeX + x];
			}
			pfDst[x] = fVal;
		}
	}
	delete[] pfKernel;
}

//--------------------------------------------------------------------
// Median and median absolute deviation. pfVals is reordered.
//--------------------------------------------------------------------
void CDetectFeatures2D::mPatchStats
(	float* pfVals,
	int iCount,
	float* pfMedian,
	float* pfMad
)
{	if(iCount <= 0)
	{	pfMedian[0] = 0.0f;
		pfMad[0] = 0.0f;
		return;
	}
	int iMid = iCount / 2;
	std::nth_element(pfVals, pfVals + iMid, pfVals + iCount);
	float fMedian = pfVals[iMid];
	for(int i=0; i<iCount; i++) pfVals[i] = fabsf(pfVals[i] - fMedian);
	std::nth_element(pfVals, pfVals + iMid, pfVals + iCount);
	pfMedian[0] = fMedian;
	pfMad[0] = pfVals[iMid];
}

void CDetectFeatures2D::mGetPatchRect
(	int iPatch,
	int* piStart,
	int* piEnd
)
{	int aiPat[2] = { iPatch % m_aiNumPatches[0],
	   iPatch / m_aiNumPatches[0] };
	for(int i=0; i<2; i++)
	{	piStart[i] = (int)(aiPat[i] * m_afPatSize[i]);
		piEnd[i] = (int)((aiPat[i] + 1) * m_afPatSize[i]);
		if(piEnd[i] > m_aiBinnedSize[i]) piEnd[i] = m_aiBinnedSize[i];
	}
}

//--------------------------------------------------------------------
// Pick the local maximum of the score map near the patch center
// that has the highest score. Refine it with a parabolic fit. If
// no pixel is prominent, keep the geometric center.
//--------------------------------------------------------------------
void CDetectFeatures2D::mFindPatchFeature(int iPatch)
{
	int iSizeX = m_aiBinnedSize[0];
	float fCentX = (iPatch % m_aiNumPatches[0] + 0.5f) * m_afPatSize[0];
	float fCentY = (iPatch / m_aiNumPatches[0] + 0.5f) * m_afPatSize[1];
	float* pfCent = m_pfCenters + 5 * iPatch;
	pfCent[0] = fCentX;
	pfCent[1] = fCentY;
	pfCent[2] = -1.0f;
	pfCent[3] = 0.0f;
	pfCent[4] = 0.0f;
	//-----------------
	float afHalf[2] = { s_fSearchFrac * m_afPatSize[0],
	   s_fSearchFrac * m_afPatSize[1] };
	int aiStart[2], aiEnd[2];
	float afCent[2] = {fCentX, fCentY};
	//-----------------------------------------------
	// Keep the moved center at least half a patch
	// (+1 px for sub-pixel refinement and rounding)
	// from the frame border so that the extracted
	// patch fits inside the frame without clamping.
	//-----------------------------------------------
	for(int i=0; i<2; i++)
	{	int iBorder = (int)ceilf(0.5f * m_afPatSize[i]) + 1;
		int iLow = std::max(1, m_aiMargin[i]);
		iLow = std::max(iLow, iBorder);
		int iHigh = m_aiBinnedSize[i] - 1 - m_aiMargin[i];
		if(iHigh > m_aiBinnedSize[i] - 1) iHigh = m_aiBinnedSize[i] - 1;
		iHigh = std::min(iHigh, m_aiBinnedSize[i] - iBorder);
		aiStart[i] = std::max(iLow, (int)(afCent[i] - afHalf[i]));
		aiEnd[i] = std::min(iHigh, (int)(afCent[i] + afHalf[i]) + 1);
	}
	//-----------------
	float fBestScore = 0.0f;
	int iBestX = -1, iBestY = -1;
	float* Z = m_pfScore;
	for(int y=aiStart[1]; y<aiEnd[1]; y++)
	{	for(int x=aiStart[0]; x<aiEnd[0]; x++)
		{	int i = y * iSizeX + x;
			float fZ = Z[i];
			if(fZ <= fBestScore) continue; // also rejects 0
			//---------------
			bool bMax = true;
			for(int j=-1; j<=1 && bMax; j++)
			{	for(int k=-1; k<=1; k++)
				{	if(j == 0 && k == 0) continue;
					if(Z[i + j * iSizeX + k] > fZ)
					{	bMax = false;
						break;
					}
				}
			}
			if(!bMax) continue;
			//---------------
			fBestScore = fZ;
			iBestX = x;
			iBestY = y;
		}
	}
	if(iBestX < 0) return;
	//-----------------
	int i = iBestY * iSizeX + iBestX;
	float fZc = Z[i];
	float fOffX = mParabolicPeak(Z[i-1], fZc, Z[i+1]);
	float fOffY = mParabolicPeak(Z[i-iSizeX], fZc, Z[i+iSizeX]);
	pfCent[0] = iBestX + 0.5f + fOffX;
	pfCent[1] = iBestY + 0.5f + fOffY;
	pfCent[2] = fZc;
	pfCent[3] = m_pfDark[i];
	pfCent[4] = m_pfStdRatio[i];
}

float CDetectFeatures2D::mParabolicPeak
(	float fLeft,
	float fCent,
	float fRight
)
{	float fDenom = fLeft - 2.0f * fCent + fRight;
	if(fDenom >= 0.0f) return 0.0f;
	float fOffset = 0.5f * (fLeft - fRight) / fDenom;
	if(fOffset > 0.5f) fOffset = 0.5f;
	else if(fOffset < -0.5f) fOffset = -0.5f;
	return fOffset;
}
