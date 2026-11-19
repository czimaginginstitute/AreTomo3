#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>

using namespace McAreTomo::AreTomo::MrcUtil;

//--------------------------------------------------------------------
// Tunable parameters.
// 1. The tilt range is given to DoIt (-BadTiltRange, default
//    [-20, 20]); only images within it are tested.
// 2. s_fBinSize: images are binned so their larger dimension is
//    about this many pixels.
// 3. s_fLowSigma, s_fHighSigma: Gaussian bandpass in cycles per
//    binned pixel.
// 4. s_fMinOverlap: shifts with less overlap than this fraction of
//    the mask area are not considered. With the 5% mask margin,
//    0.2 keeps shifts up to ~50% in both x and y measurable.
// 5. s_fMinNCC: pairs with lower NCC carry no information.
// 6. s_fMinInlier: an image whose fraction of consistent pairs is
//    below this cannot be located reliably.
// 7. An image is bad (shift) if its leave-one-out deviation from
//    the trend passes both tests:
//    a) s_fJumpTol: absolute floor, exceeds this fraction of the
//       image width in x or of the image height in y;
//    b) s_fJumpSigmas: the deviation magnitude exceeds this many
//       robust per-axis sigmas of the trend residuals, so the
//       threshold adapts to the scatter of each tilt series.
//--------------------------------------------------------------------
static float s_fBinSize = 1024.0f;
static float s_fLowSigma = 0.15f;
static float s_fHighSigma = 0.008f;
static float s_fMinOverlap = 0.20f;
static float s_fMinNCC = 0.05f;
static float s_fMinInlier = 0.50f;
static float s_fJumpTol = 0.05f;
static float s_fJumpSigmas = 6.0f;
static int s_iMinImgs = 5;
static int s_iIterations = 10;

//--------------------------------------------------------------------
// Solves pdA * x = pdB by Gaussian elimination with partial pivoting.
// pdA is iN x iN (row major) and pdB is iN x iRhs. Both are
// destroyed. The solution is returned in pdB.
//--------------------------------------------------------------------
static bool sSolveLinear(double* pdA, double* pdB, int iN, int iRhs)
{
	for(int c=0; c<iN; c++)
	{	int iPivot = c;
		for(int r=c+1; r<iN; r++)
		{	if(fabs(pdA[r*iN+c]) > fabs(pdA[iPivot*iN+c])) iPivot = r;
		}
		if(fabs(pdA[iPivot*iN+c]) < 1e-12) return false;
		if(iPivot != c)
		{	for(int k=0; k<iN; k++)
			{	std::swap(pdA[c*iN+k], pdA[iPivot*iN+k]);
			}
			for(int k=0; k<iRhs; k++)
			{	std::swap(pdB[c*iRhs+k], pdB[iPivot*iRhs+k]);
			}
		}
		for(int r=0; r<iN; r++)
		{	if(r == c) continue;
			double dF = pdA[r*iN+c] / pdA[c*iN+c];
			if(dF == 0) continue;
			for(int k=c; k<iN; k++) pdA[r*iN+k] -= dF * pdA[c*iN+k];
			for(int k=0; k<iRhs; k++)
			{	pdB[r*iRhs+k] -= dF * pdB[c*iRhs+k];
			}
		}
	}
	for(int r=0; r<iN; r++)
	{	for(int k=0; k<iRhs; k++) pdB[r*iRhs+k] /= pdA[r*iN+r];
	}
	return true;
}

static float sMedian(float* pfVals, int iNum)
{
	if(iNum <= 0) return 0.0f;
	float* pfBuf = new float[iNum];
	memcpy(pfBuf, pfVals, sizeof(float) * iNum);
	std::sort(pfBuf, pfBuf + iNum);
	float fMed = (iNum % 2 == 1) ? pfBuf[iNum/2] :
	   0.5f * (pfBuf[iNum/2 - 1] + pfBuf[iNum/2]);
	delete[] pfBuf;
	return fMed;
}

//--------------------------------------------------------------------
// IRLS weight for residual fRes given robust sigma fSigma. The first
// s_iIterations use Huber, which never zeroes a weight, so that an
// outlier in the initial least-squares fit cannot push good data out
// of the Tukey window. The remaining iterations use Tukey biweight,
// which removes outliers completely.
//--------------------------------------------------------------------
static float sRobustWeight(float fRes, float fSigma, int iIter)
{
	if(iIter < s_iIterations)
	{	float fC = 1.345f * fSigma;
		return (fRes <= fC) ? 1.0f : fC / fRes;
	}
	float fC = 4.685f * fSigma;
	if(fRes >= fC) return 0.0f;
	float fR = fRes / fC;
	float fW = 1.0f - fR * fR;
	return fW * fW;
}

CDetectBadShifts::CDetectBadShifts(void)
{
	m_piFrmIdxs = 0L;
	m_pfTilts = 0L;
	m_ppfBinImgs = 0L;
	m_pfPairShifts = 0L;
	m_pfPairNCCs = 0L;
	m_pfPairWgts = 0L;
	m_pfShifts = 0L;
	m_pfInliers = 0L;
	m_pfLooErrs = 0L;
	m_pbBadImgs = 0L;
	m_iNumImgs = 0;
}

CDetectBadShifts::~CDetectBadShifts(void)
{
	mClean();
}

void CDetectBadShifts::mClean(void)
{
	if(m_ppfBinImgs != 0L)
	{	for(int i=0; i<m_iNumImgs; i++)
		{	if(m_ppfBinImgs[i] != 0L) delete[] m_ppfBinImgs[i];
		}
		delete[] m_ppfBinImgs;
	}
	if(m_piFrmIdxs != 0L) delete[] m_piFrmIdxs;
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_pfPairShifts != 0L) delete[] m_pfPairShifts;
	if(m_pfPairNCCs != 0L) delete[] m_pfPairNCCs;
	if(m_pfPairWgts != 0L) delete[] m_pfPairWgts;
	if(m_pfShifts != 0L) delete[] m_pfShifts;
	if(m_pfInliers != 0L) delete[] m_pfInliers;
	if(m_pfLooErrs != 0L) delete[] m_pfLooErrs;
	if(m_pbBadImgs != 0L) delete[] m_pbBadImgs;
	m_ppfBinImgs = 0L;
	m_piFrmIdxs = 0L;
	m_pfTilts = 0L;
	m_pfPairShifts = 0L;
	m_pfPairNCCs = 0L;
	m_pfPairWgts = 0L;
	m_pfShifts = 0L;
	m_pfInliers = 0L;
	m_pfLooErrs = 0L;
	m_pbBadImgs = 0L;
	m_iNumImgs = 0;
}

void CDetectBadShifts::DoIt(int iNthGpu, float* pfTiltRange)
{
	mClean();
	m_iNthGpu = iNthGpu;
	m_afTiltRange[0] = pfTiltRange[0];
	m_afTiltRange[1] = pfTiltRange[1];
	if(m_afTiltRange[1] <= m_afTiltRange[0]) return;
	MD::CTsPackage* pPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	m_pTiltSeries = pPackage->GetSeries(0);
	//-----------------
	if(!mFindLowTilts())
	{	mClean();
		return;
	}
	mBinImages();
	mMeasurePairs();
	mSolveShifts();
	mTestTrend();
	mFlagImages();
	mClean();
}

//--------------------------------------------------------------------
// Collect the non-dark images within m_afTiltRange. The series is
// sorted by tilt angle, so the collected images are too.
//--------------------------------------------------------------------
bool CDetectBadShifts::mFindLowTilts(void)
{
	CBadImages* pBadImages = CBadImages::GetInstance(m_iNthGpu);
	int iAllFrms = m_pTiltSeries->m_aiStkSize[2];
	m_piFrmIdxs = new int[iAllFrms];
	m_pfTilts = new float[iAllFrms];
	//-----------------
	m_iNumImgs = 0;
	for(int i=0; i<iAllFrms; i++)
	{	float fTilt = m_pTiltSeries->m_pfTilts[i];
		if(fTilt < m_afTiltRange[0]) continue;
		if(fTilt > m_afTiltRange[1]) continue;
		if(pBadImages->IsDarkImg(i)) continue;
		m_piFrmIdxs[m_iNumImgs] = i;
		m_pfTilts[m_iNumImgs] = fTilt;
		m_iNumImgs += 1;
	}
	//-----------------
	if(m_iNumImgs < s_iMinImgs)
	{	printf("GPU %d: skip bad-shift detection, only %d images "
		   "within [%.1f, %.1f] deg.\n\n", m_iNthGpu, m_iNumImgs,
		   m_afTiltRange[0], m_afTiltRange[1]);
		return false;
	}
	return true;
}

//--------------------------------------------------------------------
// Bin and bandpass each image, then normalize it to zero mean and
// unit variance inside the mask and zero it outside. The margin
// keeps edge artifacts (bandpass wrap-around, detector borders)
// out of the correlation because they would favor zero shift.
//--------------------------------------------------------------------
void CDetectBadShifts::mBinImages(void)
{
	int* piRawSize = m_pTiltSeries->m_aiStkSize;
	int iMaxDim = std::max(piRawSize[0], piRawSize[1]);
	float fBin = std::max(1.0f, iMaxDim / s_fBinSize);
	MU::GFourierResize2D::GetBinnedImgSize(piRawSize, fBin,
	   m_aiBinSize);
	m_afBinning[0] = piRawSize[0] / (float)m_aiBinSize[0];
	m_afBinning[1] = piRawSize[1] / (float)m_aiBinSize[1];
	//-----------------
	for(int i=0; i<2; i++)
	{	int iMargin = std::max(8, (int)(m_aiBinSize[i] * 0.05f));
		m_aiMaskStart[i] = iMargin;
		m_aiMaskSize[i] = m_aiBinSize[i] - 2 * iMargin;
	}
	//-----------------
	GBinBandpass2D binBandpass;
	binBandpass.Setup(piRawSize, m_aiBinSize, s_fLowSigma,
	   s_fHighSigma);
	int iPixels = m_aiBinSize[0] * m_aiBinSize[1];
	m_ppfBinImgs = new float*[m_iNumImgs];
	//-----------------
	for(int i=0; i<m_iNumImgs; i++)
	{	float* pfImg = new float[iPixels];
		m_ppfBinImgs[i] = pfImg;
		float* pfFrm = (float*)m_pTiltSeries->GetFrame(m_piFrmIdxs[i]);
		binBandpass.DoIt(pfFrm, pfImg);
		//----------------
		double dSum = 0.0, dSum2 = 0.0;
		for(int y=0; y<m_aiMaskSize[1]; y++)
		{	float* pfRow = pfImg + (y + m_aiMaskStart[1]) *
			   m_aiBinSize[0] + m_aiMaskStart[0];
			for(int x=0; x<m_aiMaskSize[0]; x++)
			{	dSum += pfRow[x];
				dSum2 += pfRow[x] * pfRow[x];
			}
		}
		double dN = (double)m_aiMaskSize[0] * m_aiMaskSize[1];
		double dMean = dSum / dN;
		double dStd = dSum2 / dN - dMean * dMean;
		dStd = (dStd > 0) ? sqrt(dStd) : 1.0;
		//----------------
		for(int y=0; y<m_aiBinSize[1]; y++)
		{	bool bInY = (y >= m_aiMaskStart[1]) &&
			   (y < m_aiMaskStart[1] + m_aiMaskSize[1]);
			float* pfRow = pfImg + y * m_aiBinSize[0];
			for(int x=0; x<m_aiBinSize[0]; x++)
			{	bool bIn = bInY && (x >= m_aiMaskStart[0]) &&
				   (x < m_aiMaskStart[0] + m_aiMaskSize[0]);
				if(bIn) pfRow[x] = (float)((pfRow[x] - dMean) / dStd);
				else pfRow[x] = 0.0f;
			}
		}
	}
}

//--------------------------------------------------------------------
// All-pairs overlap-normalized NCC. m_pfPairShifts[(i*N+j)*2] is
// the position of image j relative to image i in raw pixels.
//--------------------------------------------------------------------
void CDetectBadShifts::mMeasurePairs(void)
{
	int iN = m_iNumImgs;
	m_pfPairShifts = new float[iN * iN * 2];
	m_pfPairNCCs = new float[iN * iN];
	memset(m_pfPairShifts, 0, sizeof(float) * iN * iN * 2);
	for(int i=0; i<iN*iN; i++) m_pfPairNCCs[i] = 0.0f;
	//-----------------
	GMaskedNCC2D maskedNCC;
	maskedNCC.Setup(m_aiBinSize, m_aiMaskStart, m_aiMaskSize,
	   s_fMinOverlap);
	float afShift[2] = {0.0f};
	for(int i=0; i<iN; i++)
	{	maskedNCC.SetRef(m_ppfBinImgs[i]);
		for(int j=i+1; j<iN; j++)
		{	float fNCC = maskedNCC.DoIt(m_ppfBinImgs[j], afShift);
			float fSx = afShift[0] * m_afBinning[0];
			float fSy = afShift[1] * m_afBinning[1];
			m_pfPairShifts[(i*iN+j)*2] = fSx;
			m_pfPairShifts[(i*iN+j)*2+1] = fSy;
			m_pfPairShifts[(j*iN+i)*2] = -fSx;
			m_pfPairShifts[(j*iN+i)*2+1] = -fSy;
			m_pfPairNCCs[i*iN+j] = fNCC;
			m_pfPairNCCs[j*iN+i] = fNCC;
		}
	}
}

//--------------------------------------------------------------------
// Minimize sum_ij w_ij |p_j - p_i - s_ij|^2 subject to sum_k p_k = 0.
// No image is used as the reference since the zero-tilt image is
// often the bad one. w_ij is the pair NCC times a Tukey weight
// updated from the residuals.
// Afterwards m_pfInliers[k] is the fraction of image k's pairs that
// are both informative and consistent with the solution.
//--------------------------------------------------------------------
void CDetectBadShifts::mSolveShifts(void)
{
	int iN = m_iNumImgs;
	m_pfShifts = new float[iN * 2];
	m_pfPairWgts = new float[iN * iN];
	m_pfInliers = new float[iN];
	memset(m_pfShifts, 0, sizeof(float) * iN * 2);
	for(int i=0; i<iN*iN; i++) m_pfPairWgts[i] = 1.0f;
	//-----------------
	double* pdA = new double[iN * iN];
	double* pdB = new double[iN * 2];
	float* pfRes = new float[iN * iN];
	float fMinSigma = 2.0f * std::max(m_afBinning[0], m_afBinning[1]);
	//-----------------
	for(int iter=0; iter<2*s_iIterations; iter++)
	{	memset(pdA, 0, sizeof(double) * iN * iN);
		memset(pdB, 0, sizeof(double) * iN * 2);
		for(int i=0; i<iN; i++)
		{	for(int j=i+1; j<iN; j++)
			{	float fNCC = m_pfPairNCCs[i*iN+j];
				if(fNCC < s_fMinNCC) continue;
				double dW = fNCC * m_pfPairWgts[i*iN+j];
				float* pfS = m_pfPairShifts + (i*iN+j) * 2;
				pdA[i*iN+i] += dW;
				pdA[j*iN+j] += dW;
				pdA[i*iN+j] -= dW;
				pdA[j*iN+i] -= dW;
				pdB[j*2] += dW * pfS[0];
				pdB[j*2+1] += dW * pfS[1];
				pdB[i*2] -= dW * pfS[0];
				pdB[i*2+1] -= dW * pfS[1];
			}
		}
		//-----------------------------------------------
		// Gauge: the pair terms only fix relative shifts.
		// Adding dLambda * (sum_k p_k)^2 makes the mean
		// shift zero without favoring any image, because
		// the entries of pdB sum to zero. The small
		// diagonal term keeps isolated images solvable;
		// they are flagged by their inliers.
		//-----------------------------------------------
		double dLambda = 0.0;
		for(int k=0; k<iN; k++) dLambda += pdA[k*iN+k];
		dLambda = (dLambda > 0) ? dLambda / iN : 1.0;
		for(int k=0; k<iN*iN; k++) pdA[k] += dLambda;
		for(int k=0; k<iN; k++) pdA[k*iN+k] += 1e-6 * dLambda;
		if(!sSolveLinear(pdA, pdB, iN, 2)) break;
		for(int k=0; k<iN*2; k++) m_pfShifts[k] = (float)pdB[k];
		//-----------------
		int iCount = 0;
		for(int i=0; i<iN; i++)
		{	for(int j=i+1; j<iN; j++)
			{	if(m_pfPairNCCs[i*iN+j] < s_fMinNCC) continue;
				float* pfS = m_pfPairShifts + (i*iN+j) * 2;
				float fDx = m_pfShifts[j*2] - m_pfShifts[i*2] - pfS[0];
				float fDy = m_pfShifts[j*2+1] - m_pfShifts[i*2+1]
				   - pfS[1];
				pfRes[iCount] = sqrtf(fDx * fDx + fDy * fDy);
				iCount += 1;
			}
		}
		float fSigma = 1.4826f * sMedian(pfRes, iCount);
		fSigma = std::max(fSigma, fMinSigma);
		//-----------------
		for(int i=0; i<iN; i++)
		{	for(int j=i+1; j<iN; j++)
			{	float* pfS = m_pfPairShifts + (i*iN+j) * 2;
				float fDx = m_pfShifts[j*2] - m_pfShifts[i*2] - pfS[0];
				float fDy = m_pfShifts[j*2+1] - m_pfShifts[i*2+1]
				   - pfS[1];
				float fW = sRobustWeight(sqrtf(fDx * fDx
				   + fDy * fDy), fSigma, iter);
				m_pfPairWgts[i*iN+j] = fW;
				m_pfPairWgts[j*iN+i] = fW;
			}
		}
	}
	//-----------------
	for(int k=0; k<iN; k++)
	{	int iGood = 0;
		for(int j=0; j<iN; j++)
		{	if(j == k) continue;
			if(m_pfPairNCCs[k*iN+j] < s_fMinNCC) continue;
			if(m_pfPairWgts[k*iN+j] <= 0.0f) continue;
			iGood += 1;
		}
		m_pfInliers[k] = iGood / (float)(iN - 1);
	}
	delete[] pdA;
	delete[] pdB;
	delete[] pfRes;
}

//--------------------------------------------------------------------
// Fit p(t) = c0 + c1 t + c2 t^2 to the solved shifts, where t is
// (tilt - center) / half-width of m_afTiltRange. Image iSkip (if >= 0) is left out. The order
// drops when there are too few images with nonzero weight.
//--------------------------------------------------------------------
void CDetectBadShifts::mFitTrend
(	float* pfWeights,
	int iSkip,
	double* pdCoeffX,
	double* pdCoeffY
)
{	int iUsed = 0;
	for(int k=0; k<m_iNumImgs; k++)
	{	if(k != iSkip && pfWeights[k] > 0) iUsed += 1;
	}
	int iOrder = (iUsed >= 6) ? 3 : ((iUsed >= 3) ? 2 : 1);
	//-----------------
	double adA[9] = {0.0}, adB[6] = {0.0};
	for(int k=0; k<m_iNumImgs; k++)
	{	if(k == iSkip || pfWeights[k] <= 0) continue;
		double dT = mNormTilt(m_pfTilts[k]);
		double adT[3] = {1.0, dT, dT * dT};
		for(int r=0; r<iOrder; r++)
		{	for(int c=0; c<iOrder; c++)
			{	adA[r*iOrder+c] += pfWeights[k] * adT[r] * adT[c];
			}
			adB[r*2] += pfWeights[k] * adT[r] * m_pfShifts[k*2];
			adB[r*2+1] += pfWeights[k] * adT[r] * m_pfShifts[k*2+1];
		}
	}
	for(int r=0; r<3; r++)
	{	pdCoeffX[r] = 0.0;
		pdCoeffY[r] = 0.0;
	}
	if(!sSolveLinear(adA, adB, iOrder, 2)) return;
	for(int r=0; r<iOrder; r++)
	{	pdCoeffX[r] = adB[r*2];
		pdCoeffY[r] = adB[r*2+1];
	}
}

float CDetectBadShifts::mNormTilt(float fTilt)
{
	float fCent = 0.5f * (m_afTiltRange[0] + m_afTiltRange[1]);
	float fHalf = 0.5f * (m_afTiltRange[1] - m_afTiltRange[0]);
	return (fTilt - fCent) / fHalf;
}

float CDetectBadShifts::mCalcTrend(double* pdCoeff, float fTilt)
{
	double dT = mNormTilt(fTilt);
	double dVal = pdCoeff[0] + pdCoeff[1] * dT + pdCoeff[2] * dT * dT;
	return (float)dVal;
}

//--------------------------------------------------------------------
// 1. Robust trend: Huber then Tukey IRLS over images that can be
//    located.
// 2. Leave-one-out: each image is compared with the trend fitted
//    to the other images using their robust weights.
//--------------------------------------------------------------------
void CDetectBadShifts::mTestTrend(void)
{
	int iN = m_iNumImgs;
	m_pfLooErrs = new float[iN * 2];
	float* pfWeights = new float[iN];
	float* pfErrs = new float[iN];
	float* pfBuf = new float[iN];
	double adCx[3], adCy[3];
	float fMinSigma = 2.0f * std::max(m_afBinning[0], m_afBinning[1]);
	m_fTrendSigma = fMinSigma;
	float fIrlsSigma = fMinSigma;
	float fMedErr = 0.0f;
	//-----------------
	for(int k=0; k<iN; k++)
	{	pfWeights[k] = (m_pfInliers[k] >= s_fMinInlier) ? 1.0f : 0.0f;
	}
	for(int iter=0; iter<2*s_iIterations; iter++)
	{	mFitTrend(pfWeights, -1, adCx, adCy);
		int iCount = 0;
		for(int k=0; k<iN; k++)
		{	float fDx = m_pfShifts[k*2] - mCalcTrend(adCx, m_pfTilts[k]);
			float fDy = m_pfShifts[k*2+1] - mCalcTrend(adCy,
			   m_pfTilts[k]);
			pfErrs[k] = sqrtf(fDx * fDx + fDy * fDy);
			if(m_pfInliers[k] >= s_fMinInlier)
			{	pfBuf[iCount] = pfErrs[k];
				iCount += 1;
			}
		}
		fMedErr = sMedian(pfBuf, iCount);
		fIrlsSigma = std::max(1.4826f * fMedErr, fMinSigma);
		for(int k=0; k<iN; k++)
		{	if(m_pfInliers[k] < s_fMinInlier) continue;
			pfWeights[k] = sRobustWeight(pfErrs[k], fIrlsSigma,
			   iter);
		}
	}
	//-----------------------------------------------
	// Per-axis sigma for flagging. pfErrs are 2-D
	// magnitudes: for Gaussian x, y residuals of
	// sigma s, the median magnitude is s*sqrt(2 ln 2)
	// = 1.1774 s. 1.4826 (the 1-D factor) would
	// overstate s by 1.75x.
	//-----------------------------------------------
	m_fTrendSigma = std::max(fMedErr / 1.1774f, fMinSigma);
	//-----------------
	for(int k=0; k<iN; k++)
	{	mFitTrend(pfWeights, k, adCx, adCy);
		float fDx = m_pfShifts[k*2] - mCalcTrend(adCx, m_pfTilts[k]);
		float fDy = m_pfShifts[k*2+1] - mCalcTrend(adCy, m_pfTilts[k]);
		m_pfLooErrs[k*2] = fDx;
		m_pfLooErrs[k*2+1] = fDy;
	}
	delete[] pfWeights;
	delete[] pfErrs;
	delete[] pfBuf;
}

//--------------------------------------------------------------------
// An image is bad if it cannot be located consistently, or if its
// leave-one-out deviation from the trend is large both in absolute
// terms and relative to the scatter of the other images. If more
// than half of the images would be flagged, the trend itself is
// unreliable and nothing is flagged.
//--------------------------------------------------------------------
void CDetectBadShifts::mFlagImages(void)
{
	int iN = m_iNumImgs;
	m_pbBadImgs = new bool[iN];
	int* piRawSize = m_pTiltSeries->m_aiStkSize;
	float afTol[2] = {0.0f};
	afTol[0] = s_fJumpTol * piRawSize[0];
	afTol[1] = s_fJumpTol * piRawSize[1];
	float fSigmaTol = s_fJumpSigmas * m_fTrendSigma;
	//-----------------
	int iNumBad = 0;
	for(int k=0; k<iN; k++)
	{	bool bUnreliable = (m_pfInliers[k] < s_fMinInlier);
		float fLooX = m_pfLooErrs[k*2];
		float fLooY = m_pfLooErrs[k*2+1];
		bool bAbsJump = (fabsf(fLooX) > afTol[0]) ||
		   (fabsf(fLooY) > afTol[1]);
		bool bSigJump = sqrtf(fLooX * fLooX + fLooY * fLooY)
		   > fSigmaTol;
		bool bJump = bAbsJump && bSigJump;
		m_pbBadImgs[k] = bUnreliable || bJump;
		if(m_pbBadImgs[k]) iNumBad += 1;
	}
	//-----------------
	printf("GPU %d: bad-shift detection within [%.1f, %.1f] deg, "
	   "tolerance %.0f%% of image size (%.1f, %.1f) pixels and "
	   "%.1f x trend sigma (%.1f) = %.1f pixels\n",
	   m_iNthGpu, m_afTiltRange[0], m_afTiltRange[1],
	   s_fJumpTol * 100.0f, afTol[0], afTol[1],
	   s_fJumpSigmas, m_fTrendSigma, fSigmaTol);
	printf("  frm    tilt    shift-x   shift-y   loo-x%%  loo-y%%"
	   "  inlier  max-ncc\n");
	for(int k=0; k<iN; k++)
	{	float fMaxNCC = 0.0f;
		for(int j=0; j<iN; j++)
		{	if(j == k) continue;
			fMaxNCC = std::max(fMaxNCC, m_pfPairNCCs[k*iN+j]);
		}
		const char* pcFlag = "";
		if(m_pfInliers[k] < s_fMinInlier) pcFlag = "bad (unreliable)";
		else if(m_pbBadImgs[k]) pcFlag = "bad (shift)";
		printf(" %4d  %7.2f  %9.2f %9.2f  %6.1f  %6.1f   %5.2f  %6.3f"
		   "  %s\n", m_piFrmIdxs[k], m_pfTilts[k], m_pfShifts[k*2],
		   m_pfShifts[k*2+1], 
		   100.0f * m_pfLooErrs[k*2] / piRawSize[0],
		   100.0f * m_pfLooErrs[k*2+1] / piRawSize[1],
		   m_pfInliers[k], fMaxNCC, pcFlag);
	}
	//-----------------
	if(iNumBad * 2 > iN)
	{	printf("GPU %d: warning, %d of %d low-tilt images look bad, "
		   "the trend is unreliable, nothing is flagged.\n\n",
		   m_iNthGpu, iNumBad, iN);
		return;
	}
	if(iNumBad == 0)
	{	printf("GPU %d: no bad-shift images detected.\n\n", m_iNthGpu);
		return;
	}
	//-----------------
	CBadImages* pBadImages = CBadImages::GetInstance(m_iNthGpu);
	for(int k=0; k<iN; k++)
	{	if(!m_pbBadImgs[k]) continue;
		int iFrm = m_piFrmIdxs[k];
		pBadImages->AddBadImg(iFrm, m_pTiltSeries->m_piSecIndices[iFrm],
		   m_pfTilts[k]);
	}
	printf("GPU %d: %d bad-shift images flagged.\n\n",
	   m_iNthGpu, iNumBad);
}
