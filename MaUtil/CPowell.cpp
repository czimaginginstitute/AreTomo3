#include "CMaUtilInc.h"
#include <math.h>
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::MaUtil;

CPowell::CPowell(void)
{
	m_fTiny = (float)1e-25;
	m_iIterations = 200;
	m_fTol = 0.001f;
	m_iDim = 0;
	m_iNumSteps = 0;
	m_fInitVal = 0.0f;
	m_fBestVal = 0.0f;
	//--------------
	m_pfBestPoint = 0L;
	m_pfInitPoint = 0L;
	m_pfPointMin = 0L;
	m_pfPointMax = 0L;
	m_pfVectors = 0L;
}

CPowell::~CPowell(void)
{
	this->Clean();
}

float CPowell::Eval(float* pfPoint)
{
	return 0.0f;
}

void CPowell::Clean(void)
{
	if(m_pfBestPoint != 0L) delete[] m_pfBestPoint;
	if(m_pfInitPoint != 0L) delete[] m_pfInitPoint;
	if(m_pfPointMin != 0L) delete[] m_pfPointMin;
	if(m_pfPointMax != 0L) delete[] m_pfPointMax;
	if(m_pfVectors != 0L) delete[] m_pfVectors;
	m_pfBestPoint = 0L;
	m_pfInitPoint = 0L;
	m_pfPointMin = 0L;
	m_pfPointMax = 0L;
	m_pfVectors = 0L;
}

void CPowell::Setup(int iDim, int iIterations, float fTol)
{
	this->Clean();
	//------------
	m_iDim = iDim;
	m_pfInitPoint = new float[m_iDim];
	m_pfBestPoint = new float[m_iDim];
	m_pfPointMin = new float[m_iDim];
	m_pfPointMax = new float[m_iDim];
	m_pfVectors = new float[m_iDim * m_iDim];
	//---------------------------------------
	m_iIterations = iIterations;
	m_fTol = fTol;
}

float CPowell::DoIt
(	float* pfInitPoint,
	float* pfSearchRange,
	int iNumSteps
)
{	//-----------------------------------------------------------
	// Line search needs at least 3 samples for the parabolic
	// interpolation to be meaningful.
	//-----------------------------------------------------------
	m_iNumSteps = (iNumSteps < 3) ? 3 : iNumSteps;
	memcpy(m_pfInitPoint, pfInitPoint, sizeof(float) * m_iDim);
	//---------------------------------------------------------
	for(int i=0; i<m_iDim; i++)
	{	float fHalf = (float)fabs(pfSearchRange[i]) * 0.5f;
		m_pfPointMax[i] = m_pfInitPoint[i] + fHalf;
		m_pfPointMin[i] = m_pfInitPoint[i] - fHalf;
	}
	//-------------------------------------------------------------------
	memset(m_pfVectors, 0, sizeof(float) * m_iDim * m_iDim);
	for(int i=0; i<m_iDim; i++)
	{	float* pfVector = m_pfVectors + i * m_iDim;
		pfVector[i] = 1.0f;
	}
	m_fBestVal = mDoIt();
	return m_fBestVal;
}

float CPowell::mDoIt(void)
{
	float* pfPoint0 = new float[m_iDim]; // Point 0
	float* pfPointN = m_pfBestPoint;     // Point N - last point
	float* pfPointE = new float[m_iDim]; // extrapolated point
	float* pfVector = new float[m_iDim];
	//----------------------------------
	int iVectBytes = sizeof(float) * m_iDim;
	m_fBestVal = this->Eval(m_pfInitPoint);
	m_fInitVal = m_fBestVal;
	//-------------------------------------
	memcpy(pfPoint0, m_pfInitPoint, iVectBytes);
	memcpy(pfPointN, m_pfInitPoint, iVectBytes);
	//------------------------------------------
	int iIter = 0;
	for(iIter=0; iIter<m_iIterations; iIter++)
	{	float fMaxDrop = 0.0f;
		int iMaxVector = 0;
		float fLastBest = m_fBestVal;
		//-----------------------------------------------------------
		for(int i=0; i<m_iDim; i++) // find Point N
		{	memcpy(pfVector, m_pfVectors+i*m_iDim, iVectBytes);
			float fLastBest1 = m_fBestVal;
			mLineMinimize(pfPointN, pfVector);
			float fDrop = fLastBest1 - m_fBestVal;
			//------------------------------------
			if(fDrop <= fMaxDrop) continue;
			fMaxDrop = fDrop;
			iMaxVector = i;
		}
		float fErr = 2.0f * (fLastBest - m_fBestVal);
		double dTol = m_fTol * (fabs(fLastBest) + fabs(m_fBestVal))
		   + m_fTiny;
		if(fErr <= dTol) break;
		//--------------------------------------------------------
		// 1. Construct a vector that is the averaged direction of
		//    the point movement. Unew = PointN - Point0.
		// 2. Extrapolate PointN along this direction to PointE.
		//    PointE is clamped into the search box so that Eval
		//    is never called outside the allowed range.
		// 3. Replace Point0 with PointN for next iteration.
		//--------------------------------------------------------
		float* pfNewVect = pfVector;
		for(int i=0; i<m_iDim; i++)
		{	pfNewVect[i] = pfPointN[i] - pfPoint0[i];
			float fE = pfPointN[i] + pfNewVect[i];
			if(fE < m_pfPointMin[i]) fE = m_pfPointMin[i];
			else if(fE > m_pfPointMax[i]) fE = m_pfPointMax[i];
			pfPointE[i] = fE;
		}
		memcpy(pfPoint0, pfPointN, iVectBytes);
		//---------------------------------------
		// PointE is only a probe. Do NOT assign fValE to
		// m_fBestVal: m_fBestVal must remain the value at
		// pfPointN (m_pfBestPoint).
		//---------------------------------------
		float fValE = this->Eval(pfPointE);
		if(fValE >= fLastBest) continue; // new vector bad, discard
		//-------------------------------------------------------------
		// 1. The following checks if it is worth of keeping the
		//    new direction PointN - Point0.
		// 2. If fValE is close to fLastBest, it means the drop along
		//    the new direction is not significant. We should throw
		//    away this new direction.
		// 3. When fValE is close to fLastBest, fDelta2 is close to 0.
		//    fMaxDrop * fDelta2 * fDelt2 is then close to zero. t
		//    is then likely positive. This results in discarding the
		//    new direction.
		//-------------------------------------------------------------
		float fDelta1 = fLastBest - m_fBestVal - fMaxDrop;
		float fDelta2 = fLastBest - fValE;
		float t = 2.0f * (fLastBest - 2.0f * m_fBestVal + fValE)
			* fDelta1 * fDelta1 - fMaxDrop * fDelta2 * fDelta2;
		if(t >= 0) continue;
		//------------------------------------------------------------
		// 1. The new direction is good. Find the minimum along this
		//    direction. Replace Umax with Un and then replace Un
		//    with the new direction (PointN - Point0).
		//------------------------------------------------------------
		mLineMinimize(pfPointN, pfNewVect);
		float* pfVectN = m_pfVectors + (m_iDim - 1) * m_iDim;
		float* pfVectMax = m_pfVectors + iMaxVector * m_iDim;
		memcpy(pfVectMax, pfVectN, iVectBytes);
		memcpy(pfVectN, pfNewVect, iVectBytes);
	}
	//---------------------------------------------------------------
	if(pfPoint0 != 0L) delete[] pfPoint0;
	if(pfPointE != 0L) delete[] pfPointE;
	if(pfVector != 0L) delete[] pfVector;
	return m_fBestVal;
}

//-------------------------------------------------------------------
// Samples m_iNumSteps points uniformly over the allowable stride
// range [afRange[0], afRange[1]] (both ends inclusive), so every
// sample lies inside [m_pfPointMin, m_pfPointMax]. pfPoint is moved
// to the best location found and m_fBestVal is updated.
//-------------------------------------------------------------------
float CPowell::mLineMinimize(float* pfPoint, float* pfVector)
{
	float afRange[2] = {0.0f};
	mNormVector(pfVector); // vector normalized to unit vector
	mFindAllowableRange(pfPoint, pfVector, afRange);
	float fSpan = afRange[1] - afRange[0];
	if(fSpan <= 0) return m_fBestVal;
	//-------------------------------
	int iBestStep = -1;
	float* pfNewPoint = new float[m_iDim];
	float* pfVals = new float[m_iNumSteps];
	float fStep = fSpan / (m_iNumSteps - 1);
	//--------------------------------------
	for(int i=0; i<m_iNumSteps; i++)
	{	float fStride = afRange[0] + fStep * i;
		mCalcNewPoint(pfPoint, pfVector, fStride, pfNewPoint);
		pfVals[i] = this->Eval(pfNewPoint);
		if(pfVals[i] >= m_fBestVal) continue;
		m_fBestVal = pfVals[i];
		iBestStep = i;
	}
	if(iBestStep < 0)
	{	delete[] pfVals;
		delete[] pfNewPoint;
		return m_fBestVal;
	}
	//----------------------------------
	float fStride = afRange[0] + fStep * iBestStep;
	mCalcNewPoint(pfPoint, pfVector, fStride, pfPoint);
	if(iBestStep == 0 || iBestStep == (m_iNumSteps - 1))
	{	delete[] pfVals;
		delete[] pfNewPoint;
		return m_fBestVal;
	}
	//--------------------------------------------------------------
	// Numerical Recipes: Parabolic Interpolation and Brent's Method
	// |fFract| < 1 keeps the interpolated point between the two
	// neighboring samples, hence inside the allowed range.
	//--------------------------------------------------------------
	float fa = pfVals[iBestStep - 1];
	float fb = pfVals[iBestStep];
	float fc = pfVals[iBestStep + 1];
	delete[] pfVals;
	float fNorm = -0.5f * (fc - fa);
	float fDenorm = fa - 2.0f * fb + fc;
	float fFract = -10.0f;
	if(fDenorm != 0) fFract = fNorm / fDenorm;
	//---------------------------
	if(fFract <= -1 || fFract >= 1 || fFract == 0)
	{	delete[] pfNewPoint;
		return m_fBestVal;
	}
	//------------------------
	fStride = fStep * fFract;
	mCalcNewPoint(pfPoint, pfVector, fStride, pfNewPoint);
	float fValInt = this->Eval(pfNewPoint);
	if(fValInt < m_fBestVal)
	{	m_fBestVal = fValInt;
		memcpy(pfPoint, pfNewPoint, sizeof(float) * m_iDim);
	}
	delete[] pfNewPoint;
	return m_fBestVal;
}

//-------------------------------------------------------------------
// The result is clamped to [m_pfPointMin, m_pfPointMax] to guard
// against floating-point round-off at the range boundaries.
//-------------------------------------------------------------------
void CPowell::mCalcNewPoint
(	float* pfOldPoint, // starting point
	float* pfVector,   // must be unit vector
	float fStride,     // movement along pfVector
	float* pfNewPoint  // new point at the step
)
{	for(int i=0; i<m_iDim; i++)
	{	float fVal = pfOldPoint[i] + fStride * pfVector[i];
		if(fVal < m_pfPointMin[i]) fVal = m_pfPointMin[i];
		else if(fVal > m_pfPointMax[i]) fVal = m_pfPointMax[i];
		pfNewPoint[i] = fVal;
	}
}

//-------------------------------------------------------------------
// 1. pfRange[0] (<= 0) and pfRange[1] (>= 0) are the minimum and
//    maximum strides along pfVector starting from pfStartPoint
//    such that pfStartPoint + stride * pfVector stays inside
//    [m_pfPointMin, m_pfPointMax] for every dimension.
// 2. Components of pfVector may be positive or negative; both
//    cases are handled.
//-------------------------------------------------------------------
void CPowell::mFindAllowableRange
(	float* pfStartPoint,
	float* pfVector,
	float* pfRange
)
{	float fMaxNeg = (float)1e30; // max |stride| in -pfVector direction
	float fMaxPos = (float)1e30; // max stride in +pfVector direction
	bool bNonZero = false;
	//--------------------
	for(int i=0; i<m_iDim; i++)
	{	if(pfVector[i] == 0) continue;
		bNonZero = true;
		float fToMin = pfStartPoint[i] - m_pfPointMin[i];
		float fToMax = m_pfPointMax[i] - pfStartPoint[i];
		if(fToMin < 0) fToMin = 0.0f;
		if(fToMax < 0) fToMax = 0.0f;
		//---------------------------
		float fPos, fNeg;
		if(pfVector[i] > 0)
		{	fPos = fToMax / pfVector[i];
			fNeg = fToMin / pfVector[i];
		}
		else
		{	fPos = fToMin / -pfVector[i];
			fNeg = fToMax / -pfVector[i];
		}
		if(fPos < fMaxPos) fMaxPos = fPos;
		if(fNeg < fMaxNeg) fMaxNeg = fNeg;
	}
	if(!bNonZero) fMaxNeg = fMaxPos = 0.0f;
	pfRange[0] = -fMaxNeg;
	pfRange[1] = fMaxPos;
}

void CPowell::mNormVector(float* pfVector)
{
	float fMag = 0.0;
	for(int i=0; i<m_iDim; i++)
	{	fMag += (pfVector[i] * pfVector[i]);
	}
	if(fMag == 0) return;
	//---------------------------
	fMag = (float)sqrtf(fMag);
	for(int i=0; i<m_iDim; i++)
	{	pfVector[i] /= fMag;
	}
}
