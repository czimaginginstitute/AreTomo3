#include "CPatchAlignInc.h"
#include "../CAreTomoInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <Util/Util_LinEqs.h>
#include <memory.h>
#include <math.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::PatchAlign;

static float s_fD2R = 0.01745f;
static float s_fR2D = 57.2958f;

CFitPatchShifts::CFitPatchShifts(void)
{
	// ------------------------------------------------------------------
	// Locations of each patch at each tilt angle. U and V are horizontal
	// and vertical axes on image plane, respectively.
	// -----------------------------------------------
	m_pfMeasuredUs = 0L;
	m_pfCosTilts = 0L; // cosine of tilt angles
	m_pfCosRots = 0L; // refined rotation of each projection
}

CFitPatchShifts::~CFitPatchShifts(void)
{
	this->Clean();
}

void CFitPatchShifts::Clean(void)
{
	if(m_pfMeasuredUs != 0L) delete[] m_pfMeasuredUs;
	if(m_pfCosTilts != 0L) delete[] m_pfCosTilts;
	if(m_pfCosRots != 0L) delete[] m_pfCosRots;
	m_pfMeasuredUs = 0L;
	m_pfCosTilts = 0L;
	m_pfCosRots = 0L;
}

void CFitPatchShifts::Setup
(	MAM::CAlignParam* pFullParam,
	int iNumPatches
)
{	this->Clean();
	//------------
	m_pFullParam = pFullParam;
	m_iNumPatches = iNumPatches;
	m_iNumTilts = m_pFullParam->m_iNumFrames;
	m_iZeroTilt = m_pFullParam->GetFrameIdxFromTilt(0.0f);
	//----------------------------------------------------
	int iSize = m_iNumPatches * m_iNumTilts;
	m_pfMeasuredUs = new float[iSize * 3];
	m_pfMeasuredVs = m_pfMeasuredUs + iSize;
	//--------------------------------------
	m_pfCosTilts = new float[m_iNumTilts * 2];
	m_pfSinTilts = m_pfCosTilts + m_iNumTilts;
	float* pfTilts = m_pFullParam->GetTilts(false);
	for(int i=0; i<m_iNumTilts; i++)
	{	double dTilt = pfTilts[i] * s_fD2R;
		m_pfCosTilts[i] = (float)cos(dTilt);
		m_pfSinTilts[i] = (float)sin(dTilt);
	}
	//------------------------------------------
	m_pfCosRots = new float[m_iNumTilts * 3];
	m_pfSinRots = m_pfCosRots + m_iNumTilts;
	m_pfDeltaRots = m_pfCosRots + m_iNumTilts * 2;
}

float CFitPatchShifts::DoIt
(	MAM::CPatchShifts* pPatchShifts,
	MAM::CLocalAlignParam* pLocalParam
)
{	printf("Fit local motion.\n");
	m_pPatchShifts = pPatchShifts;
	m_pLocalParam = pLocalParam;
	//--------------------------
	m_pPatchShifts->GetPatShifts(m_pfMeasuredUs, m_pfMeasuredVs);
	//------------------------------------------------------
	memset(m_pfDeltaRots, 0, sizeof(float) * m_iNumTilts);
	mCalcLocalShifts();
	return m_fErr;
}

void CFitPatchShifts::mCalcSinCosRots(void)
{
	for(int i=0; i<m_iNumTilts; i++)
	{	float fTiltAxis = m_pFullParam->GetTiltAxis(i);
		float fRot = fTiltAxis * s_fD2R;
		m_pfCosRots[i] = (float)cos(fRot);
		m_pfSinRots[i] = (float)sin(fRot);
        }
}

float CFitPatchShifts::mCalcZs(void)
{
	float fMaxErr = 0.0f;
	for(int p=0; p<m_iNumPatches; p++)
	{	float fErr = mCalcPatchZ(p);
		if(fErr > fMaxErr) fMaxErr = fErr;
	}
	return fMaxErr;
}

float CFitPatchShifts::mCalcPatchZ(int iPatch)
{	
	int iOffset = iPatch * m_iNumTilts;
	bool* pbBadShifts = m_pPatchShifts->m_pbBadShifts;	
	//------------------------------------------------
	float afPatCent[3] = {0.0f};
	m_pPatchShifts->GetRotCenter(iPatch, afPatCent);
	//----------------------------------------------
	double dSumSin2 = 0, dSumSinCos = 0;
	double dSumTop = 0.0;
	float afGlobalShift[2] = {0.0f};
	//------------------------------
	for(int t=0; t<m_iNumTilts; t++)
	{	int i = iOffset + t;
		if(pbBadShifts[i]) continue;
		//--------------------------
		float fSin2 = m_pfSinTilts[t] * m_pfSinTilts[t];
		float fSinCos = m_pfSinTilts[t] * m_pfCosTilts[t];
		dSumSin2 += fSin2;
		dSumSinCos += fSinCos;
		//--------------------
		m_pFullParam->GetShift(t, afGlobalShift);
		float fResU = m_pfMeasuredUs[i] - afGlobalShift[0];
		float fResV = m_pfMeasuredVs[i] - afGlobalShift[1];
		dSumTop += (fResU * m_pfCosRots[t]
		   + fResV * m_pfSinRots[t]) * m_pfSinTilts[t];
	}
	dSumTop = afPatCent[0] * dSumSinCos - dSumTop;
	float fZ = (float)(dSumTop / (dSumSin2 + 1e-30));
	//-----------------------------------------------
	m_pPatchShifts->SetRotCenterZ(iPatch, fZ);
	//----------------------------------------	
	double dErr = fabs(fZ - afPatCent[2]);
	return (float)dErr;	
}

float CFitPatchShifts::mRefineTiltAxis(void)
{
	CAtInput* pInput = CAtInput::GetInstance();
	if(pInput->m_afTiltAxis[1] < 0) return 0.0f;	
	//-----------------
	float fMinDelta = 0.0f;
	float fMinErr = (float)1e30;
	for(int i=-100; i<=100; i++)
	{	float fDelta = i * 0.05f; 
		float fErr = mCalcTiltAxis(fDelta);
		if(fErr < fMinErr)
		{	fMinErr = fErr;
			fMinDelta = fDelta;
		}
		//printf(" %3d  %8.2f  %9.4e\n", i, fDelta, fErr);
	}
	//------------------------------------------------------
	float fOldTiltAxis = m_pFullParam->GetTiltAxis(0);
	float fNewTiltAxis = fOldTiltAxis + fMinDelta;
	m_pFullParam->SetTiltAxisAll(fNewTiltAxis);
	//-----------------------------------------
	for(int i=0; i<m_iNumPatches; i++)
	{	MAM::CAlignParam* pAlignParam = 
		   m_pPatchShifts->GetAlignParam(i);
		pAlignParam->SetTiltAxisAll(fNewTiltAxis);
	}
	return fMinDelta;
}

float CFitPatchShifts::mCalcTiltAxis(float fDelta)
{
	float fTiltAxis = m_pFullParam->GetTiltAxis(0);
	fTiltAxis = (fTiltAxis + fDelta) * s_fD2R;
	float fCosTA = (float)cos(fTiltAxis);
	float fSinTA = (float)sin(fTiltAxis);
	float fCosDT = (float)cos(fDelta * s_fD2R);
	float fSinDT = (float)sin(fDelta * s_fD2R);
	//-----------------------------------------
	bool* pbBadShifts = m_pPatchShifts->m_pbBadShifts;
	float afGlobalS[2] = {0.0f}, afPatchS[2] = {0.0f};
	float afPatCent[3] = {0.0f};
	double dError = 0.0;
	int iCount = 0;
	//------------------
	for(int p=0; p<m_iNumPatches; p++)
	{	m_pPatchShifts->GetRotCenter(p, afPatCent);
		float fPatX = afPatCent[0] * fCosDT + afPatCent[1] * fSinDT;
		float fPatY = afPatCent[1] * fCosDT - afPatCent[0] * fSinDT;
		for(int t=0; t<m_iNumTilts; t++)
		{	int i = t * m_iNumPatches + p;
			int j = p * m_iNumTilts + t;
			if(m_pPatchShifts->m_pbBadShifts[j]) continue;
			//--------------------------------------------
			m_pFullParam->GetShift(t, afGlobalS);
			m_pPatchShifts->GetShift(p, t, afPatchS);
			afPatchS[0] -= afGlobalS[0];
			afPatchS[1] -= afGlobalS[1];
			//--------------------------
			float fX = afPatCent[0] * m_pfCosTilts[t]
			   - afPatCent[2] * m_pfSinTilts[t];
			float fU = fX * fCosTA - fPatY * fSinTA;
			float fV = fX * fSinTA + fPatY * fCosTA;
			fU -= afPatchS[0];
			fV -= afPatchS[1];
			double dS = sqrtf(fU * fU + fV * fV);
			//if(dS > 100) continue;
			dError += dS;
			iCount += 1;
		}
	}
	float fErr = (float)(dError / (iCount + 1e-30));
	return fErr;
}

void CFitPatchShifts::mCalcLocalShifts(void)
{
	mCalcSinCosRots();
	float fErrZ = mCalcZs();
	//----------------------
	for(int p=0; p<m_iNumPatches; p++)
	{	mCalcPatchLocalShifts(p);
	}
	//-------------------------------
	int iSize = m_iNumTilts * m_iNumPatches;
	for(int iter=0; iter<10; iter++)
	{	memset(m_pPatchShifts->m_pbBadShifts, 0, sizeof(bool) * iSize);
		for(int t=0; t<m_iNumTilts; t++)
		{	mScreenTiltLocalShifts(t);
		}
		for(int p=0; p<m_iNumPatches; p++)
		{	mScreenPatchLocalShifts(p);
		}
		//--------------------------------
		fErrZ = mCalcZs();
		for(int p=0; p<m_iNumPatches; p++)
		{	mCalcPatchLocalShifts(p);
		}
		if(fErrZ < 1.0f) break;
	}
	for(int iter=0; iter<100; iter++)
	{	mRefineTiltAxis();
		mCalcSinCosRots(); // Missing before, added 07-18-2023
		float fTiltAxis = m_pFullParam->GetTiltAxis(0);
		float fErrZ = mCalcZs();
		printf("Refine tilt aixs and z: %8.2f  %9.5e\n", 
		   fTiltAxis, fErrZ);
		if(fErrZ < 1.0f && iter > 5) break;
	}
	//-----------------------------
	for(int p=0; p<m_iNumPatches; p++)
	{	int iOffset = p * m_iNumTilts;
		for(int t=0; t<m_iNumTilts; t++)
		{	m_pLocalParam->SetBad(t, p,
			   m_pPatchShifts->m_pbBadShifts[iOffset+t]);
		}
	}
	printf("\n");
}

void CFitPatchShifts::mCalcPatchLocalShifts(int iPatch)
{
	float fX = 0.0f, fU = 0.0f, fV = 0.0f, fSu = 0, fSv = 0;
	float afPatCent[3] = {0.0f}, afGlobalShift[2] = {0.0f};
	m_pPatchShifts->GetRotCenter(iPatch, afPatCent);
	//-----------------
	for(int t=0; t<m_iNumTilts; t++)
	{	fX = afPatCent[0] * m_pfCosTilts[t] 
		   - afPatCent[2] * m_pfSinTilts[t];
		fU = fX * m_pfCosRots[t] - afPatCent[1] * m_pfSinRots[t];
		fV = fX * m_pfSinRots[t] + afPatCent[1] * m_pfCosRots[t];
		//----------------
		int i = iPatch * m_iNumTilts + t;
		int j = t * m_iNumPatches + iPatch;
		m_pLocalParam->SetCoordXY(t, iPatch, fU, fV);
		//----------------
		m_pFullParam->GetShift(t, afGlobalShift);
		fSu = m_pfMeasuredUs[i] - afGlobalShift[0] - fU;
		fSv = m_pfMeasuredVs[i] - afGlobalShift[1] - fV;
		m_pLocalParam->SetShift(t, iPatch, fSu, fSv);
	}
}

void CFitPatchShifts::mScreenPatchLocalShifts(int iPatch)
{
        float* pfShiftS = new float[m_iNumTilts];
        double dMean = 0.0, dStd = 0.0;
        int iCount = 0;
        for(int t=0; t<m_iNumTilts; t++)
        {       int i = t * m_iNumPatches + iPatch;
                float fSx = m_pLocalParam->m_pfShiftXs[i];
                float fSy = m_pLocalParam->m_pfShiftYs[i];
                pfShiftS[t] = (float)sqrtf(fSx * fSx + fSy * fSy);
                //------------------------------------------------
                if(fabs(m_pFullParam->GetTilt(t)) > 30.5f) continue;
                dMean += pfShiftS[t];
                dStd += (pfShiftS[t] * pfShiftS[t]);
                iCount += 1;
        }
        dMean /= iCount;
        dStd = dStd / iCount - dMean * dMean;
        if(dStd < 0) dStd = 0;
        else dStd = sqrtf(dStd);
        float fTol = (float)(dMean + 4 * dStd);
        if(fTol < 50) fTol = 50.0f;
        else if(fTol > 200) fTol = 200.0f;
        //--------------------------------
        for(int t=0; t<m_iNumTilts; t++)
        {       if(pfShiftS[t] < fTol) continue;
                int j = iPatch * m_iNumTilts + t;
                m_pPatchShifts->m_pbBadShifts[j] = true;
                //---------------------------------------
                int i = t * m_iNumPatches + iPatch;
		/*
                printf("bad: %3d  %4d  %8.2f  %8.2f  %8.2f  %8.2f\n",
                   iPatch, t, m_pLocalParam->m_pfShiftXs[i],
                   m_pLocalParam->m_pfShiftYs[i], pfShiftS[t], fTol);
		*/
        }
        delete[] pfShiftS;
}


void CFitPatchShifts::mScreenTiltLocalShifts(int iTilt)
{
	int iOffset = iTilt * m_iNumPatches;
	double dSum1 = 0.0, dSum2 = 0.0;
	//------------------------------
	for(int p=0; p<m_iNumPatches; p++)
	{	int i = iOffset + p;
		float fSx = m_pLocalParam->m_pfShiftXs[i];
		float fSy = m_pLocalParam->m_pfShiftYs[i];
		float fS2 = fSx * fSx + fSy * fSy;
		dSum1 += sqrtf(fS2);
		dSum2 += fS2;
	}
	float fMean = (float)(dSum1 / m_iNumPatches);
	float fStd = (float)(dSum2 / m_iNumPatches - fMean * fMean);
	if(fStd <= 0) fStd = 0.0;
	else fStd = (float)sqrtf(fStd);
	//-----------------------------
	float fTol = fMean + 4.0f * fStd;
	if(fTol < 50) fTol = 50.0f;
	else if(fTol > 200) fTol = 200.0f;
	//--------------------------------
	bool* pbBadShifts = m_pPatchShifts->m_pbBadShifts;
	//------------------------------------------------
	for(int p=0; p<m_iNumPatches; p++)
	{	int i = iOffset + p;
		float fSx = m_pLocalParam->m_pfShiftXs[i];
		float fSy = m_pLocalParam->m_pfShiftYs[i];
		float fS = (float)sqrtf(fSx * fSx + fSy * fSy);
		//---------------------------------------------
		int j = p * m_iNumTilts + iTilt;
		if(fS > fTol) pbBadShifts[j] = true;
	}
}

