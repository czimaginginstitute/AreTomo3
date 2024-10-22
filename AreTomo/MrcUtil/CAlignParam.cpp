#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::MrcUtil;

CAlignParam* CAlignParam::m_pInstances = 0L;
int CAlignParam::m_iNumGpus = 0;
static float s_fD2R = 0.017453f;

void CAlignParam::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CAlignParam[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CAlignParam::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CAlignParam* CAlignParam::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

void CAlignParam::RotShift(float* pfInShift, 
	float fRotAngle, float* pfOutShift
)
{	double dAngle = 3.1415926 * fRotAngle / 180.0;
        float fCos = (float)cos(dAngle);
        float fSin = (float)sin(dAngle);
        float fSx = pfInShift[0] * fCos - pfInShift[1] * fSin;
        float fSy = pfInShift[0] * fSin + pfInShift[1] * fCos;
	//------------------
	pfOutShift[0] = fSx;
	pfOutShift[1] = fSy;
}

CAlignParam::CAlignParam(void)
{
	m_piSecIndex = 0L;
	m_pfTilts = 0L;
	m_pfTiltAxis = 0L;
	m_pfShiftXs = 0L;
	m_pfShiftYs = 0L;
	//-----------------
	m_fAlphaOffset = 0.0f;
	m_fBetaOffset = 0.0f;
	m_iThickness = 0;
	//-----------------
	m_iNumFrames = 0;
	m_fZ0 = 0.0f;
}

CAlignParam::~CAlignParam(void)
{
	this->Clean();
}

void CAlignParam::Clean(void)
{
	if(m_piSecIndex != 0L) delete[] m_piSecIndex;
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_pfTiltAxis != 0L) delete[] m_pfTiltAxis;
	if(m_pfShiftXs != 0L) delete[] m_pfShiftXs;
	//---------------------------------------------
	m_piSecIndex = 0L;
	m_pfTilts = 0L;
	m_pfTiltAxis = 0L;
	m_pfShiftXs = 0L;
	m_pfShiftYs = 0L;
}

void CAlignParam::Create(int iNumFrames)
{
	this->Clean();
	m_iNumFrames = iNumFrames;	
	//------------------------
	m_piSecIndex = new int[m_iNumFrames];
	for(int i=0; i<m_iNumFrames; i++) m_piSecIndex[i] = i;
	//-----------------------------------------------------
	m_pfTilts = new float[m_iNumFrames];
	memset(m_pfTilts , 0, sizeof(float) * m_iNumFrames);
	//--------------------------------------------------
	m_pfTiltAxis = new float[m_iNumFrames];
	memset(m_pfTiltAxis, 0, sizeof(float) * m_iNumFrames);
	//----------------------------------------------------
	int iSize = m_iNumFrames * 2;
	m_pfShiftXs = new float[iSize];
	m_pfShiftYs = m_pfShiftXs + m_iNumFrames;
	memset(m_pfShiftXs, 0, sizeof(float) * iSize);
}


void CAlignParam::SetSecIndex(int iFrame, int iSecIdx)
{
	m_piSecIndex[iFrame] = iSecIdx;
}

void CAlignParam::SetTilt(int iFrame, float fTilt)
{
	m_pfTilts[iFrame] = fTilt;
}

void CAlignParam::SetTiltAxis(int iFrame, float fTiltAxis)
{
	m_pfTiltAxis[iFrame] = fTiltAxis;
}

void CAlignParam::SetTiltAxisAll(float fTiltAxis)
{
	for(int i=0; i<m_iNumFrames; i++)
	{	m_pfTiltAxis[i] = fTiltAxis;
	}
}

void CAlignParam::SetShift(int iFrame, float* pfShift)
{
	m_pfShiftXs[iFrame] = pfShift[0];
	m_pfShiftYs[iFrame] = pfShift[1];
}

void CAlignParam::SetCenter(float fCentX, float fCentY)
{
	m_afCenter[0] = fCentX;
	m_afCenter[1] = fCentY;
}

void CAlignParam::SetTiltRange(float fEndAng1, float fEndAng2)
{
	if(fEndAng1 < fEndAng2)
	{	m_afTiltRange[0] = fEndAng1;
		m_afTiltRange[1] = fEndAng2;
	}
	else
	{	m_afTiltRange[0] = fEndAng2;
		m_afTiltRange[1] = fEndAng1;
	}
}

int CAlignParam::GetSecIndex(int iFrame)
{
	return m_piSecIndex[iFrame];
}

float CAlignParam::GetTilt(int iFrame)
{
	return m_pfTilts[iFrame];
}

float CAlignParam::GetTiltAxis(int iFrame)
{
	return m_pfTiltAxis[iFrame];
}

void CAlignParam::GetShift(int iFrame, float* pfShift)
{
	pfShift[0] = m_pfShiftXs[iFrame];
	pfShift[1] = m_pfShiftYs[iFrame];
}

int CAlignParam::GetFrameIdxFromTilt(float fTilt)
{
	int iFrameIdx = 0;
	float fMin = (float)fabs(fTilt - m_pfTilts[0]);
	for(int i=1; i<m_iNumFrames; i++)
	{	float fDiff = (float)fabs(fTilt - m_pfTilts[i]);
		if(fDiff >= fMin) continue;
		fMin = fDiff;
		iFrameIdx = i;
	}
	return iFrameIdx;
}

float* CAlignParam::GetTilts(bool bCopy)
{
	if(!bCopy) return m_pfTilts;
	//--------------------------
	float* pfTilts = new float[m_iNumFrames];
	memcpy(pfTilts, m_pfTilts, sizeof(float) * m_iNumFrames);
	return pfTilts;
}

float CAlignParam::GetMinTilt(void)
{
	return m_afTiltRange[0];
}

float CAlignParam::GetMaxTilt(void)
{
	return m_afTiltRange[1];
}

void CAlignParam::AddAlphaOffset(float fTiltOffset)
{
	for(int i=0; i<m_iNumFrames; i++)
	{	m_pfTilts[i] += fTiltOffset;
	}
	m_fAlphaOffset += fTiltOffset;
}

void CAlignParam::AddBetaOffset(float fTiltOffset)
{
	m_fBetaOffset = fTiltOffset;
}

void CAlignParam::AddShift(int iFrame, float* pfShift)
{
	m_pfShiftXs[iFrame] += pfShift[0];
	m_pfShiftYs[iFrame] += pfShift[1];
}

void CAlignParam::AddShift(float* pfShift)
{
	for(int i=0; i<m_iNumFrames; i++)
	{	m_pfShiftXs[i] += pfShift[0];
		m_pfShiftYs[i] += pfShift[1];
	}
}

void CAlignParam::MultiplyShift(float fFactX, float fFactY)
{
	for(int i=0; i<m_iNumFrames; i++)
	{	m_pfShiftXs[i] *= fFactX;
		m_pfShiftYs[i] *= fFactY;
	}
}

void CAlignParam::RotateShift(int iFrame, float fAngle)
{
	float afShift[] = {0.0f, 0.0f};
	this->GetShift(iFrame, afShift);
	//------------------------------
	CAlignParam::RotShift(afShift, fAngle, afShift);
        this->SetShift(iFrame, afShift);
}

void CAlignParam::MakeRelative(int iRefFrame)
{
	for(int i=0; i<m_iNumFrames; i++)
	{	m_pfShiftXs[i] -= m_pfShiftXs[iRefFrame];
		m_pfShiftYs[i] -= m_pfShiftYs[iRefFrame];
	}
}

void CAlignParam::ResetShift(void)
{
	if(m_pfShiftXs == 0L) return;
	int iBytes = sizeof(float) * m_iNumFrames * 2;
	memset(m_pfShiftXs, 0, iBytes);
}

void CAlignParam::SortByTilt(void)
{
        for(int iStart=0; iStart<m_iNumFrames; iStart++)
	{	float fMinTilt = this->GetTilt(iStart);
		int iMinFrame = iStart;
		//---------------------
		for(int i=iStart; i<m_iNumFrames; i++)
                {       float fTilt = this->GetTilt(i);
                        if(fTilt >= fMinTilt) continue;
                        fMinTilt = fTilt;
                        iMinFrame = i;
                }
                mSwap(iStart, iMinFrame);
        }
}

void CAlignParam::SortBySecIndex(void)
{
	for(int iStart=0; iStart<m_iNumFrames; iStart++)
	{	int iMinSec = this->GetSecIndex(iStart);
		int iMinFrm = iStart;
		//-------------------
		for(int i=iStart; i<m_iNumFrames; i++)
		{	int iSec = this->GetSecIndex(i);
			if(iSec >= iMinSec) continue;
			iMinSec = iSec;
			iMinFrm = i;
		}
		mSwap(iStart, iMinFrm);
	}
}

void CAlignParam::RemoveFrame(int iFrame)
{
	for(int i=iFrame+1; i<m_iNumFrames; i++)
	{	int j = i - 1;
		m_piSecIndex[j] = m_piSecIndex[i];
		m_pfTilts[j] = m_pfTilts[i];
		m_pfTiltAxis[j] = m_pfTiltAxis[i];
		//--------------------------------
		m_pfShiftXs[j] = m_pfShiftXs[i];
		m_pfShiftYs[j] = m_pfShiftYs[i];
	}
	m_iNumFrames--;
}

CAlignParam* CAlignParam::GetCopy(void)
{
	CAlignParam* pAlignParam = new CAlignParam;
	pAlignParam->Create(m_iNumFrames);
	//-----------------
	int iBytes = sizeof(int) * m_iNumFrames;
	memcpy(pAlignParam->m_piSecIndex, m_piSecIndex, iBytes);
	//-----------------
	iBytes = sizeof(float) * m_iNumFrames;
	memcpy(pAlignParam->m_pfTilts, m_pfTilts, iBytes);
	memcpy(pAlignParam->m_pfTiltAxis, m_pfTiltAxis, iBytes);
	memcpy(pAlignParam->m_pfShiftXs, m_pfShiftXs, iBytes * 2);
	pAlignParam->m_fX0 = m_fX0;
	pAlignParam->m_fY0 = m_fY0;
	pAlignParam->m_fZ0 = m_fZ0;
	return pAlignParam;
}

CAlignParam* CAlignParam::GetCopy(int iStartFm, int iNumFms)
{
	CAlignParam* pAlignParam = new CAlignParam;
	pAlignParam->Create(iNumFms);
	//-----------------
	int iBytes = sizeof(int) * iNumFms;
	memcpy(pAlignParam->m_piSecIndex, m_piSecIndex+iStartFm, iBytes);
	//-----------------
	iBytes = sizeof(float) * iNumFms;
	memcpy(pAlignParam->m_pfTilts, m_pfTilts+iStartFm, iBytes);
	memcpy(pAlignParam->m_pfTiltAxis, m_pfTiltAxis+iStartFm, iBytes);
	memcpy(pAlignParam->m_pfShiftXs, m_pfShiftXs+iStartFm, iBytes);
	memcpy(pAlignParam->m_pfShiftYs, m_pfShiftYs+iStartFm, iBytes);
	//-----------------
	pAlignParam->m_fX0 = m_fX0;
	pAlignParam->m_fY0 = m_fY0;
	pAlignParam->m_fZ0 = m_fZ0;
	return pAlignParam;
}

CAlignParam* CAlignParam::GetCopy(float fStartTilt, float fEndTilt)
{
	int iStartFm = this->GetFrameIdxFromTilt(fStartTilt);
	int iEndFm = this->GetFrameIdxFromTilt(fEndTilt);
	int iNumFms = iEndFm - iStartFm + 1;
	return this->GetCopy(iStartFm, iNumFms);
}

void CAlignParam::Set(CAlignParam* pAlignParam)
{
	if(m_iNumFrames != pAlignParam->m_iNumFrames)
	{	this->Create(pAlignParam->m_iNumFrames);
	}
	//-----------------
	int iBytes = sizeof(int) * m_iNumFrames;
	memcpy(m_piSecIndex, pAlignParam->m_piSecIndex, iBytes);
	//-----------------
	iBytes = sizeof(float) * m_iNumFrames;
	memcpy(m_pfTilts, pAlignParam->m_pfTilts, iBytes);
	memcpy(m_pfTiltAxis, pAlignParam->m_pfTiltAxis, iBytes);
	memcpy(m_pfShiftXs, pAlignParam->m_pfShiftXs, iBytes * 2);
	m_fX0 = pAlignParam->m_fX0;
	m_fY0 = pAlignParam->m_fY0;
	m_fZ0 = pAlignParam->m_fZ0;
}	

void CAlignParam::mSwap(int iFrame1, int iFrame2)
{
	if(iFrame1 == iFrame2) return;
	//-----------------
	int iSecIdx1 = this->GetSecIndex(iFrame1);
	int iSecIdx2 = this->GetSecIndex(iFrame2);
	this->SetSecIndex(iFrame1, iSecIdx2);
	this->SetSecIndex(iFrame2, iSecIdx1);
	//-----------------
	float fTiltAxis1 = this->GetTiltAxis(iFrame1);
	float fTiltAxis2 = this->GetTiltAxis(iFrame2);
	this->SetTiltAxis(iFrame1, fTiltAxis2);
	this->SetTiltAxis(iFrame2, fTiltAxis1);
	//-----------------
	float fTilt1 = this->GetTilt(iFrame1);
	float fTilt2 = this->GetTilt(iFrame2);
	this->SetTilt(iFrame1, fTilt2);
	this->SetTilt(iFrame2, fTilt1);
	//-----------------
	float afShift1[2] = {0.0f}, afShift2[2] = {0.0f};
	this->GetShift(iFrame1, afShift1);
	this->GetShift(iFrame2, afShift2);
	this->SetShift(iFrame1, afShift2);
	this->SetShift(iFrame2, afShift1);
}

void CAlignParam::LogShift(char* pcLogFile)
{
	if(pcLogFile == 0L) return;
	FILE* pFile = fopen(pcLogFile, "wt");
	if(pFile == 0L) return;
	//---------------------
	fprintf(pFile, "# Concised version of aln file\n");
	for(int i=0; i<m_iNumFrames; i++)
	{	fprintf(pFile, "%4d %8.2f % 8.2f %7.2f %7.2f %9.2f %9.2f\n",
		   i, m_afCenter[0], m_afCenter[1], 
		   m_pfTilts[i], m_pfTiltAxis[i], 
		   m_pfShiftXs[i], m_pfShiftYs[i]);
	}
	fclose(pFile);
}

void CAlignParam::FitRotCenterZ(void)
{
	float afRotCent[3] = {0.0f};
	this->GetRotationCenter(afRotCent);
	//---------------------------------
        float afRawShift[2] = {0.0f}, afRotShift[2] = {0.0f};
        double dSin, dCos, dA = 0.0, dB = 0.0;
        //------------------------------------
        for(int i=0; i<m_iNumFrames; i++)
        {       this->GetShift(i, afRawShift);
                CAlignParam::RotShift(afRawShift, -m_pfTiltAxis[i],
                   afRotShift);
                //-------------
                dSin = sin(m_pfTilts[i] * s_fD2R);
		dCos = cos(m_pfTilts[i] * s_fD2R);
		dA += (afRotCent[0] * dCos - afRotShift[0]) * dSin;
                dB += (dSin * dSin);
        }
	float fOldZ = m_fZ0;
	m_fZ0 = (float)(dA / (dB + 1e-30));
        printf("Rot center Z: %8.2f  %8.2f  %8.2f\n\n", 
	   fOldZ, m_fZ0, m_fZ0 - fOldZ);
}

void CAlignParam::FitRotCenterX(void)
{
	float afRotCent[3] = {0.0f};
	this->GetRotationCenter(afRotCent);
	//---------------------------------
	float afRawShift[2] = {0.0f}, afRotShift[2] = {0.0f};
        double dSin, dCos, dA = 0, dB = 0;
	//--------------------------------
	for(int i=0; i<m_iNumFrames; i++)
        {       this->GetShift(i, afRawShift);
                CAlignParam::RotShift(afRawShift, -m_pfTiltAxis[i],
                   afRotShift);
                //-------------
                dSin = sin(m_pfTilts[i] * s_fD2R);
                dCos = cos(m_pfTilts[i] * s_fD2R);
		dA += (afRotCent[2] * dCos * dSin + dCos * afRotShift[0]);
		dB += (dCos * dCos);
	}
	float fNewX0 = (float)(dA / dB);
	m_fX0 = fNewX0 - afRotCent[0];
	printf("Rot center X: %8.2f  %8.2f  %8.2f\n\n", 
	   afRotCent[0], fNewX0, m_fX0);
}

void CAlignParam::GetRotationCenter(float* pfCenter)
{
	int iZeroTilt = this->GetFrameIdxFromTilt(0.0f);
        float afS0[2] = {0.0f};
        this->GetShift(iZeroTilt, afS0);
        CAlignParam::RotShift(afS0, -m_pfTiltAxis[iZeroTilt], afS0);
        pfCenter[0] = afS0[0] / cos(m_pfTilts[iZeroTilt] * s_fD2R);
        pfCenter[1] = afS0[1];
	pfCenter[2] = m_fZ0;
}

void CAlignParam::RemoveOffsetX(float fFact)
{
	if(m_fX0 == 0) return;
	//--------------------
	float afRawShift[2] = {0.0f}, afShift[2] = {0.0f};
	for(int i=0; i<m_iNumFrames; i++)
	{	this->GetShift(i, afRawShift);
		//----------------------------
		double dCos = cos(m_pfTilts[i] * s_fD2R);
                afShift[0] = (float)(m_fX0 * dCos);
		afShift[1] = 0.0f;
		CAlignParam::RotShift(afShift, m_pfTiltAxis[i], afShift);
		afRawShift[0] += (afShift[0] * fFact);
		afRawShift[1] += (afShift[1] * fFact);
		this->SetShift(i, afRawShift);
	}
}

void CAlignParam::RemoveOffsetZ(float fFact)
{
	if(m_fZ0 == 0) return;
	//--------------------
	int iZeroTilt = this->GetFrameIdxFromTilt(0.0f);
	float afRawShift[2] = {0.0f}, afInducedS[2] = {0.0f};
	for(int i=0; i<m_iNumFrames; i++)
	{	if(i == iZeroTilt) continue;
		//--------------------------
		this->GetShift(i, afRawShift);
		this->CalcZInducedShift(i, afInducedS);
		afRawShift[0] += (afInducedS[0] * fFact);
		afRawShift[1] += (afInducedS[1] * fFact);
		this->SetShift(i, afRawShift);
	}
}

void CAlignParam::CalcZInducedShift(int iFrame, float* pfShift)
{
	memset(pfShift, 0, sizeof(float) * 2);
	if(m_fZ0 == 0) return;
	//--------------------
	float fTilt = m_pfTilts[iFrame] * s_fD2R;
	pfShift[0] = (float)(-m_fZ0 * sin(m_pfTilts[iFrame] * s_fD2R));
	CAlignParam::RotShift(pfShift, m_pfTiltAxis[iFrame], pfShift);
}	
