#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo;
using namespace McAreTomo::DataUtil;

CTiltSeries::CTiltSeries(void)
{
	m_pfTilts = 0L;
	m_piAcqIndices = 0L;
	m_piSecIndices = 0L;
	m_ppfCenters = 0L;
	m_ppfImages = 0L;
	m_bLoaded = false;
	memset(m_aiStkSize, 0, sizeof(m_aiStkSize));
}

CTiltSeries::~CTiltSeries(void)
{
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	if(m_piAcqIndices != 0L) delete[] m_piAcqIndices;
	m_pfTilts = 0L;
	m_piAcqIndices = 0L;
	m_piSecIndices = 0L;
	//--------------------------------------------
	// do NOT free each image. They will be freed
	// in the base class (m_ppvFrames).
	//--------------------------------------------
	if(m_ppfImages != 0L) delete[] m_ppfImages;
	m_ppfImages = 0L;
	//-----------------
	mCleanCenters();
}

void CTiltSeries::Create(int* piStkSize)
{
	this->Create(piStkSize, piStkSize[2]);
}

void CTiltSeries::Create(int* piImgSize, int iNumTilts)
{
	mCleanCenters();
	//-----------------
	int aiStkSize[] = {piImgSize[0], piImgSize[1], iNumTilts};
	CMrcStack::Create(2, aiStkSize);
	//-----------------
	if(m_pfTilts != 0L) delete[] m_pfTilts;
	m_pfTilts = new float[iNumTilts];
	memset(m_pfTilts, 0, sizeof(float) * iNumTilts);
	//----------------
	if(m_piAcqIndices != 0L) delete[] m_piAcqIndices;
	m_piAcqIndices = new int[iNumTilts * 2];
	m_piSecIndices = &m_piAcqIndices[iNumTilts];
	memset(m_piAcqIndices, 0, sizeof(int) * iNumTilts * 2);
	//----------------
	m_ppfCenters = new float*[m_aiStkSize[2]];
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	float* pfCent = new float[2];
		pfCent[0] = 0.5f * m_aiStkSize[0];
		pfCent[1] = 0.5f * m_aiStkSize[1];
		m_ppfCenters[i] = pfCent;
	} 
	//-----------------
	if(m_ppfImages != 0L) delete[] m_ppfImages;
	m_ppfImages = new float*[m_aiStkSize[2]];
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	m_ppfImages[i] = (float*)m_ppvFrames[i];
	}
	//-----------------
	m_bLoaded = false;
}

void CTiltSeries::SortByTilt(void)
{
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	float fMinT = m_pfTilts[i];
		int iMin = i;
		for(int j=i+1; j<m_aiStkSize[2]; j++)
		{	if(m_pfTilts[j] >= fMinT) continue;
			fMinT = m_pfTilts[j];
			iMin = j;
		}
		mSwap(i, iMin);
	}
}

void CTiltSeries::SortByAcq(void)
{
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	int iMinAcq = m_piAcqIndices[i];
		int iMin = i;
		for(int j=i+1; j<m_aiStkSize[2]; j++)
		{	if(m_piAcqIndices[j] >= iMinAcq) continue;
			iMinAcq = m_piAcqIndices[j];
			iMin = j;
		}
		mSwap(i, iMin);
	}
}

void CTiltSeries::SetTilts(float* pfTilts)
{
	int iBytes = sizeof(float) * m_aiStkSize[2];
	if(iBytes == 0) return;
	memcpy(m_pfTilts, pfTilts, iBytes);
}

void CTiltSeries::SetAcqs(int* piAcqIndices)
{
	int iBytes = sizeof(int) * m_aiStkSize[2];
	if(iBytes <= 0) return;
	memcpy(m_piAcqIndices, piAcqIndices, iBytes);
}

void CTiltSeries::SetSecs(int* piSecIndices)
{
	int iBytes = sizeof(int) * m_aiStkSize[2];
	if(iBytes <= 0) return;
	memcpy(m_piSecIndices, piSecIndices, iBytes);
}

void CTiltSeries::SetImage(int iTilt, void* pvImage)
{
	float* pfImg = m_ppfImages[iTilt];
	memcpy(pfImg, pvImage, m_tFmBytes);
}

void CTiltSeries::SetCenter(int iTilt, float* pfCent)
{
	float* pfDstCent = m_ppfCenters[iTilt];	
	pfDstCent[0] = pfCent[0];
	pfDstCent[1] = pfCent[1];	
}

void CTiltSeries::GetCenter(int iTilt, float* pfCent)
{
	float* pfSrcCent = m_ppfCenters[iTilt];
	pfCent[0] = pfSrcCent[0];
	pfCent[1] = pfSrcCent[1];
}

CTiltSeries* CTiltSeries::GetSubSeries(int* piStart, int* piSize)
{
	CTiltSeries* pSubSeries = new CTiltSeries;
	pSubSeries->Create(piSize, piSize[2]);
	//-----------------
	int iBytes = sizeof(float) * piSize[0];
	int iOffset = piStart[1] * m_aiStkSize[0] + piStart[0];
	//-----------------
	float afCent[2] = {0.0f};
	afCent[0] = piStart[0] + 0.5f * piSize[0];
	afCent[1] = piStart[1] + 0.5f * piSize[1];
	//----------------------------------------
	for (int i=0; i<piSize[2]; i++)
	{	int iSrcProj = i + piStart[2];
		float* pfSrcFrm = (float*)m_ppvFrames[iSrcProj] + iOffset;
		float* pfDstFrm = (float*)pSubSeries->GetFrame(i);
		for(int y=0; y<piSize[1]; y++)
		{	float* pfSrc = pfSrcFrm + y * m_aiStkSize[0];
			float* pfDst = pfDstFrm + y * piSize[0];
			memcpy(pfDst, pfSrc, iBytes);
		}
		pSubSeries->SetCenter(i, afCent);
	}
	return pSubSeries;		
}

void CTiltSeries::RemoveFrame(int iFrame)
{
	void* pvFrm = m_ppvFrames[iFrame];
	float* pfImg = m_ppfImages[iFrame];
	float* pfCent = m_ppfCenters[iFrame];
	//-----------------
	for(int i=iFrame+1; i<m_aiStkSize[2]; i++)
	{	int k = i - 1;
		m_ppvFrames[k] = m_ppvFrames[i];
		m_ppfImages[k] = m_ppfImages[i];
		m_ppfCenters[k] = m_ppfCenters[i];
		m_pfTilts[k] = m_pfTilts[i];
		m_piAcqIndices[k] = m_piAcqIndices[i];
		m_piSecIndices[k] = m_piSecIndices[i];
	};
	//-----------------
	int iLast = m_aiStkSize[2] - 1;
	m_ppvFrames[iLast] = pvFrm;
	m_ppfImages[iLast] = pfImg;
	m_ppfCenters[iLast] = pfCent;
	m_aiStkSize[2] = iLast;
}

void CTiltSeries::GetAlignedSize(float fTiltAxis, int* piAlnSize)
{
	memcpy(piAlnSize, m_aiStkSize, sizeof(int) * 2);
	double dRot = fabs(sin(fTiltAxis * 3.14 / 180));
	if(dRot <= 0.707) return;
	piAlnSize[0] = m_aiStkSize[1];
	piAlnSize[1] = m_aiStkSize[0];
}

float* CTiltSeries::GetAccDose(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	float fTiltDose = pAtInput->m_fTotalDose / m_aiStkSize[2];
	//-----------------
	float* pfAccDose = new float[m_aiStkSize[2]];
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	pfAccDose[i] = fTiltDose * m_piAcqIndices[i];
	}
	return pfAccDose;
}

int CTiltSeries::GetTiltIdx(float fTilt)
{
	int iMin = 0;
	float fMin = (float)fabs(m_pfTilts[0] - fTilt);
	for(int i=1; i<m_aiStkSize[2]; i++)
	{	float fDif = (float)fabs(m_pfTilts[i] - fTilt);
		if(fDif < fMin)
		{	fMin = fDif;
			iMin = i;
		}
	}
	return iMin;
}

bool CTiltSeries::bEmpty(void)
{
	if(m_aiStkSize[0] == 0) return true;
	if(m_aiStkSize[1] == 0) return true;
	if(m_aiStkSize[0] == 0) return true;
	if(m_ppfImages == 0L) return true;
	return false;
}

float** CTiltSeries::GetImages(void)
{
	return m_ppfImages;
}

//--------------------------------------------------------------------
// 1. This is called in CProcessThread::mProcessTsPackage().
// 2. Since the tilt series is sorted by tilt angles and then saved
//    into MRC files, its section indices are in ascending order
//    as the tilt angles.
// 3. If not sorted by tilt angles, the section indices should be
//    the same as acquisition indices.
//--------------------------------------------------------------------
void CTiltSeries::ResetSecIndices(void)
{
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	m_piSecIndices[i] = i;
	}
}

//--------------------------------------------------------------------
// 1. Generate a new volume that swaps y and z dimensions by flipping
//    of rotation.
// 2. When flipping is used, the handedness will be changed. Rotation
//    maintains the handedness.
// 3. Flipping generates a volume that matches the one by rotation
//    around X axis in IMOD user interface.
//--------------------------------------------------------------------
CTiltSeries* CTiltSeries::FlipVol(bool bFlip)
{
	CTiltSeries* pNewSeries = mGenVolXZY();
	int* piNewSize = pNewSeries->m_aiStkSize;
	//-----------------
	int iBytes = m_aiStkSize[0] * sizeof(float);
	int iEndOldY = m_aiStkSize[1] - 1;
	//-----------------
	for(int y=0; y<m_aiStkSize[1]; y++)
	{	int iNewFrm = bFlip ? (iEndOldY - y) : y;
		float* pfNewFrm = (float*)pNewSeries->GetFrame(iNewFrm);
		for(int z=0; z<m_aiStkSize[2]; z++)
		{	float* pfOldFrm = (float*)this->GetFrame(z);
			memcpy(pfNewFrm + z * piNewSize[0],
			   pfOldFrm + y * m_aiStkSize[0], iBytes);
		}
	}
	//-----------------
	return pNewSeries;
}

void CTiltSeries::mSwap(int k1, int k2)
{
	if(k1 == k2) return;
	//-----------------
	float fTilt1 = m_pfTilts[k1];
	int iAcqIdx1 = m_piAcqIndices[k1];
	int iSecIdx1 = m_piSecIndices[k1];
	void* pvFrm1 = m_ppvFrames[k1];
	float* pfImg1 = m_ppfImages[k1];
	//-----------------
	m_pfTilts[k1] = m_pfTilts[k2];
	m_piAcqIndices[k1] = m_piAcqIndices[k2];
	m_piSecIndices[k1] = m_piSecIndices[k2];
	m_ppvFrames[k1] = m_ppvFrames[k2];
	m_ppfImages[k1] = m_ppfImages[k2];
	//-----------------
	m_pfTilts[k2] = fTilt1;
	m_piAcqIndices[k2] = iAcqIdx1;
	m_piSecIndices[k2] = iSecIdx1;
	m_ppvFrames[k2] = pvFrm1;
	m_ppfImages[k2] = pfImg1;
	//-----------------
	float* pfCent1 = m_ppfCenters[k1];
	m_ppfCenters[k1] = m_ppfCenters[k2];
	m_ppfCenters[k2] = pfCent1;
}

void CTiltSeries::mCleanCenters(void)
{
	if(m_ppfCenters == 0L) return;
	for(int i=0; i<m_aiStkSize[2]; i++)
	{	if(m_ppfCenters[i] == 0L) continue;
		delete[] m_ppfCenters[i];
	}
	delete[] m_ppfCenters;
	m_ppfCenters = 0L;
}

CTiltSeries* CTiltSeries::mGenVolXZY(void)
{
	CTiltSeries* pNewSeries = new CTiltSeries;
	int aiNewSize[] = {m_aiStkSize[0], 
	   m_aiStkSize[2], m_aiStkSize[1]};
	//-----------------
	pNewSeries->Create(aiNewSize);
	pNewSeries->m_fPixSize = m_fPixSize;
	return pNewSeries;
}
