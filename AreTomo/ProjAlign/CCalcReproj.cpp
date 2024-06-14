#include "CProjAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::ProjAlign;

static float s_fD2R = 0.01745f;

CCalcReproj::CCalcReproj(void)
{
	m_pfPinnedBuf = 0L;
	m_gfBuf = 0L;
	m_fMaxDiff = 20.0f;
	m_fMaxStch = 1.0 / cos(30.0 * s_fD2R);
}

CCalcReproj::~CCalcReproj(void)
{
	this->Clean();
}

void CCalcReproj::Clean(void)
{
	if(m_pfPinnedBuf != 0L) cudaFreeHost(m_pfPinnedBuf);
	if(m_gfBuf != 0L) cudaFree(m_gfBuf);
	m_aGReproj.Clean();
	m_pfPinnedBuf = 0L;
	m_gfBuf = 0L;
}

void CCalcReproj::Setup(int* piTomoStkSize, int iVolZ, int iNthGpu)
{
	this->Clean();
	//-----------------
	memcpy(m_aiProjSize, piTomoStkSize, sizeof(int) * 2);
	m_iNumProjs = piTomoStkSize[2];
	mAllocBuf();
	int iVolX = m_aiProjSize[0] * 2;
	m_aGReproj.SetSizes(m_aiProjSize[0], m_iNumProjs, iVolX, iVolZ);
}

void CCalcReproj::DoIt
(	float** ppfProjs, float* pfTiltAngles, 
	bool* pbSkipProjs, int iProjIdx,
	float* pfReproj
)
{
	m_ppfProjs = ppfProjs;
	m_iProjIdx = iProjIdx;
	m_pfReproj = pfReproj;
	//-----------------
	mFindProjRange(pfTiltAngles, pbSkipProjs);
	//-----------------
	size_t tBytes = sizeof(float) * m_iNumProjs;
	cudaMemcpy(m_gfTiltAngles, pfTiltAngles, tBytes, cudaMemcpyDefault);
	//-----------------
	for(int y=0; y<m_aiProjSize[1]; y++)
	{	mGetSinogram(y);	
		mReproj(y, pfTiltAngles[m_iProjIdx]);
	}
}

//--------------------------------------------------------------------
// 1. This function searches the index range of tilt images that have
//    lower tilt angles than that to be reprojected.
// 2. pbSkipProjs: the projections at higher tilt angles than the
//    projection angle are excluded in reprojection.
//--------------------------------------------------------------------
void CCalcReproj::mFindProjRange(float* pfTiltAngles, bool* pbSkipProjs)
{
	float fRefRange = 20.5f;
	float fRefStretch = 1.20f; 
	float fProjA = pfTiltAngles[m_iProjIdx];
	float fCosProjA = (float)cos(fProjA * s_fD2R);
	//-----------------
	int iStart = -1;
	for(int i=0; i<m_iNumProjs; i++)
	{	if(pbSkipProjs[i]) continue;
		//----------------
		float fTiltA = pfTiltAngles[i];
		float fDiffA = fProjA - fTiltA;
		if(fabs(fDiffA) > fRefRange) continue;
		//----------------
		float fStretch = (float)(cos(fTiltA * s_fD2R) / fCosProjA);
		if(fStretch > fRefStretch) continue;
		//----------------
		iStart = i; break;
	}
	int iEnd = -1;
	for(int i=iStart; i<m_iNumProjs; i++)
	{	if(pbSkipProjs[i]) continue;
		//----------------
		float fTiltA = pfTiltAngles[i];
		float fDiffA = fProjA - fTiltA;
		if(fabs(fDiffA) > fRefRange) continue;
		//----------------
		float fStretch = (float)(cos(fTiltA * s_fD2R) / fCosProjA);
		if(fStretch > fRefStretch) continue;
		//----------------
		iEnd = i;
	}
	//-----------------------------------------------
	// 1) Prevent an unusual situation where the
	// angular step is so large causing stretching
	// factor larger that allowed. 2) In that case
	// we just use the nearest lower tilt image.
	//-----------------------------------------------
	if(iStart < 0 || iEnd < 0)
	{	int iSign = (fProjA > 0) ? 1 : -1;
		iStart = m_iProjIdx - iSign;
		iEnd = iStart;
	}
	//-----------------
	if((iEnd - iStart) > 9) iEnd = iStart + 9;
	m_aiProjRange[0] = iStart;
	m_aiProjRange[1] = iEnd;
	//-----------------
	/*	
	printf("%.2f  %d  %d  %d  %.2f  %.2f\n", pfTiltAngles[m_iProjIdx],
           m_aiProjRange[0], m_aiProjRange[1],
           m_aiProjRange[1] - m_aiProjRange[0] + 1,
	   pfTiltAngles[m_aiProjRange[0]], pfTiltAngles[m_aiProjRange[1]]);
	*/	
} 

void CCalcReproj::mGetSinogram(int iY)
{
	size_t tPixels = m_aiProjSize[0] * m_aiProjSize[1];
	size_t tLineBytes = sizeof(float) * m_aiProjSize[0];
	size_t tSinoBytes = tLineBytes * m_iNumProjs;
	int iOffsetY = iY * m_aiProjSize[0];
	//----------------------------------
	for(int i=m_aiProjRange[0]; i<=m_aiProjRange[1]; i++)
	{	float* pfProj = m_ppfProjs[i];
		float* pfSrc = pfProj + iOffsetY;
		float* gfDst = m_gfSinogram + i * m_aiProjSize[0];
		cudaMemcpy(gfDst, pfSrc, tLineBytes, cudaMemcpyDefault);
	}
}

void CCalcReproj::mReproj(int iY, float fProjAngle)
{	
	m_aGReproj.DoIt(m_gfSinogram, m_gfTiltAngles, 
	   m_aiProjRange, fProjAngle);
	//----------------------------
	size_t tLineBytes = sizeof(float) * m_aiProjSize[0];
	float* pfLine = m_pfReproj + iY * m_aiProjSize[0];
	cudaMemcpy(pfLine, m_aGReproj.m_gfReproj, tLineBytes, 
	   cudaMemcpyDefault);
}

void CCalcReproj::mAllocBuf(void)
{
	int iSinoPixels = m_aiProjSize[0] * m_iNumProjs;
	size_t tSinoBytes = iSinoPixels * sizeof(float);
	cudaMallocHost(&m_pfPinnedBuf, tSinoBytes);
	//-----------------------------------------
	size_t tTiltAngBytes = (m_iNumProjs) * sizeof(float);
	cudaMalloc(&m_gfBuf, tSinoBytes + tTiltAngBytes);
	//-----------------------------------------------
	m_gfSinogram = m_gfBuf;
	m_gfTiltAngles = m_gfBuf + iSinoPixels;
}
