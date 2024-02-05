#include "CPatchAlignInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

using namespace McAreTomo::AreTomo::PatchAlign;

CDetectFeatures* CDetectFeatures::m_pInstances = 0L;
int CDetectFeatures::m_iNumGpus = 0;

void CDetectFeatures::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CDetectFeatures[iNumGpus];
	m_iNumGpus = iNumGpus;
}

void CDetectFeatures::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CDetectFeatures* CDetectFeatures::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CDetectFeatures::CDetectFeatures(void)
{
	m_aiBinnedSize[0] = 128;
	m_aiBinnedSize[1] = 128;
	m_pbFeatures = 0L;
	m_pbUsed = 0L;
	m_pfCenters = 0L; // patch centers
}

CDetectFeatures::~CDetectFeatures(void)
{
	mClean();
}

void CDetectFeatures::mClean(void)
{
	if(m_pbFeatures != 0L) delete[] m_pbFeatures;
	if(m_pbUsed != 0L) delete[] m_pbUsed;
	if(m_pfCenters != 0L) delete[] m_pfCenters;
	m_pbFeatures = 0L;
	m_pbUsed = 0L;
	m_pfCenters = 0L;
}


void CDetectFeatures::SetSize(int* piImgSize, int* piNumPatches)
{
	mClean();
	int iBinnedSize = m_aiBinnedSize[0] * m_aiBinnedSize[1];
	m_pbFeatures = new bool[iBinnedSize];
	m_pbUsed = new bool[iBinnedSize];
	//-------------------------------
	int iNumPatches = piNumPatches[0] * piNumPatches[1];
	m_pfCenters = new float[iBinnedSize * 3];
	//---------------------------------------
	m_aiImgSize[0] = piImgSize[0];
	m_aiImgSize[1] = piImgSize[1];
	m_aiNumPatches[0] = piNumPatches[0];
	m_aiNumPatches[1] = piNumPatches[1];
	//----------------------------------
	m_afPatSize[0] = m_aiBinnedSize[0] * 1.0f / m_aiNumPatches[0];
	m_afPatSize[1] = m_aiBinnedSize[1] * 1.0f / m_aiNumPatches[1];
	//------------------------------------------------------------
	m_aiSeaRange[0] = (int)(m_afPatSize[0] * 0.5f - 0.5f);
	if(m_aiSeaRange[0] < 4) m_aiSeaRange[0] = 4;
	m_aiSeaRange[1] = m_aiBinnedSize[0] - m_aiSeaRange[0];
	m_aiSeaRange[2] = (int)(m_afPatSize[1] * 0.5f - 0.5f);
	if(m_aiSeaRange[2] < 4) m_aiSeaRange[2] = 4;
	m_aiSeaRange[3] = m_aiBinnedSize[1] - m_aiSeaRange[2];
}

void CDetectFeatures::DoIt(float* pfImg)
{
	size_t tBytes = (m_aiImgSize[0] / 2 + 1) * 2 *
	   m_aiImgSize[1] * sizeof(float);
	float* gfPadImg = (float*)MU::GetGpuBuf(tBytes, false);
	//-----------------
	MU::CPad2D aPad2D;
	aPad2D.Pad(pfImg, m_aiImgSize, gfPadImg);
	//-----------------
	int aiOutCmpSize[] = {m_aiBinnedSize[0] / 2 + 1, m_aiBinnedSize[1]};
	int iBytes = sizeof(cufftComplex) * aiOutCmpSize[0] * aiOutCmpSize[1];
	cufftComplex* gCmpCropped = 0L;
	cudaMalloc(&gCmpCropped, iBytes);
	//-------------------------------	
	MU::CCufft2D aCufft2D;
	aCufft2D.CreateForwardPlan(m_aiImgSize, false);
	aCufft2D.Forward(gfPadImg, true);
	cufftComplex* gCmpImg = reinterpret_cast<cufftComplex*>(gfPadImg);
	int aiInCmpSize[] = {m_aiImgSize[0] / 2 + 1, m_aiImgSize[1]};
	//-----------------
	MU::GFourierResize2D fourierResize;
	bool bSum = false;
	fourierResize.DoIt(gCmpImg, aiInCmpSize,
	   gCmpCropped, aiOutCmpSize, bSum);
	cudaFree(gfPadImg);
	//-----------------
	aCufft2D.CreateInversePlan(aiOutCmpSize, true);
	aCufft2D.Inverse(gCmpCropped);
	float* gfBinnedImg = reinterpret_cast<float*>(gCmpCropped);
	//---------------------------------------------------------
	int aiOutPadSize[] = { aiOutCmpSize[0] * 2, aiOutCmpSize[1] };
	MU::GFindMinMax2D gFindMinMax2D;
	gFindMinMax2D.SetSize(aiOutPadSize, true);
	float fMin = gFindMinMax2D.DoMin(gfBinnedImg, true, 0);
	//-----------------------------------------------------
	MU::GPositivity2D gPositivity;
	gPositivity.AddVal(gfBinnedImg, aiOutPadSize, 1.0f - fMin);
	//---------------------------------------------------------
	int aiWinSize[] = {5, 5};
	GNormByStd2D gNormByStd;
	gNormByStd.DoIt(gfBinnedImg, aiOutPadSize, true, aiWinSize);	
	//----------------------------------------------------------
	MU::GCalcMoment2D gCalcMoment;
	gCalcMoment.SetSize(aiOutPadSize, true);
	float fMean = gCalcMoment.DoIt(gfBinnedImg, 1, true);
	float fStd = gCalcMoment.DoIt(gfBinnedImg, 2, true);
	fStd = fStd - fMean * fMean;
	if(fStd < 0) fStd = 0.0f;
	else fStd = (float)sqrtf(fStd);
	//-----------------------------
	int iBinnedSize = m_aiBinnedSize[0] * m_aiBinnedSize[1];
	float* pfBinnedImg = new float[iBinnedSize];
	iBytes = sizeof(float) * m_aiBinnedSize[0];
	int iPadSize = aiOutCmpSize[0] * 2;
	for(int y=0; y<m_aiBinnedSize[1]; y++)
	{	cudaMemcpy(pfBinnedImg + y * m_aiBinnedSize[0],
		   gfBinnedImg + y * iPadSize, iBytes, cudaMemcpyDefault);
	}
	cudaFree(gCmpCropped);
	//--------------------
	fMin = fMean - 4.0f * fStd;
	float fMax = fMean + 4.0f * fStd;
	double dMean = 0, dStd = 0;
	int iCount = 0;
	for(int i=0; i<iBinnedSize; i++)
	{	if(pfBinnedImg[i] < fMin) continue;
		else if(pfBinnedImg[i] > fMax) continue;
		dMean += pfBinnedImg[i];
		dStd += (pfBinnedImg[i] * pfBinnedImg[i]);
		iCount += 1;
	}
	dMean /= (iCount + 1e-30);
	dStd = dStd / (iCount + 1e-30) - dMean * dMean;
	if(dStd < 0) dStd = 0;
	else dStd = sqrtf(dStd);
	fMin = (float)(dMean - 4.0 * dStd);
	fMax = (float)(dMean - 0.5f * dStd);
	//-----------------------------------
	for(int i=0; i<iBinnedSize; i++)
	{	if(pfBinnedImg[i] < fMin) 
		{	m_pbFeatures[i] = false;
			if(pfBinnedImg[i] < -1e10) pfBinnedImg[i] = fMin;
		}
		else if(pfBinnedImg[i] > fMax) 
		{	m_pbFeatures[i] = false;
		}
		else 
		{	m_pbFeatures[i] = true;
		}
	}
	/*
	Util::CSaveTempMrc saveTempMrc;
        saveTempMrc.SetFile("/home/szheng/Temp/Test_Norm", ".mrc");
        saveTempMrc.DoIt(pfBinnedImg, 2, m_aiBinnedSize);
	delete[] pfBinnedImg;
	*/
	//-------------------
	mFindCenters();
	/*
	printf("# Feature map\n");
	float fBinX = (pXcfBuffer->m_aiCmpSize[0] - 1) * 2.0f
	   / m_aiBinnedSize[0];
	float fBinY = pXcfBuffer->m_aiCmpSize[1] * 1.0f
	   / m_aiBinnedSize[1];
	for(int y=0; y<m_aiBinnedSize[1]; y+=5)
	{	int i = y * m_aiBinnedSize[0];
		float fY = (y + 0.5f) * fBinY;
		for(int x=0; x<m_aiBinnedSize[0]; x+=5)
		{	if(!m_pbFeatures[i+x]) continue;
			float fX = (x + 0.5f) * fBinX;
			printf("%6d  %9.2f  %9.2f\n", i+x, fX, fY);
		}
	}
	printf("\n");
	*/
}

void CDetectFeatures::GetCenter(int iPatch, int* piCent)
{
	float fBinX = m_aiImgSize[0] * 1.0f / m_aiBinnedSize[0];
	float fBinY = m_aiImgSize[1] * 1.0f / m_aiBinnedSize[1];
	int i = 3 * iPatch;
	piCent[0] = (int)(m_pfCenters[i] * fBinX);
	piCent[1] = (int)(m_pfCenters[i+1] * fBinY);
}

void CDetectFeatures::mFindCenters(void)
{
	int iBinnedSize = m_aiBinnedSize[0] * m_aiBinnedSize[1];	
	int iNumPatches = m_aiNumPatches[0] * m_aiNumPatches[1];
	memset(m_pbUsed, 0, sizeof(bool) * iBinnedSize);
	memset(m_pfCenters, 0, sizeof(float) * 3 * iNumPatches);
	//------------------------------------------------------
	int iBadPatches = 0;
	for(int i=0; i<iNumPatches; i++)
	{	int x = i % m_aiNumPatches[0];
		int y = i / m_aiNumPatches[0];
		float fCentX = (x + 0.5f) * m_afPatSize[0];
		float fCentY = (y + 0.5f) * m_afPatSize[1];
		bool bHasFeature = mCheckFeature(fCentX, fCentY);
		//-----------------------------------------------
		int j = 3 * i;
		m_pfCenters[j] = fCentX;
		m_pfCenters[j+1] = fCentY;
		if(bHasFeature) 
		{	m_pfCenters[j+2] = 1.0f;
			mSetUsed(fCentX, fCentY);
		}
		else 
		{	m_pfCenters[j+2] = -1.0f;
			iBadPatches += 1;
		}
	}
	if(iBadPatches = 0) return;
	//-------------------------
	for(int i=0; i<iNumPatches; i++)
	{	float* pfCenter = m_pfCenters + 3 * i;
		if(pfCenter[2] > 0) continue;
		mFindCenter(pfCenter);
	}
}

bool CDetectFeatures::mCheckFeature(float fCentX, float fCentY)
{
	int iCentX = (int)fCentX;
	int iCentY = (int)fCentY;
	if(iCentX < m_aiSeaRange[0]) return false;
	else if(iCentX > m_aiSeaRange[1]) return false;
	//---------------------------------------------
	if(iCentY < m_aiSeaRange[2]) return false;
	else if(iCentY > m_aiSeaRange[3]) return false;
	//---------------------------------------------
	int i = iCentY * m_aiBinnedSize[0] + iCentX;
	return m_pbFeatures[i];
}

void CDetectFeatures::mFindCenter(float* pfCenter)
{
	int iWinX = m_aiBinnedSize[0] / 2;
	int iWinY = m_aiBinnedSize[1] / 2;
	int iStartX = (int)(pfCenter[0] - iWinX * 0.5f + 0.5f);
	int iStartY = (int)(pfCenter[1] - iWinY * 0.5f + 0.5f);
	iStartX = mCheckRange(iStartX, iWinX, &m_aiSeaRange[0]);
	iStartY = mCheckRange(iStartY, iWinY, &m_aiSeaRange[2]);
	//------------------------------------------------------
	float fMinR = (float)1e20;
	float afCent[2] = {-1.0f};
	int iOffset = iStartY * m_aiBinnedSize[0] + iStartX;
	bool* pbFeatures = &m_pbFeatures[iOffset];
	bool* pbUsed = &m_pbUsed[iOffset];
	//--------------------------------
	float fX, fY, fDx, fDy, fR;
	for(int y=0; y<iWinY; y++)
	{	int i = y * m_aiBinnedSize[0];
		for(int x=0; x<iWinX; x++)
		{	int j = i + x;
			if(pbUsed[j]) continue;
			if(!pbFeatures[j]) continue;
			//----------------------------
			fX = x + iStartX + 0.5f;
			fY = y + iStartY + 0.5f;
			fDx = fX - pfCenter[0];
			fDy = fY - pfCenter[1]; 
			fR = (float)sqrtf(fDx * fDx + fDy * fDy);
			if(fR >= fMinR) continue;
			//-----------------------
			fMinR = fR;
			afCent[0] = fX;
			afCent[1] = fY;
		}
	}
	//-----------------------------
	if(afCent[0] > 0)
	{	pfCenter[0] = afCent[0];
		pfCenter[1] = afCent[1];
	}
	mSetUsed(pfCenter[0], pfCenter[1]);
}


void CDetectFeatures::mSetUsed(float fCentX, float fCentY)
{
	int iWinX = (int)(m_afPatSize[0] * 0.8f);
	int iWinY = (int)(m_afPatSize[1] * 0.8f);
	int iStartX = (int)(fCentX - iWinX / 2);
	int iStartY = (int)(fCentY - iWinY / 2);
	iStartX = mCheckRange(iStartX, iWinX, &m_aiSeaRange[0]);
	iStartY = mCheckRange(iStartY, iWinY, &m_aiSeaRange[2]);
	//------------------------------------------------------
	bool* pbUsed = &m_pbUsed[iStartY * m_aiBinnedSize[0] + iStartX];
	for(int y=0; y<iWinY; y++)
	{	int i = y * m_aiBinnedSize[0];
		for(int x=0; x<iWinX; x++)
		{	pbUsed[i + x] = true;
		}
	}
}

int CDetectFeatures::mCheckRange(int iStart, int iSize, int* piRange)
{
	if(iStart < piRange[0]) return piRange[0];
	if((iStart + iSize) > piRange[1]) return (piRange[1] - iSize);
	return iStart;
}
