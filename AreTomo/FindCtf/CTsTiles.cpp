#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

CTsTiles* CTsTiles::m_pInstances[] = {0L};

CTsTiles* CTsTiles::GetInstance(int iNthGpu)
{
	if(m_pInstances[iNthGpu] != 0L) return m_pInstances[iNthGpu];
	m_pInstances[iNthGpu] = new CTsTiles;
	m_pInstances[iNthGpu]->m_iNthGpu = iNthGpu;
	return m_pInstances[iNthGpu];
}

void CTsTiles::DeleteInstance(int iNthGpu)
{
	if(m_pInstances[iNthGpu] == 0L) return;
	delete m_pInstances[iNthGpu];
	m_pInstances[iNthGpu] = 0L;
}

void CTsTiles::DeleteAll(void)
{
	int iNumInsts = sizeof(m_pInstances) / sizeof(CTsTiles*);
	for(int i=0; i<iNumInsts; i++)
	{	if(m_pInstances[i] == 0L) continue;
		delete m_pInstances[i];
		m_pInstances[i] = 0L;
	}
}

CTsTiles::CTsTiles(void)
{
	m_fOverlap = 0.50f;
	m_gfTileSpect = 0L;
	m_pGCalcMoment2D = 0L;
	m_pGCalcSpectrum = 0L;
	//-----------------
	m_pTiles = 0L;
	m_iNthGpu = 0;
	//-----------------
	memset(m_aiImgTiles, 0, sizeof(m_aiImgTiles));
	m_iNumTilts = 0;
	m_iTileSize = 512;
}

CTsTiles::~CTsTiles(void)
{
	this->Clean();
}

void CTsTiles::Clean(void)
{
	if(m_gfTileSpect != 0L) cudaFree(m_gfTileSpect);
	if(m_pGCalcMoment2D != 0L) delete m_pGCalcMoment2D;
	if(m_pGCalcSpectrum != 0L) delete m_pGCalcSpectrum;
	if(m_pTiles != 0L) delete[] m_pTiles;
	//-----------------
	m_gfTileSpect = 0L;
	m_pGCalcMoment2D = 0L;
	m_pGCalcSpectrum = 0L;
	m_pTiles = 0L;
}

int CTsTiles::GetAllTiles(void)
{
	int iAllTiles = m_iNumTilts * this->GetImgTiles();
	return iAllTiles;
}

int CTsTiles::GetImgTiles(void)
{
	int iImgTiles = m_aiImgTiles[0] * m_aiImgTiles[1];
	return iImgTiles;
}

void CTsTiles::GetImgCent(int* piImgCent)
{
	piImgCent[0] = m_aiImgSize[0] * 0.5f;
	piImgCent[1] = m_aiImgSize[1] * 0.5f;
}

CTile* CTsTiles::GetTile(int iTile)
{
	return &m_pTiles[iTile];
}

CTile* CTsTiles::GetTile(int iTilt, int iImgTile)
{
	int iTile = iTilt * m_aiImgTiles[0] * m_aiImgTiles[1] + iImgTile;
	return this->GetTile(iTile);
}

float CTsTiles::GetTilt(int iTilt)
{
	CTile* pTile = this->GetTile(iTilt, 0);
	return pTile->GetTilt();
}

int CTsTiles::GetTiltIdx(float fTilt)
{
	int iMinIdx = -1;
	float fMinDiff = 10000.0f;
	for(int i=0; i<m_iNumTilts; i++)
	{	float fDiff = (float)fabs(GetTilt(i) - fTilt);
		if(fDiff >= fMinDiff) continue;
		fMinDiff = fDiff;
		iMinIdx = i;
	}
	return iMinIdx;
}

void CTsTiles::Generate(int iTileSize)
{
	this->Clean();
	m_iTileSize = iTileSize;
	mSetSize();
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	int iImgTiles = m_aiImgTiles[0] * m_aiImgTiles[1];
	int iTsTiles = m_iNumTilts * iImgTiles;
	m_pTiles = new CTile[iTsTiles];
	//-----------------
	if(m_gfTileSpect != 0L) cudaFree(m_gfTileSpect);
	int aiCmpSize[] = {m_iTileSize / 2 + 1, m_iTileSize};
	int iCmpSize = aiCmpSize[0] * aiCmpSize[1];
	size_t tBytes = sizeof(float) * iCmpSize * 3;
        cudaMalloc(&m_gfTileSpect, tBytes);
	m_gfPadTile = m_gfTileSpect + iCmpSize;
	//-----------------
	m_pGCalcMoment2D = new MU::GCalcMoment2D;
	m_pGCalcSpectrum = new GCalcSpectrum;
	int aiPadSize[] = {aiCmpSize[0] * 2, aiCmpSize[1]};
	m_pGCalcMoment2D->SetSize(aiPadSize, true);
	m_pGCalcSpectrum->SetSize(aiPadSize, true);
	//-----------------
	for(int i=0; i<m_iNumTilts; i++)
	{	mDoTilt(i);
	}
	//-----------------
	if(m_gfTileSpect != 0L) cudaFree(m_gfTileSpect);
	if(m_pGCalcMoment2D != 0L) delete m_pGCalcMoment2D;
	if(m_pGCalcSpectrum != 0L) delete m_pGCalcSpectrum;
	m_gfTileSpect = 0L;
	m_pGCalcMoment2D = 0L;
	m_pGCalcSpectrum = 0L;
}

void CTsTiles::mSetSize(void)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	m_aiImgSize[0] = pTiltSeries->m_aiStkSize[0];
	m_aiImgSize[1] = pTiltSeries->m_aiStkSize[1];
	m_iNumTilts = pTiltSeries->m_aiStkSize[2];
	m_fPixSize = pTiltSeries->m_fPixSize;
	//-----------------------------------------------
	// After this, m_aiImgSize, m_fPixSize are
	// changed according to m_fBinning.
	//-----------------------------------------------
	mCalcBinning();
	//-----------------
	m_iOverlap = (int)(m_iTileSize * m_fOverlap);
	m_iOverlap = m_iOverlap / 2 * 2;
	//-----------------
	int iSize = m_iTileSize - m_iOverlap;
	m_aiImgTiles[0] = (m_aiImgSize[0] - m_iOverlap) / iSize;
	m_aiImgTiles[1] = (m_aiImgSize[1] - m_iOverlap) / iSize;
	//-----------------
	m_aiOffset[0] = (m_aiImgSize[0] - m_aiImgTiles[0] * m_iTileSize
	   + (m_aiImgTiles[0] - 1) * m_iOverlap) / 2;
	m_aiOffset[1] = (m_aiImgSize[1] - m_aiImgTiles[1] * m_iTileSize
	   + (m_aiImgTiles[1] - 1) * m_iOverlap) / 2;
}

void CTsTiles::mCalcBinning(void)
{
	m_fBinning = 1.0f;
	if(m_fPixSize >= 1.0f) return;
	//-----------------
	m_fBinning = 1.0f / m_fPixSize;
	if(m_fBinning > 2.0f) m_fBinning = 2.0f;
	//-----------------
	m_aiImgSize[0] = (int)(m_aiImgSize[0] / m_fBinning) / 2 * 2;
	m_aiImgSize[1] = (int)(m_aiImgSize[1] / m_fBinning) / 2 * 2;
	//-----------------
	m_fPixSize *= m_fBinning;
}

void CTsTiles::mDoTilt(int iTilt)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	float* pfImg = (float*)pTiltSeries->GetFrame(iTilt);
	float fTilt = pTiltSeries->m_pfTilts[iTilt];
	//-----------------
	float* pfBinnedImg = pfImg;
	if(m_fBinning > 1.0f) 
	{	int iBinnedSize = m_aiImgSize[0] * m_aiImgSize[1];
		pfBinnedImg = new float[iBinnedSize];
		mDoBinning(iTilt, pfBinnedImg);
	}
	//-----------------
	int iImgTiles = m_aiImgTiles[0] * m_aiImgTiles[1];
	int aiSpectSize[] = {m_iTileSize / 2 + 1, m_iTileSize};
	int iOffset = iTilt * iImgTiles;
	//-----------------
	float* pfStds = new float[iImgTiles];
	//-----------------
	for(int i=0; i<iImgTiles; i++)
	{	int j = i + iOffset;
		m_pTiles[j].SetSize(aiSpectSize);
		m_pTiles[j].SetTilt(fTilt);
		//----------------
		mExtractPadTile(iTilt, i, pfBinnedImg);
		pfStds[i] = mGenTileSpect(iTilt, i);
	}
	//-----------------
	if(m_fBinning > 1.0f) delete[] pfBinnedImg;
	//-----------------
	float fMean = 0.0f, fStd = 0.0f;
	for(int i=0; i<iImgTiles; i++)
	{	fMean += pfStds[i];
		fStd += (pfStds[i] * pfStds[i]);
	}
	fMean = fMean / iImgTiles;
	fStd = fStd / iImgTiles - fMean * fMean;
	if(fStd < 0) fStd = 0.0f;
	else fStd = (float)sqrtf(fStd);
	float fTol = fmax(fMean - 2.0f * fStd, 0.2f * fMean);
	//-----------------
	for(int i=0; i<iImgTiles; i++)
	{	int j = i + iOffset;
		if(pfStds[i] == 0) m_pTiles[j].SetGood(false);
		else if(pfStds[i] < fTol) m_pTiles[j].SetGood(false);
		else m_pTiles[j].SetGood(true);
	}
	if(pfStds != 0L) delete[] pfStds;
}

void CTsTiles::mDoBinning(int iTilt, float* pfBinnedImg)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
        MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
        float* pfRawImg = (float*)pTiltSeries->GetFrame(iTilt);
	//-----------------
	MU::GFourierResize2D gFtResize;
	gFtResize.Setup(pTiltSeries->m_aiStkSize, m_aiImgSize);
	gFtResize.DoIt(pfRawImg, pfBinnedImg);
}

float CTsTiles::mGenTileSpect(int iTilt, int iTile)
{	
	float fMean = m_pGCalcMoment2D->DoIt(m_gfPadTile, 1, true);
	float fStd = m_pGCalcMoment2D->DoIt(m_gfPadTile, 2, true);
	fStd = fStd - fMean * fMean;
	if(fStd < 0) fStd = 0.0f;
	else fStd = sqrtf(fStd);
	float fStd1 = (fStd == 0) ? 1.0f : fStd;
	//-----------------
	MU::GNormalize2D gNorm;
	int aiPadSize[] = {(m_iTileSize / 2 + 1) * 2, m_iTileSize};
	gNorm.DoIt(m_gfPadTile, aiPadSize, true, fMean, fStd1);
	//-----------------
	GRoundEdge aGRoundEdge;
	float afCent[] = {m_iTileSize * 0.5f, m_iTileSize * 0.5f};
	float afSize[] = {m_iTileSize * 1.0f, m_iTileSize * 1.0f};
	aGRoundEdge.SetMask(afCent, afSize);
	aGRoundEdge.DoIt(m_gfPadTile, aiPadSize, false);
	//-----------------
	bool bLogrith = true;
	int i = iTilt * m_aiImgTiles[0] * m_aiImgTiles[1] + iTile;
	float* qfTile = m_pTiles[i].GetTile();
	m_pGCalcSpectrum->DoPad(m_gfPadTile, qfTile, !bLogrith);
	//-----------------
	return fStd;
}

void CTsTiles::mExtractPadTile(int iTilt, int iImgTile, float* pfImg)
{
	int iTileX = iImgTile % m_aiImgTiles[0];
	int iTileY = iImgTile / m_aiImgTiles[0];
	int iStartX = m_aiOffset[0] + iTileX * (m_iTileSize - m_iOverlap);
	int iStartY = m_aiOffset[1] + iTileY * (m_iTileSize - m_iOverlap);
	//-----------------
	size_t tBytes = sizeof(float) * m_iTileSize;
	int iOffset = iStartY * m_aiImgSize[0] + iStartX;
	int iTilePadX = (m_iTileSize / 2 + 1) * 2;
	//-----------------
	for(int y=0; y<m_iTileSize; y++)
	{	float* pfSrc = pfImg + y * m_aiImgSize[0] + iOffset;
		float* gfDst = m_gfPadTile + y * iTilePadX;
		cudaMemcpy(gfDst, pfSrc, tBytes, cudaMemcpyDefault);
	}
	//-----------------------------------------------
	// Tile center is relative to the image center.
	//-----------------------------------------------
	float fCentX = iStartX + m_iTileSize * 0.5f - m_aiImgSize[0] * 0.5f;
	float fCentY = iStartY + m_iTileSize * 0.5f - m_aiImgSize[1] * 0.5f;
	//-----------------
	int i = iTilt * m_aiImgTiles[0] * m_aiImgTiles[1] + iImgTile;
	m_pTiles[i].SetCentX(fCentX);
	m_pTiles[i].SetCentY(fCentY);
	m_pTiles[i].SetCentZ(0.0f);
	m_pTiles[i].SetPixSize(m_fPixSize);
}	
