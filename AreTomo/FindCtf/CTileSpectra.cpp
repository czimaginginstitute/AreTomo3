#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.01745329f;

CTileSpectra* CTileSpectra::m_pInstances = 0L;
int CTileSpectra::m_iNumGpus = 0;

CTileSpectra* CTileSpectra::GetInstance(int iNthGpu)
{
	if(m_pInstances != 0L) return &m_pInstances[iNthGpu];
	else return 0L;
}

void CTileSpectra::CreateInstances(int iNumGpus)
{
	if(iNumGpus == m_iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	//-----------------
	m_pInstances = new CTileSpectra[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	//-----------------
	m_iNumGpus = iNumGpus;
}

void CTileSpectra::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
}

CTileSpectra::CTileSpectra(void)
{
        m_ppfSpectra = 0L;
	m_pfTileXs = 0L;
	m_pfTileYs = 0L;
	m_pfTileZs = 0L;
	//-----------------
	m_pfMeans = 0L;
	m_pfStds = 0L;
	m_pfWeights = 0L;
	//-----------------
        m_iBufSize = 0;
        m_iSeriesTiles = 0;
        m_iNumTilts = 0;
	m_iTilesPerTilt = 0;
	//-----------------
	memset(m_aiSpectSize, 0, sizeof(m_aiSpectSize));
	m_iNthGpu = 0;
}

CTileSpectra::~CTileSpectra(void)
{
	this->Clean();
}

void CTileSpectra::Clean(void)
{
	if(m_ppfSpectra != 0L)
	{	for(int i=0; i<m_iBufSize; i++)
		{	float* pfSpect = m_ppfSpectra[i];
			if(pfSpect != 0L) cudaFreeHost(pfSpect);
		}
		delete[] m_ppfSpectra;
        }
	m_ppfSpectra = 0L;
	//-----------------
	if(m_pfTileXs != 0L) delete[] m_pfTileXs;
	if(m_pfMeans != 0L) delete[] m_pfMeans;
	m_pfTileXs = 0L;
	m_pfMeans = 0L;
	m_pfWeights = 0L;
	//-----------------
	m_iSeriesTiles = 0;
        m_iBufSize = 0;
}

void CTileSpectra::Create(int iTilesPerTilt, int iNumTilts, int iTileSize)
{
        this->Clean();
	//-----------------
	m_iTilesPerTilt = iTilesPerTilt;
	m_iNumTilts = iNumTilts;
	//-----------------
	m_aiSpectSize[0] = iTileSize / 2 + 1;
	m_aiSpectSize[1] = iTileSize;
	//-----------------
        mExpandBuf();
}

void CTileSpectra::Adjust(int iTilesPerTilt, int iNumTilts)
{	
	m_iTilesPerTilt = iTilesPerTilt;
	m_iNumTilts = iNumTilts;
	mExpandBuf();
}

void CTileSpectra::SetSpect(int iTile, int iTilt, float* gfSpect)
{
	float* pfSpect = m_ppfSpectra[iTilt * m_iTilesPerTilt + iTile];
	size_t tBytes = this->GetSpectSize() * sizeof(float);
	cudaMemcpy(pfSpect, gfSpect, tBytes, cudaMemcpyDefault);
}

void CTileSpectra::SetAvgSpect(int iTilt, float* gfSpect)
{
	size_t tBytes = this->GetSpectSize() * sizeof(float);
	float* pfSpect = m_ppfSpectra[m_iSeriesTiles + iTilt];
	cudaMemcpy(pfSpect, gfSpect, tBytes, cudaMemcpyDefault);
}

void CTileSpectra::SetStat(int iTile, int iTilt, float* pfStat)
{
	int i = iTilt * m_iTilesPerTilt + iTile;
	m_pfMeans[i] = pfStat[0];
	m_pfStds[i] = pfStat[1];
}

void CTileSpectra::Screen(void)
{
	//-----------------------------------------------
	// 1) Too dark or too bright tiles are rejected.
	//-----------------------------------------------	
	float afMeanStat[2] = {0.0f};
	mCalcStat(m_pfMeans, afMeanStat);
	//-----------------
	float afStdStat[2] = {0.0f};
	mCalcStat(m_pfStds, afStdStat);
	//-----------------
	int iSize = m_iNumTilts * m_iTilesPerTilt;
	if(afMeanStat[1] == 0 || afStdStat[1] == 0)
	{	for(int i=0; i<iSize; i++) m_pfWeights[i] = 1.0f;
		return;
	}
	//-----------------
	float fFact1 = 1.0f / (afMeanStat[1] * sqrtf(6.24f));
	float fFact2 = 1.0f / (afStdStat[1] * sqrtf(6.24f));
	double dV1 = 0.0f, dV2 = 0.0f, dV = 0.0, dMax = -1e30;
	//-----------------
	for(int i=0; i<iSize; i++)
	{	dV1 = (m_pfMeans[i] - afMeanStat[0]) / afMeanStat[1];
		dV2 = (m_pfStds[i] - afStdStat[0]) / afStdStat[1];
		dV = -0.5 * (dV1 * dV1 + dV2 * dV2);
		m_pfWeights[i] = (float)(fFact1 * fFact2 * exp(dV));
	}
}

//---------------------------------------------------------------
// pfXY is the coordinates at center of the tile and relative
// to the center of the raw image. Therefore the coordinates
// can be positive or negative.
//---------------------------------------------------------------
void CTileSpectra::SetXY(int iTile, int iTilt, float* pfXY)
{
        int i = iTilt * m_iTilesPerTilt + iTile;
        m_pfTileXs[i] = pfXY[0];
        m_pfTileYs[i] = pfXY[1];
}

float* CTileSpectra::GetSpect(int iTile, int iTilt)
{
	int i = iTilt * m_iTilesPerTilt + iTile;
	return m_ppfSpectra[i];
}

float* CTileSpectra::GetAvgSpect(int iTilt)
{
	int i = m_iSeriesTiles + iTilt;
	return m_ppfSpectra[i];
}

float CTileSpectra::GetWeight(int iTile, int iTilt)
{
	int i = iTilt * m_iTilesPerTilt + iTile;
	return m_pfWeights[i];
}

void CTileSpectra::GetXYZ(int iTile, int iTilt, float* pfXYZ)
{
        int i = iTilt * m_iTilesPerTilt + iTile;
        pfXYZ[0] = m_pfTileXs[i];
        pfXYZ[1] = m_pfTileYs[i];
        pfXYZ[2] = m_pfTileZs[i];
}

void CTileSpectra::GetXY(int iTile, int iTilt, float* pfXY)
{
        int i = iTilt * m_iTilesPerTilt + iTile;
        pfXY[0] = m_pfTileXs[i];
        pfXY[1] = m_pfTileYs[i];
}

float CTileSpectra::GetZ(int iTile, int iTilt)
{
        int i = iTilt * m_iTilesPerTilt + iTile;
        return m_pfTileZs[i];
}

int CTileSpectra::GetSpectSize(void)
{
	int iSize = m_aiSpectSize[0] * m_aiSpectSize[1];
	return iSize;
}

void CTileSpectra::CalcZs
(	float* pfTilts,
	float fTiltAxis,
	float fTiltOffset,
	float fBetaOffset
)
{	m_fTiltAxis = fTiltAxis;
	m_fTiltOffset = fTiltOffset;
	m_fBetaOffset = fBetaOffset;
	//-----------------
	for(int i=0; i<m_iNumTilts; i++)
	{	mCalcTiltZs(i, pfTilts[i]);
	}
}

//--------------------------------------------------------------------
// 1. There are two types of spectrums 1) Series tile spectrums are
//    ones of tiles extracted from tilt images. 2) Averaged tile 
//    spectrums are the averaged spectrums at each tilts.
// 2. Averaged spectra are placed at the end of series spectra.
//--------------------------------------------------------------------
void CTileSpectra::mExpandBuf(void)
{
	m_iSeriesTiles = m_iTilesPerTilt * m_iNumTilts;
	int iAllTiles = m_iSeriesTiles + m_iNumTilts;
        if(m_iBufSize >= iAllTiles) return;
        //----------------
	float* pfTileXs = new float[m_iSeriesTiles * 3];
	float* pfTileYs = &pfTileXs[m_iSeriesTiles];
	float* pfTileZs = &pfTileXs[m_iSeriesTiles * 2];
        //----------------
	float* pfMeans = new float[m_iSeriesTiles * 3];
	float* pfStds = &pfMeans[m_iSeriesTiles];
	float* pfWeights = &pfMeans[m_iSeriesTiles * 2];
	//----------------
        float** ppfSpectra = new float*[iAllTiles];
	memset(ppfSpectra, 0, sizeof(float*) * iAllTiles);
	//----------------
        for(int i=0; i<m_iBufSize; i++)
        {       ppfSpectra[i] = m_ppfSpectra[i];
		m_ppfSpectra[i] = 0L;
	}
        //----------------
        size_t tBytes = sizeof(float) * this->GetSpectSize();
        for(int i=m_iBufSize; i<iAllTiles; i++)
        {       float* pfSpect = 0L;
                cudaMallocHost(&pfSpect, tBytes);
                ppfSpectra[i] = pfSpect;
        }
        //-----------------
        if(m_ppfSpectra != 0L) delete[] m_ppfSpectra;
	if(m_pfTileXs != 0L) delete[] m_pfTileXs;
	if(m_pfMeans != 0L) delete[] m_pfMeans;
	//-----------------
        m_ppfSpectra = ppfSpectra;
	m_pfTileXs = pfTileXs;
	m_pfTileYs = pfTileYs;
	m_pfTileZs = pfTileZs;
	//-----------------
	m_pfMeans = pfMeans;
	m_pfStds = pfStds;
	m_pfWeights = pfWeights;
	//-----------------
        m_iBufSize = iAllTiles;
}

void CTileSpectra::mCalcStat(float* pfVals, float* pfStat)
{
	int iSize = m_iNumTilts * m_iTilesPerTilt;
	double dSum1 = 0, dSum2 = 0;
	for(int i=0; i<iSize; i++)
	{	dSum1 += pfVals[i];
		dSum2 += (pfVals[i] * pfVals[i]);
	}
	pfStat[0] = (float)(dSum1 / iSize);
	pfStat[1] = (float)(dSum2 / iSize - pfStat[0] * pfStat[0]);
	if(pfStat[1] < 0) pfStat[1] = 0;
	pfStat[1] = (float)sqrtf(pfStat[1]);
}

//--------------------------------------------------------------------
// 1. Positive alpha tilt induces -z on samples at positive x.
// 2. Positive beta tilt indices +z on sample at positive y.
//--------------------------------------------------------------------
void CTileSpectra::mCalcTiltZs(int iTilt, float fTilt)
{
	float fCosTx = (float)cos(m_fTiltAxis * s_fD2R);
	float fSinTx = (float)sin(m_fTiltAxis * s_fD2R);
	//-----------------
	float fTanA = (float)tan((fTilt + m_fTiltOffset) * s_fD2R);
	float fTanB = (float)tan(m_fBetaOffset * s_fD2R);
	//-----------------
	float fXp = 0.0f, fYp = 0.0f;
	int iOffset = iTilt * m_iTilesPerTilt;
	//-----------------
	for(int i=0; i<m_iTilesPerTilt; i++)
	{	int j = i + iOffset;
		fXp = m_pfTileXs[j] * fCosTx + m_pfTileYs[j] * fSinTx;
		fYp = -m_pfTileXs[j] * fSinTx + m_pfTileYs[j] * fCosTx;
		m_pfTileZs[j] = -fXp * fTanA + fYp * fTanB;
	}
}
