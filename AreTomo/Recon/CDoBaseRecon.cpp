#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::Recon;

CDoBaseRecon::CDoBaseRecon(void)
{
	m_gfPadSinogram = 0L;
	m_pfPadSinogram = 0L;
	m_gfVolXZ = 0L;
	m_pfVolXZ = 0L;
}

CDoBaseRecon::~CDoBaseRecon(void)
{
	this->Clean();
}

void CDoBaseRecon::Clean(void)
{
	if(m_gfPadSinogram != 0L) cudaFree(m_gfPadSinogram);
	if(m_pfPadSinogram != 0L) cudaFreeHost(m_pfPadSinogram);
	if(m_gfVolXZ != 0L) cudaFree(m_gfVolXZ);
	if(m_pfVolXZ != 0L) cudaFreeHost(m_pfVolXZ);
	m_gfPadSinogram = 0L;
	m_pfPadSinogram = 0L;
	m_gfVolXZ = 0L;
	m_pfVolXZ = 0L;
}

float* CDoBaseRecon::mCalcProjXY(void)
{
	if(m_pVolSeries == 0L) return 0L;
	int iPixels = m_pVolSeries->m_aiStkSize[0] *
	   m_pVolSeries->m_aiStkSize[1];
	float* pfProjXY = new float[iPixels];
	memset(pfProjXY, 0, sizeof(float) * iPixels);
	//-----------------
	for(int z=0; z<m_pVolSeries->m_aiStkSize[2]; z++)
	{	float* pfSlice = (float*)m_pVolSeries->GetFrame(z);
		for(int i=0; i<iPixels; i++)
		{	pfProjXY[i] += pfSlice[i];
		}
	}
	return pfProjXY;
}

float* CDoBaseRecon::mCalcProjXZ(void)
{
	if(m_pVolSeries == 0L) return 0L;
	int iSizeXZ = m_pVolSeries->m_aiStkSize[0] *
	   m_pVolSeries->m_aiStkSize[2];
	int iSizeXY = m_pVolSeries->m_aiStkSize[0] *
	   m_pVolSeries->m_aiStkSize[1];
	float* pfProjXZ = new float[iSizeXZ];
	memset(pfProjXZ, 0, sizeof(float) * iSizeXZ);
	//----------------
	for(int z=0; z<m_pVolSeries->m_aiStkSize[2]; z++)
	{	float* pfSliceXY = (float*)m_pVolSeries->GetFrame(z);
		float* pfLineZ = pfProjXZ + z * m_pVolSeries->m_aiStkSize[0];
		for(int i=0; i<iSizeXY; i++)
		{	int x = i % m_pVolSeries->m_aiStkSize[0];
			int y = i / m_pVolSeries->m_aiStkSize[0];
			pfLineZ[x] += pfSliceXY[i];
		}
	}
	return pfProjXZ;
}
