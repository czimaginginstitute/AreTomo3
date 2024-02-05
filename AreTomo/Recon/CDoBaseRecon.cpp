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
