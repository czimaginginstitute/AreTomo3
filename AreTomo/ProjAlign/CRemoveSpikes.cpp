#include "CProjAlignInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo::ProjAlign;

CRemoveSpikes::CRemoveSpikes(void)
{
}

CRemoveSpikes::~CRemoveSpikes(void)
{
}

void CRemoveSpikes::DoIt(MD::CTiltSeries* pTiltSeries)
{
	bool bPadded = true;
	bool bGpu = true;
	int iWinSize = 11;
	m_tFmBytes = sizeof(float) * pTiltSeries->GetPixels();
	cudaMalloc(&m_gfInFrm, m_tFmBytes);
	cudaMalloc(&m_gfOutFrm, m_tFmBytes);
	//-----------------
	MAU::GRemoveSpikes2D removeSpikes;
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float* pfFrm = (float*)pTiltSeries->GetFrame(i);
		cudaMemcpy(m_gfInFrm, pfFrm, m_tFmBytes, cudaMemcpyDefault);
		removeSpikes.DoIt(m_gfInFrm, pTiltSeries->m_aiStkSize,
		   !bPadded, iWinSize, m_gfOutFrm);
		cudaMemcpy(pfFrm, m_gfOutFrm, m_tFmBytes, cudaMemcpyDefault);
	}
	//-----------------
	cudaFree(m_gfInFrm);
	cudaFree(m_gfOutFrm);
}
