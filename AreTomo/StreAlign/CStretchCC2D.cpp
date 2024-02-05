#include "CStreAlignInc.h"
#include "../Util/CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::StreAlign;

CStretchCC2D::CStretchCC2D(void)
{
	m_gfRefImg = 0L;
	m_gfImg = 0L;
	m_gfBuf = 0L;
	m_aiSize[0] = 0;
	m_aiSize[1] = 0;
}

CStretchCC2D::~CStretchCC2D(void)
{
	this->Clean();
}

void CStretchCC2D::Clean(void)
{
	if(m_gfRefImg != 0L) cudaFree(m_gfRefImg);
	if(m_gfImg != 0L) cudaFree(m_gfImg);
	if(m_gfBuf != 0L) cudaFree(m_gfBuf);
	m_gfRefImg = 0L;
	m_gfImg = 0L;
	m_gfBuf = 0L;
}

void CStretchCC2D::SetSize(int* piSize, bool bPadded)
{
	this->Clean();
	m_aiSize[0] = piSize[0];
	m_aiSize[1] = piSize[1];
	m_bPadded = bPadded;
	//-----------------
	size_t tBytes = sizeof(float) * piSize[0] * piSize[1];
	cudaMalloc(&m_gfRefImg, tBytes);
	cudaMalloc(&m_gfImg, tBytes);
	cudaMalloc(&m_gfBuf, tBytes);
}
	
float CStretchCC2D::DoIt
(	float* pfRefImg,
	float* pfImg,
	float fRefTilt,
	float fTilt,
	float fTiltAxis
)
{	size_t tBytes = sizeof(float) * m_aiSize[0] * m_aiSize[1];
	cudaMemcpy(m_gfBuf, pfImg, tBytes, cudaMemcpyDefault);
	cudaMemcpy(m_gfRefImg, pfRefImg, tBytes, cudaMemcpyDefault);
	//-----------------
	double dRad = 4 * atan(1.0) / 180.0;
	double dStretch = cos(dRad * fRefTilt) / cos(dRad * fTilt);
	//-----------------
	MAU::GTiltStretch tiltStretch;
	tiltStretch.DoIt(m_gfBuf, m_aiSize, m_bPadded,
	   (float)dStretch, fTiltAxis, m_gfImg);
	//-----------------
	MAU::GRealCC2D aGRealCC2D;
	float fCC = aGRealCC2D.DoIt(m_gfRefImg, m_gfImg, m_gfBuf, 
	   m_aiSize, m_bPadded);
	return fCC;
}

