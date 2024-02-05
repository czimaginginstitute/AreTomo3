#include "CDataUtilInc.h"
#include "../MaUtil/CMaUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/types.h>
#include <errno.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::DataUtil;

CStackBuffer::CStackBuffer(void)
{
	m_pGpuBuffer = 0L;
}

CStackBuffer::~CStackBuffer(void)
{
	this->Clean();
}

void CStackBuffer::Clean(void)
{
	if(m_pGpuBuffer == 0L) return;
	delete m_pGpuBuffer;
	m_pGpuBuffer = 0L;
}

//-----------------------------------------------------------------------------
// CStackBuffer is responsible to distribute stack frames to the memories of
// all available GPUs. If not enough, CPU memory is used to buffer the
// remaining frames. Each GPU buffer is abstracted in CGpuBuffer that hosts
// a subset of frames. CPU buffer is also abstracted in CGpuBuffer with its
// ID set to -1.
//-----------------------------------------------------------------------------
void CStackBuffer::Create
(	int* piCmpSize,
	int iNumFrames,
	int iGpuID
)
{	this->Clean();
	//-----------------
	m_aiCmpSize[0] = piCmpSize[0];
	m_aiCmpSize[1] = piCmpSize[1];
	m_iNumFrames = iNumFrames;
	m_iGpuID = iGpuID;
	m_tFmBytes = sizeof(cufftComplex) * 
	   m_aiCmpSize[0] * m_aiCmpSize[1];
	//-----------------
	m_pGpuBuffer = new CGpuBuffer;
	m_pGpuBuffer->Create(m_tFmBytes, m_iNumFrames, m_iGpuID);
}

void CStackBuffer::Adjust(int iNumFrames)
{
	if(iNumFrames == m_iNumFrames) return;
	m_iNumFrames = iNumFrames;
	m_pGpuBuffer->AdjustBuffer(m_iNumFrames);
}

bool CStackBuffer::IsGpuFrame(int iFrame)
{
	if(iFrame < m_pGpuBuffer->m_iNumGpuFrames) return true;
	else return false;
}

cufftComplex* CStackBuffer::GetFrame(int iFrame)
{
	void* pvFrame = m_pGpuBuffer->GetFrame(iFrame);
	cufftComplex* pCmpFrame = reinterpret_cast<cufftComplex*>(pvFrame);
	return pCmpFrame;
}
