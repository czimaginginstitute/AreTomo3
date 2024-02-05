#include "CDataUtilInc.h"
#include <Mrcfile/CMrcFileInc.h>
#include <memory.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::DataUtil;

CMrcStack::CMrcStack(void)
{
	memset(m_aiStkSize, 0, sizeof(m_aiStkSize));
	m_iMode = -1;
	m_tFmBytes = 0;
	m_iBufSize = 0;
	//-----------------
	m_iBufSize = 0;
	m_ppvFrames = 0L;
}

CMrcStack::~CMrcStack(void)
{
	mCleanFrames();
}

void CMrcStack::Create(int iMode, int* piStkSize)
{
	//----------------------------------------------------
	// reallocate only when the new frame byte is larger.
	//----------------------------------------------------	
	size_t tFmBytes = Mrc::C4BitImage::GetImgBytes(iMode, piStkSize);
	if(tFmBytes > m_tFmBytes)
	{	mCleanFrames();
		m_tFmBytes = tFmBytes;
		mExpandBuf(piStkSize[2]);
	}
	//-----------------------------------------
	// increase number of frames of the buffer
	//-----------------------------------------
	else
	{	m_tFmBytes = tFmBytes;
		if(piStkSize[2] > m_iBufSize) mExpandBuf(piStkSize[2]);
	}
	m_iMode = iMode;
	memcpy(m_aiStkSize, piStkSize, sizeof(int) * 3);
}

void* CMrcStack::GetFrame(int iFrame)
{
	if(m_ppvFrames == 0L) return 0L;
	else if(iFrame >= m_aiStkSize[2]) return 0L;
	else return m_ppvFrames[iFrame];
}

int CMrcStack::GetPixels(void)
{
	return m_aiStkSize[0] * m_aiStkSize[1];
}

void CMrcStack::mCleanFrames(void)
{
	if(m_ppvFrames == 0L) return;
	for(int i=0; i<m_iBufSize; i++)
	{	char* pcFrame = (char*)m_ppvFrames[i];
		if(pcFrame != 0L) delete[] pcFrame;
	}
	//-----------------
	delete[] (char**)m_ppvFrames;
	m_ppvFrames = 0L;
	m_iBufSize = 0;
}

void CMrcStack::mExpandBuf(int iNumFrames)
{
	if(iNumFrames <= m_iBufSize) return;
	//-----------------
	void** ppvFrames = new void*[iNumFrames];
	for(int i=0; i<m_iBufSize; i++)
	{	ppvFrames[i] = m_ppvFrames[i];
	}
	//-----------------
	for(int i=m_iBufSize; i<iNumFrames; i++)
	{	ppvFrames[i] = new char[m_tFmBytes];
	}
	//-----------------
	if(m_ppvFrames != 0L) delete[] (char**)m_ppvFrames;
	m_ppvFrames = ppvFrames;
	m_iBufSize = iNumFrames;
}

