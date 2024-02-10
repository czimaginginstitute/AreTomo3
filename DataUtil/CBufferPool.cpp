#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include "../MaUtil/CMaUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/types.h>
#include <errno.h>
#include <Util/Util_Time.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::DataUtil;

CBufferPool* CBufferPool::m_pInstances = 0L;
int CBufferPool::m_iNumGpus = 0;

void CBufferPool::CreateInstances(int iNumGpus)
{	
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	//-----------------
	m_pInstances = new CBufferPool[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CBufferPool::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CBufferPool* CBufferPool::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CBufferPool::CBufferPool(void)
{
	mInit();
	m_bCreated = false;
}

CBufferPool::~CBufferPool(void)
{
	this->Clean();
}

void CBufferPool::Clean(void)
{
	if(m_pTmpBuffer != 0L) delete m_pTmpBuffer;
	if(m_pSumBuffer != 0L) delete m_pSumBuffer;
	if(m_pFrmBuffer != 0L) delete m_pFrmBuffer;
	if(m_pXcfBuffer != 0L) delete m_pXcfBuffer;
	if(m_pPatBuffer != 0L) delete m_pPatBuffer;
	if(m_avPinnedBuf[0] != 0L) cudaFreeHost(m_avPinnedBuf[0]);
	if(m_avPinnedBuf[1] != 0L) cudaFreeHost(m_avPinnedBuf[1]);
	memset(m_avPinnedBuf, 0, sizeof(m_avPinnedBuf));
	//-----------------
	if(m_pCufft2Ds != 0L)
	{	m_pCufft2Ds[0].DestroyPlan();
		m_pCufft2Ds[1].DestroyPlan();
	}
	if(m_pCudaStreams != 0L)
	{	cudaStreamDestroy(m_pCudaStreams[0]);
		cudaStreamDestroy(m_pCudaStreams[1]);
	}
	if(m_pCufft2Ds != 0L) delete[] m_pCufft2Ds;
	if(m_pCudaStreams != 0L) delete[] m_pCudaStreams;
	//------------------
	mInit();
	m_bCreated = false;
}

//------------------------------------------------------------------------------
// 1. This is used for the first-time creation of the buffer. It should be
//    called only once throughout the lifetime once the program is started.
// 2. The frame size cannot be changed.
// 3. The number of frames in a movie can be changed using Adjust(...);
//------------------------------------------------------------------------------
void CBufferPool::Create(int* piStkSize)
{
	if(m_bCreated) 
	{	this->Adjust(piStkSize[2]);
		return;
	}
	//-----------------
	this->Clean();
	CInput* pInput = CInput::GetInstance();
	m_iGpuID = pInput->m_piGpuIDs[m_iNthGpu];
	//-----------------
	memcpy(m_aiStkSize, piStkSize, sizeof(m_aiStkSize));
	m_pCufft2Ds = new MU::CCufft2D[2];
	//-----------------
	m_pCudaStreams = new cudaStream_t[2];
        cudaStreamCreate(&m_pCudaStreams[0]);
        cudaStreamCreate(&m_pCudaStreams[1]);
	//-----------------
	Util_Time aTimer;
	aTimer.Measure();
	//---------------------------------------------------------------
	// If the proc queue does not have the package, this is because
	// the loading is not done and the package has not been placed
	// from the load queue to the proc queue. This happens in the
	// single processing or the first movie of the batch processing.
	//---------------------------------------------------------------
	mCreateSumBuffer();
	mCreateTmpBuffer();
	mCreateXcfBuffer();
	mCreatePatBuffer();
	mCreateFrmBuffer();
	m_bCreated = true;
	//-----------------
	float fTime = aTimer.GetElapsedSeconds();
	printf("Create buffers: %.2f seconds\n\n", fTime);	
}

void CBufferPool::Adjust(int iNumFrames)
{
	if(iNumFrames == m_aiStkSize[2]) return;
	else m_aiStkSize[2] = iNumFrames;
	//-----------------
	m_pFrmBuffer->Adjust(iNumFrames);
	m_pXcfBuffer->Adjust(iNumFrames);
	m_pPatBuffer->Adjust(iNumFrames);

}

CStackBuffer* CBufferPool::GetBuffer(int iBuf)
{
	if(iBuf == EBuffer::tmp) return m_pTmpBuffer;
	else if(iBuf == EBuffer::sum) return m_pSumBuffer;
	else if(iBuf == EBuffer::frm) return m_pFrmBuffer;
	else if(iBuf == EBuffer::xcf) return m_pXcfBuffer;
	else if(iBuf == EBuffer::pat) return m_pPatBuffer;
	else return 0L;
}

void* CBufferPool::GetPinnedBuf(int iFrame)
{
	return m_avPinnedBuf[iFrame];
}

MU::CCufft2D* CBufferPool::GetCufft2D(bool bForward)
{
	if(m_pCufft2Ds == 0L) return 0L;
	else if(bForward) return &m_pCufft2Ds[0];
	else return &m_pCufft2Ds[1];
}

cudaStream_t CBufferPool::GetCudaStream(int iStream)
{
	if(m_pCudaStreams == 0L) return (cudaStream_t)0;
	return m_pCudaStreams[iStream];
}

void CBufferPool::mCreateSumBuffer(void)
{
	CMcPackage* pMcPackage = CMcPackage::GetInstance(m_iNthGpu);
	m_iNumSums = pMcPackage->m_pAlnSums->m_iNumSums;
	//-----------------
	m_pSumBuffer = new CStackBuffer;
	int aiCmpSize[] = {m_aiStkSize[0] / 2 + 1, m_aiStkSize[1]};
	m_pSumBuffer->Create(aiCmpSize, m_iNumSums, m_iGpuID); 
}

void CBufferPool::mCreateTmpBuffer(void)
{
	int iNumFrames = 4;
	//-----------------
	int iSizeX = (m_aiStkSize[0] > 4096) ? m_aiStkSize[0] : 4096;
	int iSizeY = (m_aiStkSize[1] > 4096) ? m_aiStkSize[1] : 4096;
	int aiCmpSize[] = {iSizeX / 2 + 1, iSizeY};
	//-----------------	
	m_pTmpBuffer = new CStackBuffer;
	m_pTmpBuffer->Create(aiCmpSize, iNumFrames, m_iGpuID);
	//-----------------
	cudaMallocHost(&m_avPinnedBuf[0], m_pTmpBuffer->m_tFmBytes);
	cudaMallocHost(&m_avPinnedBuf[1], m_pTmpBuffer->m_tFmBytes);
}

void CBufferPool::mCreateXcfBuffer(void)
{
	float fBinX = m_aiStkSize[0] / 2048.0f;
	float fBinY = m_aiStkSize[1] / 2048.0f;
	float fBin = fmin(fBinX, fBinY);
	if(fBin < 1) fBin = 1.0f;
	//-----------------
	int iXcfX = (int)(m_aiStkSize[0] / fBin + 0.5f);
	int iXcfY = (int)(m_aiStkSize[1] / fBin + 0.5f);
	//-----------------
	int aiXcfCmpSize[] = {iXcfX / 2 + 1, iXcfY / 2 * 2};
	m_afXcfBin[0] = (m_pSumBuffer->m_aiCmpSize[0] - 1.0f)
	   / (aiXcfCmpSize[0] - 1.0f);
	m_afXcfBin[1] = m_pSumBuffer->m_aiCmpSize[1]
	   * 1.0f / aiXcfCmpSize[1];
	//-----------------
	m_pXcfBuffer = new CStackBuffer;
	m_pXcfBuffer->Create(aiXcfCmpSize, m_aiStkSize[2], m_iGpuID); 
}

void CBufferPool::mCreatePatBuffer(void)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(pMcInput->m_aiNumPatches[0] <= 1) return;
	if(pMcInput->m_aiNumPatches[1] <= 1) return;
	if(m_pXcfBuffer == 0L) return;
	//----------------------------
	int iXcfX = (m_pXcfBuffer->m_aiCmpSize[0] - 1) * 2;
	int iXcfY = m_pXcfBuffer->m_aiCmpSize[1];
	int iPatX = iXcfX / pMcInput->m_aiNumPatches[0];
	int iPatY = iXcfY / pMcInput->m_aiNumPatches[1];
	if(iPatX > (iXcfX / 2)) return;
	if(iPatY > (iXcfY / 2)) return;
	//-----------------------------
	int aiPatCmpSize[2] = {iPatX / 2 + 1, iPatY / 2 * 2};
	m_pPatBuffer = new CStackBuffer;
	m_pPatBuffer->Create(aiPatCmpSize, 
	   m_pXcfBuffer->m_iNumFrames, m_iGpuID);
}

void CBufferPool::mCreateFrmBuffer(void)
{
	m_pFrmBuffer = new CStackBuffer;
	int aiCmpSize[] = {m_aiStkSize[0] / 2 + 1, m_aiStkSize[1]};
	m_pFrmBuffer->Create(aiCmpSize, m_aiStkSize[2], m_iGpuID);
}

void CBufferPool::mInit(void)
{
	m_pTmpBuffer = 0L;
	m_pSumBuffer = 0L;
	m_pFrmBuffer = 0L;
	m_pXcfBuffer = 0L;
	m_pPatBuffer = 0L;
	m_pCufft2Ds = 0L;
	m_pCudaStreams = 0L;
	memset(m_avPinnedBuf, 0, sizeof(m_avPinnedBuf));
	memset(m_aiStkSize, 0, sizeof(m_aiStkSize));
}
