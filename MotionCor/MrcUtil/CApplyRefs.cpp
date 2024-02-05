#include "CMrcUtilInc.h"
#include "../CMotionCorInc.h"
#include "../Util/CUtilInc.h"
#include "../DataUtil/CDataUtilInc.h"
#include <Util/Util_Time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::MrcUtil;
using namespace McAreTomo::MotionCor;
namespace MMU = McAreTomo::MotionCor::Util;

CApplyRefs::CApplyRefs(void)
{	
	m_gfDark = 0L;
	m_gfGain = 0L;
	m_pvMrcFrames[0] = 0L;
        m_pvMrcFrames[1] = 0L;
}

CApplyRefs::~CApplyRefs(void)
{
}

//-----------------------------------------------------------------------------
// 1. This is the place where GFFTStack first gets created. However, the
//    frames are padded but not yet transformed into Fourier space.
// 2. Even if there is no gain reference, i.e. pfGain is null, the original
//    frames are stilled padded and put into GFFTStack.
//-----------------------------------------------------------------------------
void CApplyRefs::DoIt
(	float* pfGain,
	float* pfDark,
	int iNthGpu
)
{	if(pfGain == 0L && pfDark == 0L) return;
	nvtxRangePushA("CApplyRefs::DoIt");
	m_iNthGpu = iNthGpu;
	//------------------
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	m_pFrmBuffer = pBufPool->GetBuffer(MD::EBuffer::frm);
	m_pTmpBuffer = pBufPool->GetBuffer(MD::EBuffer::tmp);
	m_pSumBuffer = pBufPool->GetBuffer(MD::EBuffer::sum);
	//-----------------
	m_streams[0] = pBufPool->GetCudaStream(0);
	m_streams[1] = pBufPool->GetCudaStream(1);
	//-----------------
	m_pvMrcFrames[0] = pBufPool->GetPinnedBuf(0);
	m_pvMrcFrames[1] = pBufPool->GetPinnedBuf(1);
	//-----------------
	MD::CMcPackage* pMcPackage = MD::CMcPackage::GetInstance(m_iNthGpu);
	m_pRawStack = pMcPackage->m_pRawStack;
	//-----------------
	mCopyRefs(pfDark, pfGain);
	mCorrectCpuFrames();
	mCorrectGpuFrames();
	//-----------------
	cudaStreamSynchronize(m_streams[0]);
	cudaStreamSynchronize(m_streams[1]);
	//-----------------
	m_pvMrcFrames[0] = 0L;
	m_pvMrcFrames[1] = 0L;
	m_gfGain = 0L;
	m_gfDark = 0L;
	//-----------------
	nvtxRangePop();
}

void CApplyRefs::mCopyRefs(float* pfDark, float* pfGain)
{
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	cufftComplex* gCmpDark = m_pTmpBuffer->GetFrame(0);
	cufftComplex* gCmpGain = m_pTmpBuffer->GetFrame(1);
	//-----------------
	int iFrmSizeX = (m_pFrmBuffer->m_aiCmpSize[0] - 1) * 2;
	int iFrmSizeY = m_pFrmBuffer->m_aiCmpSize[1];
	//--------------------------------------------
	// This is needed when users specify cropping
	// image at the command line.
	//--------------------------------------------
	int iStartX = (m_pRawStack->m_aiStkSize[0] - iFrmSizeX) / 2;
	int iStartY = (m_pRawStack->m_aiStkSize[1] - iFrmSizeY) / 2;
	int iOffset = iStartY * m_pRawStack->m_aiStkSize[0] + iStartX;
	//-----------------
	int aiPadSize[] = {1, m_pFrmBuffer->m_aiCmpSize[1]};
	aiPadSize[0] = m_pFrmBuffer->m_aiCmpSize[0] * 2;
	int iCpySizeX = (m_pFrmBuffer->m_aiCmpSize[0] - 1) * 2;
	//-----------------
	m_aGAppRefsToFrame.SetSizes(m_pRawStack->m_aiStkSize, 
	   aiPadSize, true);
	//-----------------
	int iRawSizeX = m_pRawStack->m_aiStkSize[0];
	if(pfDark != 0L)
	{	m_gfDark = reinterpret_cast<float*>(gCmpDark);
		float* pfSrc = pfDark + iOffset;
		MU::GPartialCopy::DoIt(pfSrc, iRawSizeX,
		   m_gfDark, iCpySizeX, aiPadSize, m_streams[0]);
	}
	if(pfGain != 0L)
	{	m_gfGain = reinterpret_cast<float*>(gCmpGain);
		float* pfSrc = pfGain + iOffset;
		MU::GPartialCopy::DoIt(pfSrc, iRawSizeX, 
		   m_gfGain, iCpySizeX, aiPadSize, m_streams[0]); 
	}
	m_aGAppRefsToFrame.SetRefs(m_gfGain, m_gfDark);
}

void CApplyRefs::mCorrectGpuFrames(void)
{
	int iCount = 0;
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	if(!m_pFrmBuffer->IsGpuFrame(i)) continue;
		int iStream = iCount % 2; 
		m_iFrame = i;
		//----------------
		cufftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(i);
		mApplyRefs(gCmpFrm, iStream);
		iCount += 1;
	}
}

void CApplyRefs::mCorrectCpuFrames(void)
{
	cufftComplex* pCmpFrm = 0L;
	cufftComplex* gCmpBufs[2] = {0L};
	gCmpBufs[0] = m_pTmpBuffer->GetFrame(2);
	gCmpBufs[1] = m_pSumBuffer->GetFrame(0);
	//-----------------
	int iCount = 0;
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	if(m_pFrmBuffer->IsGpuFrame(i)) continue;
		int iStream = iCount % 2;
		m_iFrame = i;
		//----------------
		mApplyRefs(gCmpBufs[iStream], iStream);
		//----------------
		pCmpFrm = m_pFrmBuffer->GetFrame(i);
		cudaMemcpyAsync(pCmpFrm, gCmpBufs[iStream], 
		   m_pFrmBuffer->m_tFmBytes, cudaMemcpyDefault, 
		   m_streams[iStream]);
		iCount += 1;
	}
}

void CApplyRefs::mApplyRefs(cufftComplex* gCmpFrm, int iStream)
{	
	void* pvRawFrm = m_pRawStack->GetFrame(m_iFrame);
	float* gfPadFrm = reinterpret_cast<float*>(gCmpFrm);
	//----------------------------------------------------
	// Synchornize to make sure m_pvMrcFrames[iStream] is
	// done with the previous operation.
	//----------------------------------------------------
	cudaStreamSynchronize(m_streams[iStream]);
	memcpy(m_pvMrcFrames[iStream], pvRawFrm, m_pRawStack->m_tFmBytes);
	//----------------
	int* piCmpSize = ((MD::CStackBuffer*)m_pFrmBuffer)->m_aiCmpSize;
	int aiPadSize[] = {piCmpSize[0] * 2, piCmpSize[1]};
	int iMode = m_pRawStack->m_iMode;
	//-----------------
	m_aGAppRefsToFrame.DoIt(m_pvMrcFrames[iStream], iMode,
	   gfPadFrm, m_streams[iStream]);
}

