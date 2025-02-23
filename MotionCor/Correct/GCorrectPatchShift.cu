#include "CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <Util/Util_Time.h>
#include <nvToolsExt.h>

using namespace McAreTomo::MotionCor::Correct;
using namespace McAreTomo::MotionCor;

// padded sizeX, sizeY, number of frames, number of patches
static __device__ __constant__ int giSizes[4];

static __global__ void mGCorrect3D
(	float fBFactor,
	float* gfPadFrmIn,
	float* gfPatCenters,
	float* gfPatShifts,
	bool* gbBadShifts,
	int iUpSample,
	float* gfPadFrmOut
)
{	int x = 0;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(y >= giSizes[1]) return;
	int iOut = y * giSizes[0] + blockIdx.x;
	//-------------------------------------
	float afXYZ[2] = {0.0f};
	float fSx = 0.0f, fSy = 0.0f, fW = 0.0f;
	for(int p=0; p<giSizes[3]; p++)
	{	if(gbBadShifts != 0L && gbBadShifts[p]) continue;
		//-----------------------------------------------
		int k =  p * 2;
		afXYZ[0] = (blockIdx.x - gfPatCenters[k]) / gridDim.x;
		afXYZ[1] = (y - gfPatCenters[k+1]) / giSizes[1];
		afXYZ[0] = sqrtf(afXYZ[0] * afXYZ[0] + afXYZ[1] * afXYZ[1]);
		if(afXYZ[0] > 0.5f) continue;
		//----------------------------
		afXYZ[0] = expf(-fBFactor * afXYZ[0]);
		fW += afXYZ[0];
		//-------------
		fSx += gfPatShifts[p * 2] * afXYZ[0];
		fSy += gfPatShifts[p * 2 + 1] * afXYZ[0];
	}
	if(fW > 0)
	{	fSx = fSx / fW;
		fSy = fSy / fW;
	}
	//---------------------------
	x = (int)((blockIdx.x - fSx) * iUpSample);
	y = (int)((y - fSy) * iUpSample);
	int iSizeX = gridDim.x * iUpSample;
	int iSizeY = giSizes[1] * iUpSample;
	//---------------------------
	if(x < 0 || y < 0 || x >= iSizeX || y >= iSizeY)
	{	x = (x < 0) ? -x : x;
		y = (y < 0) ? -y : y;
		x = (811 * x) % iSizeX;
		y = (811 * y) % iSizeY;
	}
	//---------------------------
	iSizeX = (iSizeX / 2 + 1) * 2;
	gfPadFrmOut[iOut] = gfPadFrmIn[y * iSizeX + x];
}

GCorrectPatchShift::GCorrectPatchShift(void)
{
	m_aBlockDim.x = 1;
	m_aBlockDim.y = 64;
	m_iUpsample = 1;
	//---------------------------
	m_gCmpUpsampled = 0L;
}

GCorrectPatchShift::~GCorrectPatchShift(void)
{
	mClean();
}

void GCorrectPatchShift::mClean(void)
{
	if(m_gCmpUpsampled != 0L) 
	{	cudaFree(m_gCmpUpsampled);
		m_gCmpUpsampled = 0L;
	}
}

void GCorrectPatchShift::DoIt
(	MMD::CPatchShifts* pPatchShifts,
	int iNthGpu
)
{	Util_Time utilTime; utilTime.Measure();
	//---------------------------
	m_pPatchShifts = pPatchShifts;
	CCorrectFullShift::Setup(m_pPatchShifts->m_pFullShift, iNthGpu);
	//---------------------------
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
        for(int i=0; i<m_pSumBuffer->m_iNumFrames; i++)
        {       cufftComplex* gCmpSum = m_pSumBuffer->GetFrame(i);
                cudaMemsetAsync(gCmpSum, 0, tBytes, m_streams[0]);
        }
	mSetupUpSample();
	//---------------------------
	mDoIt();
	mCorrectMag();
	mUnpadSums();
	cudaStreamSynchronize(m_streams[0]);
	cudaStreamSynchronize(m_streams[1]);
	//---------------------------
	mClean();
	float fSecs = utilTime.GetElapsedSeconds();
	printf("Correction of local motion: %f sec\n\n", fSecs); 
}

void GCorrectPatchShift::mDoIt(void)
{
	m_pForwardFFT->CreateForwardPlan(m_aiInPadSize, true);
	//-----------------
	int aiSizes[] = 
	{ m_aiInPadSize[0], m_aiInPadSize[1], 
	  m_pPatchShifts->m_aiFullSize[2],
	  m_pPatchShifts->m_iNumPatches
	};
	cudaMemcpyToSymbol(giSizes, aiSizes, sizeof(aiSizes));
	//-----------------
	int iNumPoints = m_pPatchShifts->m_iNumPatches *
	   m_pPatchShifts->m_aiFullSize[2];
	int iBytes = iNumPoints * (2 * sizeof(float) + sizeof(bool));
	cudaMalloc(&m_gfPatShifts, iBytes);
	m_gbBadShifts = (bool*)(m_gfPatShifts + iNumPoints * 2);
	//-----------------
	m_pPatchShifts->CopyShiftsToGpu(m_gfPatShifts);
	m_pPatchShifts->CopyFlagsToGpu(m_gbBadShifts);
	//-----------------
	iBytes = m_pPatchShifts->m_iNumPatches * 2 * sizeof(float);
	cudaMalloc(&m_gfPatCenters, iBytes);
	//-----------------
	m_pPatchShifts->CopyCentersToGpu(m_gfPatCenters);
	//-----------------
	m_aGridDim.x = (m_aiInCmpSize[0] - 1) * 2;
	m_aGridDim.y = (m_aiInCmpSize[1] + m_aBlockDim.y - 1) / m_aBlockDim.y;
	//-----------------
	mCorrectCpuFrames();
	mCorrectGpuFrames();
	cudaStreamSynchronize(m_streams[0]);
	cudaStreamSynchronize(m_streams[1]);
	//------------------
	if(m_gfPatShifts != 0L) cudaFree(m_gfPatShifts);
	if(m_gfPatCenters != 0L) cudaFree(m_gfPatCenters);
}

void GCorrectPatchShift::mCorrectCpuFrames(void)
{
	int iCount = 0;
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
	cufftComplex* pCmpFrm = 0L;
	cufftComplex* gCmpBuf = m_pTmpBuffer->GetFrame(0); 
	cufftComplex* gCmpAln = m_pTmpBuffer->GetFrame(1);
	//-----------------
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	if(m_pFrmBuffer->IsGpuFrame(i)) continue;
		pCmpFrm = m_pFrmBuffer->GetFrame(i);
		//----------------
		m_iFrame = i;
		int iStream = iCount % 2;
		//----------------
		if(iStream == 1) cudaStreamSynchronize(m_streams[0]);
		cudaMemcpyAsync(gCmpBuf, pCmpFrm, tBytes, 
		   cudaMemcpyDefault, m_streams[iStream]);
		if(iStream == 1) cudaStreamSynchronize(m_streams[1]);
		//-----------------
		mAlignFrame(gCmpBuf);
		mGenSums(gCmpAln);
		iCount += 1;	
	}
}

void GCorrectPatchShift::mCorrectGpuFrames(void)
{
	cufftComplex* gCmpAln = m_pTmpBuffer->GetFrame(1);
	for(int i=0; i<m_pFrmBuffer->m_iNumFrames; i++)
	{	if(!m_pFrmBuffer->IsGpuFrame(i)) continue;
		//----------------
		m_iFrame = i;
		cufftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(i);
		mAlignFrame(gCmpFrm);
		mGenSums(gCmpAln);	
	}
}
/*
void GCorrectPatchShift::mAlignFrame(cufftComplex* gCmpFrm)
{
	float fBFactor = 100.0f;
	float* gfPadFrm = reinterpret_cast<float*>(gCmpFrm);
	//------------------
	MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	float* gfPadAln = (float*)m_pTmpBuffer->GetFrame(1);
	//------------------
	int iOffset = m_iFrame * m_pPatchShifts->m_iNumPatches;
	float* gfPatShifts = m_gfPatShifts + iOffset * 2;
	bool* gbBadShifts = m_gbBadShifts + iOffset;
	//------------------
	mGCorrect3D<<<m_aGridDim, m_aBlockDim, 0, m_streams[0]>>>(fBFactor,
	   gfPadFrm, m_gfPatCenters, gfPatShifts, 
	   gbBadShifts, m_iUpsample, gfPadAln);
	//------------------
	bool bNorm = true;
	m_pForwardFFT->Forward(gfPadAln, bNorm, m_streams[0]);
}
*/

void GCorrectPatchShift::mAlignFrame(cufftComplex* gCmpFrm)
{
	float fBFactor = 100.0f;
	mUpSample(gCmpFrm);
	float* gfUpsampled = reinterpret_cast<float*>(m_gCmpUpsampled);
        //--------------------
        MD::CBufferPool* pBufferPool = MD::CBufferPool::GetInstance(m_iNthGpu);
        float* gfPadAln = (float*)m_pTmpBuffer->GetFrame(1);
        //------------------
        int iOffset = m_iFrame * m_pPatchShifts->m_iNumPatches;
        float* gfPatShifts = m_gfPatShifts + iOffset * 2;
        bool* gbBadShifts = m_gbBadShifts + iOffset;
        //------------------
        mGCorrect3D<<<m_aGridDim, m_aBlockDim, 0, m_streams[0]>>>(fBFactor,
           gfUpsampled, m_gfPatCenters, gfPatShifts,
           gbBadShifts, m_iUpsample, gfPadAln);
        //------------------
        bool bNorm = true;
        m_pForwardFFT->Forward(gfPadAln, bNorm, m_streams[0]);
}

//--------------------------------------------------------------------
// Upsample the input frame if the motioncor binning is 1 and the
// frame size does not exceed 8K x 8k.
//--------------------------------------------------------------------
void GCorrectPatchShift::mSetupUpSample(void)
{
	m_iUpsample = 2;
	CMcInput* pMcInput = CMcInput::GetInstance();
	if(pMcInput->m_fMcBin >= 1.5f) m_iUpsample = 1;
	//---------------------------
	int iImgSize = (m_aiInCmpSize[0] - 1) * 2;
	if(iImgSize > m_aiInCmpSize[1]) iImgSize = m_aiInCmpSize[1];
	if(iImgSize >= 8192) m_iUpsample = 1;
	//---------------------------
	m_iUpsample = 1;
	m_aiUpCmpSize[0] = (m_aiInCmpSize[0] - 1) * m_iUpsample + 1;
	m_aiUpCmpSize[1] = m_aiInCmpSize[1] * m_iUpsample;
	//---------------------------
	size_t tBytes = sizeof(cufftComplex) * m_aiUpCmpSize[0] 
	   * m_aiUpCmpSize[1];
	cudaMalloc(&m_gCmpUpsampled, tBytes);
	//---------------------------
	m_pInverseFFT->CreateInversePlan(m_aiUpCmpSize, true);
}

void GCorrectPatchShift::mUpSample(cufftComplex* gCmpFrm)
{
	MU::GFtResize2D ftResize2D;
	ftResize2D.UpSample(gCmpFrm, m_aiInCmpSize,
	   m_gCmpUpsampled, m_aiUpCmpSize, m_streams[0]);
	//---------------------------
	m_pInverseFFT->Inverse(m_gCmpUpsampled, m_streams[0]);
}


