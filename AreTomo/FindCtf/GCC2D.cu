#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::FindCtf;
static __constant__ int c_aiCmpSize[2];
static __constant__ float c_afResRange[2];  // pix_size/res
static __constant__ float c_afIceRange[2];  // pix_size/res

//-----------------------------------------------------------------------------
// 1. The zero-frequency component is at (x=0, y=iCmpY/2). The frequency
//    range in y direction is [-CmpY/2, CmpY/2).
// 2. fFreqLow, fFreqHigh are in the range of [0, 0.5f] of unit 1/pixel.
//-----------------------------------------------------------------------------
static __global__ void mGCalc2D
(       float* gfCTF2D,
        float* gfSpectrum,
        float fBFactor,
        float* gfRes
)
{       extern __shared__ float s_afShared[];
        float* s_afSumStd1 = &s_afShared[blockDim.x];
        float* s_afSumStd2 = &s_afSumStd1[blockDim.x];
        //-------------------------------------
        float fSumCC = 0.0f;
        float fSumStd1 = 0.0f;
        float fSumStd2 = 0.0f;
        int iOffset = 0, i = 0;
        //-------------------------------------
	for(int y=blockIdx.x; y<c_aiCmpSize[1]; y+=gridDim.x)
        {       float fY = (y - c_aiCmpSize[1] * 0.5f) / c_aiCmpSize[1];
                iOffset = y * c_aiCmpSize[0];
                //-----------------------------
                for(int x=threadIdx.x; x<c_aiCmpSize[0]; x+=blockDim.x)
                {       float fX = (0.5f * x) / (c_aiCmpSize[0] - 1);
                        float fR = sqrtf(fX * fX + fY * fY);
                        //---------------------
                        bool bIce = (fR >= c_afIceRange[0] &&
                           fR <= c_afIceRange[1]);
                        if(fR < c_afResRange[0] || bIce) continue;
                        //---------------------
			fR = fR; // - c_afResRange[0];
			fR = expf(-fBFactor * fR * fR);
                        i = iOffset + x;
                        float fC = (fabsf(gfCTF2D[i]) - 0.5f) * fR;
                        float fS = gfSpectrum[i];
                        fSumCC += (fC * fS);
                        fSumStd1 += (fC * fC);
                        fSumStd2 += (fS * fS);
                }
        }
        s_afShared[threadIdx.x] = fSumCC;
        s_afSumStd1[threadIdx.x] = fSumStd1;
        s_afSumStd2[threadIdx.x] = fSumStd2;
        __syncthreads();
        //----------------------------------
        iOffset = blockDim.x / 2;
        while(iOffset > 0)
        {       if(threadIdx.x < iOffset)
                {       i = iOffset + threadIdx.x;
                        s_afShared[threadIdx.x] += s_afShared[i];
                        s_afSumStd1[threadIdx.x] += s_afSumStd1[i];
                        s_afSumStd2[threadIdx.x] += s_afSumStd2[i];
                }
                __syncthreads();
                iOffset /= 2;
        }
        //-------------------
        if(threadIdx.x != 0) return;
        i = blockIdx.x * 3;
        gfRes[i] = s_afShared[0];
        gfRes[i+1] = s_afSumStd1[0];
        gfRes[i+2] = s_afSumStd2[0];
}

static __global__ void mGCalc1D(float* gfSum)
{
        extern __shared__ float s_afShared[];
        float* s_afSumStd1 = &s_afShared[blockDim.x];
        float* s_afSumStd2 = &s_afSumStd1[blockDim.x];
        //--------------------------------------------
        int i = threadIdx.x * 3;
        s_afShared[threadIdx.x] = gfSum[i];
        s_afSumStd1[threadIdx.x] = gfSum[i+1];
        s_afSumStd2[threadIdx.x] = gfSum[i+2];
        __syncthreads();
        //------------------------------------
        int iOffset = blockDim.x / 2;
        while(iOffset > 0)
        {       if(threadIdx.x < iOffset)
                {       i = threadIdx.x + iOffset;
                        s_afShared[threadIdx.x] += s_afShared[i];
                        s_afSumStd1[threadIdx.x] += s_afSumStd1[i];
                        s_afSumStd2[threadIdx.x] += s_afSumStd2[i];
                }
                __syncthreads();
                iOffset /= 2;
        }
        //---------------------
        if(threadIdx.x != 0) return;
        float fStd = sqrtf(s_afSumStd1[0] * s_afSumStd2[0]);
        if(fStd == 0) gfSum[0] = 0.0f;
        else gfSum[0] = s_afShared[0] / fStd;
}

GCC2D::GCC2D(void)
{
	m_fBFactor = 1.0f;
	m_gfRes = 0L;
}

GCC2D::~GCC2D(void)
{
	if(m_gfRes != 0L) cudaFree(m_gfRes);
}

void GCC2D::SetResRange
(       float* pfResRange, // ex: [30A, 4A]
        float fPixSize    // angstrom
)
{       float afResRange[2] = {0.0f};
        afResRange[0] = fPixSize / pfResRange[0];
        afResRange[1] = fPixSize / pfResRange[1];
        cudaMemcpyToSymbol(c_afResRange, afResRange, sizeof(float) * 2);
        //---------------------------
        float afIceRange[] = {1.0f, 2.0f};
        cudaMemcpyToSymbol(c_afIceRange, afIceRange, sizeof(float) * 2);
}

void GCC2D::SetSize(int* piCmpSize)
{
        if(m_gfRes != 0L) cudaFree(m_gfRes);
        m_aiCmpSize[0] = piCmpSize[0];
        m_aiCmpSize[1] = piCmpSize[1];
        //----------------------------------
        int iSize = m_aiCmpSize[0] * m_aiCmpSize[1];
        double dSize = sqrtf(iSize);
        if(dSize > 512) m_iBlockDimX = 512;
        else if(dSize > 256) m_iBlockDimX = 256;
        else if(dSize > 128) m_iBlockDimX = 128;
        else m_iBlockDimX = 64;
        //---------------------------------------
        m_iGridDimX = (iSize + m_iBlockDimX - 1)/ m_iBlockDimX;
        if(m_iGridDimX > 512) m_iGridDimX = 512;
        else if(m_iGridDimX > 256) m_iGridDimX = 256;
        else if(m_iGridDimX > 128) m_iGridDimX = 128;
        else m_iGridDimX = 64;
        //-------------------------------------------
        cudaMalloc(&m_gfRes, 3 * m_iGridDimX * sizeof(float));
        cudaMemcpyToSymbol(c_aiCmpSize, m_aiCmpSize, sizeof(int) * 2);
}

float GCC2D::DoIt
(	float* gfCTF, 
	float* gfSpectrum
)
{	dim3 aBlockDim(m_iBlockDimX, 1);
        dim3 aGridDim(m_iGridDimX, 1);
        size_t tSmBytes = sizeof(float) * aBlockDim.x * 3;
        //---------------------------
        mGCalc2D<<<aGridDim, aBlockDim, tSmBytes>>>(
           gfCTF, gfSpectrum,
           m_fBFactor,
           m_gfRes);
        //---------------------------
        aBlockDim.x = aGridDim.x; aBlockDim.y = 1;
        aGridDim.x = 1; aGridDim.y = 1;
        tSmBytes = sizeof(float) * aBlockDim.x * 3;
        mGCalc1D<<<aGridDim, aBlockDim, tSmBytes>>>(m_gfRes);
        //---------------------------
        float fCC = 0.0f;
        cudaMemcpy(&fCC, m_gfRes, sizeof(float), cudaMemcpyDefault);
        return fCC;
}

