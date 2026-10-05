#include "CFindCtfInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZEX 512

using namespace McAreTomo::AreTomo::FindCtf;
static __constant__ float c_afResRange[2];  // pix_size/res
static __constant__ float c_afIceRange[2];  // pix_size/res

//-----------------------------------------------------------------------------
// 1. The zero-frequency component is at (x=0, y=iCmpY/2). The frequency
//    range in y direction is [-CmpY/2, CmpY/2).
// 2. fFreqLow, fFreqHigh are in the range of [0, 0.5f] of unit 1/pixel.
//-----------------------------------------------------------------------------
static __global__ void mGCalculate
(       float* gfCTF,
        float* gfSpectrum,
        int iSize,
        float fBFactor,
        float* gfRes
)
{       extern __shared__ char s_cArray[];
        float* s_gfCC = (float*)&s_cArray[0];
        float* s_gfStd1 = &s_gfCC[blockDim.x];
        float* s_gfStd2 = &s_gfStd1[blockDim.x];
        s_gfCC[threadIdx.x] = 0.0f;
        s_gfStd1[threadIdx.x] = 0.0f;
        s_gfStd2[threadIdx.x] = 0.0f;
        __syncthreads();
        //---------------------------
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if(x >= iSize) return;
        //---------------------------
        float fX = x / ((iSize - 1.0f) * 2.0f);
        bool bNotIce = (fX < c_afIceRange[0] || fX > c_afIceRange[1]);
        //---------------------------
        if(fX >= c_afResRange[0] && bNotIce)
        {       float fW = fX; // - c_afResRange[0];
		fW = expf(-fBFactor * fW * fW);
                float fSpec = gfSpectrum[x];
                float fCTF = (fabsf(gfCTF[x]) - 0.5f) * fW;
                //-------------------
                s_gfCC[threadIdx.x] = fCTF * fSpec;
                s_gfStd1[threadIdx.x] = fCTF * fCTF;
                s_gfStd2[threadIdx.x] = fSpec * fSpec;
        }
        __syncthreads();
        //--------------
        x = blockDim.x / 2;
	while(x > 0)
        {       if(threadIdx.x < x)
                {       int j = x + threadIdx.x;
                        s_gfCC[threadIdx.x] += s_gfCC[j];
                        s_gfStd1[threadIdx.x] += s_gfStd1[j];
                        s_gfStd2[threadIdx.x] += s_gfStd2[j];
                }
                __syncthreads();
                x /= 2;
        }
        //-------------
        if(threadIdx.x == 0)
        {       x = 3 * blockIdx.x;
                gfRes[x] = s_gfCC[0];
                gfRes[x+1] = s_gfStd1[0];
                gfRes[x+2] = s_gfStd2[0];
        }
}

GCC1D::GCC1D(void)
{
	m_fBFactor = 1.0f;
	m_gfRes = 0L;
}

GCC1D::~GCC1D(void)
{
	if(m_gfRes != 0L) cudaFree(m_gfRes);
}

void GCC1D::SetResRange
(	float* pfResRange, // [low, high] (A)
	float fPixSize
)
{	float afResRange[2] = {0.0f};
        afResRange[0] = fPixSize / pfResRange[0];
        afResRange[1] = fPixSize / pfResRange[1];
        cudaMemcpyToSymbol(c_afResRange, afResRange, sizeof(float) * 2);
        //---------------------------
        float afIceRange[] = {1.0f, 2.0f};
        cudaMemcpyToSymbol(c_afIceRange, afIceRange, sizeof(float) * 2);	
}

void GCC1D::SetSize(int iSize)
{
	if(m_gfRes != 0L) cudaFree(m_gfRes);
	//----------------------------------
	m_iSize = iSize;
	cudaMalloc(&m_gfRes, sizeof(float) * m_iSize);
} 

float GCC1D::DoIt(float* gfCTF, float* gfSpectrum)
{
	dim3 aBlockDim(256, 1);
        dim3 aGridDim(1, 1);
        aGridDim.x = (m_iSize + aBlockDim.x - 1) / aBlockDim.x;
        //---------------------------
        size_t tBytes = sizeof(float) * aGridDim.x * 3;
        cudaMemset(m_gfRes, 0, tBytes);
        //---------------------------
        tBytes = sizeof(float) * aBlockDim.x * 3;
        mGCalculate<<<aGridDim, aBlockDim, tBytes>>>(
           gfCTF, gfSpectrum, m_iSize,
           m_fBFactor,
           m_gfRes);
        //---------------------------
        float* pfRes = new float[aGridDim.x * 3];
        tBytes = sizeof(float) * aGridDim.x * 3;
        cudaMemcpy(pfRes, m_gfRes, tBytes, cudaMemcpyDefault);
        //---------------------------
        double dCC = 0.0, dStd1 = 0.0, dStd2 = 0.0;
        for(int i=0; i<aGridDim.x; i++)
        {       int j = 3 * i;
                dCC += pfRes[j];
                dStd1 += pfRes[j+1];
                dStd2 += pfRes[j+2];
        }
        if(pfRes != 0L) delete[] pfRes;
        //-----------------------------
        if(dStd1 > 0 && dStd2 > 0) dCC /= sqrt(dStd1 * dStd2);
        else dCC = 0.0;
        return (float)dCC;
}
