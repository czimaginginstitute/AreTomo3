#include "CUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>

namespace MMU = McAreTomo::MotionCor::Util;

size_t MMU::GetUCharBytes(int* piSize)
{
	size_t tBytes = sizeof(char) * piSize[0];
	tBytes *= piSize[1];
        return tBytes;
}

size_t MMU::GetFloatBytes(int* piSize)
{
	size_t tBytes = sizeof(float) * piSize[0];
	tBytes *= piSize[1];
	return tBytes;
}

size_t MMU::GetCmpBytes(int* piSize)
{
	size_t tBytes = sizeof(cufftComplex) * piSize[0];
	tBytes *= piSize[1];
	return tBytes;
}

unsigned char* MMU::GGetUCharBuf(int* piSize, bool bZero)
{
	size_t tBytes = MMU::GetUCharBytes(piSize);
	unsigned char* gucBuf = 0L;
	cudaMalloc(&gucBuf, tBytes);
	if(bZero) cudaMemset(gucBuf, 0, tBytes);
	return gucBuf;
}

unsigned char* MMU::CGetUCharBuf(int* piSize, bool bZero)
{
	int iPixels = piSize[0] * piSize[1];
	unsigned char* pucBuf = new unsigned char[iPixels];
	if(bZero) memset(pucBuf, 0, iPixels * sizeof(char));
	return pucBuf;
}

float* MMU::GGetFloatBuf(int* piSize, bool bZero)
{
	size_t tBytes = MMU::GetFloatBytes(piSize);
	float* gfBuf = 0L;
	cudaMalloc(&gfBuf, tBytes);
	if(bZero) cudaMemset(gfBuf, 0, tBytes);
	return gfBuf;
}

float* MMU::CGetFloatBuf(int* piSize, bool bZero)
{
	int iPixels = piSize[0] * piSize[1];
	float* pfBuf = new float[iPixels];
	if(bZero) memset(pfBuf, 0, iPixels * sizeof(float));
	return pfBuf;
}

void* MMU::GetPinnedBuf(int* piSize, int iPixelBytes, bool bZero)
{
	void* pvBuf = 0L;
	size_t tBytes = (size_t)piSize[0] * piSize[1] * iPixelBytes;
	cudaMallocHost(&pvBuf, tBytes);
	if(bZero) memset(pvBuf, 0, tBytes);
	return pvBuf;
}

void* MMU::GetGpuBuf(int* piSize, int iPixelBytes, bool bZero)
{
	void* gvBuf = 0L;
	size_t tBytes = (size_t)piSize[0] * piSize[1] * iPixelBytes;
	cudaMalloc(&gvBuf, tBytes);
	if(bZero) cudaMemset(gvBuf, 0, tBytes);
	return gvBuf;
}

cufftComplex* MMU::GGetCmpBuf(int* piSize, bool bZero)
{
	size_t tBytes = MMU::GetCmpBytes(piSize);
	cufftComplex* gCmpBuf = 0L;
	cudaMalloc(&gCmpBuf, tBytes);
	if(bZero) cudaMemset(gCmpBuf, 0, tBytes);
	return gCmpBuf;
}

cufftComplex* MMU::CGetCmpBuf(int* piSize, bool bZero)
{
	int iPixels = piSize[0] * piSize[1];
	cufftComplex* pCmpBuf = new cufftComplex[iPixels];
	if(bZero) memset(pCmpBuf, 0, iPixels * sizeof(cufftComplex));
	return pCmpBuf;
}

unsigned char* MMU::GCopyFrame
(	unsigned char* pucSrc, 
	int* piSize, 
	cudaStream_t stream
)
{	unsigned char* gucDst = MMU::GGetUCharBuf(piSize, false);
	MMU::CopyFrame(pucSrc, gucDst, piSize, stream);
	return gucDst;
}

unsigned char* MMU::CCopyFrame
(	unsigned char* pucSrc, 
	int* piSize, 
	cudaStream_t stream
)
{	unsigned char* pucDst = MMU::CGetUCharBuf(piSize, false);
	MMU::CopyFrame(pucSrc, pucDst, piSize, stream);
	return pucDst;
}

void MMU::CopyFrame
(	unsigned char* pucSrc, 
	unsigned char* pucDst, 
	int* piSize, 
	cudaStream_t stream
)
{	size_t tBytes = sizeof(char) * piSize[0] * piSize[1];
	cudaMemcpyAsync(pucDst, pucSrc, tBytes, cudaMemcpyDefault, stream);
}

float* MMU::GCopyFrame(float* pfSrc, int* piSize, cudaStream_t stream)
{
	float* gfDst = MMU::GGetFloatBuf(piSize, false);
	MMU::CopyFrame(pfSrc, gfDst, piSize, stream);
	return gfDst;
}

float* MMU::CCopyFrame(float* pfSrc, int* piSize, cudaStream_t stream)
{
        float* pfDst = MMU::CGetFloatBuf(piSize, false);
	MMU::CopyFrame(pfSrc, pfDst, piSize, stream);
        return pfDst;
}

void MMU::CopyFrame
(	float* pfSrc, float* pfDst, int* piSize, 
	cudaStream_t stream
)
{       size_t tBytes = sizeof(float) * piSize[0] * piSize[1];
        cudaMemcpyAsync(pfDst, pfSrc, tBytes, cudaMemcpyDefault, stream);
}

cufftComplex* MMU::GCopyFrame
(	cufftComplex* pCmpSrc, int* piSize, 
	cudaStream_t stream
)
{	cufftComplex* gCmpDst = MMU::GGetCmpBuf(piSize, false);
	MMU::CopyFrame(pCmpSrc, gCmpDst, piSize, stream);
	return gCmpDst;
}

cufftComplex* MMU::CCopyFrame
(	cufftComplex* pCmpSrc, int* piSize, 
	cudaStream_t stream
)
{	cufftComplex* pCmpDst = MMU::CGetCmpBuf(piSize, false);
	MMU::CopyFrame(pCmpSrc, pCmpDst, piSize, stream);
	return pCmpDst;
}

void MMU::CopyFrame
(	cufftComplex* pCmpSrc,
	cufftComplex* pCmpDst, 
	int* piSize, 
	cudaStream_t stream
)
{	size_t tBytes = sizeof(cufftComplex) * piSize[0] * piSize[1];
	cudaMemcpyAsync(pCmpDst, pCmpSrc, tBytes, cudaMemcpyDefault, stream);
}

size_t MMU::GetGpuMemory(int iGpuId)
{	
	cudaDeviceProp aDeviceProp;
	cudaGetDeviceProperties(&aDeviceProp, iGpuId);
	return aDeviceProp.totalGlobalMem;
}

int MMU::CalcNumGpuFrames
(	int* piFrmSize,
	int iGpuId,
	double dOccupancy
)
{	size_t tGpuMem = MMU::GetGpuMemory(iGpuId);
	size_t tFrmMem = sizeof(float) * piFrmSize[0] * piFrmSize[1];
	int iNumFrames = (int)(tGpuMem * dOccupancy / tFrmMem);
	return iNumFrames;
}

void MMU::PrintGpuMemoryUsage(const char* pcInfo)
{
	size_t tTotal = 0, tFree = 0;
	cudaError_t tErr = cudaMemGetInfo(&tFree, &tTotal);
	//-------------------------------------------------
	tTotal /= (1024 * 1024);
	tFree /= (1024 * 1024);
	double dUsed = 100.0 - tFree * 100.0 / tTotal;
	printf("%s", pcInfo);
	printf("  total GPU memory: %ld MB\n", tTotal);
	printf("  free GPU memory:  %ld MB\n", tFree);
	printf("  used GPU memory: %8.2f%%\n\n", dUsed);
}

float MMU::GetGpuMemoryUsage(void)
{
	size_t tTotal = 0, tFree = 0;
	cudaError_t tErr = cudaMemGetInfo(&tFree, &tTotal);
	double dUsed = 1.0 - tFree * 1.0 / tTotal;
	return (float)dUsed;
}

void MMU::CheckCudaError(const char* pcLocation)
{
	cudaError_t cuErr = cudaGetLastError();
	if(cuErr == cudaSuccess) return;
	//------------------------------
	fprintf(stderr, "%s: %s\n\t\n\n", pcLocation,
		cudaGetErrorString(cuErr));
	cudaDeviceReset();
	assert(0);
}

void MMU::CheckRUsage(const char* pcLocation)
{
	struct rusage resUsage;
	int iRet = getrusage(RUSAGE_SELF, &resUsage);
	double dMemUsageGB = resUsage.ru_maxrss / (1024.0 * 1024);
	printf("%s: total memory allocation %.2f GB\n", 
	   pcLocation, dMemUsageGB);
}
