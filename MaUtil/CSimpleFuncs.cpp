#include "CMaUtilInc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <pwd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>


size_t MU::GetGpuMemory(int iGpuId)
{	
	cudaDeviceProp aDeviceProp;
	cudaGetDeviceProperties(&aDeviceProp, iGpuId);
	return aDeviceProp.totalGlobalMem;
}

void MU::PrintGpuMemoryUsage(const char* pcInfo)
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

float MU::GetGpuMemoryUsage(void)
{
	size_t tTotal = 0, tFree = 0;
	cudaError_t tErr = cudaMemGetInfo(&tFree, &tTotal);
	double dUsed = 1.0 - tFree * 1.0 / tTotal;
	return (float)dUsed;
}

void MU::CheckCudaError(const char* pcLocation)
{
	cudaError_t cuErr = cudaGetLastError();
	if(cuErr == cudaSuccess) return;
	//------------------------------
	fprintf(stderr, "%s: %s\n\t\n\n", pcLocation,
		cudaGetErrorString(cuErr));
	cudaDeviceReset();
	assert(0);
}

void MU::CheckRUsage(const char* pcLocation)
{
	struct rusage resUsage;
	int iRet = getrusage(RUSAGE_SELF, &resUsage);
	double dMemUsageGB = resUsage.ru_maxrss / (1024.0 * 1024);
	printf("%s: total memory allocation %.2f GB\n", 
	   pcLocation, dMemUsageGB);
}

void* MU::GetGpuBuf(size_t tBytes, bool bZero)
{
	void* gvBuf = 0L;
	cudaMalloc(&gvBuf, tBytes);
	if(bZero) cudaMemset(gvBuf, 0, tBytes);
	return gvBuf;
}

const char* MU::GetHomeDir(void)
{
	const char* pcHomeDir = getenv("HOME");
	if(pcHomeDir != 0L) return pcHomeDir;
	//-----------------
	pcHomeDir = getpwuid(getuid())->pw_dir;
	return pcHomeDir;
}

bool MU::GetCurrDir(char* pcRet, int iSize)
{
	if(getcwd(pcRet, iSize) != 0L) return true;
	else return false;	
}

void MU::UseFullPath(char* pcPath)
{
        if(pcPath == 0L || strlen(pcPath) == 0) return;
        if(pcPath[0] == '/') return;
        //-----------------
        char acDir[256] = {'\0'};
        bool bExit = MU::GetCurrDir(acDir, 256);
        if(!bExit) return;
        //-----------------
        if(pcPath[0] == '.')
        {       strcat(acDir, &pcPath[1]);
        }
        else
        {       strcat(acDir, "/");
                strcat(acDir, pcPath);
        }
        strcpy(pcPath, acDir);
}

