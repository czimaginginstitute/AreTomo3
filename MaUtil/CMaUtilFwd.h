#pragma once
#include <cufft.h>

namespace McAreTomo::MaUtil
{
	size_t GetGpuMemory(int iGpuId);
	void PrintGpuMemoryUsage(const char* pcInfo);
	float GetGpuMemoryUsage(void);
	void CheckCudaError(const char* pcLocation);
	void CheckRUsage(const char* pcLocaltion);
	void* GetGpuBuf(size_t tBytes, bool bZero);
	//-----------------
	class CParseArgs;
	class CCufft2D;
	class CPad2D;
	class CPeak2D;
	class GAddFrames;
	class GCalcMoment2D;
	class GCorrLinearInterp;
	class GPad2D;
	class GRoundEdge;
	class GNormalize2D;
	class GThreshold2D;
	class GFourierResize2D;
	class GPositivity2D;
	class GFFTUtil2D;
	class GFindMinMax2D;
	class GPartialCopy;
	class GPhaseShift2D;
	class GGriddingCorrect;
}

namespace MU = McAreTomo::MaUtil;
