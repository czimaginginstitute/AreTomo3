#pragma once
#include "../MaUtil/CMaUtilFwd.h"
#include <Util/Util_Thread.h>
#include <Mrcfile/CMrcFileInc.h>
#include <queue>
#include <cuda.h>

namespace McAreTomo::DataUtil
{
	class CMrcStack;
	class CTiltSeries;
	class CAlnSums; 
	class CGpuBuffer;
	class CStackBuffer;
	class CBufferPool;
	class CCtfResults;
	class CMcPackage;
	class CReadMdoc;
	class CTsPackage;
	class CStackFolder;
}

namespace MD = McAreTomo::DataUtil;
