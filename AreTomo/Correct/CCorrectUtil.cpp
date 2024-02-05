#include "CCorrectInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::Correct;
namespace MU = McAreTomo::MaUtil;
namespace MAU = McAreTomo::AreTomo::Util;

void CCorrectUtil::CalcAlignedSize
(	int* piRawSize, float fTiltAxis,
	int* piAlnSize
)
{	memcpy(piAlnSize, piRawSize, sizeof(int) * 2);
	double dRot = fabs(sin(fTiltAxis * 3.14 / 180.0));
	if(dRot <= 0.707) return;
	//-----------------------
	piAlnSize[0] = piRawSize[1];
	piAlnSize[1] = piRawSize[0];		
}

void CCorrectUtil::CalcBinnedSize
(	int* piRawSize, float fBinning, bool bFourierCrop,
	int* piBinnedSize
)
{	if(bFourierCrop)
	{	MU::GFourierResize2D::GetBinnedImgSize(piRawSize,
		   fBinning, piBinnedSize);
	}
	else
	{	int iBin = (int)(fBinning + 0.5f);
		bool bPadded = true;
		MAU::GBinImage2D::GetBinSize(piRawSize, !bPadded,
		   iBin, piBinnedSize, !bPadded);
	}
}

