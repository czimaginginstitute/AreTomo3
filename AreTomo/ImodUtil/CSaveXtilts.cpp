#include "CImodUtilInc.h"
#include "../CAreTomoInc.h"
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace McAreTomo::AreTomo::ImodUtil;

CSaveXtilts::CSaveXtilts(void)
{
}

CSaveXtilts::~CSaveXtilts(void)
{
}

void CSaveXtilts::DoIt
(	int iNthGpu,
	const char* pcFileName
)
{	FILE* pFile = fopen(pcFileName, "wt");
	if(pFile == 0L) return;
	//---------------------
	CAtInput* pInput = CAtInput::GetInstance();
	MAM::CDarkFrames* pDarkFrames = 
	   MAM::CDarkFrames::GetInstance(iNthGpu);
	MAM::CAlignParam* pAlnParam =
	   MAM::CAlignParam::GetInstance(iNthGpu);
	int iTilts = pAlnParam->m_iNumFrames;
	if(pInput->m_iOutImod == 2) 
	{	iTilts = pDarkFrames->m_aiRawStkSize[2];
	}
	//----------------------------------------------
	int iLast = iTilts - 1;
	for(int i=0; i<iLast; i++)
	{	fprintf(pFile, "0\n");
	}
	if(iLast > 0) fprintf(pFile, "0");
	fclose(pFile);
}
