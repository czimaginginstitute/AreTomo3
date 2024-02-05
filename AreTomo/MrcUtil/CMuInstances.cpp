#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <string.h>

using namespace McAreTomo::AreTomo::MrcUtil;

void CMuInstances::CreateInstances(int iNumGpus)
{
	CAlignParam::CreateInstances(iNumGpus);
	CLocalAlignParam::CreateInstances(iNumGpus);
	CPatchShifts::CreateInstances(iNumGpus);
	CDarkFrames::CreateInstances(iNumGpus);
}

void CMuInstances::DeleteInstances(void)
{
	CAlignParam::DeleteInstances();
	CLocalAlignParam::DeleteInstances();
	CPatchShifts::DeleteInstances();
	CDarkFrames::DeleteInstances();
}

