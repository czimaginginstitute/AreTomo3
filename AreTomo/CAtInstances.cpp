#include "CAreTomoInc.h"
#include "CommonLine/CCommonLineInc.h"
#include "ImodUtil/CImodUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"
#include "ProjAlign/CProjAlignInc.h"
#include "PatchAlign/CPatchAlignInc.h"

using namespace McAreTomo::AreTomo;

void CAtInstances::CreateInstances(int iNumGpus)
{
	CTsMetrics::CreateInstances();
	CommonLine::CCommonLineParam::CreateInstances(iNumGpus);
	ImodUtil::CImodUtil::CreateInstances(iNumGpus);
	MrcUtil::CMuInstances::CreateInstances(iNumGpus);
	PatchAlign::CPatchAlignMain::CreateInstances(iNumGpus);
	ProjAlign::CParam::CreateInstances(iNumGpus);
}

void CAtInstances::DeleteInstances(void)
{
	CommonLine::CCommonLineParam::DeleteInstances();
	ImodUtil::CImodUtil::DeleteInstances();
	MrcUtil::CMuInstances::DeleteInstances();
	PatchAlign::CPatchAlignMain::DeleteInstances();
	ProjAlign::CParam::DeleteInstances();
	CTsMetrics::DeleteInstances();
}

