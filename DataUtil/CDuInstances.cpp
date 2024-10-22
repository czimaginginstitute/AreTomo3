#include "CDataUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo::DataUtil;

void CDuInstances::CreateInstances(int iNumGpus)
{
	CBufferPool::CreateInstances(iNumGpus);
	CCtfResults::CreateInstances(iNumGpus);
	CMcPackage::CreateInstances(iNumGpus);
	CReadMdoc::CreateInstances(iNumGpus);
	CAsyncSaveVol::CreateInstances(iNumGpus);
	CTsPackage::CreateInstances(iNumGpus);
	CLogFiles::CreateInstances(iNumGpus);
}

void CDuInstances::DeleteInstances(void)
{
	CBufferPool::DeleteInstances();
	CCtfResults::DeleteInstances();
	CMcPackage::DeleteInstances();
	CReadMdoc::DeleteInstances();
	CTsPackage::DeleteInstances();
	CStackFolder::DeleteInstance();
	CLogFiles::DeleteInstances();
	CAsyncSaveVol::DeleteInstances();
}
