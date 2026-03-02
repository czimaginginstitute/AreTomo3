#include "CMotionCorInc.h"
#include "Align/CAlignInc.h"
#include "BadPixel/CBadPixelInc.h"
#include "Correct/CCorrectInc.h"
#include "Util/CUtilInc.h"
#include "DataUtil/CDataUtilInc.h"
#include "TiffUtil/CTiffUtilInc.h"
#include "EerUtil/CEerUtilInc.h"
#include <stdio.h>

using namespace McAreTomo::MotionCor;

void CMcInstances::CreateInstances(int iNumGpus)
{

	MMD::CFmIntParam::CreateInstances(iNumGpus);
	MMD::CFmGroupParam::CreateInstances(iNumGpus);
	//-----------------
	MMB::CDetectMain::CreateInstances(iNumGpus);
	MMB::CCorrectMain::CreateInstances(iNumGpus);
	//-----------------
	MMA::CPatchCenters::CreateInstances(iNumGpus);
	MMA::CDetectFeatures::CreateInstances(iNumGpus);
	MMA::CSaveAlign::CreateInstances(iNumGpus);
}

void CMcInstances::DeleteInstances(void)
{
	MMD::CFmGroupParam::DeleteInstances();
	MMD::CFmIntParam::DeleteInstances();
	//-----------------
	MMB::CDetectMain::DeleteInstances();
	MMB::CCorrectMain::DeleteInstances();
	//-----------------
	MMA::CPatchCenters::DeleteInstances();
	MMA::CSaveAlign::DeleteInstances();
	MMA::CDetectFeatures::DeleteInstances();
}

