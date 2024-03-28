#include "CFindCtfInc.h"
#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

CCorrCtfMain::CCorrCtfMain(void)
{
	m_pCorrImgCtf = new CCorrImgCtf;
}

CCorrCtfMain::~CCorrCtfMain(void)
{
	if(m_pCorrImgCtf != 0L) delete m_pCorrImgCtf;
}

//--------------------------------------------------------------------
// 1. CTF estimation is done on raw tilt series including dark images.
//    However, the correction is done on dark-removed images.
// 2. CCorrImgCtf maps the tilt image to the estimated CTF using
//    tilt angle rather than image index in CTiltSeries.
// 3. The tilt angles must be raw, not ones corrected with tilt
//    angle offset.
//--------------------------------------------------------------------
void CCorrCtfMain::DoIt(int iNthGpu)
{
	if(!CFindCtfMain::bCheckInput()) return;
	//-----------------	
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	m_pCorrImgCtf->Setup(pTiltSeries->m_aiStkSize, m_iNthGpu);
	//-----------------
	for(int i=0; i<MD::CAlnSums::m_iNumSums; i++)
	{	mCorrTiltSeries(i);
	}
}

void CCorrCtfMain::mCorrTiltSeries(int iSeries)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(iSeries);

	if(iSeries == 0) 
        {       MU::CSaveTempMrc saveMrc;
                saveMrc.SetFile("/home/shawn.zheng/szheng/Temp/TestNoCTF", 
                   ".mrc");
                void** ppvImgs = pTiltSeries->GetFrames();
                saveMrc.DoMany(ppvImgs, 2, pTiltSeries->m_aiStkSize);
                printf("Save CTF corrected tilt series done.\n");
        }


	//-----------------
	MAM::CAlignParam* pAlignParam = 
	   MAM::CAlignParam::GetInstance(m_iNthGpu);
	float fTiltAxis = pAlignParam->GetTiltAxis(0);
	//-----------------
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float* pfImage = (float*)pTiltSeries->GetFrame(i);
		float fTilt = pTiltSeries->m_pfTilts[i];
		m_pCorrImgCtf->DoIt(pfImage, fTilt, fTiltAxis);
	}
	
	if(iSeries == 0)
	{	MU::CSaveTempMrc saveMrc;
		saveMrc.SetFile("/home/shawn.zheng/szheng/Temp/TestYesCTF", 
		   ".mrc");
		void** ppvImgs = pTiltSeries->GetFrames();
		saveMrc.DoMany(ppvImgs, 2, pTiltSeries->m_aiStkSize);
		printf("Save CTF corrected tilt series done.\n");
	}
	
}
