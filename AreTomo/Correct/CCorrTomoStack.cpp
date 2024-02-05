#include "CCorrectInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Util/CUtilInc.h"
#include "../Recon/CReconInc.h"
#include "../PatchAlign/CPatchAlignInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <math.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::Correct;

static void sCalcPadSize(int* piSize, int* piPadSize)
{
	piPadSize[0] = (piSize[0] / 2 + 1) * 2;
	piPadSize[1] = piSize[1];
}

static void sCalcCmpSize(int* piSize, int* piCmpSize)
{
	piCmpSize[0] = piSize[0] / 2 + 1;
	piCmpSize[1] = piSize[1];
}

CCorrTomoStack::CCorrTomoStack(void)
{
	m_gfLocalParam = 0L;
	m_pOutSeries = 0L;
	m_pGRWeight = 0L;
	m_iNthGpu = -1;
	m_bForRecon = false;
}

CCorrTomoStack::~CCorrTomoStack(void)
{
	this->Clean();
}

void CCorrTomoStack::Clean(void)
{
	if(m_gfLocalParam != 0L) cudaFree(m_gfLocalParam);
	if(m_pOutSeries != 0L) delete m_pOutSeries;
	if(m_pGRWeight != 0L) delete m_pGRWeight;
	m_gfLocalParam = 0L;
	m_pOutSeries = 0L;
	m_pGRWeight = 0L;
}

void CCorrTomoStack::GetBinning(float* pfBinning)
{
	pfBinning[0] = m_afBinning[0];
	pfBinning[1] = m_afBinning[1];
}

void CCorrTomoStack::Set0(int iNthGpu)
{
	this->Clean();
	m_iNthGpu = iNthGpu;
	//-----------------
	MD::CTsPackage* pPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pPkg->GetSeries(0);
	memcpy(m_aiStkSize, pTiltSeries->m_aiStkSize, sizeof(int) * 3);
}

//-----------------------------------------------------------------------------
// Note: In case of shift only, 0 must be passed in for fTiltAxis
//-----------------------------------------------------------------------------
void CCorrTomoStack::Set1(int iNumPatches, float fTiltAxis)
{
	CCorrectUtil::CalcAlignedSize(m_aiStkSize, fTiltAxis, m_aiAlnSize);
	m_aiAlnSize[2] = m_aiStkSize[2];
	//-----------------
	MD::CBufferPool* pBufPool = MD::CBufferPool::GetInstance(m_iNthGpu);
	MD::CStackBuffer* pTmpBuf = pBufPool->GetBuffer(MD::EBuffer::tmp);
	//-----------------
	m_gfRawProj = (float*)pTmpBuf->GetFrame(0);
	//-----------------
	int* piCorrSize = m_aiAlnSize;
	if(m_aiAlnSize[1] < m_aiStkSize[1]) piCorrSize = m_aiStkSize;
	m_gfCorrProj = (float*)pTmpBuf->GetFrame(1);
	m_gfBinProj = (float*)pTmpBuf->GetFrame(2);
	//-----------------
	bool bPadded = true;
        int aiAlnPadSize[2] = {0};
        sCalcPadSize(m_aiAlnSize, aiAlnPadSize);
        m_aGCorrPatchShift.SetSizes(m_aiStkSize, !bPadded,
           aiAlnPadSize, bPadded, iNumPatches);
	//-----------------
	if(m_gfLocalParam != 0L) cudaFree(m_gfLocalParam);
	m_gfLocalParam = 0L;
	if(iNumPatches <= 0) return;
	//-----------------
	int iBytes = iNumPatches * 5 * sizeof(float);
	cudaMalloc(&m_gfLocalParam, iBytes);
}

void CCorrTomoStack::Set2(float fOutBin, bool bFourierCrop, bool bRandFill)
{
	m_fOutBin = fOutBin;
	m_bFourierCrop = bFourierCrop;
	m_bRandomFill = m_bFourierCrop ? true : bRandFill;
	//-----------------
	CCorrectUtil::CalcBinnedSize(m_aiAlnSize, m_fOutBin,
	   m_bFourierCrop, m_aiBinnedSize);
	m_aiBinnedSize[2] = m_aiStkSize[2];
	//-----------------
	m_afBinning[0] = m_aiAlnSize[0] / (float)m_aiBinnedSize[0];
	m_afBinning[1] = m_aiAlnSize[1] / (float)m_aiBinnedSize[1];
	//-----------------
	if(m_pOutSeries != 0L) delete m_pOutSeries;
	m_pOutSeries = new MD::CTiltSeries;
	m_pOutSeries->Create(m_aiBinnedSize);
	//-----------------
	if(m_bFourierCrop)
	{	m_aFFTCropImg.Setup(m_iNthGpu, m_aiAlnSize, m_fOutBin);
	}
	else
	{	bool bPadded = true;
		int aiAlnPadSize[2], aiBinPadSize[2];
		sCalcPadSize(m_aiAlnSize, aiAlnPadSize);
		sCalcPadSize(m_aiBinnedSize, aiBinPadSize);
		m_aGBinImg2D.SetupSizes(aiAlnPadSize, bPadded,
		   aiBinPadSize, bPadded);
	}
}

void CCorrTomoStack::Set3(bool bShiftOnly, bool bCorrInt, bool bRWeight)
{	
	m_bShiftOnly = bShiftOnly;
	//------------------------
	int aiBinPadSize[2], aiAlnPadSize[2];
	sCalcPadSize(m_aiAlnSize, aiAlnPadSize);
	sCalcPadSize(m_aiBinnedSize, aiBinPadSize);
	if(bCorrInt)
	{	//m_pCorrInt = new CCorrLinearInterp;
		//m_pCorrInt->Setup(m_aiStkSize);
	}
	//-------------------------------------
	if(bRWeight)
	{	m_pGRWeight = new Recon::GRWeight;
		m_pGRWeight->SetSize(aiBinPadSize[0], aiBinPadSize[1]);
	}
}

void CCorrTomoStack::Set4(bool bForRecon)
{
	m_bForRecon = bForRecon;
}

MD::CTiltSeries* CCorrTomoStack::GetCorrectedStack(bool bClean)
{
	MD::CTiltSeries* pRetStack = m_pOutSeries;
	if(bClean) m_pOutSeries = 0L;
	return pRetStack;
}

void CCorrTomoStack::DoIt(int iNthSeries, MAM::CAlignParam* pAlignParam)
{
	m_iSeries = iNthSeries;
	m_pAlignParam = pAlignParam;
	if(m_pAlignParam == 0L)
	{	m_pAlignParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	}
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pSeries = pTsPkg->GetSeries(m_iSeries);
	//-----------------
	m_pOutSeries->SetTilts(pSeries->m_pfTilts);
	m_pOutSeries->SetAcqs(pSeries->m_piAcqIndices);
	//-----------------
	for(int i=0; i<pSeries->m_aiStkSize[2]; i++)
	{	mCorrectProj(i);
	}
}

void CCorrTomoStack::mCorrectProj(int iProj)
{
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
        MD::CTiltSeries* pRawSeries = pTsPkg->GetSeries(m_iSeries);
	//-----------------
	MAM::CLocalAlignParam* pLocalParam = 
	   MAM::CLocalAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	float afShift[2] = {0.0f};
	m_pAlignParam->GetShift(iProj, afShift);
	//-----------------
	float fTiltAxis = m_pAlignParam->GetTiltAxis(iProj);
	if(m_bShiftOnly) fTiltAxis = 0.0f;
	//-----------------
	float* pfProj = (float*)pRawSeries->GetFrame(iProj);
	size_t tBytes = sizeof(float) * pRawSeries->GetPixels();
	cudaMemcpy(m_gfRawProj, pfProj, tBytes, cudaMemcpyDefault);
	//-----------------
	/*
	if(!m_bShiftOnly && m_pCorrInt != 0L)
	{	m_pCorrInt->DoIt(m_gfRawProj, m_gfCorrProj);
	}*/
	//-----------------
	if(m_gfLocalParam != 0L) 
	{	pLocalParam->GetParam(iProj, m_gfLocalParam);
	}
	//-----------------
	m_aGCorrPatchShift.DoIt(m_gfRawProj, afShift, fTiltAxis, 
	   m_gfLocalParam, m_bRandomFill, m_gfCorrProj);
	//-----------------
	if(m_bFourierCrop)
	{	m_aFFTCropImg.DoPad(m_gfCorrProj, m_gfBinProj);
	}
	else m_aGBinImg2D.DoIt(m_gfCorrProj, m_gfBinProj);
	//-----------------
	if(m_pGRWeight != 0L) m_pGRWeight->DoIt(m_gfBinProj);
	//-----------------
        int aiPadSize[] = {0, m_pOutSeries->m_aiStkSize[1]};
	aiPadSize[0] = (m_pOutSeries->m_aiStkSize[0] / 2 + 1) * 2;
	float* pfProjOut = (float*)m_pOutSeries->GetFrame(iProj);
        MU::CPad2D pad2D;
	pad2D.Unpad(m_gfBinProj, aiPadSize, pfProjOut);
}

