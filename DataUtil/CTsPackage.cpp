#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo::DataUtil;

CTsPackage* CTsPackage::m_pInstances = 0L;
int CTsPackage::m_iNumGpus = 0;

void CTsPackage::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	//-----------------
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CTsPackage[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CTsPackage::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CTsPackage* CTsPackage::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CTsPackage::CTsPackage(void)
{
	m_pcMdocFile = 0L;
	m_ppTsStacks = new CTiltSeries*[CAlnSums::m_iNumSums];
	m_ppVolStacks = new CTiltSeries*[CAlnSums::m_iNumSums];
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	m_ppTsStacks[i] = new CTiltSeries;
		m_ppVolStacks[i] = 0L;
	}
}

CTsPackage::~CTsPackage(void)
{
	if(m_pcMdocFile != 0L) delete[] m_pcMdocFile;
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	if(m_ppTsStacks[i] != 0L) delete m_ppTsStacks[i];
		if(m_ppVolStacks[i] != 0L) delete m_ppVolStacks[i];
	}
	delete[] m_ppTsStacks;
	delete[] m_ppVolStacks;
}

void CTsPackage::SetMdoc(char* pcMdocFile)
{
	if(m_pcMdocFile != 0L) delete[] m_pcMdocFile;
	m_pcMdocFile = 0L;
	memset(m_acMrcMain, 0, sizeof(m_acMrcMain));
	//-----------------
	if(pcMdocFile == 0L || strlen(pcMdocFile) == 0) return;
	m_pcMdocFile = new char[256];
	strcpy(m_pcMdocFile, pcMdocFile);
	//-----------------
	MU::CFileName fileName(m_pcMdocFile);
	fileName.GetName(m_acMrcMain);
}

void CTsPackage::CreateTiltSeries(void)
{
	CReadMdoc* pReadMdoc = CReadMdoc::GetInstance(m_iNthGpu);
	CMcPackage* pMcPackage = CMcPackage::GetInstance(m_iNthGpu);
	CMrcStack* pAlnSums = pMcPackage->m_pAlnSums;
	//-----------------
	mCreateTiltSeries(pAlnSums->m_aiStkSize, 
	   pReadMdoc->m_iNumTilts, pAlnSums->m_fPixSize);
}

bool CTsPackage::LoadTiltSeries(void)
{
	char acMrcFile[256] = {'0'};
	mGenFullPath(".mrc", acMrcFile);
        //-----------------
        Mrc::CLoadMrc loadMrc;
        bool bLoaded = loadMrc.OpenFile(acMrcFile);
        if(!bLoaded) return false;
        //-----------------
        int iMode = loadMrc.m_pLoadMain->GetMode();
        if(iMode != Mrc::eMrcFloat) return false;
        //-----------------
        int aiStkSize[3] = {0};
        loadMrc.m_pLoadMain->GetSize(aiStkSize, 3);
        if(aiStkSize[0] <= 0 || aiStkSize[1] <= 0
           || aiStkSize[2] <= 0) return false;
	//-----------------
	float fPixSize = loadMrc.GetPixelSize();
	if(fPixSize <= 0) fPixSize = 1.0f;
	//--------------------------------------------------
	// 1) Full tilt series must exist for subsequent
	// processing. 2) ODD and EVN tilt series can be
	// missing. If so, no ODD and EVN tomograms will
	// be reconstructed.
	//--------------------------------------------------
	loadMrc.CloseFile();
	mCreateTiltSeries(aiStkSize, aiStkSize[2], fPixSize);
	//-----------------
        bLoaded = mLoadMrc(".mrc", m_ppTsStacks[0]);
	if(!bLoaded) return false;
	//-----------------
	mLoadMrc("_EVN.mrc", m_ppTsStacks[1]);
        mLoadMrc("_ODD.mrc", m_ppTsStacks[2]);
	//-----------------
	bLoaded = mLoadTiltFile();
	if(!bLoaded) return false;
	return true;
}

void CTsPackage::mCreateTiltSeries
(	int* piImgSize, 
	int iNumTilts,
	float fPixSize
)
{	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->Create(piImgSize, iNumTilts);
		pSeries->m_fPixSize = fPixSize;
	}
}

void CTsPackage::SetTiltAngle(int iTilt, float fTiltAngle)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->m_pfTilts[iTilt] = fTiltAngle;
	}
}

void CTsPackage::SetAcqIdx(int iTilt, int iAcqIdx)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->m_piAcqIndices[iTilt] = iAcqIdx;
	}
}

void CTsPackage::SetSecIdx(int iTilt, int iSecIdx)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->m_piSecIndices[iTilt] = iSecIdx;
	}
}

void CTsPackage::SetSums(int iTilt, CAlnSums* pAlnSums)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	void* pvImg = pAlnSums->GetFrame(i);
		CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->SetImage(iTilt, pvImg);
	}
}

void CTsPackage::SetImgDose(float fImgDose)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		if(i == 0) pSeries->m_fImgDose = fImgDose;
		else pSeries->m_fImgDose = fImgDose * 0.5f;
	}
}

CTiltSeries* CTsPackage::GetSeries(int iSeries)
{
	return m_ppTsStacks[iSeries];
}

void CTsPackage::SortTiltSeries(int iOrder)
{
	for(int i=0; i<3; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		if(iOrder == 0) pSeries->SortByTilt();
		else pSeries->SortByAcq();
	}
}

void CTsPackage::ResetSectionIndices(void)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		pSeries->ResetSecIndices();
        }
}

void CTsPackage::SaveVol(CTiltSeries* pVol, int iVol)
{
	char acExt[32] = {'\0'};
	if(iVol == 0) strcpy(acExt, "_Vol.mrc");
	else if(iVol == 1) strcpy(acExt, "_EVN_Vol.mrc");
	else if(iVol == 2) strcpy(acExt, "_ODD_Vol.mrc");
	//-----------------
	mSaveMrc(acExt, pVol);	
}

void CTsPackage::SaveTiltSeries(void)
{
	mSaveTiltFile(m_ppTsStacks[0]);
	mSaveMrc(".mrc", m_ppTsStacks[0]);
	mSaveMrc("_EVN.mrc", m_ppTsStacks[1]);
	mSaveMrc("_ODD.mrc", m_ppTsStacks[2]);
}

void CTsPackage::mSaveMrc
(	const char* pcExt, 
	CTiltSeries* pTiltSeries
)
{	char acMrcFile[256] = {'0'};
	mGenFullPath(pcExt, acMrcFile);
	//-----------------
	Mrc::CSaveMrc saveMrc;
	saveMrc.OpenFile(acMrcFile);
	saveMrc.SetMode(Mrc::eMrcFloat);
	saveMrc.SetExtHeader(0, 32, 0);
	saveMrc.SetImgSize(pTiltSeries->m_aiStkSize,
	   pTiltSeries->m_aiStkSize[2], 1,
	   pTiltSeries->m_fPixSize);
	saveMrc.m_pSaveMain->DoIt();
	//-----------------
	float** ppfImages = pTiltSeries->GetImages();
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float fTilt = pTiltSeries->m_pfTilts[i];
		saveMrc.m_pSaveExt->SetTilt(i, &fTilt, 1);
		saveMrc.m_pSaveExt->DoIt();
		saveMrc.m_pSaveImg->DoIt(i, ppfImages[i]);
	}
	saveMrc.CloseFile();
}

void CTsPackage::mSaveTiltFile(CTiltSeries* pTiltSeries)
{	
	char acTiltFile[256] = {'0'};
	mGenFullPath("_TLT.txt", acTiltFile);
	//-----------------
	FILE* pFile = fopen(acTiltFile, "wt");
	if(pFile == 0L)
	{	printf("GPU %d warning: Unable to save tilt angles\n\n",
		   m_iNthGpu, acTiltFile);
		return;
	}
	//-----------------------------------------------
	// 1) Check if iAcqIdx is 0-based. If yes,
	// convert it to 1-based to be consistent
	// with Relion 4.
	//-----------------------------------------------
	int iMinAcq = pTiltSeries->m_piAcqIndices[0];
	for(int i=1; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	int iAcqIdx = pTiltSeries->m_piAcqIndices[i];
		if(iMinAcq < iAcqIdx) continue;
		else iMinAcq = iAcqIdx;
	}
	int iAdd = (iMinAcq == 0) ? 1 : 0;
	//-----------------
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	float fTilt = pTiltSeries->m_pfTilts[i];
		int iAcqIdx = pTiltSeries->m_piAcqIndices[i] + iAdd;
		fprintf(pFile, "%8.2f  %4d\n", fTilt, iAcqIdx);
	}
	fclose(pFile);
}

bool CTsPackage::mLoadMrc
(	const char* pcExt,
	CTiltSeries* pTiltSeries
)
{	char acMrcFile[256] = {'0'};
	mGenFullPath(pcExt, acMrcFile);
	//-----------------
	Mrc::CLoadMrc loadMrc;
	bool bLoaded = loadMrc.OpenFile(acMrcFile);
	if(!bLoaded) return false;
	//-----------------
	int iMode = loadMrc.m_pLoadMain->GetMode();
	if(iMode != Mrc::eMrcFloat) return false;
	//-----------------
	int aiStkSize[3] = {0};
	loadMrc.m_pLoadMain->GetSize(aiStkSize, 3);
	if(aiStkSize[2] != pTiltSeries->m_aiStkSize[2]) return false;
	if(aiStkSize[1] != pTiltSeries->m_aiStkSize[1]) return false;
	if(aiStkSize[0] != pTiltSeries->m_aiStkSize[0]) return false;
	//-----------------
	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
	{	void* pvFrm = pTiltSeries->GetFrame(i);
		loadMrc.m_pLoadImg->DoIt(i, pvFrm);
	}
	return true;
}

bool CTsPackage::mLoadTiltFile(void)
{
	char acTiltFile[256] = {'0'};
	mGenFullPath("_TLT.txt", acTiltFile);
	FILE* pFile = fopen(acTiltFile, "rt");
        if(pFile == 0L)
        {       printf("GPU %d warning: Unable to load tilt angles from\n"
		   "   file: %s\n", m_iNthGpu, acTiltFile);
		//----------------
		mGenFullPath(".rawtlt", acTiltFile);
		printf("GPU %d: try loading tilt angles from\n"
		   "   file: %s\n\n", m_iNthGpu, acTiltFile);
		pFile = fopen(acTiltFile, "rt");
		//----------------
		if(pFile == 0L)
		{	fprintf(stderr, "GPU %d error: unable to load "
			   "tilt angles. Tilt angle files not "
			   "found.\n\n", m_iNthGpu);
			return false;
		}
        }
	//-----------------
	float fTilt = 0.0f;
	int iAcqIdx = 0, iCount = 0;
	//-----------------
	char acBuf[256] = {'\0'};
	while(!feof(pFile))
	{	memset(acBuf, 0, sizeof(acBuf));
		char* pcRet = fgets(acBuf, 256, pFile);
		if(pcRet == 0L) continue;
		//----------------
		int iItems = sscanf(acBuf, "%f %d", &fTilt, &iAcqIdx);
		if(iItems < 1) continue;
		//----------------
		this->SetTiltAngle(iCount, fTilt);
		this->SetAcqIdx(iCount, iAcqIdx);
		//----------------
		iCount += 1;
		if(iCount == m_ppTsStacks[0]->m_aiStkSize[2]) break;
	}
	if(iCount < m_ppTsStacks[0]->m_aiStkSize[2]) return false;
	return true;
}

void CTsPackage::mGenFullPath(const char* pcSuffix, char* pcFullPath)
{
	CInput* pInput = CInput::GetInstance();
	strcpy(pcFullPath, pInput->m_acOutDir);
        strcat(pcFullPath, m_acMrcMain);
	if(pcSuffix != 0L && strlen(pcSuffix) > 0)
	{	strcat(pcFullPath, pcSuffix);
	}
}
