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
	m_ppTsStacks = new CTiltSeries*[CAlnSums::m_iNumSums];
	m_ppVolStacks = new CTiltSeries*[CAlnSums::m_iNumSums];
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	m_ppTsStacks[i] = new CTiltSeries;
		m_ppVolStacks[i] = 0L;
	}
}

CTsPackage::~CTsPackage(void)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	if(m_ppTsStacks[i] != 0L) delete m_ppTsStacks[i];
		if(m_ppVolStacks[i] != 0L) delete m_ppVolStacks[i];
	}
	delete[] m_ppTsStacks;
	delete[] m_ppVolStacks;
}

void CTsPackage::SetInFile(char* pcInFile)
{
	memset(m_acInFile, 0, sizeof(m_acInFile));
	memset(m_acInDir, 0, sizeof(m_acInDir));
	//-----------------
	memset(m_acMrcMain, 0, sizeof(m_acMrcMain));
	//-----------------
	if(pcInFile == 0L || strlen(pcInFile) == 0) return;
	strcpy(m_acInFile, pcInFile);
	//-----------------
	MU::CFileName fileName(m_acInFile);
	fileName.GetName(m_acMrcMain);
	fileName.GetFolder(m_acInDir);
	fileName.GetExt(m_acMrcExt);
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

void CTsPackage::SetLoaded(bool bLoaded)
{
	for(int i=0; i<MD::CAlnSums::m_iNumSums; i++)
	{	m_ppTsStacks[i]->m_bLoaded = bLoaded;
	}
}

bool CTsPackage::LoadTiltSeries(void)
{
	char* pcExt = strrchr(m_acInFile, '.');
	if(pcExt == 0L) return false;	
	//-----------------------------------------------
	// 1) If the input is a mdoc file, load the tilt
	// series from the output directory.
	// 2) Otherwise, it is a mrc file file, load
	// from the input directory.
	//-----------------------------------------------
	char acMrcFile[256] = {'0'};
	if(strstr(pcExt, ".mdoc") != 0L) mGenOutPath(".mrc", acMrcFile);
	else strcpy(acMrcFile, m_acInFile);
        //-----------------
        Mrc::CLoadMrc loadMrc;
        bool bLoaded = loadMrc.OpenFile(acMrcFile);
        if(!bLoaded) return false;
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
	char acExt[32] = {'\0'};
	strcpy(acExt, pcExt);
        bLoaded = mLoadMrc(acExt, m_ppTsStacks[0]);
	if(!bLoaded) return false;
	//-----------------
	strcpy(acExt, "_EVN");
	strcat(acExt, pcExt);
	mLoadMrc(acExt, m_ppTsStacks[1]);
	//-----------------
	strcpy(acExt, "_ODD");
	strcat(acExt, pcExt);
        mLoadMrc(acExt, m_ppTsStacks[2]);
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

void CTsPackage::SetImgDose(int iTilt, float fImgDose)
{
	for(int i=0; i<CAlnSums::m_iNumSums; i++)
	{	CTiltSeries* pSeries = this->GetSeries(i);
		if(pSeries == 0L) continue;
		if(i == 0) pSeries->m_pfDoses[iTilt] =  fImgDose;
		else pSeries->m_pfDoses[iTilt] = fImgDose * 0.5f;
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
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_iSplitSum == 0)
	{	if(iVol == 1) return;
		if(iVol == 2) return;
	}
	//-----------------
	char acExt[32] = {'\0'};
	if(iVol == 0) strcpy(acExt, "_Vol.mrc");
	else if(iVol == 1) strcpy(acExt, "_EVN_Vol.mrc");
	else if(iVol == 2) strcpy(acExt, "_ODD_Vol.mrc");
	else if(iVol == 3) strcpy(acExt, "_2ND_Vol.mrc");
	//-----------------
	mSaveMrc(acExt, pVol);	
}

void CTsPackage::SaveTiltSeries(void)
{
	CInput* pInput = CInput::GetInstance();
	mSaveTiltFile(m_ppTsStacks[0]);
	mSaveMrc(".mrc", m_ppTsStacks[0]);
	if(pInput->m_iSplitSum == 0) return;
	//-----------------
	mSaveMrc("_EVN.mrc", m_ppTsStacks[1]);
	mSaveMrc("_ODD.mrc", m_ppTsStacks[2]);
}

void CTsPackage::mSaveMrc
(	const char* pcExt, 
	CTiltSeries* pTiltSeries
)
{	char acMrcFile[256] = {'0'};
	mGenOutPath(pcExt, acMrcFile);
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
	mGenOutPath("_TLT.txt", acTiltFile);
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
		float fDose = pTiltSeries->m_pfDoses[i];
		int iAcqIdx = pTiltSeries->m_piAcqIndices[i] + iAdd;
		fprintf(pFile, "%8.2f  %4d  %8.2f\n", 
		   fTilt, iAcqIdx, fDose);
	}
	fclose(pFile);
}

bool CTsPackage::mLoadMrc
(	const char* pcExt,
	CTiltSeries* pTiltSeries
)
{	char acMrcFile[256] = {'0'};
	mGenInPath(pcExt, acMrcFile);
	//-----------------
	pTiltSeries->m_bLoaded = false;
	Mrc::CLoadMrc loadMrc;
	bool bLoaded = loadMrc.OpenFile(acMrcFile);
	if(!bLoaded) return false;
	//-----------------
	int iMode = loadMrc.m_pLoadMain->GetMode();
	int aiStkSize[3] = {0};
	loadMrc.m_pLoadMain->GetSize(aiStkSize, 3);
	//-----------------
	if(iMode == Mrc::eMrcFloat)
	{	for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
		{	void* pvFrm = pTiltSeries->GetFrame(i);
			loadMrc.m_pLoadImg->DoIt(i, pvFrm);
		}
		pTiltSeries->m_bLoaded = true;
		return true;
	}
	//----------------
	int iPixels = aiStkSize[0] * aiStkSize[1];
	if(iMode == 1)
	{	short* psBuf = new short[iPixels];
		for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
		{	loadMrc.m_pLoadImg->DoIt(i, (void*)psBuf);
			float* pfImg = (float*)pTiltSeries->GetFrame(i);
			for(int k=0; k<iPixels; k++) pfImg[k] = psBuf[k];
		}
		pTiltSeries->m_bLoaded = true;
		delete[] psBuf;
		return true;
	}
	else if(iMode == 6)
	{	unsigned short* pusBuf = new unsigned short[iPixels];
                for(int i=0; i<pTiltSeries->m_aiStkSize[2]; i++)
                {       loadMrc.m_pLoadImg->DoIt(i, (void*)pusBuf);
                        float* pfImg = (float*)pTiltSeries->GetFrame(i);
                        for(int k=0; k<iPixels; k++) pfImg[k] = pusBuf[k];
                }
		pTiltSeries->m_bLoaded = true;
		delete[] pusBuf;
                return true;
	}
	return false;
}

bool CTsPackage::mLoadTiltFile(void)
{
	char acTlt[256] = {'\0'}, acRawTlt[256] = {'\0'};
	mGenInPath("_TLT.txt", acTlt);
	mGenInPath(".rawtlt", acRawTlt);
	FILE* pFile = fopen(acTlt, "rt");
        if(pFile == 0L) pFile = fopen(acRawTlt, "rt");
	//-----------------
	if(pFile == 0L)
	{	fprintf(stderr, "Error (GPU %d): Unable to load tilt "
		   " angles from TLT.txt or rawtlt files\n"
		   "   %s\n   %s\n\n", m_iNthGpu, acTlt, acRawTlt);
		return false;
	}
	//-----------------
	float fTilt = 0.0f;
	float fDose = 0.0f;
	int iAcqIdx = 0, iCount = 0;
	//-----------------
	char acBuf[256] = {'\0'};
	while(!feof(pFile))
	{	memset(acBuf, 0, sizeof(acBuf));
		char* pcRet = fgets(acBuf, 256, pFile);
		if(pcRet == 0L) continue;
		//----------------
		int iItems = sscanf(acBuf, "%f %d %f", 
		   &fTilt, &iAcqIdx, &fDose);
		if(iItems < 1) continue;
		//----------------
		this->SetTiltAngle(iCount, fTilt);
		if(iItems >= 2) this->SetAcqIdx(iCount, iAcqIdx);
		if(iItems >= 3) this->SetImgDose(iCount, fDose);
		//----------------
		iCount += 1;
		if(iCount == m_ppTsStacks[0]->m_aiStkSize[2]) break;
	}
	if(iCount < m_ppTsStacks[0]->m_aiStkSize[2]) return false;
	return true;
}

void CTsPackage::mGenInPath(const char* pcSuffix, char* pcInPath)
{
	//-----------------------------------------------
	// 1) when the input is mdoc file, we search the
	// output directory for input tilt series.
	//-----------------------------------------------
	char* pcExt = strrchr(m_acInFile, '.');
	if(pcExt != 0L && strcasestr(pcExt, ".mdoc") != 0L)
	{	mGenOutPath(pcSuffix, pcInPath);
		return;
	}
	//-----------------------------------------------
	// 2) otherwise we search the input directory
	// for input tilt series.
	//-----------------------------------------------
	strcpy(pcInPath, m_acInDir);
	strcat(pcInPath, m_acMrcMain);
	if(pcSuffix != 0L && strlen(pcSuffix) > 0)
	{	strcat(pcInPath, pcSuffix);
	}
}

void CTsPackage::mGenOutPath(const char* pcSuffix, char* pcOutPath)
{
	CInput* pInput = CInput::GetInstance();
	strcpy(pcOutPath, pInput->m_acOutDir);
        strcat(pcOutPath, m_acMrcMain);
	if(pcSuffix != 0L && strlen(pcSuffix) > 0)
	{	strcat(pcOutPath, pcSuffix);
	}
}
