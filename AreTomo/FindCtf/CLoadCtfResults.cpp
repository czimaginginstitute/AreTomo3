#include "CFindCtfInc.h"
#include <stdio.h>
#include <memory.h>

using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.017453f;

CLoadCtfResults::CLoadCtfResults(void)
{
	m_bLoaded = false;
}

CLoadCtfResults::~CLoadCtfResults(void)
{
}
 
bool CLoadCtfResults::DoIt(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	m_bLoaded = false;
	//-----------------
	CInput* pInput = CInput::GetInstance();
	CAtInput* pAtInput = CAtInput::GetInstance();
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	MD::CCtfParam ctfParam;
	ctfParam.Setup(pInput->m_iKv, pInput->m_fCs, 
	   pAtInput->m_fAmpContrast, pTiltSeries->m_fPixSize);
	//-----------------
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	int aiTileSize[] = {pAtInput->m_iCtfTileSize, pAtInput->m_iCtfTileSize};
	pCtfResults->Setup(pTiltSeries->m_aiStkSize[2], aiTileSize, &ctfParam);
	//-----------------
	char acCtfFile[256] = {'\0'};
	CSaveCtfResults::GenFileName(m_iNthGpu, acCtfFile);
	strcat(acCtfFile, ".txt");
	m_bLoaded = mLoadFittings(acCtfFile);
	//-----------------
	return m_bLoaded;
}

bool CLoadCtfResults::mLoadFittings(const char* pcCtfFile)
{
	MD::CCtfResults* pCtfResults = MD::CCtfResults::GetInstance(m_iNthGpu);
	int iNumImgs = pCtfResults->m_iNumImgs;
	//-----------------
	MD::CTsPackage* pTsPkg = MD::CTsPackage::GetInstance(m_iNthGpu);
	MD::CTiltSeries* pTiltSeries = pTsPkg->GetSeries(0);
	//-----------------
	FILE* pFile = fopen(pcCtfFile, "rt");
	if(pFile == 0L) return false;
	//-----------------
	int* piTilts = new int[iNumImgs];
	float* pfRes = new float[6 * iNumImgs];
	memset(piTilts, 0, sizeof(int) * iNumImgs);
	memset(pfRes, 0, sizeof(float) * iNumImgs * 6);
	//-----------------
	int iNumReads = 0;
	int iMinIdx = 1000000;
	char acLine[256] = {'\0'};
	//-----------------
	while(!feof(pFile))
	{	memset(acLine, 0, sizeof(acLine));
		char* pcRet = fgets(acLine, 256, pFile);
		if(pcRet == 0L) continue;
		if(acLine[0] == '#') continue;
		//----------------
		int j = iNumReads * 6;
		int iNumItems = sscanf(&acLine[1], "%d %f %f %f %f "
		   "%f  %f", &piTilts[iNumReads], &pfRes[j], &pfRes[j+1],
		   &pfRes[j+2], &pfRes[j+3], &pfRes[j+4], &pfRes[j+5]);
		if(iNumItems != 7) continue;
		//----------------
		if(piTilts[iNumReads] < iMinIdx) 
		{	iMinIdx = piTilts[iNumReads];
		}
		iNumReads += 1;
	}
	fclose(pFile);
	//-----------------
	if(iNumReads == iNumImgs)
	{	for(int i=0; i<iNumImgs; i++)
		{	int j = i * 6;
			int iTilt = piTilts[i] - iMinIdx;
			float fPhase = pfRes[j+3] / s_fD2R;
			//---------------
			pCtfResults->SetTilt(iTilt, 
			   pTiltSeries->m_pfTilts[iTilt]);
			pCtfResults->SetDfMax(iTilt, pfRes[j]);
			pCtfResults->SetDfMin(iTilt, pfRes[j+1]);
			pCtfResults->SetAzimuth(iTilt, pfRes[j+2]);
			pCtfResults->SetExtPhase(iTilt, fPhase);
			pCtfResults->SetScore(iTilt, pfRes[j+4]);
			pCtfResults->SetCtfRes(iTilt, pfRes[j+5]);
		}
		m_bLoaded = true;
	}
	else 
	{	m_bLoaded = false;
		pCtfResults->Clean();
	}
	//-----------------
	if(piTilts != 0L) delete[] piTilts;
	if(pfRes != 0L) delete[] pfRes;
	//-----------------
	return m_bLoaded;
}
