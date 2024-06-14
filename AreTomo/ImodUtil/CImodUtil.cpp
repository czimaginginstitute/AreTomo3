#include "CImodUtilInc.h"
#include "../Correct/CCorrectInc.h"
#include "../FindCtf/CFindCtfInc.h"
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <dirent.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::ImodUtil;

CImodUtil* CImodUtil::m_pInstances= 0L;
int CImodUtil::m_iNumGpus = 0;

void CImodUtil::CreateInstances(int iNumGpus)
{
	if(m_iNumGpus == iNumGpus) return;
	if(m_pInstances != 0L) delete[] m_pInstances;
	m_pInstances = new CImodUtil[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	m_pInstances[i].m_iNthGpu = i;
	}
	m_iNumGpus = iNumGpus;
}

void CImodUtil::DeleteInstances(void)
{
	if(m_pInstances == 0L) return;
	delete[] m_pInstances;
	m_pInstances = 0L;
	m_iNumGpus = 0;
}

CImodUtil* CImodUtil::GetInstance(int iNthGpu)
{
	return &m_pInstances[iNthGpu];
}

CImodUtil::CImodUtil(void)
{
}

CImodUtil::~CImodUtil(void)
{
}

bool CImodUtil::bFolderExist(void)
{
	mGenFolderName();
	struct stat st = {0};
	if (stat(m_acOutFolder, &st) == -1) return false;
	else return true;
}

int CImodUtil::FindOutImodVal(void)
{
	if(!bFolderExist()) return 0;
	DIR* pDir = opendir(m_acOutFolder);
	if(pDir == 0L) return 0;
	//-----------------
	int iOutImod = 0;
	struct dirent* pDirent;
	while(true)
	{	pDirent = readdir(pDir);
		if(pDirent == 0L) break;
		if(pDirent->d_name[0] == '.') continue;
		//----------------
		char* pcMrcFile = strstr(pDirent->d_name, "_st.mrc");
		if(pcMrcFile != 0L)
		{	iOutImod = 2; // aligned tilt series exists
			break;
		}
		pcMrcFile = strstr(pDirent->d_name, ".mrc");
		if(pcMrcFile != 0L)
		{	iOutImod = 1; // dark-removed and unaligned
			break;
		}
	}
	closedir(pDir);
	return iOutImod;
}

void CImodUtil::CreateFolder(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	if(pAtInput->m_iOutImod == 0) return;
	if(!this->bFolderExist()) mkdir(m_acOutFolder, 0700);
	//----------------
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	strcpy(m_acTltFile, pTsPackage->m_acMrcMain);
	strcat(m_acTltFile, "_st.tlt");
	//-----------------
	strcpy(m_acCsvFile, pTsPackage->m_acMrcMain);
	if(pAtInput->m_iOutImod == 1) strcat(m_acCsvFile, "_order_list.csv");
	else strcat(m_acCsvFile, "_st_order_list.csv");
	//-----------------
	strcpy(m_acXfFile, pTsPackage->m_acMrcMain);
	strcat(m_acXfFile, "_st.xf");
	//------------------------
	strcpy(m_acAliFile, pTsPackage->m_acMrcMain);
	strcat(m_acAliFile, "_st.ali");
	//--------------------------
	strcpy(m_acXtiltFile, pTsPackage->m_acMrcMain);
	strcat(m_acXtiltFile, "_st.xtilt");
	//------------------------------
	strcpy(m_acRecFile, pTsPackage->m_acMrcMain);
	strcat(m_acRecFile, "_st.rec");
	//-----------------------------
	strcpy(m_acCtfFile, pTsPackage->m_acMrcMain);
	strcat(m_acCtfFile, "_ctf.txt");
	//-----------------
	mGenMrcName();
}

void CImodUtil::SaveTiltSeries(MD::CTiltSeries* pTiltSeries)
{	
	CAtInput* pAtInput = CAtInput::GetInstance();
	if(pAtInput->m_iOutImod == 0) return;
	//-----------------
	m_pTiltSeries = pTiltSeries;
	m_pGlobalParam = MAM::CAlignParam::GetInstance(m_iNthGpu);
	//-----------------
	m_fPixelSize = 0.0f;
	if(m_pTiltSeries != 0L) m_fPixelSize = m_pTiltSeries->m_fPixSize;
	int iCentFrm = m_pGlobalParam->m_iNumFrames / 2;
	m_fTiltAxis = m_pGlobalParam->GetTiltAxis(iCentFrm);
	//-----------------
	char acFile[256] = {'\0'};
	mCreateFileName(m_acTltFile, acFile);
	CSaveTilts aSaveTilts;
	aSaveTilts.DoIt(m_iNthGpu, acFile);
	//-----------------
	mCreateFileName(m_acCsvFile, acFile);
	CSaveCsv saveCsv;
	saveCsv.DoIt(m_iNthGpu, acFile);
	//-----------------
	mCreateFileName(m_acXfFile, acFile);
	CSaveXF aSaveXF;
	aSaveXF.DoIt(m_iNthGpu, acFile);
	//-----------------	
	mCreateFileName(m_acXtiltFile, acFile);
	CSaveXtilts aSaveXtilts;
	aSaveXtilts.DoIt(m_iNthGpu, acFile);
	//-----------------
	mSaveTiltSeries();
	mSaveNewstComFile();
	mSaveTiltComFile();
}

void CImodUtil::mSaveTiltSeries(void)
{
	if(m_pTiltSeries == 0L) return;
	//-----------------
	char acFile[256];
	mCreateFileName(m_acInMrcFile, acFile);
	//-----------------
	MAM::CSaveStack aSaveStack;
	bool bOpen = aSaveStack.OpenFile(acFile);
	if(!bOpen) return;
	//-----------------
	printf("GPU %d: Save tilt series in Imod folder, please "
	   "wait .....\n", m_iNthGpu);
	bool bVolume = true;
	aSaveStack.DoIt(m_pTiltSeries, m_pGlobalParam,
	   m_fPixelSize, 0L, !bVolume);
	printf("GPU %d: Aligned series saved in Imod folder.\n", m_iNthGpu);
}

void CImodUtil::mSaveNewstComFile(void)
{
	char acComFile[256];
	mCreateFileName("newst.com", acComFile);
	FILE* pFile = fopen(acComFile, "wt");
	if(pFile == 0L) return;
	//---------------------
	fprintf(pFile, "$newstack -StandardInput\n");
	fprintf(pFile, "InputFile	%s\n", m_acInMrcFile);
	fprintf(pFile, "OutputFile	%s\n", m_acAliFile);
	fprintf(pFile, "TransformFile	%s\n", m_acXfFile);     
	fprintf(pFile, "TaperAtFill     1,0\n");
	fprintf(pFile, "AdjustOrigin\n");
	fprintf(pFile, "OffsetsInXandY  0.0,0.0\n");
	fprintf(pFile, "#DistortionField        .idf\n");
	fprintf(pFile, "ImagesAreBinned 1.0\n");
	fprintf(pFile, "BinByFactor     1\n");
	fprintf(pFile, "#GradientFile   hc20211206_804.maggrad\n");
	fprintf(pFile, "$if (-e ./savework) ./savework");
	//-----------------------------------------------
	fclose(pFile);	
}

void CImodUtil::mSaveTiltComFile(void)
{
	char acComFile[256];
	mCreateFileName("tilt.com", acComFile);
	FILE* pFile = fopen(acComFile, "wt");
	if(pFile == 0L) return;
	//---------------------
	MAM::CDarkFrames* pDarkFrames =
	   MAM::CDarkFrames::GetInstance(m_iNthGpu);
	int* piRawSize = pDarkFrames->m_aiRawStkSize;
	int aiAlnSize[] = {0, 0};
	Correct::CCorrectUtil::CalcAlignedSize(piRawSize, 
	   m_fTiltAxis, aiAlnSize);
	CAtInput* pInput = CAtInput::GetInstance();
	//-----------------------------------------------
	fprintf(pFile, "$tilt -StandardInput\n");
	fprintf(pFile, "InputProjections %s\n", m_acAliFile);
	fprintf(pFile, "OutputFile %s\n", m_acRecFile);
	fprintf(pFile, "IMAGEBINNED 1\n");
	fprintf(pFile, "TILTFILE %s\n", m_acTltFile);
	fprintf(pFile, "THICKNESS %d\n", pInput->m_iVolZ);
	fprintf(pFile, "RADIAL 0.35 0.035\n");
	fprintf(pFile, "FalloffIsTrueSigma 1\n");
	fprintf(pFile, "XAXISTILT 0.0\n");
	fprintf(pFile, "LOG 0.0\n");
	fprintf(pFile, "SCALE 0.0 250.0\n");
	fprintf(pFile, "PERPENDICULAR\n");
	fprintf(pFile, "Mode 2\n");
	fprintf(pFile, "FULLIMAGE %d %d\n", aiAlnSize[0], aiAlnSize[1]);
	fprintf(pFile, "SUBSETSTART 0 0\n");
	fprintf(pFile, "AdjustOrigin\n");
	fprintf(pFile, "LOCALFILE %s\n", m_acXfFile);
	fprintf(pFile, "ActionIfGPUFails 1,2\n");
	fprintf(pFile, "XTILTFILE %s\n", m_acXtiltFile);
	fprintf(pFile, "OFFSET 0.0\n");
	fprintf(pFile, "SHIFT 0.0 0.0\n");
	//--------------------------------
	if(pInput->m_iOutImod == 1 && pDarkFrames->m_iNumDarks > 0)
	{	char acExclude[128] = {'\0'};
		pDarkFrames->GenImodExcludeList(acExclude, 128);
		fprintf(pFile, "%s\n", acExclude);
	}
	fprintf(pFile, "$if (-e ./savework) ./savework");
	//-----------------------------------------------
	fclose(pFile);	
}

void CImodUtil::SaveCtfFile(void)
{
	MD::CCtfResults* pCtfResults = 
	   MD::CCtfResults::GetInstance(m_iNthGpu);
	if(pCtfResults->m_iNumImgs <= 0) return;
	//-----------------
	char acFile[256] = {'\0'};	
	mCreateFileName(m_acCtfFile, acFile);
	FILE* pFile = fopen(acFile, "w");
	if(pFile == 0L) return;
 	//--------------------------------------
	float fExtPhase = pCtfResults->GetExtPhase(0);
	if(fExtPhase == 0) fprintf(pFile, "1  0  0.0  0.0  0.0  3\n");
	else fprintf(pFile, "5  0  0.0  0.0  0.0  3\n");
	//----------------------------------------------
	const char *pcFormat1 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f\n";
	const char *pcFormat2 = "%4d  %4d  %7.2f  %7.2f  %8.2f  "
	   "%8.2f  %7.2f  %8.2f\n";
	float fDfMin, fDfMax;
	if(fExtPhase == 0)
	{	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
		{	float fTilt = pCtfResults->GetTilt(i);
			fDfMin = pCtfResults->GetDfMin(i) * 0.1f;
			fDfMax = pCtfResults->GetDfMax(i) * 0.1f;
			fprintf(pFile, pcFormat1, i+1, i+1, fTilt, fTilt,
			   fDfMin, fDfMax, pCtfResults->GetAzimuth(i));
		}
	}
	else
	{	for(int i=0; i<pCtfResults->m_iNumImgs; i++)
		{	float fTilt = pCtfResults->GetTilt(i);
			fDfMin = pCtfResults->GetDfMin(i) * 0.1f;
			fDfMax = pCtfResults->GetDfMax(i) * 0.1f;
			fprintf(pFile, pcFormat2, i+1, i+1, fTilt, fTilt,
			   fDfMin, fDfMax, pCtfResults->GetAzimuth(i),
			   pCtfResults->GetExtPhase(i));
		}
	}
	fclose(pFile);
}

void CImodUtil::mCreateFileName(const char* pcInFileName, char* pcOutFileName)
{
	strcpy(pcOutFileName, m_acOutFolder);
	strcat(pcOutFileName, pcInFileName);
}

void CImodUtil::mGenFolderName(void)
{
        McAreTomo::CInput* pInput = CInput::GetInstance();
        MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
        //-----------------
        strcpy(m_acOutFolder, pInput->m_acOutDir);
        strcat(m_acOutFolder, pTsPackage->m_acMrcMain);
        strcat(m_acOutFolder, "_Imod/");
}

void CImodUtil::mGenMrcName(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	MD::CTsPackage* pTsPackage = MD::CTsPackage::GetInstance(m_iNthGpu);
	strcpy(m_acInMrcFile, pTsPackage->m_acMrcMain);
	//-----------------
        if(pAtInput->m_iOutImod == 1)
        {       strcat(m_acInMrcFile, ".mrc");
        }
        else if(pAtInput->m_iOutImod >= 2)
        {       strcat(m_acInMrcFile, "_st.mrc");
        }
}

