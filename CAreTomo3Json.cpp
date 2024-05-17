#include "CMcAreTomoInc.h"
#include "MaUtil/CMaUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo;
namespace MU = McAreTomo::MaUtil;

CAreTomo3Json::CAreTomo3Json(void)
{
	m_iNumLines = 1024;
	m_iLineSize = 256;
	m_iLineCount = 0;
	m_iIndent = 5;
	m_pcJson = 0L;
}

CAreTomo3Json::~CAreTomo3Json(void)
{
	if(m_pcJson != 0L) delete[] m_pcJson;
}

void CAreTomo3Json::Create(char* pcVersion)
{
	int iSize = m_iNumLines * m_iLineSize;
	m_pcJson = new char[iSize];
	memset(m_pcJson, 0, sizeof(char) * iSize);
	m_iLineCount = 0;
	//-----------------
	char acBuf[256] = {"{\n"};
	mAddLine(acBuf);
	//-----------------
	strcpy(acBuf, "AreTomo3");
	mAddLine("Name", "AreTomo3");
	mAddLine("Version", pcVersion);
	//-----------------
	mAddInput();
	mAddMcInput();
	mAddAtInput();
	//--------------------------------------------------
	// Add data and time as the last entry
	//--------------------------------------------------
	time_t aTime = time(0L);
	struct tm* pTm = localtime(&aTime);
	char acTime[128] = {'\0'};
	strftime(acTime, sizeof(acTime), "%c", pTm);
	//-----------------
	char* pcLine = &m_pcJson[strlen(m_pcJson)];
	sprintf(pcLine, "     \"Time\":  \"%s\"\n", acTime);
	//-----------------
	strcpy(acBuf, "}\n");	
	mAddLine(acBuf);
	//-----------------
	CInput* pInput = CInput::GetInstance();
	sprintf(acBuf, "%sAreTomo3_Session.json", pInput->m_acOutDir);
	FILE* pFile = fopen(acBuf, "wt");
	if(pFile != 0L) 
	{	fprintf(pFile, "%s", m_pcJson);
		fclose(pFile);
	}
	//-----------------
	delete[] m_pcJson;
	m_pcJson = 0L;
}

void CAreTomo3Json::mAddInput(void)
{
	CInput* pInput = CInput::GetInstance();
	mAddLine(pInput->m_acInMdocTag+1, pInput->m_acInMdoc);
	mAddLine(pInput->m_acInSuffixTag+1, pInput->m_acInSuffix);
	mAddLine(pInput->m_acInSkipsTag+1, pInput->m_acInSkips, true);
	mAddLine(pInput->m_acOutDirTag+1, pInput->m_acOutDir);
	mAddLine(pInput->m_acPixSizeTag+1, pInput->m_fPixSize);
	mAddLine(pInput->m_acKvTag+1, pInput->m_iKv);
	mAddLine(pInput->m_acCsTag+1, pInput->m_fCs);
	mAddLine(pInput->m_acFmDoseTag+1, pInput->m_fFmDose);
	mAddLine(pInput->m_acSerialTag+1, pInput->m_iSerial);
	mAddLine(pInput->m_acCmdTag+1, pInput->m_iCmd);
	mAddLine(pInput->m_acResumeTag+1, pInput->m_iResume);
	mAddLine(pInput->m_acGpuIDTag+1, pInput->m_piGpuIDs,
	   pInput->m_iNumGpus);
}

void CAreTomo3Json::mAddMcInput(void)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	mAddLine(pMcInput->m_acFmIntFileTag+1, pMcInput->m_acFmIntFile);
	mAddLine(pMcInput->m_acGainFileTag+1, pMcInput->m_acGainFile);
	mAddLine(pMcInput->m_acDarkMrcTag+1, pMcInput->m_acDarkMrc);
	mAddLine(pMcInput->m_acDefectFileTag+1, pMcInput->m_acDefectFile);
	mAddLine(pMcInput->m_acEerSamplingTag+1, pMcInput->m_iEerSampling);
	mAddLine(pMcInput->m_acPatchesTag+1, pMcInput->m_aiNumPatches, 2);
	mAddLine(pMcInput->m_acIterTag+1, pMcInput->m_iMcIter);
	mAddLine(pMcInput->m_acTolTag+1, pMcInput->m_fMcTol);
	mAddLine(pMcInput->m_acMcBinTag+1, pMcInput->m_fMcBin);
	mAddLine(pMcInput->m_acGroupTag+1, pMcInput->m_aiGroup, 2);
	mAddLine(pMcInput->m_acFmRefTag+1, pMcInput->m_iFmRef);
	mAddLine(pMcInput->m_acRotGainTag+1, pMcInput->m_iRotGain);
	mAddLine(pMcInput->m_acFlipGainTag+1, pMcInput->m_iFlipGain);
	mAddLine(pMcInput->m_acInvGainTag+1, pMcInput->m_iInvGain);
	mAddLine(pMcInput->m_acMagTag+1, pMcInput->m_afMag, 3);
	mAddLine(pMcInput->m_acInFmMotionTag+1, pMcInput->m_iInFmMotion);
}

void CAreTomo3Json::mAddAtInput(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	mAddLine(pAtInput->m_acTotalDoseTag+1, pAtInput->m_fTotalDose);
	mAddLine(pAtInput->m_acAlignZTag+1, pAtInput->m_iAlignZ);
	mAddLine(pAtInput->m_acVolZTag+1, pAtInput->m_iVolZ);
	mAddLine(pAtInput->m_acAtBinTag+1, pAtInput->m_fAtBin);
	mAddLine(pAtInput->m_acTiltAxisTag+1, pAtInput->m_afTiltAxis, 2);
	mAddLine(pAtInput->m_acTiltCorTag+1, pAtInput->m_afTiltCor, 2);
	mAddLine(pAtInput->m_acAtPatchTag+1, pAtInput->m_aiAtPatches, 2);
	mAddLine(pAtInput->m_acAlignTag+1, pAtInput->m_iAlign);
	mAddLine(pAtInput->m_acReconRangeTag+1, pAtInput->m_afReconRange, 2);
	mAddLine(pAtInput->m_acFlipVolTag+1, pAtInput->m_iFlipVol);
	mAddLine(pAtInput->m_acSartTag+1, pAtInput->m_aiSartParam, 2);
	mAddLine(pAtInput->m_acWbpTag+1, pAtInput->m_iWbp);
	mAddLine(pAtInput->m_acDarkTolTag+1, pAtInput->m_fDarkTol);
	mAddLine(pAtInput->m_acOutImodTag+1, pAtInput->m_iOutImod);
	mAddLine(pAtInput->m_acAmpContrastTag+1, pAtInput->m_fAmpContrast);
	mAddLine(pAtInput->m_acExtPhaseTag+1, pAtInput->m_afExtPhase, 2);
	mAddLine(pAtInput->m_acCorrCTFTag+1, pAtInput->m_aiCorrCTF, 2);
}

void CAreTomo3Json::mAddLine(char* pcLine)
{
	int iSize = strlen(m_pcJson);
	char* pcJson = &m_pcJson[iSize];
	sprintf(pcJson, "%s", pcLine);
};

void CAreTomo3Json::mAddLine
(	const char* pcKey, 
	const char* pcVal, 
	bool bList
)
{	char acBuf[256] = {""};
	if(pcVal != 0L && strlen(pcVal) > 0) strcpy(acBuf, pcVal);
	//-----------------
	int iCount = 0;
	char* pcTokens[32] = {0L};
	char* pcToken = strtok(acBuf, ", ");
	while(pcToken != 0L)
	{	pcTokens[iCount] = pcToken;
		iCount += 1;
		pcToken = strtok(0L, ", ");
	}
	//-----------------
	int iSize = strlen(m_pcJson);
	char* pcJson = &m_pcJson[iSize];
	//-----------------
	if(!bList) 
	{	sprintf(pcJson, "     \"%s\":  \"", pcKey);
	       	if(pcVal != 0L) strcat(pcJson, pcVal);
		strcat(pcJson, "\"");
	}
	else
	{	char acVal[128] = {'\0'};
		mCreateVal(pcTokens, iCount, acVal);
		sprintf(pcJson, "     \"%s\":  %s", pcKey, acVal);
	}
	//-----------------
	strcat(pcJson, ",\n");
	m_iLineCount += 1;
}

void CAreTomo3Json::mAddLine(char* pcKey, int iVal)
{
	int iSize = strlen(m_pcJson);
	char* pcJson = &m_pcJson[iSize];
	sprintf(pcJson, "     \"%s\":  %d", pcKey, iVal);
	//-----------------
	strcat(pcJson, ",\n");
	m_iLineCount += 1;
}

void CAreTomo3Json::mAddLine(char* pcKey, float fVal)
{
	int iSize = strlen(m_pcJson);
        char* pcJson = &m_pcJson[iSize];
        sprintf(pcJson, "     \"%s\":  %f", pcKey, fVal);
	//-----------------
	strcat(pcJson, ",\n");
        m_iLineCount += 1;
}

void CAreTomo3Json::mAddLine(char* pcKey, int* piVals, int iNumVals)
{	
	char acVal[128] = {'\0'};
	mCreateVal(piVals, iNumVals, acVal);
	//-----------------
	int iSize = strlen(m_pcJson);
	char* pcJson = &m_pcJson[iSize];
	sprintf(pcJson, "     \"%s\":  %s", pcKey, acVal);
	//-----------------
	strcat(pcJson, ",\n");
	m_iLineCount += 1;
}

void CAreTomo3Json::mAddLine(char* pcKey, float* pfVals, int iNumVals)
{	
	char acVal[128] = {'\0'};
	mCreateVal(pfVals, iNumVals, acVal);
	//-----------------
	int iSize = strlen(m_pcJson);
	char* pcJson = &m_pcJson[iSize];
	sprintf(pcJson, "     \"%s\":  %s", pcKey, acVal);
	//-----------------
	strcat(pcJson, ",\n");
	m_iLineCount += 1;
}

void CAreTomo3Json::mCreateVal(int* piVals, int iNumVals, char* pcVal)
{
	int iLast = iNumVals - 1;
	char acBuf[32] = {'\0'};
	//-----------------
	strcpy(pcVal, "[");
	for(int i=0; i<iNumVals; i++) 
	{	sprintf(acBuf, "%d", piVals[i]);
                strcat(pcVal, acBuf);
                if(i < iLast) strcat(pcVal, ", ");
        }
        strcat(pcVal, "]");
}

void CAreTomo3Json::mCreateVal(float* pfVals, int iNumVals, char* pcVal)
{
	int iLast = iNumVals - 1;
	char acBuf[32] = {'\0'};
	//-----------------
	strcpy(pcVal, "[");
	for(int i=0; i<iNumVals; i++)
	{	sprintf(acBuf, "%f", pfVals[i]);
		strcat(pcVal, acBuf);
		if(i < iLast) strcat(pcVal, ", ");
	}
	strcat(pcVal, "]");
}

void CAreTomo3Json::mCreateVal(char** pcToks, int iNumToks, char* pcVal)
{
	int iLast = iNumToks - 1;
	strcpy(pcVal, "[");
	for(int i=0; i<iNumToks; i++)
	{	strcat(pcVal, pcToks[i]);
		if(i < iLast) strcat(pcVal, ", ");
	}
	strcat(pcVal, "]");
}
