#include "CMcAreTomoInc.h"
#include "MaUtil/CMaUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo;
namespace MU = McAreTomo::MaUtil;

CAreTomo3Json::CAreTomo3Json(void)
{
	m_pcJson = 0L;
}

CAreTomo3Json::~CAreTomo3Json(void)
{
	if(m_pcJson != 0L) delete[] m_pcJson;
}

void CAreTomo3Json::Create(char* pcVersion)
{
	int iSize = 256 * 512;
	if(m_pcJson == 0L) m_pcJson = new char[iSize];
	memset(m_pcJson, 0, iSize * sizeof(char));
	//-----------------
	strcpy(m_pcJson, "{\n");
	//-----------------
	mGenSoftware(pcVersion);
	mGenInput();
	mGenOutput();
	mGenParams();
	//-----------------
	time_t aTime = time(0L);
        struct tm* pTm = localtime(&aTime);
        char acTime[128] = {'\0'};
        strftime(acTime, sizeof(acTime), "%c", pTm);
	//-----------------
	char acBuf[256] = {'\0'};
	mCreateKey("Time", 5, acBuf);
	mAddStrVal(acTime, true, acBuf);
	strcat(m_pcJson, acBuf);
	//-----------------
	strcat(m_pcJson, "}");
	//-----------------
	CInput* pInput = CInput::GetInstance();
        sprintf(acBuf, "%sAreTomo3_Session.json", pInput->m_acOutDir);
        FILE* pFile = fopen(acBuf, "wt");
        if(pFile != 0L) 
        {       fprintf(pFile, "%s", m_pcJson);
                fclose(pFile);
        }
        //-----------------
        delete[] m_pcJson;
        m_pcJson = 0L;
}

void CAreTomo3Json::mGenSoftware(char* pcVersion)
{
	char acBuf[256] = {'\0'};
	mCreateKey("software", 5, acBuf);
	strcat(m_pcJson, acBuf);
	strcat(m_pcJson, "{\n");
	//-----------------
	bool bEnd = true, bList = true;
	mAddKeyValPair("name", "AreTomo3", 10, !bList, !bEnd);
	//-----------------	
	mCreateKey("version", 10, acBuf);
	mAddStrVal(pcVersion, bEnd, acBuf);
	strcat(m_pcJson, acBuf);
	//-----------------
	mAddEndBrace(5, !bEnd);
}

void CAreTomo3Json::mGenInput(void)
{
	bool bEnd = true;
	bool bList = true;
	CInput* pInput = CInput::GetInstance();
	CMcInput* pMcInput = CMcInput::GetInstance();
	//-----------------
	char acBuf[256] = {'\0'};
	mCreateKey("input", 5, acBuf);
	strcat(m_pcJson, acBuf);
	strcat(m_pcJson, "{\n");
	//-----------------
	strcpy(acBuf, pInput->m_acInPrefix);
	MU::UseFullPath(acBuf);
	mAddKeyValPair(pInput->m_acInPrefixTag + 1, acBuf, 10, !bList, !bEnd);
	//-----------------
	strcpy(acBuf, pMcInput->m_acFmIntFile);
	MU::UseFullPath(acBuf);
	mAddKeyValPair(pMcInput->m_acFmIntFileTag + 1, acBuf, 
	   10, !bList, !bEnd);
	//-----------------
	strcpy(acBuf, pMcInput->m_acGainFile);
	MU::UseFullPath(acBuf);
        mAddKeyValPair(pMcInput->m_acGainFileTag + 1, acBuf, 
	   10, !bList, !bEnd);
	//-----------------
	strcpy(acBuf, pMcInput->m_acDarkMrc);
	MU::UseFullPath(acBuf);
        mAddKeyValPair(pMcInput->m_acDarkMrcTag + 1, acBuf,
	   10, !bList, !bEnd);
	//-----------------
	strcpy(acBuf, pMcInput->m_acDefectFile);
	MU::UseFullPath(acBuf);
        mAddKeyValPair(pMcInput->m_acDefectFileTag + 1, acBuf, 
	   10, !bList, bEnd);
	//-----------------
	mAddEndBrace(5, !bEnd);
}

void CAreTomo3Json::mGenOutput(void)
{
	bool bEnd = true, bList = true;
	CInput* pInput = CInput::GetInstance();
	//-----------------
	char acBuf[256] = {'\0'};
	mCreateKey("output", 5, acBuf);
	strcat(m_pcJson, acBuf);
	strcat(m_pcJson, "{\n");
	//-----------------
	strcpy(acBuf, pInput->m_acTmpDir);
	MU::UseFullPath(acBuf);
	mAddKeyValPair(pInput->m_acTmpDirTag + 1, acBuf, 10, !bList, !bEnd);
	//-----------------
	strcpy(acBuf, pInput->m_acOutDir);
	MU::UseFullPath(acBuf);
	mAddKeyValPair(pInput->m_acOutDirTag + 1, acBuf, 10, !bList, bEnd);
	//-----------------
	mAddEndBrace(5, !bEnd);
}

void CAreTomo3Json::mGenParams(void)
{
	bool bEnd = true;
	char acBuf[256] = {'\0'};
	mCreateKey("parameters", 5, acBuf);
	strcat(m_pcJson, acBuf);
	strcat(m_pcJson, "{\n");
	//-----------------
	mAddMainInput();
	mAddMcInput();
	mAddAtInput();
	//-----------------
	mAddEndBrace(5, !bEnd);
}

void CAreTomo3Json::mAddMainInput(void)
{
	bool bEnd = true;
	bool bList = true;
	CInput* pInput = CInput::GetInstance();
	//-----------------
	mAddKeyValPair(pInput->m_acInSuffixTag + 1, 
	   pInput->m_acInSuffix, 10, !bList, !bEnd);
	//-----------------
	mAddKeyValPair(pInput->m_acInSkipsTag + 1, 
	   pInput->m_acInSkips, 10, bList, !bEnd);
	//-----------------
	mAddKeyFloatPair(pInput->m_acPixSizeTag + 1,
           &(pInput->m_fPixSize), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pInput->m_acKvTag + 1,
	   &(pInput->m_iKv), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyFloatPair(pInput->m_acCsTag + 1,
	   &(pInput->m_fCs), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyFloatPair(pInput->m_acFmDoseTag + 1,
	   &(pInput->m_fFmDose), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pInput->m_acSerialTag + 1,
	   &(pInput->m_iSerial), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pInput->m_acCmdTag + 1,
	   &(pInput->m_iCmd), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pInput->m_acResumeTag + 1,
	   &(pInput->m_iResume), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pInput->m_acGpuIDTag + 1,
	   pInput->m_piGpuIDs, pInput->m_iNumGpus, 10, bList, !bEnd);	
}

void CAreTomo3Json::mAddMcInput(void)
{
	CMcInput* pMcInput = CMcInput::GetInstance();
	bool bList = true, bEnd = true;
	//-----------------
	mAddKeyIntPair(pMcInput->m_acEerSamplingTag + 1, 
	   &(pMcInput->m_iEerSampling), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pMcInput->m_acPatchesTag + 1, 
	   pMcInput->m_aiNumPatches, 2, 10, bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pMcInput->m_acIterTag + 1, 
	   &(pMcInput->m_iMcIter), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyFloatPair(pMcInput->m_acTolTag + 1, 
	   &(pMcInput->m_fMcTol), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyFloatPair(pMcInput->m_acMcBinTag + 1, 
	   &(pMcInput->m_fMcBin), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pMcInput->m_acGroupTag + 1, 
	   pMcInput->m_aiGroup, 2, 10, bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pMcInput->m_acFmRefTag + 1, 
	   &(pMcInput->m_iFmRef), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pMcInput->m_acRotGainTag + 1, 
	   &(pMcInput->m_iRotGain), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pMcInput->m_acFlipGainTag + 1, 
	   &(pMcInput->m_iFlipGain), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pMcInput->m_acInvGainTag + 1, 
	   &(pMcInput->m_iInvGain), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyFloatPair(pMcInput->m_acMagTag + 1, 
	   pMcInput->m_afMag, 3, 10, bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pMcInput->m_acInFmMotionTag + 1, 
	   &(pMcInput->m_iInFmMotion), 1, 10, !bList, !bEnd);
}

void CAreTomo3Json::mAddAtInput(void)
{
	CAtInput* pAtInput = CAtInput::GetInstance();
	bool bList = true, bEnd = true;
	//-----------------
	mAddKeyFloatPair(pAtInput->m_acTotalDoseTag + 1, 
	   &(pAtInput->m_fTotalDose), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyIntPair(pAtInput->m_acAlignZTag + 1, 
	   &(pAtInput->m_iAlignZ), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyIntPair(pAtInput->m_acVolZTag + 1, 
	   &(pAtInput->m_iVolZ), 1, 10, !bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pAtInput->m_acExtZTag + 1,
           &(pAtInput->m_iExtZ), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyFloatPair(pAtInput->m_acAtBinTag + 1, 
	   pAtInput->m_afAtBin, 3, 10, bList, !bEnd);
	//-----------------
        mAddKeyFloatPair(pAtInput->m_acTiltAxisTag + 1, 
	   pAtInput->m_afTiltAxis, 2, 10, bList, !bEnd);
	//-----------------
        mAddKeyFloatPair(pAtInput->m_acTiltCorTag + 1, 
	   pAtInput->m_afTiltCor, 2, 10, bList, !bEnd);
	//-----------------
        mAddKeyIntPair(pAtInput->m_acAtPatchTag + 1, 
	   pAtInput->m_aiAtPatches, 2, 10, bList, !bEnd);
	//-----------------
        mAddKeyIntPair(pAtInput->m_acAlignTag + 1, 
	   &(pAtInput->m_iAlign), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyFloatPair(pAtInput->m_acReconRangeTag + 1, 
	   pAtInput->m_afReconRange, 2, 10, bList, !bEnd);
	//-----------------
        mAddKeyIntPair(pAtInput->m_acFlipVolTag + 1, 
	   &(pAtInput->m_iFlipVol), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyIntPair(pAtInput->m_acSartTag + 1, 
	   pAtInput->m_aiSartParam, 2, 10, !bList, !bEnd);
	//-----------------
        mAddKeyIntPair(pAtInput->m_acWbpTag + 1, 
	   &(pAtInput->m_iWbp), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyFloatPair(pAtInput->m_acDarkTolTag + 1, 
	   &(pAtInput->m_fDarkTol), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyIntPair(pAtInput->m_acOutImodTag + 1, 
	   &(pAtInput->m_iOutImod), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyFloatPair(pAtInput->m_acAmpContrastTag + 1, 
	   &(pAtInput->m_fAmpContrast), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyFloatPair(pAtInput->m_acExtPhaseTag + 1, 
	   pAtInput->m_afExtPhase, 2, 10, bList, !bEnd);
	//-----------------
	mAddKeyIntPair(pAtInput->m_acDfHandTag + 1,
	   &(pAtInput->m_iDfHand), 1, 10, !bList, !bEnd);
	//-----------------
        mAddKeyIntPair(pAtInput->m_acCorrCTFTag + 1, 
	   pAtInput->m_aiCorrCTF, 2, 10, bList, bEnd);
}


void CAreTomo3Json::mAddKeyValPair
(	const char* pcKey,
	const char* pcVal,
	int iNumSpaces,
	bool bList,
	bool bEnd
)
{	char acKey[256] = {'\0'};
	mCreateKey(pcKey, iNumSpaces, acKey);
	//-----------------
	char acVal[256] = {'\0'};
	if(bList) mAddStrList(pcVal, bEnd, acVal);
	else mAddStrVal(pcVal, bEnd, acVal);
	//-----------------
	strcat(m_pcJson, acKey);
	strcat(m_pcJson, acVal);
}

void CAreTomo3Json::mAddKeyFloatPair
(	const char* pcKey,
	float* pfVals,
	int iNumVals,
	int iNumSpaces,
	bool bList,
	bool bEnd
)
{	char acKey[256] = {'\0'};
	mCreateKey(pcKey, iNumSpaces, acKey);
	//-----------------
	char acVal[256] = {'\0'};
	if(bList)
	{	mAddFloatList(pfVals, iNumVals, bEnd, acVal);
	}
	else
	{	mAddFloatVal(pfVals[0], bEnd, acVal);
	}
	//-----------------
	strcat(m_pcJson, acKey);
	strcat(m_pcJson, acVal);
}

void CAreTomo3Json::mAddKeyIntPair
(       const char* pcKey,
        int* piVals,
        int iNumVals,
        int iNumSpaces,
        bool bList,
        bool bEnd
)
{       char acKey[256] = {'\0'};
        mCreateKey(pcKey, iNumSpaces, acKey);
        //-----------------
        char acVal[256] = {'\0'};
        if(bList)
        {       mAddIntList(piVals, iNumVals, bEnd, acVal);
        }
        else
        {       mAddIntVal(piVals[0], bEnd, acVal);
        }
        //-----------------
        strcat(m_pcJson, acKey);
        strcat(m_pcJson, acVal);
}

void CAreTomo3Json::mCreateKey(const char* pcKey, int iNumSpaces, char* pcRet)
{
	char acKey[256] = {'\0'};
	mAddFrontSpaces("\"", iNumSpaces, acKey);
	//-----------------
	strcpy(pcRet, acKey);
	strcat(pcRet, pcKey);
	strcat(pcRet, "\": ");
}

void CAreTomo3Json::mAddStrVal(const char* pcVal, bool bEnd, char* pcRet)
{
        strcat(pcRet, "\"");
        if(pcVal != 0L) strcat(pcRet, pcVal);
        strcat(pcRet, "\"");
        if(bEnd) strcat(pcRet, "\n");
        else strcat(pcRet, ",\n");
}

void CAreTomo3Json::mAddStrList(const char* pcList, bool bEnd, char* pcRet)
{
	strcat(pcRet, "[");
	if(pcList == 0L || strlen(pcList) == 0)
	{	if(bEnd) strcat(pcRet, "]\n");
		else strcat(pcRet, "],\n");
		return;
	}
	//-------------
	char acBuf[256] = {'\0'};
        strcpy(acBuf, pcList);
        //-----------------
        int iCount = 0;
        char* pcToken = strtok(acBuf, ", ");
        while(true)
        {       strcat(pcRet, "\"");
		strcat(pcRet, pcToken);
		strcat(pcRet, "\"");
		//----------------
                pcToken = strtok(0L, ", ");
		if(pcToken == 0L) break;
		else strcat(pcRet, ",");
        }
	if(bEnd) strcat(pcRet, "]\n");
	else strcat(pcRet, "],\n");
}

void CAreTomo3Json::mAddFloatVal(float fVal, bool bEnd, char* pcRet)
{
	char acVal[64] = {'\0'};
	const char* pcFormat = bEnd? "%f\n" : "%f,\n";
	sprintf(acVal, pcFormat, fVal);
	strcpy(pcRet, acVal);
}

void CAreTomo3Json::mAddFloatList
(	float* pfVals, int iNumVals, 
	bool bEnd, char* pcRet
)
{	if(iNumVals <= 0)
	{	if(bEnd) strcpy(pcRet, "[]\n");
		else strcpy(pcRet, "[],\n");
		return;
	}
	//-----------------
	strcpy(pcRet, "[");
	int iLast = iNumVals - 1;
	char acBuf[64] = {'\0'};
	for(int i=0; i<iLast; i++)
	{	mFloatToStr(pfVals[i], acBuf);
		strcat(pcRet, acBuf);
		strcat(pcRet, ",");
	}
	mFloatToStr(pfVals[iLast], acBuf);
	strcat(pcRet, acBuf);
	//-----------------
	if(bEnd) strcat(pcRet, "]\n");
	else strcat(pcRet, "],\n");
}

void CAreTomo3Json::mAddIntVal(int iVal, bool bEnd, char* pcRet)
{
	char acVal[64] = {'\0'};
	const char* pcFormat = bEnd ? "%d\n" : "%d,\n";
	sprintf(acVal, pcFormat, iVal);
	strcpy(pcRet, acVal);
}

void CAreTomo3Json::mAddIntList
(	int* piVals, int iNumVals,
	bool bEnd, char* pcRet
)
{	if(iNumVals <= 0)
	{	if(bEnd) strcpy(pcRet, "[]\n");
		else strcpy(pcRet, "[],\n");
		return;
	}
	//-----------------
	strcpy(pcRet, "[");
	int iLast = iNumVals - 1;
	char acBuf[64] = {'\0'};
	for(int i=0; i<iLast; i++)
	{	mIntToStr(piVals[i], acBuf);
		strcat(pcRet, acBuf);
		strcat(pcRet, ",");
	}
	mIntToStr(piVals[iLast], acBuf);
	strcat(pcRet, acBuf);
	//-----------------
	if(bEnd) strcat(pcRet, "]\n");
	else strcat(pcRet, "],\n");	
}

void CAreTomo3Json::mAddEndBrace(int iNumSpaces, bool bEnd)
{
	char acEndBrace[256] = {'\0'};
        mAddFrontSpaces("}", iNumSpaces, acEndBrace);
	if(!bEnd) strcat(acEndBrace, ",");
	strcat(acEndBrace, "\n");
        strcat(m_pcJson, acEndBrace);
}


void CAreTomo3Json::mAddFrontSpaces
(	const char* pcStr, 
	int iNumSpaces,
	char* pcRet
)
{	char acSpaces[256] = {'\0'};
	sprintf(acSpaces, "%255s", " ");	
	//-----------------
	int iStrSize = 0;
	if(pcStr != 0L) iStrSize = strlen(pcStr);
	//-----------------
	if(iStrSize == 0) acSpaces[iNumSpaces] = '\0';
	else strcpy(&acSpaces[iNumSpaces], pcStr);
	//-----------------
	strcpy(pcRet, acSpaces);	
}

void CAreTomo3Json::mIntToStr(int iVal, char* pcRet)
{
	char acVal[64] = {'\0'};
	sprintf(acVal, "%d", iVal);
	strcpy(pcRet, acVal);
}

void CAreTomo3Json::mFloatToStr(float fVal, char* pcRet)
{
	char acVal[64] = {'\0'};
	sprintf(acVal, "%f", fVal);
	strcpy(pcRet, acVal);
}
