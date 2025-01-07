#include "CMrcUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace McAreTomo::AreTomo::MrcUtil;

CLoadAlignFile::CLoadAlignFile(void)
{
	m_iNumPatches = 0;
	m_fAlphaOffset = 0.0f;
	m_fBetaOffset = 0.0f;
}

CLoadAlignFile::~CLoadAlignFile(void)
{
	mClean();
}

void CLoadAlignFile::mClean(void)
{
	while(!m_aHeaderQueue.empty()) 
	{	char* pcLine = m_aHeaderQueue.front();
		m_aHeaderQueue.pop();
		if(pcLine != 0L) delete[] pcLine;
	}
	while(!m_aDataQueue.empty())
	{	char* pcLine = m_aDataQueue.front();
		m_aDataQueue.pop();
		if(pcLine != 0L) delete[] pcLine;
	}
}

bool CLoadAlignFile::DoIt(int iNthGpu)
{
	mClean();
	m_bLoaded = false;
	m_iNthGpu = iNthGpu;
	//-----------------
	char acAlnFile[256] = {'\0'};
	bool bSave = false;
	CSaveAlignFile::GenFileName(m_iNthGpu, bSave, acAlnFile);
	FILE* pFile = fopen(acAlnFile, "rt");
	if(pFile == 0L) return false;
	//-----------------
	char acBuf[256] = {'\0'};
	while(!feof(pFile))
	{	memset(acBuf, 0, sizeof(acBuf));
		char* pcRet = fgets(acBuf, 256, pFile);
		if(pcRet == 0L) break;
		else if(strlen(acBuf) < 4) continue;
		//----------------
		char* pcBuf = new char[256];
		if(acBuf[0] == '#')
		{	strcpy(pcBuf, &acBuf[1]);
			m_aHeaderQueue.push(pcBuf);
		}
		else 
		{	strcpy(pcBuf, acBuf);
			m_aDataQueue.push(pcBuf);
		}
	}
	fclose(pFile);
	//-----------------
	m_bLoaded = mParseHeader();
	if(m_bLoaded)
	{	mLoadGlobal();
		mLoadLocal();
	}
	//-----------------
	mClean();
	if(!m_bLoaded)
	{	printf("Error (GPU %d): loading alignment file\n"
		   "   %s failed.\n\n", acAlnFile);
	}
	return m_bLoaded;
}

bool CLoadAlignFile::mParseHeader(void)
{
	while(!m_aHeaderQueue.empty())
	{	char* pcLine = m_aHeaderQueue.front();
		m_aHeaderQueue.pop();
		//-------------------
		if(mParseRawSize(pcLine)) delete[] pcLine;
		else if(mParseDarkFrame(pcLine)) delete[] pcLine;
		else if(mParseNumPatches(pcLine)) delete[] pcLine;
		else if(mParseAlphaOffset(pcLine)) delete[] pcLine;
		else if(mParseBetaOffset(pcLine)) delete[] pcLine;
		else if(mParseThickness(pcLine)) delete[] pcLine;
		else delete[] pcLine;
	}
	//---------------------------------------------------------
	// 1) CDarkFrames instance's Setup must have been called
	// already prior to this method. 2) iNumAlnTilts is
	// number of tilt images after rejecting dark images.
	//---------------------------------------------------------
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance(m_iNthGpu);
	int iNumAlnTilts = pDarkFrames->GetNumAlnTilts();
	if(iNumAlnTilts <= 0) return false;
	//-----------------
	CAlignParam* pAlignParam = CAlignParam::GetInstance(m_iNthGpu); 
	pAlignParam->Create(iNumAlnTilts);
	//-----------------
	if(m_iNumPatches <= 0) return true;
	CLocalAlignParam* pLocalParam = 
	   CLocalAlignParam::GetInstance(m_iNthGpu);
	pLocalParam->Setup(iNumAlnTilts, m_iNumPatches);
	return true;
}

//--------------------------------------------------------------------
// 1. Read the size of raw tilt series saved in .aln file.
// 2. Check if it is identical to the loaded raw tilt series.
//--------------------------------------------------------------------
bool CLoadAlignFile::mParseRawSize(char* pcLine)
{
	memset(m_aiRawSize, 0, sizeof(m_aiRawSize));
	//-----------------
	char* pcRawSize = strstr(pcLine, CSaveAlignFile::m_acRawSizeTag);
	if(pcRawSize == 0L) return false;
	//-----------------
	char* pcTok = strtok(pcLine, "=");
	pcRawSize = strtok(0L, "=");
	//-----------------
	sscanf(pcRawSize, "%d %d %d", &m_aiRawSize[0],
	   &m_aiRawSize[1], &m_aiRawSize[2]);
	//-----------------
	return true;
}

//--------------------------------------------------------------------
// 1. Add each dark image to CDarkFrames.
//--------------------------------------------------------------------
bool CLoadAlignFile::mParseDarkFrame(char* pcLine)
{
	char* pcDarkFrm = strstr(pcLine, CSaveAlignFile::m_acDarkFrameTag);
	if(pcDarkFrm == 0L) return false;
	//-----------------
	char* pcTok = strtok(pcLine, "=");
	pcDarkFrm = strtok(0L, "=");
	//-----------------
	int iFrmIdx = 0, iSecIdx = 0; 
	float fTilt = 0.0f;
	sscanf(pcDarkFrm, "%d %d %f", &iFrmIdx, &iSecIdx, &fTilt);
	//-----------------
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance(m_iNthGpu);
	pDarkFrames->AddDark(iFrmIdx, iSecIdx, fTilt);
	return true;
}

bool CLoadAlignFile::mParseNumPatches(char* pcLine)
{
	char* pcNumPatches = strstr(pcLine, CSaveAlignFile::m_acNumPatchesTag);
	if(pcNumPatches == 0L) return false;
	//-----------------
	char* pcTok = strtok(pcLine, "=");
	pcNumPatches = strtok(0L, "=");
	//-----------------
	sscanf(pcNumPatches, "%d", &m_iNumPatches);
	return true;
}

bool CLoadAlignFile::mParseAlphaOffset(char* pcLine)
{
	char* pcAlphaOffset = strstr(pcLine, 
	   CSaveAlignFile::m_acAlphaOffsetTag);
	if(pcAlphaOffset == 0L) return false;
	//-----------------
	char* pcTok = strtok(pcLine, "=");
	pcAlphaOffset = strtok(0L, "=");
	//-----------------
	sscanf(pcAlphaOffset, "%f", &m_fAlphaOffset);
	return true;
}

bool CLoadAlignFile::mParseBetaOffset(char* pcLine)
{
        char* pcBetaOffset = strstr(pcLine,
           CSaveAlignFile::m_acBetaOffsetTag);
        if(pcBetaOffset == 0L) return false;
        //-----------------
        char* pcTok = strtok(pcLine, "=");
        pcBetaOffset = strtok(0L, "=");
        //-----------------
        sscanf(pcBetaOffset, "%f", &m_fBetaOffset);
        return true;
}

bool CLoadAlignFile::mParseThickness(char* pcLine)
{
	char* pcThickness = strstr(pcLine,
	   CSaveAlignFile::m_acThicknessTag);
        if(pcThickness == 0L) return false;
        //-----------------
        char* pcTok = strtok(pcLine, "=");
        pcThickness = strtok(0L, "=");
        //-----------------
        sscanf(pcThickness, "%d", &m_iThickness);
	CAlignParam* pAlnParam = CAlignParam::GetInstance(m_iNthGpu);
	pAlnParam->m_iThickness = m_iThickness;
	//-----------------
        return true;
}

void CLoadAlignFile::mLoadGlobal(void)
{
	CAlignParam* pAlignParam = CAlignParam::GetInstance(m_iNthGpu);
	CDarkFrames* pDarkFrames = CDarkFrames::GetInstance(m_iNthGpu);
	int iNumAlnTilts = pDarkFrames->GetNumAlnTilts();
	//-----------------
	int iSecIndex = 0;
	float fTilt, fTiltAxis, afShift[2];
	float GMAG, SMEAN, SFIT, SCALE, BASE;
	//-----------------
	int iNumReads = 0;
	for(int i=0; i<pAlignParam->m_iNumFrames; i++)
	{	char* pcLine = m_aDataQueue.front();
		m_aDataQueue.pop();
		//-----------------
		int iNumItems = sscanf(pcLine, "%d  %f  %f  %f  %f  "
		   "%f  %f  %f  %f  %f", &iSecIndex, &fTiltAxis, &GMAG, 
		   afShift+0, afShift+1, &SMEAN, &SFIT, &SCALE, 
		   &BASE, &fTilt);
		//----------------
		if(iNumItems == 10)
		{	pAlignParam->SetSecIndex(i, iSecIndex);
			pAlignParam->SetTilt(i, fTilt);
			pAlignParam->SetTiltAxis(i, fTiltAxis);
			pAlignParam->SetShift(i, afShift);
			iNumReads += 1;
		}
		if(pcLine != 0L) delete[] pcLine;
	}
	//-----------------
	if(iNumReads != pAlignParam->m_iNumFrames)
	{	printf("Error (GPU %d): .aln file global alignment\n"
		   "   has %d lines, %d lines are expected.\n\n", 
		   iNumReads, pAlignParam->m_iNumFrames);
	}
}

void CLoadAlignFile::mLoadLocal(void)
{
	CLocalAlignParam* pLocalParam = 
	   CLocalAlignParam::GetInstance(m_iNthGpu);
	int iSize = pLocalParam->m_iNumTilts * 
	   pLocalParam->m_iNumPatches;
	if(iSize <= 0) return;
	//-----------------
	int t = 0, p = 0, iNumReads = 0;
	float afXY[2] = {0.0f}, afS[2] = {0.0f};
	float fGood = 0.0f;
	//-----------------
	for(int i=0; i<iSize; i++)
	{	char* pcLine = m_aDataQueue.front();
		m_aDataQueue.pop();
		if(pcLine == 0L) continue;
		//----------------
		int iNumItems = sscanf(pcLine, "%d %d %f %f %f %f %f", 
		   &t, &p, &afXY[0], &afXY[1], &afS[0], &afS[1], &fGood);
		//----------------
		if(iNumItems == 7)
		{	int iGood = (int)(fGood + 0.5f);
			bool bBad = (iGood == 0) ? true : false;
			//---------------
			pLocalParam->SetCoordXY(t, p, afXY[0], afXY[1]);
			pLocalParam->SetShift(t, p, afS[0], afS[1]);
			pLocalParam->SetBad(t, p, bBad);
			iNumReads += 1;
		}
		else delete[] pcLine;
	}
	if(iNumReads < iSize)
	{	printf("Error (GPU %d): .aln file local alignment\n"
		   "   has %d lines, %d lines are expected.\n\n",
		   iNumReads, iSize);
	}
}	

