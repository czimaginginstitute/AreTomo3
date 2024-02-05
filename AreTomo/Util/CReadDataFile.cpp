#include "CUtilInc.h"
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include <stdlib.h>

using namespace McAreTomo::AreTomo::Util;

CReadDataFile::CReadDataFile(void)
{
	m_iNumRows = 0;
	m_iNumCols = 0;
	m_pfData = 0L;
}

CReadDataFile::~CReadDataFile(void)
{
	this->mClean();
}

float CReadDataFile::GetData(int iRow, int iCol)
{
	return m_pfData[iRow * m_iNumCols + iCol];
}

void CReadDataFile::GetRow(int iRow, float* pfRow)
{
	memcpy(pfRow, m_pfData+iRow*m_iNumCols, sizeof(float) * m_iNumCols);
}

bool CReadDataFile::DoIt(char* pcFileName, int iNumCols)
{
	this->mClean();
	if(pcFileName == 0L || strlen(pcFileName) == 0) return false;
	//-----------------------------------------------------------
	FILE* pFile = fopen(pcFileName, "rt");
	if(pFile == 0L) return false;
	//---------------------------
	m_iNumCols = iNumCols;
	CStrLinkedList aList;
	while(!feof(pFile))
	{	char* pcBuf = new char[256];
		fgets(pcBuf, 256, pFile);
		if(pcBuf[0] != '#') aList.Add(pcBuf);
		else delete[] pcBuf;
	}
	//--------------------------
	m_iNumRows = 0;
	int iBytes = sizeof(float) * m_iNumCols;
	float* pfVals = new float[m_iNumCols];
	m_pfData = new float[aList.m_iNumNodes * m_iNumCols];
	for(int i=0; i<aList.m_iNumNodes; i++)
	{	char* pcString = aList.GetString(i);
		int iNumVals = 0;
		char* pcTok = strtok(pcString, " ,");
		if(pcTok == 0L) continue;
		pfVals[0] = (float)atof(pcTok);
		iNumVals += 1;
		for(int j=1; j<m_iNumCols; j++)
		{	pcTok = strtok(0L, " ");
			if(pcTok == 0L) break;
			pfVals[j] = atof(pcTok);
			iNumVals++;
		}
		if(iNumVals != m_iNumCols) continue;
		//----------------------------------
		memcpy(m_pfData + m_iNumRows * m_iNumCols, pfVals, iBytes);
		m_iNumRows++;
	}
	delete[] pfVals;
	//--------------
	fclose(pFile);
	return true;
}
		
void CReadDataFile::mClean(void)
{
	if(m_pfData != 0L) delete[] m_pfData;
	m_pfData = 0L;
	m_iNumRows = 0;
}	
