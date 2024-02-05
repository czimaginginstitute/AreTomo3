#include "CCommonLineInc.h"
#include <memory.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::CommonLine;

CFindTiltAxis::CFindTiltAxis(void)
{
}

CFindTiltAxis::~CFindTiltAxis(void)
{
}

float CFindTiltAxis::DoIt
(	CPossibleLines* pPossibleLines,
	CLineSet* pLineSet
)
{	m_pPossibleLines = pPossibleLines;
	m_pLineSet = pLineSet;
	m_iNumLines = m_pPossibleLines->m_iNumLines;
	//------------------------------------------
	int iLine = mDoIt();
	float fTiltAxis = m_pPossibleLines->GetLineAngle(iLine);
	return fTiltAxis;
}

int CFindTiltAxis::mDoIt(void)
{
	int iLineMax = 0;
	m_fScore = 0.0f;
	float* pfScores = new float[m_iNumLines];
	//---------------------------------------
	printf("Scores of potential tilt axes.\n");
	for(int i=0; i<m_iNumLines; i++)
	{	mFillLineSet(i);
		pfScores[i] = mCalcScore();
		if(m_fScore < pfScores[i]) 
		{	m_fScore = pfScores[i];
			iLineMax = i;
		}
		printf("...... Score: %4d  %9.5f\n", i, pfScores[i]);
	}
	//-------------------------------------------------------------
	printf("Best tilt axis: %4d, Score: %9.5f\n\n", iLineMax, m_fScore);
	if(pfScores != 0L) delete[] pfScores;
	return iLineMax;
}

void CFindTiltAxis::mFillLineSet(int iLine)
{
	for(int i=0; i<m_pPossibleLines->m_iNumProjs; i++)
	{	cufftComplex* gCmpLine = m_pLineSet->GetLine(i);
		m_pPossibleLines->GetLine(i, iLine, gCmpLine);
	}
}

float CFindTiltAxis::mCalcScore(void)
{
	CSumLines sumLines;
	bool bClean = true;
	sumLines.DoIt(m_pLineSet);
	cufftComplex* gCmpSum = sumLines.GetSum(bClean);
	//-----------------
	CCalcScore calcScore;
	float fScore = calcScore.DoIt(m_pLineSet, gCmpSum);
	if(gCmpSum != 0L) cudaFree(gCmpSum);
	return fScore;
}
