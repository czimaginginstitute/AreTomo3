#include "CDataUtilInc.h"
#include "../CMcAreTomoInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo;
using namespace McAreTomo::DataUtil;

int CAlnSums::m_iNumSums = 3;

CAlnSums::CAlnSums(void)
{
}

CAlnSums::~CAlnSums(void)
{
	mCleanFrames();
}

void CAlnSums::Create(int* piImgSize)
{
	int aiStkSize[] = {piImgSize[0], piImgSize[1], m_iNumSums};
	CMrcStack::Create(2, aiStkSize);
}

void* CAlnSums::GetSum(void)
{
	return m_ppvFrames[0];
}

void* CAlnSums::GetSumEvn(void)
{
	return m_ppvFrames[1];
}

void* CAlnSums::GetSumOdd(void)
{
	return m_ppvFrames[2];
}

