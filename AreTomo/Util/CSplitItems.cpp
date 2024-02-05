#include "CUtilInc.h"
#include <stdio.h>

using namespace McAreTomo::AreTomo::Util;

CSplitItems::CSplitItems(void)
{
	m_piStarts = 0L;
	m_piSizes = 0L;
}

CSplitItems::~CSplitItems(void)
{
	this->Clean();
}

void CSplitItems::Clean(void)
{
	if(m_piStarts != 0L) delete[] m_piStarts;
	if(m_piSizes != 0L) delete[] m_piSizes;
	m_piStarts = 0L;
	m_piSizes = 0L;
}

void CSplitItems::Create(int iNumItems, int iNumSplits)
{
	this->Clean();
	//------------
	m_iNumItems = iNumItems;
	m_iNumSplits = iNumSplits;
	m_piStarts = new int[m_iNumSplits];
	m_piSizes = new int[m_iNumSplits];
	//--------------------------------
	int iSize = m_iNumItems / m_iNumSplits;
	for(int i=0; i<m_iNumSplits; i++)
	{	m_piSizes[i] = iSize;
	}
	//---------------------------
	int iLeft = m_iNumItems % m_iNumSplits;
	for(int i=0; i<iLeft; i++)
	{	m_piSizes[i] += 1;
	}
	//------------------------
	m_piStarts[0] = 0;
	for(int i=1; i<m_iNumSplits; i++)
	{	m_piStarts[i] = m_piStarts[i-1] + m_piSizes[i-1];
	}
}

int CSplitItems::GetStart(int iSplit)
{
	return m_piStarts[iSplit];
}

int CSplitItems::GetSize(int iSplit)
{
	return m_piSizes[iSplit];
}

