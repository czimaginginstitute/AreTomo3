#include "CUtilInc.h"
#include <stdio.h>
#include <string.h>

using namespace McAreTomo::AreTomo::Util;

CStrNode::CStrNode(void)
{	m_pcString = 0L;
	m_pNextNode = 0L;
}

CStrNode::~CStrNode(void)
{	if(m_pcString != 0L) delete[] m_pcString;
	if(m_pNextNode != 0L) delete m_pNextNode;
}

CStrLinkedList::CStrLinkedList(void)
{
	m_iNumNodes = 0;
	m_pHeadNode = 0L;
	m_pEndNode = 0L;
}

CStrLinkedList::~CStrLinkedList(void)
{
	mClean();
}

void CStrLinkedList::Add(char* pcString)
{
	if(m_pHeadNode == 0L)
	{	m_pHeadNode = new CStrNode;
		m_pHeadNode->m_pcString = pcString;
		m_pEndNode = m_pHeadNode;
	}
	else
	{	CStrNode* pNewNode = new CStrNode;
		pNewNode->m_pcString = pcString;
		m_pEndNode->m_pNextNode = pNewNode;
		m_pEndNode = pNewNode;
	}
	m_iNumNodes++;
} 

char* CStrLinkedList::GetString(int iNode)
{
	if(iNode >= m_iNumNodes) return 0L;
	CStrNode* pStrNode = m_pHeadNode;
	for(int i=0; i<iNode; i++)
	{	pStrNode = pStrNode->m_pNextNode;
	}
	return pStrNode->m_pcString;
}

void CStrLinkedList::mClean(void)
{
	if(m_iNumNodes = 0) return;
	delete m_pHeadNode;
	m_pHeadNode = 0L;
	m_pEndNode = 0L;
}
