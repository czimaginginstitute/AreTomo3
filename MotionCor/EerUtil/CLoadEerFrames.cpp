#include "CEerUtilInc.h"
#include "../CMotionCorInc.h"
#include <memory.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>

using namespace McAreTomo::MotionCor::EerUtil;
using namespace McAreTomo::MotionCor;

CLoadEerFrames::CLoadEerFrames(void)
{
	m_pTiff = 0L;
	m_pucFrames = 0L;
	m_piFrmStarts = 0L;
	m_piFrmSizes = 0L;
}

CLoadEerFrames::~CLoadEerFrames(void)
{
	this->Clean();
}

void CLoadEerFrames::Clean(void)
{
	if(m_pucFrames != 0L) delete[] m_pucFrames;
	if(m_piFrmStarts != 0L) delete[] m_piFrmStarts;
	if(m_piFrmSizes != 0L) delete[] m_piFrmSizes;
	m_pucFrames = 0L;
	m_piFrmStarts = 0L;
	m_piFrmSizes = 0L;
	//-----------------
	if(m_pTiff == 0L) return;
	TIFFCleanup(m_pTiff);
	m_pTiff = 0L;
}

bool CLoadEerFrames::DoIt
(	int iFile,
	int iNumFrames
)
{	this->Clean();
	m_iNumFrames = iNumFrames;
	//-----------------
	lseek64(iFile, 0, SEEK_SET);
        m_pTiff = TIFFFdOpen(iFile, "\0", "r");
	if(m_pTiff == 0L) return false;
	//-----------------
	struct stat buf;
	fstat(iFile, &buf);
	unsigned int uiSize = buf.st_size;
	m_pucFrames = new unsigned char[uiSize];
	memset(m_pucFrames, 0, sizeof(char) * uiSize);
	//-----------------
	int iBytes = sizeof(int) * m_iNumFrames;
	m_piFrmStarts = new int[m_iNumFrames];
	m_piFrmSizes = new int[m_iNumFrames];
	memset(m_piFrmStarts, 0, iBytes);
	memset(m_piFrmSizes, 0, iBytes);
	//-----------------
	m_iBytesRead = 0;
	CMcInput* pInput = CMcInput::GetInstance();
	if(pInput->m_iTiffOrder >= 0)
	{	for(int i=0; i<m_iNumFrames; i++) 
		{	mReadFrame(i);
		}
	}
	else
	{	int iLastFrm = m_iNumFrames - 1;
		for(int i=iLastFrm; i>=0; i--) 
		{	mReadFrame(i);
		}
	}
	//-----------------
	TIFFCleanup(m_pTiff);
        m_pTiff = 0L;
	return true;
}

unsigned char* CLoadEerFrames::GetEerFrame(int iFrame)
{
	if(m_pucFrames == 0L) return 0L;
	return m_pucFrames + m_piFrmStarts[iFrame];
}

int CLoadEerFrames::GetEerFrameSize(int iFrame)
{
	if(m_piFrmSizes == 0L) return 0;
	return m_piFrmSizes[iFrame];
}

void CLoadEerFrames::mReadFrame(int iFrame)
{
	m_piFrmStarts[iFrame] = m_iBytesRead;
	unsigned char* pcEerFrm = m_pucFrames + m_iBytesRead;
	//---------------------------------------------------
	TIFFSetDirectory(m_pTiff, iFrame);
	int iNumStrips = TIFFNumberOfStrips(m_pTiff);
	int iFrmBytes = 0;
	//----------------
	for(int i=0; i<iNumStrips; i++)
	{	int iStripBytes = TIFFRawStripSize(m_pTiff, i);
		TIFFReadRawStrip(m_pTiff, i, pcEerFrm + iFrmBytes, iStripBytes);
		iFrmBytes += iStripBytes;
	}
	m_piFrmSizes[iFrame] = iFrmBytes;
	m_iBytesRead += iFrmBytes;
}

