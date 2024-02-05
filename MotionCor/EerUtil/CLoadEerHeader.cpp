#include "CEerUtilInc.h"
#include <cstring>
#include <unistd.h>

using namespace McAreTomo::MotionCor::EerUtil;

CLoadEerHeader::CLoadEerHeader(void)
{
	m_pTiff = 0L;
	m_aiCamSize[0] = 4096;
	m_aiCamSize[1] = 4096;
	m_iNumFrames = 0;
	m_usCompression = 0;
	m_iNumBits = -1;
	m_iEerSampling = 1;
}


CLoadEerHeader::~CLoadEerHeader(void)
{
	mCleanTiff();
}

bool CLoadEerHeader::DoIt(int iFile, int iEerSampling)
{
	m_iFile = iFile;
	m_iEerSampling = iEerSampling;
	//-----------------
	lseek64(iFile, 0, SEEK_SET);
	m_pTiff = TIFFFdOpen(iFile, "\0", "r");
	if(m_pTiff == 0L)
	{	m_bLoaded = false;
		return m_bLoaded;
	}
	//-----------------
	memset(m_aiCamSize, 0, sizeof(m_aiCamSize));
	memset(m_aiFrmSize, 0, sizeof(m_aiFrmSize));
	//-----------------
	TIFFGetField(m_pTiff, TIFFTAG_IMAGEWIDTH, &m_aiCamSize[0]);
	TIFFGetField(m_pTiff, TIFFTAG_IMAGELENGTH, &m_aiCamSize[1]);
	m_iNumFrames = TIFFNumberOfDirectories(m_pTiff);
	TIFFGetField(m_pTiff, TIFFTAG_COMPRESSION, &m_usCompression);
	if(m_usCompression == 65000) m_iNumBits = 8;
	else if(m_usCompression == 65001) m_iNumBits = 7;
	else m_iNumBits = -1;
	//-----------------
	mCleanTiff();
	//-----------------
	m_bLoaded = mCheckError() ? false : true;
	if(!m_bLoaded) return m_bLoaded;
	//-----------------
	int iFact = 1;
	if(m_iEerSampling == 2) iFact = 2;
	else if(m_iEerSampling == 3) iFact = 4;
	m_aiFrmSize[0] = m_aiCamSize[0] * iFact;
	m_aiFrmSize[1] = m_aiCamSize[1] * iFact;
	return m_bLoaded;
}

bool CLoadEerHeader::mCheckError(void)
{
	bool bHasError = false;
	const char* pcErrSize = "Error: Invalid image size.";
	const char* pcErrCmp = "Error: Invalid compression.";
	if(m_aiCamSize[0] <= 0 || m_aiCamSize[0] <= 0 || m_iNumFrames <= 0)
	{	fprintf(stderr, "%s %d %d %d\n", pcErrSize,
		   m_aiCamSize[0], m_aiCamSize[1], m_iNumFrames);
		bHasError = true;
	}
	if(m_iNumBits <= 0)
	{	fprintf(stderr, "%s %d\n", pcErrCmp, m_usCompression);
		bHasError = true;
	}
	return bHasError;
}

void CLoadEerHeader::mCleanTiff(void)
{
	if(m_pTiff == 0L) return;
        TIFFCleanup(m_pTiff);
        m_pTiff = 0L;
}
