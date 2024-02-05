#pragma once
#include "../CMotionCorInc.h"
#include <tiffio.h>
#include <queue>

namespace McAreTomo::MotionCor::TiffUtil
{
class CLoadTiffHeader
{
public:
        CLoadTiffHeader(void);
        ~CLoadTiffHeader(void);
        bool DoIt(int iFile);
        int GetSizeX(void);
        int GetSizeY(void);
        int GetSizeZ(void);
        void GetSize(int* piSize, int iElems);
        int GetMode(void);
	float GetPixelSize(void);
        int GetTileSizeX(void);
        int GetTileSizeY(void);
        int GetNumTilesX(void);
	int GetNumTilesY(void);
	bool IsStrip(void);  // Read rows per strip
	bool IsTile(void);   // Read tiles
private:
	bool mReadImageSize(void);
	bool mReadMode(void);
	void mReadPixelSize(void);
	bool mReadRowsPerStrip(void);
	bool mReadTileSize(void);
	int m_aiImgSize[3];
	int m_iMode;
	float m_fPixelSize;
	int m_aiTileSize[2];
	int m_aiNumTiles[2];
	bool m_bReadImgSize;
	bool m_bReadMode;
	bool m_bRowsPerStrip;
	bool m_bTileSize;
	TIFF* m_pTiff;
};

class CLoadTiffImage
{
public:
        CLoadTiffImage(void);
        ~CLoadTiffImage(void);
        bool SetFile(int iFile);
        void* DoIt(int iNthImage);
        bool DoIt(int iNthImage, void* pvImage);
	int m_iMode;
	int m_aiSize[3];
private:
	bool mReadByStrip(int iNthImage, void* pvImage);
	bool mReadByTile(int iNthImage, void* pvImage);
        TIFF* m_pTiff;
	CLoadTiffHeader m_aLoadHeader;
        int m_iPixelBytes, m_iImgBytes;
};

class CLoadTiffMain
{
public:
	CLoadTiffMain(void);
	~CLoadTiffMain(void);
	bool DoIt(int iNthGpu); 
private:
	void mLoadHeader(void);
	void mLoadStack(void);
	void mLoadSingle(void);
	void mLoadInt(void);
	//-----------------
	int m_iNthGpu;
	int m_iMode;
	int m_aiStkSize[3];
	CLoadTiffImage* m_pLoadTiffImage;
	int m_iFile;
	bool m_bLoaded;
	float m_fLoadTime;
};

} 

