#include "CMcAreTomoInc.h"
#include "MaUtil/CMaUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo;
namespace MU = McAreTomo::MaUtil;

CMcInput* CMcInput::m_pInstance = 0L;

CMcInput* CMcInput::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CMcInput;
	return m_pInstance;
}

void CMcInput::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CMcInput::CMcInput(void)
{
	strcpy(m_acGainFileTag, "-Gain");
	strcpy(m_acDarkMrcTag, "-Dark");
	strcpy(m_acDefectFileTag, "-DefectFile");
	strcpy(m_acFmIntFileTag, "-FmIntFile");
	strcpy(m_acPatchesTag, "-McPatch");
	strcpy(m_acIterTag, "-McIter");
	strcpy(m_acTolTag, "-McTol");
	strcpy(m_acMcBinTag, "-McBin");
	strcpy(m_acGroupTag, "-Group");
	strcpy(m_acFmRefTag, "-FmRef");
	strcpy(m_acRotGainTag, "-RotGain");
	strcpy(m_acFlipGainTag, "-FlipGain");
	strcpy(m_acInvGainTag, "-InvGain");
	strcpy(m_acMagTag, "-Mag");
	strcpy(m_acInFmMotionTag, "-InFmMotion");
	strcpy(m_acEerSamplingTag, "-EerSampling");
	strcpy(m_acTiffOrderTag, "-TiffOrder");
	//------------------
	m_aiNumPatches[0] = 0;
	m_aiNumPatches[1] = 0;
	m_iMcIter = 15;
	m_fMcTol = 0.1f;
	m_fMcBin = 1.0f;
	m_iFmRef = -1;
	m_aiGroup[0] = 1; 
	m_aiGroup[1] = 4;
	m_iRotGain = 0;
	m_iFlipGain = 0;
	m_iInvGain = 0;
	m_afMag[0] = 1.0f;
	m_afMag[1] = 1.0f;
	m_afMag[2] = 0.0f;
	m_iInFmMotion = 0;
	m_iEerSampling = 1;
	m_iTiffOrder = 1;
	m_iCorrInterp = 0;
}

CMcInput::~CMcInput(void)
{
}

void CMcInput::ShowTags(void)
{
	printf("%-15s\n"
	   "1. Defect file stores entries of defects on camera.\n"
	   "2. Each entry corresponds to a rectangular region in image.\n"
	   "   The pixels in such a region are replaced by neighboring\n"
	   "   good pixel values.\n"
	   "3. Each entry contains 4 integers x, y, w, h representing\n"
	   "   the x, y coordinates, width, and heights, respectively.\n\n",
	   m_acDefectFileTag
	);
	//------------------
	printf("%-15s\n"
	   "1. MRC or TIFF file that stores the gain reference.\n"
	   "2. Falcon camera produced .gain file can also be used\n"
	   "   since it is a TIFF file.\n\n", m_acGainFileTag);
	//-----------------
	printf("%-15s\n", m_acDarkMrcTag);
	printf("  1. MRC file that stores the dark reference. If not\n");
	printf("     specified, dark subtraction will be skipped.\n");
	printf("  2. If -RotGain and/or -FlipGain is specified, the\n");
	printf("     dark reference will also be rotated and/or flipped.\n\n");
	//-----------------
	printf("%-15s\n" 
	   "  1. It is followed by numbers of patches in x and y dimensions.\n"
	   "  2. The default values are 1 1, meaning only full-frame\n"
	   "     based alignment is performed.\n\n", m_acPatchesTag);
	//-----------------
	printf("%-15s\n", m_acIterTag);
	printf(
	   "   Maximum iterations for iterative alignment,\n"
	   "   default 7 iterations.\n\n");
	//-----------------
	printf("%-15s\n", m_acTolTag);
	printf("   Tolerance for iterative alignment,\n");
	printf("   default 0.5 pixel.\n\n");
	//-----------------
	printf("%-15s\n", m_acMcBinTag);
	printf("   Binning performed in Fourier space, default 1.0.\n\n");
	//-----------------
	printf("%-15s\n", m_acGroupTag);
	printf("   1. Group every specified number of frames by adding\n");
	printf("      them together. The alignment is then performed\n");
	printf("      on the group sums. The so measured motion is\n");
	printf("      interpolated to each raw frame.\n");
	printf("   2. The 1st integer is for gobal alignment and the\n");
	printf("      2nd is for patch alignment.\n\n");
	//-----------------------------------------------
	printf("%-15s\n", m_acFmRefTag);
	printf("   Specify a frame in the input movie stack to be the\n");
	printf("   reference to which all other frames are aligned. The\n");
	printf("   reference is 1-based index in the input movie stack\n");
	printf("   regardless how many frames will be thrown. By default\n");
	printf("   the reference is set to be the central frame.\n\n");
	//-------------------------------------------------------------
	printf("%-15s\n", m_acRotGainTag);
	printf("   Rotate gain reference counter-clockwise.\n");
	printf("   0 - no rotation, default,\n");
	printf("   1 - rotate 90 degree,\n");
	printf("   2 - rotate 180 degree,\n");
	printf("   3 - rotate 270 degree.\n\n");
	//--------------------------------------
	printf("%-15s\n", m_acFlipGainTag);
	printf("   Flip gain reference after gain rotation.\n");
	printf("   0 - no flipping, default,\n");
	printf("   1 - flip upside down, \n");
	printf("   2 - flip left right.\n\n");
	//------------------------------------
	printf("%-15s\n", m_acInvGainTag);
	printf("   Inverse gain value at each pixel (1/f). If a orginal\n");
	printf("   value is zero, the inversed value is set zero.\n");
	printf("   This option can be used together with flip and\n");
	printf("   rotate gain reference.\n\n");
	//--------------------------------------
	printf("%-15s\n", m_acMagTag);
	printf("   1. Correct anisotropic magnification by stretching\n");
	printf("      image along the major axis, the axis where the\n");
	printf("      lower magificantion is detected.\n");
	printf("   2. Three inputs are needed including magnifications\n");
	printf("      along major and minor axes and the angle of the\n");
	printf("      major axis relative to the image x-axis in degree.\n");
	printf("   3. By default no correction is performed.\n\n");
	//---------------------------------------------------------
	printf("%-15s\n", m_acInFmMotionTag);
	printf("   1. 1 - Account for in-frame motion.\n");
	printf("      0 - Do not account for in-frame motion.\n\n");
}

void CMcInput::Parse(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
	//------------
	memset(m_acGainFile, 0, sizeof(m_acGainFile));
	memset(m_acDarkMrc, 0, sizeof(m_acDarkMrc));
	memset(m_acDefectFile, 0, sizeof(m_acDefectFile));
	memset(m_acFmIntFile, 0, sizeof(m_acFmIntFile));
	//-----------------
	int aiRange[2];
	MU::CParseArgs aParseArgs;
	aParseArgs.Set(argc, argv);
	//-----------------
	aParseArgs.FindVals(m_acFmIntFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acFmIntFile);
	//-----------------
	aParseArgs.FindVals(m_acGainFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acGainFile);
	//-----------------
	aParseArgs.FindVals(m_acDarkMrcTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acDarkMrc);
	//-----------------
	aParseArgs.FindVals(m_acDefectFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acDefectFile);
	//-----------------
	aParseArgs.FindVals(m_acEerSamplingTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iEerSampling);
	//-----------------
	aParseArgs.FindVals(m_acPatchesTag, aiRange);
	aParseArgs.GetVals(aiRange, m_aiNumPatches);
	if(m_aiNumPatches[0] <= 1) m_aiNumPatches[0] = 0;
	if(m_aiNumPatches[1] <= 1) m_aiNumPatches[1] = 0;
	if(m_aiNumPatches[2] < 0) m_aiNumPatches[2] = 0;
	if(m_aiNumPatches[2] > 100) m_aiNumPatches[2] = 100;
	//-----------------
	aParseArgs.FindVals(m_acIterTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iMcIter);
	//-----------------
	aParseArgs.FindVals(m_acTolTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fMcTol);
	//-----------------
	aParseArgs.FindVals(m_acMcBinTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fMcBin);
	//-----------------
	aParseArgs.FindVals(m_acGroupTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_aiGroup);
	if(m_aiGroup[0] < 1) m_aiGroup[0] = 1;
	if(m_aiGroup[1] < 1) m_aiGroup[1] = 1;
	//-----------------
	aParseArgs.FindVals(m_acFmRefTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iFmRef);
	//-----------------
	aParseArgs.FindVals(m_acRotGainTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iRotGain);
	//---------------------------------------
	aParseArgs.FindVals(m_acFlipGainTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iFlipGain);
	//----------------------------------------
	aParseArgs.FindVals(m_acInvGainTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iInvGain);
	//---------------------------------------
	m_afMag[0] = 1.0f;
	m_afMag[1] = 1.0f;
	aParseArgs.FindVals(m_acMagTag, aiRange);
	if(aiRange[1] == 3) aParseArgs.GetVals(aiRange, m_afMag);
	//-----------------
	m_iInFmMotion = 0;
	aParseArgs.FindVals(m_acInFmMotionTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iInFmMotion);
	//-----------------
	aParseArgs.FindVals(m_acTiffOrderTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iTiffOrder);
	//-----------------
	aParseArgs.FindVals(m_acCorrInterpTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iCorrInterp);
	mPrint();
}

void CMcInput::GetBinnedSize(int* piImgSize, int* piBinnedSize)
{
	piBinnedSize[0] = piImgSize[0];
	piBinnedSize[1] = piImgSize[1];
	if(m_fMcBin <= 0) return;
	//----------------------------
	piBinnedSize[0] = (int)(piImgSize[0] / m_fMcBin + 0.5f);
	piBinnedSize[1] = (int)(piImgSize[1] / m_fMcBin + 0.5f);
	piBinnedSize[0] = piBinnedSize[0] / 2  * 2;
	piBinnedSize[1] = piBinnedSize[1] / 2 * 2;
}

float CMcInput::GetFinalPixelSize(void)
{
	float fEerBin = 1.0f;
	if(m_iEerSampling == 2) fEerBin = 0.5f;
	else if(m_iEerSampling == 3) fEerBin = 0.25f;
	//-----------------
	CInput* pInput = CInput::GetInstance();
	float fFinalSize = pInput->m_fPixSize * m_fMcBin * fEerBin;
	if(m_afMag[0] > m_afMag[1]) fFinalSize /= m_afMag[0];
	else fFinalSize /= m_afMag[1];
	return fFinalSize;
}

bool CMcInput::bLocalAlign(void)
{
	int iNumPatches = m_aiNumPatches[0] * m_aiNumPatches[1];
	if(iNumPatches < 4) return false;
	else return true;
}

void CMcInput::mPrint(void)
{
	printf("Motion correction input parameters\n");
	printf("----------------------------------\n");
	printf("%-15s  %s\n", m_acFmIntFileTag, m_acFmIntFile);
	printf("%-15s  %s\n", m_acGainFileTag, m_acGainFile);
	printf("%-15s  %s\n", m_acDarkMrcTag, m_acDarkMrc);
	printf("%-15s  %s\n", m_acDefectFileTag, m_acDefectFile);
	printf("%-15s  %d\n", m_acEerSamplingTag, m_iEerSampling);
	printf("%-15s  %d  %d  %d\n", m_acPatchesTag,
           m_aiNumPatches[0], m_aiNumPatches[1], m_aiNumPatches[2]);
	printf("%-15s  %d\n", m_acIterTag, m_iMcIter);
	printf("%-15s  %.2f\n", m_acTolTag, m_fMcTol);
	printf("%-15s  %.2f\n", m_acMcBinTag, m_fMcBin);
	printf("%-15s  %d  %d\n", m_acGroupTag, m_aiGroup[0], m_aiGroup[1]);
	printf("%-15s  %d\n", m_acFmRefTag, m_iFmRef);
	printf("%-15s  %d\n", m_acRotGainTag, m_iRotGain);
	printf("%-15s  %d\n", m_acFlipGainTag, m_iFlipGain);
	printf("%-15s  %d\n", m_acInvGainTag, m_iInvGain);
	printf("%-15s  %.2f  %.2f  %.2f\n", m_acMagTag, m_afMag[0],
	   m_afMag[1], m_afMag[2]);
	printf("%-15s  %d\n", m_acInFmMotionTag, m_iInFmMotion);
	printf("%-15s  %d\n", m_acTiffOrderTag, m_iTiffOrder);
	printf("%-15s  %d\n", m_acCorrInterpTag, m_iCorrInterp);
	printf("\n\n");
}
