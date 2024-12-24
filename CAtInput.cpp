#include "CMcAreTomoInc.h"
#include "MaUtil/CMaUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo;
namespace MU = McAreTomo::MaUtil;

CAtInput* CAtInput::m_pInstance = 0L;


CAtInput* CAtInput::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CAtInput;
	return m_pInstance;
}

void CAtInput::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CAtInput::CAtInput(void)
{
	strcpy(m_acTotalDoseTag, "-TotalDose");
	strcpy(m_acTiltAxisTag, "-TiltAxis");
	strcpy(m_acAlignZTag, "-AlignZ");
	strcpy(m_acVolZTag, "-VolZ");
	strcpy(m_acExtZTag, "-ExtZ");
	strcpy(m_acAtBinTag, "-AtBin");
	strcpy(m_acTiltCorTag, "-TiltCor");
	strcpy(m_acReconRangeTag, "-ReconRange");
	strcpy(m_acAmpContrastTag, "-AmpContrast");
	strcpy(m_acExtPhaseTag, "-ExtPhase");
	strcpy(m_acFlipVolTag, "-FlipVol");
	strcpy(m_acFlipIntTag, "-FlipInt");
	strcpy(m_acSartTag, "-Sart");
	strcpy(m_acWbpTag, "-Wbp");
	strcpy(m_acAtPatchTag, "-AtPatch");
	strcpy(m_acOutXFTag, "-OutXF");
	strcpy(m_acAlignTag, "-Align");
	strcpy(m_acCropVolTag, "-CropVol");
	strcpy(m_acOutImodTag, "-OutImod");
	strcpy(m_acDarkTolTag, "-DarkTol");
	strcpy(m_acBFactorTag, "-Bft");
	strcpy(m_acIntpCorTag, "-IntpCor");
	strcpy(m_acCorrCTFTag, "-CorrCTF");
	strcpy(m_acDfHandTag, "-DfHand");
	//-----------------
	m_fTotalDose = 0.0f;
	m_afTiltAxis[0] = 0.0f;
	m_afTiltAxis[1] = 1.0f;
	m_iAlignZ = 0;
	m_iVolZ = -1;
	m_iExtZ = 300;
	m_afAtBin[0] = 1.0f;  // 1st res
	m_afAtBin[1] = 0.0f;  // no second res
	m_afAtBin[2] = 0.0f;  // no third res
	m_afTiltCor[0] = 0.0f;
	m_afTiltCor[1] = 0.0f;
	m_afReconRange[0] = -90.0f;
	m_afReconRange[1] = 90.0f;
	m_fAmpContrast = 0.07f;
	m_iFlipVol = 0;
	m_iFlipInt = 0;
	m_aiSartParam[0] = 20;
	m_aiSartParam[1] = 5;
	m_iWbp = 0;
	m_iOutXF = 0;
	m_iAlign = 1;
	m_fDarkTol = 0.7f;
	m_bIntpCor = false;
	m_iCtfTileSize = 512;
	m_aiCorrCTF[0] = 1;
	m_aiCorrCTF[1] = 15;
	m_iDfHand = 0;
	//-----------------
	memset(m_afExtPhase, 0, sizeof(m_afExtPhase));
	memset(m_aiAtPatches, 0, sizeof(m_aiAtPatches));
	memset(m_aiCropVol, 0, sizeof(m_aiCropVol));
}

CAtInput::~CAtInput(void)
{
}

void CAtInput::ShowTags(void)
{
	printf("%-10s\n", m_acTiltAxisTag);
	printf("   1. User provided angle of tilt axis in degree. If users\n"
	   "      do not provide one, AreTomo3 will search in full range.\n"
	   "   2. If users provide one and do not want AreTomo3 to refine\n"
	   "      it, add -1 after the provided tilt axis.\n"
	   "   3. Otherwise, AreTomo3 regines the provided value in a\n"
	   "      in a smaller range.\n\n");	   
	//-----------------
	printf("%-10s\n", m_acAlignZTag);
	printf("   Volume height for alignment, default 256\n\n");
	//-----------------
	printf("%-10s\n", m_acVolZTag);
	printf("   1. Volume z height for reconstrunction. It must be\n");
	printf("      greater than 0 to reconstruct a volume.\n");
	printf("   2. Default is 0, only aligned tilt series will\n");
	printf("      generated.\n\n");
	//-----------------
	printf("%-10s\n", m_acExtZTag);
        printf("   1. Extra volume z height for reconstrunction. This is\n"
	   "      the  space added to the estimated sample thickness for\n"
	   "      the final reconstruction of tomograms.\n"
	   "   2. This setting is relevant only when -VolZ -1 is set,\n"
	   "      which means users want to use the estimated sample\n"
	   "      thickness.\n\n");
	//-----------------
        printf("      greater than 0 to reconstruct a volume.\n");
        printf("   2. Default is 0, only aligned tilt series will\n");
        printf("      generated.\n\n");
	//-----------------
	printf("%-10s\n", m_acAtBinTag);
	printf("   1. Binnings for tomograms with respect to motion\n"
	   "      corrected tilt series. Users can specify two floats.\n"
	   "       2. The first number is required and 1.0 is default.\n"
	   "       3. The second number is optional. By default, the\n"
	   "      second resolution is disabled. It is actived only\n"
	   "      when users provide a different number.\n\n");
	//-----------------
	printf("%-10s\n", m_acTiltCorTag);
        printf("   1. Correct the offset of tilt angle.\n");
        printf("   2. This argument can be followed by two values. The\n"
           "      first value can be -1, 0, or 1. and the  default is 0,\n"
	   "      indicating the tilt offset is measured for alignment\n"
	   "      only  When the value is 1, the offset is applied to\n"
	   "      reconstion too. When a negative value is given, tilt\n"
	   "      is not measured not applied.\n"
           "   3. The second value is user provided tilt offset. When it\n"
	   "      is given, the measurement is disabled.\n\n");
	//-----------------
	printf("%-10s\n", m_acReconRangeTag);
	printf("   1. It specifies the min and max tilt angles from which\n");
	printf("      a 3D volume will be reconstructed. Any tilt image\n");
	printf("      whose tilt ange is outside this range is exclueded\n");
	printf("      in the reconstruction.\n\n");
	//-----------------
	printf("%-10s\n", m_acAmpContrastTag);
	printf("   1. Amplitude contrast, default 0.07\n\n");
	printf("%-10s\n", m_acExtPhaseTag);
	printf("   1. Guess of phase shift and search range in degree.\n");
	printf("   2. Only required for CTF estimation and with\n");
	printf("   3. Phase plate installed.\n\n");
	//-----------------------------------------
	printf("%-10s\n", m_acFlipVolTag);
	printf("   1. By giving a non-zero value, the reconstructed\n");
	printf("      volume is saved in xyz fashion. The default is\n");
	printf("      xzy.\n");
	//---------------------
	printf("%-10s\n"
	"  1. Flip the intensity of the volume to make structure white.\n"
	"     Default 0 means no flipping. Non-zero value flips.\n",
	m_acFlipIntTag);
	//--------------
	printf("%-10s\n", m_acSartTag);
	printf("   1. Specify number of SART iterations and number\n");
	printf("      of projections per update. The default values\n");
	printf("      are 15 and 5, respectively\n\n");
	//-----------------
	printf("%-10s\n", m_acWbpTag);
	printf("   1. By specifying 1, weighted back projection is enabled\n");
	printf("      to reconstruct volume.\n\n");
	//-----------------
	printf("%-10s\n", m_acDarkTolTag);
	printf("   1. Set tolerance for removing dark images. The range is\n"
	   "      in (0, 1). The default value is 0.7. The higher value is\n"
	   "      more restrictive.\n\n");
	//-----------------
	printf("%-10s\n", m_acOutXFTag);
	printf("   1. When set by giving no-zero value, IMOD compatible\n"
	   "      XF file will be generated.\n\n");
	//-----------------
	printf("%-10s\n", m_acOutImodTag);
	printf("   1. It generates the Imod files needed by Relion4 or Warp\n"
	   "      for subtomogram averaging. These files are saved in the\n"
	   "      subfolder named after the output MRC file name.\n"
	   "   2. 0: default, do not generate any IMod files.\n"
	   "   3. 1: generate IMod files needed for Relion 4.\n"
	   "   4. 2: generate IMod files needed for WARP.\n"
	   "   5. 3: generate IMod files when the aligned tilt series\n"
	   "         is used as the input for Relion 4 or WARP.\n\n");
	//-----------------
	printf("%-10s\n", m_acAlignTag);
	printf("   1. Skip alignment when followed by 0. This option is\n"
	   "      used when the input MRC file is an aligned tilt series.\n"
	   "      The default value is 1.\n\n");
	//-----------------
	printf("%-10s\n", m_acIntpCorTag);
	printf("   1. When enabled, the correction for information loss due\n"
	   "      to linear interpolation will be perform. The default\n"
	   "      setting value 1 enables the correction.\n\n");
	//-----------------
	printf("%-10s\n", m_acCorrCTFTag);
	printf("   1. When enabled, local CTF correction is performed on\n"
	   "      raw tilt series. By default this function is enabled.\n"
	   "   2. Passing 0 disables this function.\n");
	//-----------------
	printf("%-10s\n", m_acDfHandTag);
	printf("   1. Defocus handedness. 0 means unknown and AreTomo3\n"
	   "      will measured it.\n"
	   "   2. 1 and -1 denote positive and negative handedness,\n"
	   "      respectively. AreTomo3 will not measure it\n\n");
}

void CAtInput::Parse(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
	//------------
	int aiRange[2];
	MU::CParseArgs aParseArgs;
	aParseArgs.Set(argc, argv);
	//-----------------
	aParseArgs.FindVals(m_acTotalDoseTag, aiRange);
	if(aiRange[1] >= 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fTotalDose);
	//-----------------
	aParseArgs.FindVals(m_acTiltAxisTag, aiRange);
	if(aiRange[1] >= 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afTiltAxis);
	//-----------------
	aParseArgs.FindVals(m_acAlignZTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.FindVals(m_acAlignZTag, aiRange);
	aParseArgs.GetVals(aiRange, &m_iAlignZ);
	//-----------------
	aParseArgs.FindVals(m_acVolZTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iVolZ);
	//-----------------
	aParseArgs.FindVals(m_acExtZTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iExtZ);
	//-----------------
	aParseArgs.FindVals(m_acAtBinTag, aiRange);
	if(aiRange[1] > 3) aiRange[1] = 3;
	aParseArgs.GetVals(aiRange, m_afAtBin);
	if(m_afAtBin[0] < 1) m_afAtBin[0] = 1.0f;
	if(m_afAtBin[1] < 1) m_afAtBin[1] = 0.0f;
	if(m_afAtBin[2] < 1) m_afAtBin[2] = 0.0f;
	//-----------------
	aParseArgs.FindVals(m_acTiltCorTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afTiltCor);	
	//-----------------
	aParseArgs.FindVals(m_acReconRangeTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afReconRange);
	//-----------------
	aParseArgs.FindVals(m_acAmpContrastTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fAmpContrast);
	//-----------------
	aParseArgs.FindVals(m_acExtPhaseTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_afExtPhase);
	//----------------------------------------
	aParseArgs.FindVals(m_acFlipVolTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iFlipVol);
	//---------------------------------------
	aParseArgs.FindVals(m_acFlipIntTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iFlipInt);
	//---------------------------------------
	aParseArgs.FindVals(m_acSartTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_aiSartParam);
	if(m_aiSartParam[0] <= 0) m_aiSartParam[0] = 15;
	if(m_aiSartParam[1] < 1) m_aiSartParam[1] = 5;
	//--------------------------------------------
	aParseArgs.FindVals(m_acWbpTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iWbp);
	//-----------------------------------
	aParseArgs.FindVals(m_acAtPatchTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_aiAtPatches);
	//------------------------------------------
	aParseArgs.FindVals(m_acOutXFTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iOutXF);
	//-----------------
	aParseArgs.FindVals(m_acAlignTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iAlign);
	//-----------------
	aParseArgs.FindVals(m_acCropVolTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_aiCropVol);
	//-----------------
	aParseArgs.FindVals(m_acOutImodTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iOutImod);
	//-----------------
	aParseArgs.FindVals(m_acDarkTolTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fDarkTol);
	//-----------------
	aParseArgs.FindVals(m_acIntpCorTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	int iIntpCor = 0;
	aParseArgs.GetVals(aiRange, &iIntpCor);
	m_bIntpCor = (iIntpCor == 0) ? false : true;
	//-----------------
	aParseArgs.FindVals(m_acCorrCTFTag, aiRange);
	if(aiRange[1] > 2) aiRange[1] = 2;
	aParseArgs.GetVals(aiRange, m_aiCorrCTF);	
	//-----------------
	aParseArgs.FindVals(m_acDfHandTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iDfHand);
	//-----------------
	mPrint();	
}

bool CAtInput::bLocalAlign(void)
{
	int iNumPatches = m_aiAtPatches[0] * m_aiAtPatches[1];
	if(iNumPatches < 4) return false;
	return true;
}

int CAtInput::GetNumPatches(void)
{
	int iPatches = m_aiAtPatches[0] * m_aiAtPatches[1];
	return iPatches;
}

void CAtInput::mPrint(void)
{
	printf("Tomography input parameters\n");
	printf("---------------------------\n");
	printf("%-10s  %f\n", m_acTotalDoseTag, m_fTotalDose);
	//-----------------
	printf("%-10s  %d\n", m_acAlignZTag, m_iAlignZ);
	printf("%-10s  %d\n", m_acVolZTag, m_iVolZ);
	printf("%-10s  %d\n", m_acExtZTag, m_iExtZ);
	//-----------------
	printf("%-10s  %.2f  %.2f  %.2f\n", m_acAtBinTag, 
	   m_afAtBin[0], m_afAtBin[1], m_afAtBin[2]);
	printf("%-10s  %.2f  %.2f\n", m_acTiltAxisTag, 
	   m_afTiltAxis[0], m_afTiltAxis[1]);
	printf("%-10s  %.2f  %.2f\n", m_acTiltCorTag, m_afTiltCor[0],
	   m_afTiltCor[1]);
	printf( "%-10s  %.2f  %.2f\n", m_acReconRangeTag, 
	   m_afReconRange[0], m_afReconRange[1]);
	//-----------------
	printf("%-10s  %.2f\n", m_acAmpContrastTag, m_fAmpContrast);
	printf("%-10s  %.2f  %.2f\n", m_acExtPhaseTag, m_afExtPhase[0],
	   m_afExtPhase[1]);
	//-----------------
	printf("%-10s  %d\n", m_acFlipVolTag, m_iFlipVol);
	printf("%-10s  %d\n", m_acFlipIntTag, m_iFlipInt);
	//-----------------
	printf("%-10s  %d  %d\n", m_acSartTag, 
	   m_aiSartParam[0], m_aiSartParam[1]);
	printf("%-10s  %d\n", m_acWbpTag, m_iWbp);
	//-----------------
	printf("%-10s  %d  %d\n", m_acAtPatchTag, m_aiAtPatches[0],
	   m_aiAtPatches[1]);
	//-----------------
	printf("%-10s  %d\n", m_acOutXFTag, m_iOutXF);
	printf("%-10s  %d\n", m_acAlignTag, m_iAlign);
	printf("%-10s  %d  %d\n", m_acCropVolTag, m_aiCropVol[0],
	   m_aiCropVol[1]);
	//-----------------
	printf("%-10s  %d\n", m_acOutImodTag, m_iOutImod);
	printf("%-10s  %.2f\n", m_acDarkTolTag, m_fDarkTol);
	printf("%-10s  %d\n", m_acIntpCorTag, m_bIntpCor);
	printf("%-10s  %d %d\n", m_acCorrCTFTag, 
	   m_aiCorrCTF[0], m_aiCorrCTF[1]);
	printf("%-10s  %d\n", m_acDfHandTag, m_iDfHand);
	printf("\n");
}

