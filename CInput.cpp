#include "CMcAreTomoInc.h"
#include "MaUtil/CMaUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

using namespace McAreTomo;
namespace MU = McAreTomo::MaUtil;

CInput* CInput::m_pInstance = 0L;

CInput* CInput::GetInstance(void)
{
	if(m_pInstance != 0L) return m_pInstance;
	m_pInstance = new CInput;
	return m_pInstance;
}

void CInput::DeleteInstance(void)
{
	if(m_pInstance == 0L) return;
	delete m_pInstance;
	m_pInstance = 0L;
}

CInput::CInput(void)
{
	strcpy(m_acInPrefixTag, "-InPrefix");
	strcpy(m_acInSuffixTag, "-InSuffix");
	strcpy(m_acInSkipsTag, "-InSkips");
	strcpy(m_acTmpDirTag, "-TmpDir");
	strcpy(m_acLogDirTag, "-LogDir");
	strcpy(m_acOutDirTag, "-OutDir");
	//-----------------
	strcpy(m_acGpuIDTag, "-Gpu");
	//-----------------
	strcpy(m_acKvTag, "-kV");
	strcpy(m_acCsTag, "-Cs");
	strcpy(m_acPixSizeTag, "-PixSize");
	strcpy(m_acFmDoseTag, "-FmDose");
	//-----------------
	strcpy(m_acCmdTag, "-Cmd");
	strcpy(m_acResumeTag, "-Resume");
	strcpy(m_acSerialTag, "-Serial");
	//-----------------
	m_iNumGpus = 0;
	m_piGpuIDs = 0L;
	//-----------------
	m_iKv = 300;
	m_fCs = 2.7f;
	m_fPixSize = 0.0f;
	//-----------------
	m_iCmd = 0;
	m_iResume = 0;
	m_iSerial = 0;
}

CInput::~CInput(void)
{
	if(m_piGpuIDs != 0L) delete[] m_piGpuIDs;
}

void CInput::ShowTags(void)
{
	printf("     ******  Common Parameters  *****\n"); 
	printf("%-15s\n"
	   "  1. Prefix of input file name(s), ogether with Insuffix\n"
	   "     and InSkips, is used to form either a single or subset\n"
	   "     for file name(s), which are processed by AreTomo3.\n"
	   "  2. If the suffix is mdoc, any mdoc file that starts with\n"
	   "     the prefix string will be selected.\n"
	   "  3. If the suffix is mrc, any mrc file that starts with\n"
	   "     the prefix string will be selected and processed.\n"
	   "  4. The prefix can also be the path of a folder containing\n"
	   "     the movie files (tiff or eer) or tilt series (mrc).\n"
	   "  5. Note that movie files must be in the same directory\n"
	   "     as the mdoc files.\n\n", m_acInPrefixTag, m_acSerialTag);
	//-----------------
	printf("%-15s\n"
	   "  1. If MDOC files have .mdoc file name extension, then\n"
	   "     .mdoc should be given after %s. If another extension\n"
	   "     is used, it should be used instead.\n\n",
	   m_acInSuffix);
	//-----------------
	printf("%-15s\n",
	   "  1. If a MDOC file contains any string given behind %s,\n"
	   "     those MDOC files will not be processed.\n\n",
	   m_acInSkipsTag);
	//-----------------
	printf("%-15s\n"
	   "  1. Path to output folder to store generated tilt series, tomograms,\n"
	   "     and alignment files.\n\n",
	   m_acOutDirTag, m_acSerialTag);
	//-----------------
	printf("%-15s\n"
	   "  1. Pixel size in A of input stack in angstrom.\n\n",
	   m_acPixSizeTag);
	//-----------------
	printf("%-15s\n"
	   "  1. High tension in kV needed for dose weighting.\n"
	   "  2. Default is 300.\n\n", m_acKvTag);
	//-----------------
	printf("%-15s\n"
	   "  1. Spherical aberration in mm for CTF estimation.\n\n",
	   m_acCsTag);
	//-----------------
	printf("%-15s\n"
	   "  1. Per frame dose in e/A2.\n\n", m_acFmDoseTag);
	//-----------------
	printf("%-15s\n"
	   "  1. Default 0 starts processing from motion correction.\n"
	   "  2. -Cmd 1 starts processing from tilt series alignment\n"
	   "     including CTF estimation, correction, tomographic\n"
	   "     alignment and reconstruction.\n"
	   "  3. -Cmd 2 starts processing from CTF correction and\n"
	   "     then tomographic reconstruction.\n"
	   "  4. -Cmd 1 and -Cmd 2 ignore -Resume.\n\n",
	   m_acCmdTag);
	//-----------------
	printf("%-15s\n"
	   "  1. Default 0 processes all the data.\n"
	   "  2. -Resume 1 starts from what are left by skipping all the mdoc\n"
	   "     files in MdocDone.txt file in the output folder.\n\n",
	   m_acResumeTag); 
	//-----------------
	printf("%-15s\n", m_acGpuIDTag);
	printf("   GPU IDs. Default 0.\n");
	printf("   For multiple GPUs, separate IDs by space.\n");
	printf("   For example, %s 0 1 2 3 specifies 4 GPUs.\n\n",
		   m_acGpuIDTag);
}

void CInput::Parse(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
	//-----------------
	memset(m_acInPrefix, 0, sizeof(m_acInPrefix));
	memset(m_acInSuffix, 0, sizeof(m_acInSuffix));
	memset(m_acInSkips, 0, sizeof(m_acInSkips));
	memset(m_acOutDir, 0, sizeof(m_acOutDir));
	memset(m_acTmpDir, 0, sizeof(m_acTmpDir));
	memset(m_acLogDir, 0, sizeof(m_acLogDir));
	//-----------------
	int aiRange[2];
	MU::CParseArgs aParseArgs;
	aParseArgs.Set(argc, argv);
	aParseArgs.FindVals(m_acInPrefixTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInPrefix);
	//-----------------
	aParseArgs.FindVals(m_acInSuffixTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInSuffix);
	//-----------------
	aParseArgs.FindVals(m_acInSkipsTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInSkips);
	//-----------------
	aParseArgs.FindVals(m_acTmpDirTag, aiRange);
        aParseArgs.GetVal(aiRange[0], m_acTmpDir);
	//-----------------
	aParseArgs.FindVals(m_acLogDirTag, aiRange);
        aParseArgs.GetVal(aiRange[0], m_acLogDir);
	//-----------------
	aParseArgs.FindVals(m_acOutDirTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acOutDir);
	//-----------------
	aParseArgs.FindVals(m_acPixSizeTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fPixSize);
	//-----------------
	aParseArgs.FindVals(m_acKvTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iKv);
	//-----------------
	aParseArgs.FindVals(m_acCsTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fCs);
	//-----------------
	aParseArgs.FindVals(m_acFmDoseTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fFmDose);
	//-----------------
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iSerial);
	//-----------------
	if(m_piGpuIDs != 0L) delete[] m_piGpuIDs;
	aParseArgs.FindVals(m_acGpuIDTag, aiRange);
	if(aiRange[1] >= 1)
	{	m_iNumGpus = aiRange[1];
		m_piGpuIDs = new int[m_iNumGpus];
		aParseArgs.GetVals(aiRange, m_piGpuIDs);
	}
	else
	{	m_iNumGpus = 1;
		m_piGpuIDs = new int[m_iNumGpus];
		m_piGpuIDs[0] = 0;
	}
	//-----------------
	aParseArgs.FindVals(m_acSerialTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iSerial);
	//-----------------
	aParseArgs.FindVals(m_acCmdTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iCmd);
	//-----------------
	aParseArgs.FindVals(m_acResumeTag, aiRange);
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iResume);
	//-----------------
	mExtractInDir();
	mAddEndSlash(m_acOutDir);
	mAddEndSlash(m_acLogDir);
	mAddEndSlash(m_acTmpDir);
	mPrint();
}

void CInput::mPrint(void)
{
	printf("\n");
	printf("%-15s  %s\n", m_acInPrefixTag, m_acInPrefix);
	printf("%-15s  %s\n", m_acInSuffixTag, m_acInSuffix);
	printf("%-15s  %s\n", m_acInSkipsTag, m_acInSkips);
	printf("%-15s  %s\n", m_acOutDirTag, m_acOutDir);
	//-----------------
	printf("%-15s  %s\n", m_acTmpDirTag, m_acTmpDir);
	printf("%-15s  %s\n", m_acLogDirTag, m_acLogDir);
	//-----------------
	printf("%-15s  %.2f\n", m_acPixSizeTag, m_fPixSize);
	printf("%-15s  %d\n", m_acKvTag, m_iKv);
	printf("%-15s  %.2f\n", m_acCsTag, m_fCs);
	printf("%-15s  %.5f\n", m_acFmDoseTag, m_fFmDose);
	//-----------------
	printf("%-15s  %d\n", m_acSerialTag, m_iSerial);
	printf("%-15s  %d\n", m_acCmdTag, m_iCmd);
	printf("%-15s  %d\n", m_acResumeTag, m_iResume);
	//-----------------
	printf("%-15s", m_acGpuIDTag);
	for(int i=0; i<m_iNumGpus; i++)
	{	printf("  %d", m_piGpuIDs[i]);
	}
	printf("\n");
	printf("\n\n");
}

void CInput::mExtractInDir(void)
{
	MU::CFileName fileName;
	fileName.Setup(m_acInPrefix);
	strcpy(m_acInDir, fileName.m_acFolder);
}

void CInput::mAddEndSlash(char* pcDir)
{
	int iSize = strlen(pcDir);
	if(pcDir[iSize-1] != '/') strcat(pcDir, "/");
}

