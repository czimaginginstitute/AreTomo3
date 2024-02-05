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
	strcpy(m_acInMdocTag, "-InMdoc");
	strcpy(m_acInSuffixTag, "-InSuffix");
	strcpy(m_acTmpFileTag, "-TmpFile");
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
	strcpy(m_acSerialTag, "-Serial");
	//-----------------
	m_iNumGpus = 0;
	m_piGpuIDs = 0L;
	//-----------------
	m_iKv = 300;
	m_fCs = 2.7f;
	m_fPixSize = 0.0f;
	//-----------------
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
	   "  1. Input MRC file that stores dose fractionated stacks.\n"
	   "  2. It can be a MRC file containing a single stack collected\n"
           "     in Leginon or multiple stacks collected in UcsfTomo.\n"
	   "  3. It can also be the path of a folder containing multiple\n"
	   "     MRC files when %s option is turned on.\n\n",
	   m_acInMdocTag, m_acSerialTag);
	//------------------------------
	printf("%-15s\n"
	   "  1. Output MRC file that stores the frame sum.\n"
	   "  2. It can be either a MRC file name or the prefix of a series\n"
	   "     MRC files when %s option is turned on.\n\n",
	   m_acOutDirTag, m_acSerialTag);
	//-------------------------------
	printf
	( "%-15s\n"
	  "  1. Serial-processing all MRC files in a given folder whose\n"
	  "     name should be specified following %s.\n"
	  "  2. The output MRC file name emplate should be provided\n"
	  "     folllowing %s\n"
  	  "  3. 1 - serial processing, 0 - single processing, default.\n"
	  "  4. This option is only for single-particle stack files.\n\n",
	  m_acSerialTag, m_acInMdocTag, m_acOutDirTag
	);
	//-----------------
	printf
	( "%-15s\n"
	  "  1. Pixel size in A of input stack in angstrom.\n\n",
	  m_acPixSizeTag
	);
	//-----------------
	printf
	( "%-15s\n"
	   " 1. High tension in kV needed for dose weighting.\n"
	   " 2. Default is 300.\n\n", m_acKvTag
	);
	//-----------------
	printf
	( "%-15s\n"
	  "  1. Spherical aberration in mm for CTF estimation.\n\n",
	  m_acCsTag
	);
	//-----------------
	printf
	( "%-15s\n"
	  "  1. Per frame dose in e/A2.\n\n", m_acFmDoseTag
	);
	//-----------------
	printf("%-15s\n", m_acGpuIDTag);
	printf("   GPU IDs. Default 0.\n");
	printf("   For multiple GPUs, separate IDs by space.\n");
	printf("   For example, %s 0 1 2 3 specifies 4 GPUs.\n\n",
		   m_acGpuIDTag);
	//-----------------------
}

void CInput::Parse(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
	//-----------------
	memset(m_acInMdoc, 0, sizeof(m_acInMdoc));
	memset(m_acInSuffix, 0, sizeof(m_acInSuffix));
	memset(m_acOutDir, 0, sizeof(m_acOutDir));
	memset(m_acTmpFile, 0, sizeof(m_acTmpFile));
	memset(m_acLogDir, 0, sizeof(m_acLogDir));
	//-----------------
	int aiRange[2];
	MU::CParseArgs aParseArgs;
	aParseArgs.Set(argc, argv);
	aParseArgs.FindVals(m_acInMdocTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInMdoc);
	//-----------------
	aParseArgs.FindVals(m_acInSuffixTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInSuffix);
	//-----------------
	aParseArgs.FindVals(m_acTmpFileTag, aiRange);
        aParseArgs.GetVal(aiRange[0], m_acTmpFile);
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
	mExtractInDir();
	mAddEndSlash(m_acOutDir);
	mAddEndSlash(m_acLogDir);
	mPrint();
}

void CInput::mPrint(void)
{
	printf("\n");
	printf("%-15s  %s\n", m_acInMdocTag, m_acInMdoc);
	printf("%-15s  %s\n", m_acInSuffixTag, m_acInSuffix);
	printf("%-15s  %s\n", m_acOutDirTag, m_acOutDir);
	printf("%-15s  %s\n", m_acTmpFileTag, m_acTmpFile);
	printf("%-15s  %s\n", m_acLogDirTag, m_acLogDir);
	printf("%-15s  %d\n", m_acSerialTag, m_iSerial);
	printf("%-15s  %.2f\n", m_acPixSizeTag, m_fPixSize);
	printf("%-15s  %d\n", m_acKvTag, m_iKv);
	printf("%-15s  %.2f\n", m_acCsTag, m_fCs);
	printf("%-15s  %.5f\n", m_acFmDoseTag, m_fFmDose);
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
	fileName.Setup(m_acInMdoc);
	strcpy(m_acInDir, fileName.m_acFolder);
}

void CInput::mAddEndSlash(char* pcDir)
{
	int iSize = strlen(pcDir);
	if(pcDir[iSize-1] != '/') strcat(pcDir, "/");
}
