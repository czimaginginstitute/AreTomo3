#include "CInput.h"
#include "../Util/CUtilInc.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>

CInput* CInput::m_pInstance = 0L;

CInput* CInput::GetInstance(void)
{
	if(m_pInstance == 0L) m_pInstance = new CInput;
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
	strcpy(m_acInMrcTag, "-InMrc");
	strcpy(m_acOutMrcTag, "-OutMrc");
	strcpy(m_acTmpFileTag, "-TmpFile");
	strcpy(m_acLogFileTag, "-LogFile");
	strcpy(m_acAngStepTag, "-AngStep");
	strcpy(m_acNumStepsTag, "-NumSteps");
	strcpy(m_acInitTiltAxisTag, "-TiltAxis");
	//---------------------------------------
	m_bTiltAxis = false;
	m_fInitTiltAxis = 0.0f;
	m_fAngStep = 0.5f;
	m_iNumSteps = 20;
}

CInput::~CInput(void)
{
}

void CInput::ShowTags(void)
{
	printf
	(  "%-10s\n"
	   "  1. Input MRC file that stores tomo tilt series.\n\n",
	   m_acInMrcTag
	);
	//-------------
        printf
	(  "%-10s\n"
	   "  1. Output MRC file that stores the aligned tilt series.\n\n",
	   m_acOutMrcTag
	);
	//--------------
        printf
	(  "%-10s\n"
	   "  1. Temporary image file for debugging.\n\n",
	   m_acTmpFileTag
	);
	//---------------
	printf
	(  "%-10s\n"
	   "  1. Log file storing alignment data.\n\n",
	   m_acLogFileTag
	);
	//---------------
	printf("%-10s\n", m_acInitTiltAxisTag);
	printf("   Initial tilt axis used for searching.\n\n");
	//-----------------------------------------------------
	printf("%-10s\n", m_acAngStepTag);
	printf("   Angular step in searching tilt axis.\n\n");
	//----------------------------------------------------
	printf("%-10s\n", m_acNumStepsTag);
	printf("   Number of steps in searching tilt axis.\n\n");
}

void CInput::Parse(int argc, char* argv[])
{
	m_argc = argc;
	m_argv = argv;
	//------------
	memset(m_acInMrcFile, 0, sizeof(m_acInMrcFile));
	memset(m_acOutMrcFile, 0, sizeof(m_acOutMrcFile));
	memset(m_acTmpFile, 0, sizeof(m_acTmpFile));
	memset(m_acLogFile, 0, sizeof(m_acLogFile));
	//------------------------------------------
	int aiRange[2];
	Util::CParseArgs aParseArgs;
	aParseArgs.Set(argc, argv);
	aParseArgs.FindVals(m_acInMrcTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acInMrcFile);
	//-------------------------------------------
	aParseArgs.FindVals(m_acOutMrcTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acOutMrcFile);
	//--------------------------------------------
	aParseArgs.FindVals(m_acTmpFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acTmpFile);
	//-----------------------------------------
	aParseArgs.FindVals(m_acLogFileTag, aiRange);
	aParseArgs.GetVal(aiRange[0], m_acLogFile);
	//-----------------------------------------
	aParseArgs.FindVals(m_acInitTiltAxisTag, aiRange);
	if(aiRange[0] < 0) m_bTiltAxis = false;
	else m_bTiltAxis = true;
	if(aiRange[1] > 1) aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fInitTiltAxis);
	//--------------------------------------------
	aParseArgs.FindVals(m_acAngStepTag, aiRange);
	aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_fAngStep);
	//---------------------------------------
	aParseArgs.FindVals(m_acNumStepsTag, aiRange);
	aiRange[1] = 1;
	aParseArgs.GetVals(aiRange, &m_iNumSteps); 
	//----------------------------------------
	mPrint();	
}

void CInput::mPrint(void)
{
	printf("\n");
	printf("%-10s  %s\n", m_acInMrcTag, m_acInMrcFile);
	printf("%-10s  %s\n", m_acOutMrcTag, m_acOutMrcFile);
	printf("%-10s  %s\n", m_acTmpFileTag, m_acTmpFile);
	printf("%-10s  %s\n", m_acLogFileTag, m_acLogFile);
	printf("%-10s  %f\n", m_acAngStepTag, m_fAngStep);
	printf("%-10s  %d\n", m_acNumStepsTag, m_iNumSteps);
	if(m_bTiltAxis)
	{	printf("%-10s  %f\n", m_acInitTiltAxisTag, m_fInitTiltAxis);
	}
	printf("\n\n");
}
