#pragma once

class CInput
{
public:
	static CInput* GetInstance(void);
	static void DeleteInstance(void);
        ~CInput(void);
	void ShowTags(void);
        void Parse(int argc, char* argv[]);
        char m_acInMrcFile[256];
        char m_acOutMrcFile[256];
        char m_acTmpFile[256];
	char m_acLogFile[256];
	float m_fAngStep;
	int m_iNumSteps;
	float m_fInitTiltAxis;
	bool m_bTiltAxis;
	//---------------
        char m_acInMrcTag[32];
        char m_acOutMrcTag[32];
        char m_acTmpFileTag[32];
	char m_acLogFileTag[32];
	char m_acInitTiltAxisTag[32];
	char m_acAngStepTag[32];
	char m_acNumStepsTag[32];
private:
	CInput(void);
        void mPrint(void);
        int m_argc;
        char** m_argv;
	static CInput* m_pInstance;
};

