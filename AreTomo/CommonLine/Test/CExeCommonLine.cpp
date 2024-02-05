#include "CMainInc.h"
#include "Align/CAlignInc.h"
#include "Util/CUtilInc.h"
#include "MrcUtil/CMrcUtilInc.h"

using namespace CommonLine;

CMain::CMain(void)
{
}

CMain::~CMain(void)
{
}

void CMain::DoIt(void)
{
	CInput* pInput = CInput::GetInstance();
	char* pcInMrc = pInput->m_acInMrcFile;
	//------------------------------------
	MrcUtil::CLoadStack* pLoadStack = 0L;
	pLoadStack = MrcUtil::CLoadStack::GetInstance();
	pLoadStack->OpenFile(pcInMrc);
	pLoadStack->DoIt();
	MrcUtil::CTomoStack* pTomoStack = pLoadStack->GetStack(true);
	//-----------------------------------------------------------
	CProcessThread aProcessThread;
	aProcessThread.DoIt(pTomoStack);
	aProcessThread.WaitForExit(5000.0f);
}

