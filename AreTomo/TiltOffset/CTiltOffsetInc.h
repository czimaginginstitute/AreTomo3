#pragma once
#include "../CAreTomoInc.h"
#include "../MrcUtil/CMrcUtilFwd.h"
#include "../Correct/CCorrectFwd.h"
#include "../StreAlign/CStreAlignFwd.h"

namespace McAreTomo::AreTomo::TiltOffset
{
class CTiltOffsetMain
{
public:
	CTiltOffsetMain(void);
	~CTiltOffsetMain(void);
	void Setup(int iXcfBin, int iNthGpu);
	float DoIt(void);
private:
	float mSearch(int iNumSteps, float fStep, float fInitOffset);
	float mCalcAveragedCC(float fTiltOffset);
	float mCorrelate(int iRefProj, int iProj);
	//----------------------------------------
	MD::CTiltSeries* m_pTiltSeries;
	MAM::CAlignParam* m_pAlignParam;
	MAS::CStretchCC2D* m_pStretchCC2D;
	MAC::CCorrTomoStack* m_pCorrTomoStack;
};
}

namespace MAT = McAreTomo::AreTomo::TiltOffset;
