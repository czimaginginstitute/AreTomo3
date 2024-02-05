#include "CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <math.h>

using namespace Util;

CTRDecompose2D::CTRDecompose2D(void)
{
}

CTRDecompose2D::~CTRDecompose2D(void)
{
}

void CTRDecompose2D::DoIt
(	float* pfPos,
	float* pfInVector,
	float* pfOutVector
)
{	pfOutVector[0] = 0.0f;
	pfOutVector[1] = 0.0f;
	//--------------------
	float fPos = pfPos[0] * pfPos[0] + pfPos[1] * pfPos[1];
	fPos = (float)sqrt(fPos);
	if(fPos == 0) return;
	//-------------------
	float afUnitPos[] = {pfPos[0] / fPos, pfPos[1] / fPos};
	float fCompR = afUnitPos[0] * pfInVector[0] 
		+ afUnitPos[1] * pfInVector[1];
	//-------------------------------------
	float afVectR[2], afVectT[2];
	afVectR[0] = fCompR * afUnitPos[0];
	afVectR[1] = fCompR * afUnitPos[1];
	afVectT[0] = pfInVector[0] - afVectR[0];
	afVectT[1] = pfInVector[1] - afVectR[1];
	float fCompT = (float)sqrtf(afVectT[0] * afVectT[0]
		+ afVectT[1] * afVectT[1]);
	float fCurlRT = afVectR[0] * afVectT[1] 
		- afVectT[0] * afVectR[1];
	if(fCurlRT < 0) fCompT = -fCompT;
	//-------------------------------
	pfOutVector[0] = fCompT;
	pfOutVector[1] = fCompR;
}

