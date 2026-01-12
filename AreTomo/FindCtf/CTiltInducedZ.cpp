#include "CFindCtfInc.h"
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace McAreTomo::AreTomo;
using namespace McAreTomo::AreTomo::FindCtf;

static float s_fD2R = 0.0174532f;

//--------------------------------------------------------------------
// 1. making core size smaller than tilt size helps reducing the
//    checkerboard effect when we keep particles dark by flipping
//    the negative phase.
//--------------------------------------------------------------------   
CTiltInducedZ::CTiltInducedZ(void)
{
}

CTiltInducedZ::~CTiltInducedZ(void)
{
}

void CTiltInducedZ::Setup
(	float fTilt,
	float fTiltAxis,
	float fTilt0, 
	float fBeta0 
)
{	float fAlpha = fTilt + fTilt0;
	m_fTanAlpha = tan(fAlpha * s_fD2R);
	m_fCosTilt = tan(fTilt * s_fD2R);
	m_fTanBeta = tan(fBeta0 * s_fD2R);
	m_fCosTiltAxis = cos(fTiltAxis * s_fD2R);
	m_fSinTiltAxis = sin(fTiltAxis * s_fD2R);
}

float CTiltInducedZ::DoIt(float fDeltaX, float fDeltaY)
{
	float fRotX = fDeltaX * m_fCosTiltAxis + fDeltaY * m_fSinTiltAxis;
	float fRotY = -fDeltaX * m_fSinTiltAxis + fDeltaY * m_fCosTiltAxis;
	//--------------------------------------------------------------
	// 1) Assuming stage rotation is right-hand rotation around 
	// its tilt axis (y-axis), then regardless of z-axis pointing 
	// up (to electron source) or down (to detector), then positive 
	// rotation of positive fRotX always induced negative deltaZ.
	// 2) (fRotX, fRotZ) is with respect to the tilt axis corrected 
	// coordinate system of which y-axis is the tilt axis.
	//--------------------------------------------------------------
	float fDeltaZ = -fRotX * m_fTanAlpha 
	   + fRotY * m_fTanBeta * m_fCosTilt;
	return fDeltaZ;
}

