#include "CReconInc.h"
#include <memory.h>
#include <stdio.h>

using namespace McAreTomo::AreTomo::Recon;

//----------------------------------------------------------
// giPrjSize[3]: [iProjX, iPadProjX, iAllProjs]
// giVolSize[3]: [iVolX, iPadVolX, iUpsample];
//----------------------------------------------------------
static __device__ __constant__ int giPrjSize[3];
static __device__ __constant__ int giVolSize[3];

static __global__ void mGBackProjWbp
(	float* gfPadSinogram,
	float* gfCosSin,
	bool* gbNoProjs,
	float* gfPadVolXZ2
)
{	int iX = blockIdx.x * blockDim.x + threadIdx.x;
	if(iX >= giVolSize[0]) return;
	//---------------------------
	float fX = iX - giVolSize[0] / 2.0f;
	float fZ = blockIdx.y - gridDim.y / 2.0f;
	int iPrjCentX = giPrjSize[0] / 2;
	int iPrjEndX = giPrjSize[0] - 2;
	//---------------------------
        float fInt = 0.0f;
	int i, iCount = 0;
	for(i=0; i<giPrjSize[2]; i++)
	{	if(gbNoProjs[i]) continue;
		//--------------------------
		int j = 2 * i;
		float fXp = (fX * gfCosSin[j] + fZ * gfCosSin[j + 1])
		   / giVolSize[2] + iPrjCentX + 0.5f;	
		if(fXp < 0 || fXp > iPrjEndX) continue;
		//--------------------------
		int iXp = (int)fXp;
		fXp = fXp - iXp;
		j = i * giPrjSize[1] + iXp;
		//--------------------------
		fXp = gfPadSinogram[j] * (1 - fXp) 
		   + gfPadSinogram[j+1] * fXp;
		if(fXp <= (float)-1e10) continue;
		//--------------------------
		fInt += fXp;
		iCount += 1;
        }
	//---------------------------
	i = blockIdx.y * giVolSize[1] + iX;
	if(iCount <= 0) gfPadVolXZ2[i] = 0.0f;
	else gfPadVolXZ2[i] = fInt / iCount;
}

GBackProjWbp::GBackProjWbp(void)
{
	m_gfPadVol = 0L;
	m_gfPadVol2 = 0L;
	m_pForwardFFT = 0L;
	m_pInverseFFT = 0L;
}

GBackProjWbp::~GBackProjWbp(void)
{
	this->Clean();
}

void GBackProjWbp::Clean(void)
{
	if(m_gfPadVol != 0L) cudaFree(m_gfPadVol);
	if(m_gfPadVol2 != 0L) cudaFree(m_gfPadVol2);
	if(m_pForwardFFT != 0L) delete m_pForwardFFT;
	if(m_pInverseFFT != 0L) delete m_pInverseFFT;
	m_gfPadVol = 0L;
	m_gfPadVol2 = 0L;
	m_pForwardFFT = 0L;
	m_pInverseFFT = 0L;
}

void GBackProjWbp::SetSize(int* piPadPrjSize, int* piVolSize)
{
	int iPrjX = (piPadPrjSize[0] / 2 - 1) * 2;
	int aiPrjSize[] = {iPrjX, piPadPrjSize[0], piPadPrjSize[1]};
	cudaMemcpyToSymbol(giPrjSize, aiPrjSize, sizeof(giPrjSize));
	//---------------------------
	m_aiVolSize[0] = piVolSize[0];
	m_aiVolSize[1] = piVolSize[1];
	m_aiVolSize2[0] = m_aiVolSize[0] * 2;
	m_aiVolSize2[1] = m_aiVolSize[1] * 2;
	//---------------------------
	int iPadVolX2 = (m_aiVolSize2[0] / 2 + 1) * 2;	
	int aiVolSize2[] = {m_aiVolSize2[0], iPadVolX2, 2};
	cudaMemcpyToSymbol(giVolSize, aiVolSize2, sizeof(giVolSize));
	//---------------------------
	m_aBlockDim.x = 512;
	m_aBlockDim.y = 1;
	m_aGridDim.x = (m_aiVolSize2[0] + m_aBlockDim.x - 1) / m_aBlockDim.x;
	m_aGridDim.y = m_aiVolSize2[1];
	//---------------------------
	this->Clean();
	int iPadVolX = (m_aiVolSize[0] / 2 + 1) * 2;
	size_t tBytes = sizeof(float) * iPadVolX * m_aiVolSize[1];
	cudaMalloc(&m_gfPadVol, tBytes);
	//---------------------------
	tBytes = sizeof(float) * iPadVolX2 * m_aiVolSize2[1];
	cudaMalloc(&m_gfPadVol2, tBytes);
	//---------------------------
	m_pForwardFFT = new MU::CCufft2D;
	m_pInverseFFT = new MU::CCufft2D;
	m_pForwardFFT->CreateForwardPlan(m_aiVolSize2, false);
	m_pInverseFFT->CreateInversePlan(m_aiVolSize, false);
} 

void GBackProjWbp::DoIt
(	float* gfPadSinogram,
	float* gfCosSin,  // cosine and sine of all tilt angles
	bool* gbNoProjs,
	float* gfVolXZ,
	cudaStream_t stream
)
{	MaUtil::CheckCudaError("0000000");
	mGBackProjWbp<<<m_aGridDim, m_aBlockDim, 0, stream>>>(gfPadSinogram, 
	   gfCosSin, gbNoProjs, m_gfPadVol2);
	MaUtil::CheckCudaError("1111111");
	//---------------------------
	m_pForwardFFT->Forward(m_gfPadVol2, true, stream);
	MaUtil::CheckCudaError("2222222");
	cufftComplex* gCmp2 = (cufftComplex*)m_gfPadVol2;
	cufftComplex* gCmp = (cufftComplex*)m_gfPadVol;
	//---------------------------
	int aiCmpSize[] = {m_aiVolSize[0] / 2 + 1, m_aiVolSize[1]};
	int aiCmpSize2[] = {m_aiVolSize2[0] / 2 + 1, m_aiVolSize2[1]};
	MU::GFourierResize2D ftResize2D;
	ftResize2D.DoIt(gCmp2, aiCmpSize2, gCmp, aiCmpSize, false, stream);
	MaUtil::CheckCudaError("333333");
	m_pInverseFFT->Inverse(gCmp, stream);
	MaUtil::CheckCudaError("444444");
	//---------------------------
	int aiPadSize[] = {aiCmpSize[0] * 2, aiCmpSize[1]};
	MU::CPad2D pad2D;
	pad2D.Unpad(m_gfPadVol, aiPadSize, gfVolXZ);
	MaUtil::CheckCudaError("555555");
}
