PRJHOME = $(shell pwd)
CONDA = $(HOME)/miniconda3
CUDAHOME = $(HOME)/nvidia/cuda-10.1
CUDAINC = $(CUDAHOME)/include
CUDALIB = $(CUDAHOME)/lib64
PRJINC = $(PRJHOME)/LibSrc/Include
PRJLIB = $(PRJHOME)/LibSrc/Lib
#-----------------------------
CUSRCS = ./MaUtil/GAddFrames.cu \
	./MaUtil/GCalcFRC.cu \
	./MaUtil/GCalcMoment2D.cu \
	./MaUtil/GCorrLinearInterp.cu \
	./MaUtil/GFFT1D.cu \
	./MaUtil/GFFTUtil2D.cu \
	./MaUtil/GFindMinMax2D.cu \
	./MaUtil/GFourierResize2D.cu \
	./MaUtil/GFtResize2D.cu \
	./MaUtil/GNormalize2D.cu \
	./MaUtil/GPad2D.cu \
	./MaUtil/GPartialCopy.cu \
	./MaUtil/GPhaseShift2D.cu \
	./MaUtil/GPositivity2D.cu \
	./MaUtil/GRoundEdge1D.cu \
	./MaUtil/GRoundEdge2D.cu \
	./MaUtil/GThreshold2D.cu \
	./MotionCor/MrcUtil/G90Rotate2D.cu \
	./MotionCor/MrcUtil/GApplyRefsToFrame.cu \
	./MotionCor/MrcUtil/GAugmentRef.cu \
	./MotionCor/MrcUtil/GFlip2D.cu \
	./MotionCor/MrcUtil/GInverse2D.cu \
	./MotionCor/EerUtil/GAddRawFrame.cu \
	./MotionCor/BadPixel/GCombineBadMap.cu \
	./MotionCor/BadPixel/GCorrectBad.cu \
	./MotionCor/BadPixel/GDetectHot.cu \
	./MotionCor/BadPixel/GDetectPatch.cu \
	./MotionCor/BadPixel/GLabelPatch.cu \
	./MotionCor/BadPixel/GLocalCC.cu \
	./MotionCor/Align/GCC2D.cu \
	./MotionCor/Align/GCorrelateSum2D.cu \
	./MotionCor/Align/GNormByStd2D.cu \
	./MotionCor/Correct/GCorrectPatchShift.cu \
	./MotionCor/Correct/GStretch.cu \
	./MotionCor/Correct/GWeightFrame.cu \
	./MotionCor/MotionDecon/GDeconFrame.cu \
	./MotionCor/MotionDecon/GMotionWeight.cu \
	./AreTomo/Util/GBinImage2D.cu \
	./AreTomo/Util/GCC1D.cu \
	./AreTomo/Util/GCC2D.cu \
	./AreTomo/Util/GLocalCC2D.cu \
	./AreTomo/Util/GLocalRms2D.cu \
	./AreTomo/Util/GMutualMask2D.cu \
	./AreTomo/Util/GRealCC2D.cu \
	./AreTomo/Util/GRemoveSpikes2D.cu \
	./AreTomo/Util/GRotate2D.cu \
	./AreTomo/Util/GShiftRotate2D.cu \
	./AreTomo/Util/GTiltStretch.cu \
	./AreTomo/Util/GXcf2D.cu \
	./AreTomo/Massnorm/GFlipInt2D.cu \
	./AreTomo/CommonLine/GCalcCommonRegion.cu \
	./AreTomo/CommonLine/GCoherence.cu \
	./AreTomo/CommonLine/GFunctions.cu \
	./AreTomo/CommonLine/GGenCommonLine.cu \
	./AreTomo/CommonLine/GInterpolateLineSet.cu \
	./AreTomo/CommonLine/GRemoveMean.cu \
	./AreTomo/CommonLine/GSumLines.cu \
	./AreTomo/Correct/GCorrPatchShift.cu \
	./AreTomo/DoseWeight/GDoseWeightImage.cu \
	./AreTomo/FindCtf/GBackground1D.cu \
	./AreTomo/FindCtf/GCalcCTF1D.cu \
	./AreTomo/FindCtf/GCalcCTF2D.cu \
	./AreTomo/FindCtf/GCalcSpectrum.cu \
	./AreTomo/FindCtf/GCC1D.cu \
	./AreTomo/FindCtf/GCC2D.cu \
	./AreTomo/FindCtf/GSpectralCC2D.cu \
	./AreTomo/FindCtf/GCorrCTF2D.cu \
	./AreTomo/FindCtf/GRadialAvg.cu \
	./AreTomo/FindCtf/GRemoveMean.cu \
	./AreTomo/FindCtf/GRmBackground2D.cu \
	./AreTomo/FindCtf/GRmSpikes.cu \
	./AreTomo/FindCtf/GRoundEdge.cu \
	./AreTomo/FindCtf/GExtractTile.cu \
	./AreTomo/FindCtf/GScaleSpect2D.cu \
	./AreTomo/PatchAlign/GCommonArea.cu \
	./AreTomo/PatchAlign/GExtractPatch.cu \
	./AreTomo/PatchAlign/GGenXcfImage.cu \
	./AreTomo/PatchAlign/GNormByStd2D.cu \
	./AreTomo/PatchAlign/GRandom2D.cu \
	./AreTomo/ProjAlign/GProjXcf.cu \
	./AreTomo/ProjAlign/GReproj.cu \
	./AreTomo/Recon/GBackProj.cu \
	./AreTomo/Recon/GCalcRFactor.cu \
	./AreTomo/Recon/GDiffProj.cu \
	./AreTomo/Recon/GForProj.cu \
	./AreTomo/Recon/GRWeight.cu \
	./AreTomo/Recon/GWeightProjs.cu 	
CUCPPS = $(patsubst %.cu, %.cpp, $(CUSRCS))
#------------------------------------------
SRCS = ./MaUtil/CParseArgs.cpp \
	./MaUtil/CCufft2D.cpp \
	./MaUtil/CFileName.cpp \
	./MaUtil/CPad2D.cpp \
	./MaUtil/CPeak2D.cpp \
	./MaUtil/CSaveTempMrc.cpp \
	./MaUtil/CSimpleFuncs.cpp \
	./DataUtil/CAlnSums.cpp \
	./DataUtil/CAsyncSaveVol.cpp \
	./DataUtil/CBufferPool.cpp \
	./DataUtil/CCtfResults.cpp \
	./DataUtil/CDuInstances.cpp \
	./DataUtil/CGpuBuffer.cpp \
	./DataUtil/CMcPackage.cpp \
	./DataUtil/CMrcStack.cpp \
	./DataUtil/CReadMdoc.cpp \
	./DataUtil/CStackBuffer.cpp \
	./DataUtil/CReadMdocDone.cpp \
	./DataUtil/CSaveMdocDone.cpp \
	./DataUtil/CStackFolder.cpp \
	./DataUtil/CTiltSeries.cpp \
	./DataUtil/CTsPackage.cpp \
	./DataUtil/CCtfParam.cpp \
	./DataUtil/CLogFiles.cpp \
	./DataUtil/CTimeStamp.cpp \
	./MotionCor/DataUtil/CFmGroupParam.cpp \
	./MotionCor/DataUtil/CFmIntParam.cpp \
	./MotionCor/DataUtil/CPatchShifts.cpp \
	./MotionCor/DataUtil/CStackShift.cpp \
	./MotionCor/BadPixel/CCorrectMain.cpp \
	./MotionCor/BadPixel/CDetectMain.cpp \
	./MotionCor/BadPixel/CLocalCCMap.cpp \
	./MotionCor/BadPixel/CTemplate.cpp \
	./MotionCor/Align/CAlignBase.cpp \
	./MotionCor/Align/CAlignedSum.cpp \
	./MotionCor/Align/CAlignMain.cpp \
	./MotionCor/Align/CAlignParam.cpp \
	./MotionCor/Align/CAlignStack.cpp \
	./MotionCor/Align/CDetectFeatures.cpp \
	./MotionCor/Align/CExtractPatch.cpp \
	./MotionCor/Align/CFullAlign.cpp \
	./MotionCor/Align/CGenXcfStack.cpp \
	./MotionCor/Align/CIterativeAlign.cpp \
	./MotionCor/Align/CMeasurePatches.cpp \
	./MotionCor/Align/CPatchAlign.cpp \
	./MotionCor/Align/CPatchCenters.cpp \
	./MotionCor/Align/CSaveAlign.cpp \
	./MotionCor/Align/CSimpleSum.cpp \
	./MotionCor/Align/CTransformStack.cpp \
	./MotionCor/Correct/CCorrectFullShift.cpp \
	./MotionCor/Correct/CGenRealStack.cpp \
	./MotionCor/MotionDecon/CInFrameMotion.cpp \
	./MotionCor/MrcUtil/CApplyRefs.cpp \
	./MotionCor/MrcUtil/CSumFFTStack.cpp \
	./MotionCor/TiffUtil/CLoadTiffHeader.cpp \
	./MotionCor/TiffUtil/CLoadTiffImage.cpp \
	./MotionCor/TiffUtil/CLoadTiffMain.cpp \
	./MotionCor/Util/CGroupFrames.cpp \
	./MotionCor/Util/CNextItem.cpp \
	./MotionCor/Util/CRemoveSpikes1D.cpp \
	./MotionCor/Util/CSplineFit1D.cpp \
	./MotionCor/EerUtil/CLoadEerHeader.cpp \
	./MotionCor/EerUtil/CLoadEerFrames.cpp \
	./MotionCor/EerUtil/CDecodeEerFrame.cpp \
	./MotionCor/EerUtil/CRenderMrcStack.cpp \
	./MotionCor/EerUtil/CLoadEerMain.cpp \
	./MotionCor/CLoadRefs.cpp \
	./MotionCor/CMcInstances.cpp \
	./MotionCor/CMotionCorMain.cpp \
	./AreTomo/Util/CReadDataFile.cpp \
	./AreTomo/Util/CSplitItems.cpp \
	./AreTomo/Util/CStrLinkedList.cpp \
	./AreTomo/MrcUtil/CAlignParam.cpp \
	./AreTomo/MrcUtil/CCalcStackStats.cpp \
	./AreTomo/MrcUtil/CCropVolume.cpp \
	./AreTomo/MrcUtil/CDarkFrames.cpp \
	./AreTomo/MrcUtil/CLocalAlignParam.cpp \
	./AreTomo/MrcUtil/CPatchShifts.cpp \
	./AreTomo/MrcUtil/CRemoveDarkFrames.cpp \
	./AreTomo/MrcUtil/CSaveAlignFile.cpp \
	./AreTomo/MrcUtil/CLoadAlignFile.cpp \
	./AreTomo/MrcUtil/CSaveStack.cpp \
	./AreTomo/MrcUtil/CMuInstances.cpp \
	./AreTomo/CommonLine/CCalcScore.cpp \
	./AreTomo/CommonLine/CCommonLineParam.cpp \
	./AreTomo/CommonLine/CFindTiltAxis.cpp \
	./AreTomo/CommonLine/CGenLines.cpp \
	./AreTomo/CommonLine/CLineSet.cpp \
	./AreTomo/CommonLine/CPossibleLines.cpp \
	./AreTomo/CommonLine/CRefineTiltAxis.cpp \
	./AreTomo/CommonLine/CSumLines.cpp \
	./AreTomo/CommonLine/CCommonLineMain.cpp \
	./AreTomo/Correct/CBinStack.cpp \
	./AreTomo/Correct/CCorrectUtil.cpp \
	./AreTomo/Correct/CCorrProj.cpp \
	./AreTomo/Correct/CCorrTomoStack.cpp \
	./AreTomo/Correct/CFourierCropImage.cpp \
	./AreTomo/DoseWeight/CWeightTomoStack.cpp \
	./AreTomo/FindCtf/CCtfTheory.cpp \
	./AreTomo/FindCtf/CFindCtf1D.cpp \
	./AreTomo/FindCtf/CFindCtf2D.cpp \
	./AreTomo/FindCtf/CFindCtfBase.cpp \
	./AreTomo/FindCtf/CFindCtfHelp.cpp \
	./AreTomo/FindCtf/CFindCtfMain.cpp \
	./AreTomo/FindCtf/CRefineCtfMain.cpp \
	./AreTomo/FindCtf/CFindDefocus1D.cpp\
	./AreTomo/FindCtf/CFindDefocus2D.cpp \
	./AreTomo/FindCtf/CTile.cpp \
	./AreTomo/FindCtf/CCoreTile.cpp \
	./AreTomo/FindCtf/CTsTiles.cpp \
	./AreTomo/FindCtf/CExtractTiles.cpp \
	./AreTomo/FindCtf/CTiltInducedZ.cpp \
	./AreTomo/FindCtf/CCorrImgCtf.cpp \
	./AreTomo/FindCtf/CGenAvgSpectrum.cpp \
	./AreTomo/FindCtf/CSaveCtfResults.cpp \
	./AreTomo/FindCtf/CLoadCtfResults.cpp \
	./AreTomo/FindCtf/CAlignCtfResults.cpp \
	./AreTomo/FindCtf/CSpectrumImage.cpp \
	./AreTomo/FindCtf/CCorrCtfMain.cpp \
	./AreTomo/ImodUtil/CImodUtil.cpp \
	./AreTomo/ImodUtil/CSaveCsv.cpp \
	./AreTomo/ImodUtil/CSaveTilts.cpp \
	./AreTomo/ImodUtil/CSaveXF.cpp \
	./AreTomo/ImodUtil/CSaveXtilts.cpp \
	./AreTomo/Massnorm/CFlipInt3D.cpp \
	./AreTomo/Massnorm/CLinearNorm.cpp \
	./AreTomo/Massnorm/CPositivity.cpp \
	./AreTomo/ProjAlign/CCalcReproj.cpp \
	./AreTomo/ProjAlign/CCentralXcf.cpp \
	./AreTomo/ProjAlign/CParam.cpp \
	./AreTomo/ProjAlign/CRemoveSpikes.cpp \
	./AreTomo/ProjAlign/CProjAlignMain.cpp \
	./AreTomo/PatchAlign/CDetectFeatures.cpp \
	./AreTomo/PatchAlign/CFitPatchShifts.cpp \
	./AreTomo/PatchAlign/CLocalAlign.cpp \
	./AreTomo/PatchAlign/CPatchTargets.cpp \
	./AreTomo/PatchAlign/CPatchAlignMain.cpp \
	./AreTomo/Recon/CDoBaseRecon.cpp \
	./AreTomo/Recon/CDoSartRecon.cpp \
	./AreTomo/Recon/CDoWbpRecon.cpp \
	./AreTomo/Recon/CTomoBase.cpp \
	./AreTomo/Recon/CTomoSart.cpp \
	./AreTomo/Recon/CTomoWbp.cpp \
	./AreTomo/Recon/CCalcVolThick.cpp \
	./AreTomo/Recon/CAlignMetric.cpp \
	./AreTomo/StreAlign/CStretchAlign.cpp \
	./AreTomo/StreAlign/CStretchCC2D.cpp \
	./AreTomo/StreAlign/CStretchXcf.cpp \
	./AreTomo/StreAlign/CStreAlignMain.cpp \
	./AreTomo/TiltOffset/CTiltOffsetMain.cpp \
	./AreTomo/CAtInstances.cpp \
	./AreTomo/CTsMetrics.cpp \
	./AreTomo/CAreTomoMain.cpp \
	./CInput.cpp \
	./CMcInput.cpp \
	./CAtInput.cpp \
	./CAreTomo3Json.cpp \
	./CProcessThread.cpp \
	./CMcAreTomoMain.cpp \
	./CAreTomo3.cpp \
	$(CUCPPS)
OBJS = $(patsubst %.cpp, %.o, $(SRCS))
#-------------------------------------
CC = g++ -std=c++11
CFLAG = -c -g -pthread -m64
NVCC = $(CUDAHOME)/bin/nvcc -std=c++11
CUFLAG = -Xptxas -dlcm=ca -O2 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_61,code=sm_61 
#------------------------------------------
cuda: $(CUCPPS)

compile: $(OBJS)

exe: $(OBJS)
	@$(NVCC) -g -G -m64 $(OBJS) \
	$(PRJLIB)/libmrcfile.a $(PRJLIB)/libutil.a \
	-L$(CUDALIB) -L$(CUDALIB)/stubs\
	-L$(CONDA)/lib \
	-L/usr/lib64 \
	-lcufft -lcudart -lcuda -lnvToolsExt -ltiff -lc -lm -lpthread \
	-o AreTomo3
	@echo AreTomo3 has been generated.

%.cpp: %.cu
	@echo "-----------------------------------------------"
	@$(NVCC) -cuda -cudart shared \
		$(CUFLAG) -I$(PRJINC) \
		-I$(CONDA)/include $< -o $@
	@echo $< has been compiled.

%.o: %.cpp
	@echo "------------------------------------------------"
	@$(CC) $(CFLAG) -I$(PRJINC) -I$(CUDAINC) \
		-I$(CONDA)/include \
		$< -o $@
	@echo $< has been compiled.

clean:
	@rm -f $(OBJS) $(CUCPPS) *.h~ makefile~ AreTomo3

