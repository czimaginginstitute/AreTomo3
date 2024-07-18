AreTomo3 1.0.0: [01/30/2024]
----------------------------
A fully automated preprocessing pipeline that enables real time generation of cryoET tomograms with integrated correction of beam induced motion, CTF estimation, tomographic alignment and reconstruction in a single multi-GPU accelerated application.

AreTomo3 1.0.1: [02/09/2024]
----------------------------
1. MotionCor/DataUtil/CFmIntFile.cpp: Bug Fix
   It has a memory access error. Reimplemented the calculation of number of
   raw frames in each integrated frames in mCalcIntFms().
2. DataUtil/CTiltSeries.cpp::Create(...): Bug Fix
   mCleanCenters() has been moved to the first line to avoid the change
   caused of m_aiStkSize by CMrcStack::Create() 

AreTomo3 1.0.2: [02/10/2024]
----------------------------
1. CInput.cpp: Added -InSkips
   TFS data collection can produce extra mdoc files with "override" in their
   names. -InSkips can take multiple strings to exclude any mdoc files that
   contain any of these strings.
2. Bug fix: -DarkTol was hard-coded to 0.01 (for debugging).
3. AreTomo/ImodUtil/CSaveCsv.cpp: Bug fix
   The 1st column in .csv file is the acquisition order of tilt series.
   1) For Relion4 (-OutImod 1), this list must contain all the tilt images
      including
      darke images.
   2) For other 2 (-OutImod 2 or 3), this list should exclude dark images.
4. AreTomo/ImodUtil: Bugs fix
   1) in generation of .xf file. Sorted in the ordered matching the
      section index in the generated MRC tilt series file.
   2) in generation of .csv file: comma separated, no space. 1st column is
      acquisition index in ascending order, not section index.
   3) in generation of .tilt file: -OutImod 1 includes angles of all images
      including dark ones.
   4) in generation of .xtilt file. -OutImod 1 includes all images.
5. AreTomo/MrcUtil/CAlignParam, CSaveAlignFile 
   1) Added m_fAlphaOffset and m_fBetaOffset to track if the tilt angles 
      have been changed. 
   2) Save m_fAlphaOffset and m_fBetaOffset in the .aln header.
6. AreTomo/FindCtf: Bug fix
   1) When -OutImod = 1, CTF estimation needs to be done on all tilt images.
      Otherwise, it is done on dark removed images.
7. DataUtil/CReadMdoc.cpp: using 256 chars for file name instead of 128 that
   likely causes crash.

AreTomo3 1.0.3: [02-19-2024]
----------------------------
1. DataUtil/CTsPackage.cpp: Check if iAcqIdx is 0-based. If yes, add 1 to
   it before saving them into TLT file.
2. ImodUtil/CSaveCsv.cpp: When iAcqIdx becomes 1-based, it cannot be used
   as line number in m_pcOrderedList. The line number should be 0-based.

AreTomo3 1.0.4: [02-20-2024]
----------------------------
1. DataUtil/CBufferPool::Adjust: m_pPatBuffer will be NULL when patch
   align is not specified. Check NULL before calling Adjust.
2. Bug fix: DataUtil/CGpuBuffer::AdjustBuffer:
   if(iNumFrames <= m_iMaxGpuFrms)
   {	if(iNumFrames < m_iMaxGpuFrms)  <----- wrong!
	{	m_iNumGpuFrames = iNumFrames;
	} <----------------------------------- wrong!    
	m_iNumFrames = iNumFrames;
	return;
    }
3. Bug fix: DataUtil/CReadMdoc::ctr: forgot initialize m_ppcFrmPath.
4. makefile and makefile11: clean AreTomo3 

AreTomo3 1.0.5: [02-28-2024]
----------------------------
1. Reported Bugs:
   1) Some tilt series are left out without being processed.
   2) Pixel size are not saved into MRC files
   3) FlipVol is not working.
2. Fixed (1). Dropped timestamp based check of new mdoc files. It is not reliable
   since the mdoc files are the copied version.
3. Fixed (2) in Correct/CCorrTomoStack and CBinStack.
4. 03-01-2024: Bug (1) is not fixed. Add log files to save all the mdoc file names
   added to the queue and mdoc files failed in reading.

AreTomo3 1.0.6: [03-02-2024]
----------------------------
1. Bug: missing some mdoc files.
   1) A mdoc file can be incomplete when it is loaded, causing error in loading.
      In this case, it is pushed back to the queue for next loading.
   2) Added two log files MdocList.txt and MdocProcessed.txt to track both
      mdoc files loaded from directory and those being processed.
   3) The affected files are CMcAreTomoMain and CProcessThread.

AreTomo3 1.0.7: [03-06-2024]
----------------------------
1. Restored FlipVol function.
2. Segmentation Fault [03-11-2024]:
   1) It happened after processing 281 tilt series using 4 GPUs. Need to fix
      it in version 1.0.8.
   2) The MdocProcess was generated without any content. Need to flush it
      immediately.
   3) AreTomo/ProjAlign/CCalcReproj::mFindProjRange: m_aiProjRange[0] and
      m_aiProjRange[1] can be -1. Added check of -1 indices.

AreTomo3 1.0.8: [03-08-2024]
----------------------------
1. Plan to add an entry point that starts from processing tomographic tilt
   series.
2. Revision: 
   1) Renamed MdocList.txt to MdocFound.txt in CStackFolder.cpp
      to show mdoc files found the input directory. 
   2) Renamed MdocProcess.txt to MdocDone.txt in CMcAreTomoMain.cpp.
   3) flush the content immediately.
3. Add -Resume 1 to skipped the mdoc files that have been processed.
   -Resume 1 and -Cmd 0 should be used together for resumed operation.
4. Add -Cmd 1 to skip motion correction and to start from tomo
   alignment. -Resume 1 is ignored in this case.   
5. Add in AreTomo/FindCtf/GSpectralCC2D.cu to calculate Thon ring
   resolution at 0.143.

AreTomo3 1.0.9: [03-20-2024]
----------------------------
1. Added patch based CTF deconvolution
2. Bug fix: FindCtf/GCalcCTF2D.cu and GCalcCTF1D.cu: m_fAmpPhaseShift 
   calculation forgot taking square root. Corrected now.
3. Bug fix: When the gain is not provided, the motion corrected image is dark.
   MotionCor/MrcUtil/CAppyRefs::DoIt: when there is no gain and dark 
   references, we should continue. The subsequent operation copys the frames
   to GPU memory.
4. Bug fix (03-28-2024): AreTomo/FindCtf/CTile.cpp incorrectly extracts
   tiles from the image.
5. Implemented Dmitry Tegunov filter in AreTomo/FindCtf/GCorrCTF2D.cu.

AreTomo3 1.0.10: [03-30-2024]
-----------------------------
1. Revised AreTomo/FindCtf/GCorrCtf2D.cu. Invented a segmented CTF function
   to avoid large amplification near zero.
2. Added -Cmd 2 that skips motion correction, CTF estimation, and tomo
   alignment. The processing starts from loading CTF estimation and tomo
   alignment results followed by CTF correction and tomo reconstruction.
3. CTF estimation is done on full tilt series before removing dark images.

AreTomo3 1.0.11: [04-03-2024]
-----------------------------
1. Add -FlipVol 2 that rotates volume from yzx (x fastest dimention, y
   slowest dimension) to zyx without changing handedness.
2. Make -Cmd 2 more lenient: If ODD and EVN tilt series are not present,
   proceed to reconstruct full tilt series.
3. Bug fix: CAreTomoMain::mRecon: the number of patches should be retrieved
   from AreTomo/MrcUtil/CLocalAlignParam, not from CAtInput, since when
   -Cmd 2 is used, the number of patches is read from .aln file instead 
   of command line.
4. Bug fix: CTF estimation checks existence of pixel size in the object
   of DataUtil/CTiltSeries instead of CInput.

AreTomo3 1.0.12: [04-08-2024]
-----------------------------
1. Bug: When a frame integration file has an empty line at the end, AreTomo3
   fails with segmentation fault. (Fixed 04-08-2024)
2. Deconvolution kernel (FindCtf/GCorrCTF2D.cu) takes into account of less
   ocillation in higher frequency regime.
3. Added generation of a Json file containing session information.
4. Added -CorrCTF 2 for phase flipping

AreTomo3 1.0.13: [04-15-2024]
-----------------------------
1. Bug fix: -OutImod 3 does not save the aligned tilt series in Imod folder.
   Fixed on 04/15/2024.
2. Change: Save the tilt series for -OutImod 1 and 2 before doing CTF
   correction.
3. Added: Save aligned CTF resulted in Imod folder when -OutImod 3 is
   specified.
4. Added: Save unaligned but dark-removed CTF results in Imod folder when
   -OutImod 2 is specified.
5. [05-01-2024]: Revised CAreTomo3Json.cpp. For multiple arguments, use
   list even if users provide only one value. The relate option is
   -Gpu, -InSkips, -Sart, -Patch, -AtPatch 
6. [05-07-2024]: Added -Cmd 3 that repeates CTF estimation only.

AreTomo3 1.0.14: [05-15-2024]
-----------------------------
1. Bug fix (05-15-2024)
   CTomoWbp::DoIt::m_aGWeightProjs.DoIt(...):
      m_gfPadSinogram is padded, shoud use bPadded not !bPadded.
   CTomoSart::DoIt::m_aGWeightProjs.DoIt(...):
      m_gfPadSinogram is padded, should use bPadded not !bPadded.
2. Improvement on CTF estimation (05-17-2024)
   - Use multiple low-tilt CTFs rather than a single zero-tilt CTF as the
     initial values for higher tilt.
   - Increased defocus search range for CTF refinement.
3. Change (05-17-2024)
   - AreTomo/Recon/GBackProj.cu: use boolean array to specify which
     projections are included in the backprojection instead of starting
     and endind indices, a more flexible approach.
   - Added CTomoBase class as the parent class for CTomoWbp and CTomoSart.
   - Deleted Mdoc subfolder containing mdoc file examples.

AreTomo3 1.0.15: [05-23-2024]
-----------------------------
1. Bug fix (05-23-2024)
   DataUtil::CTsPackage::mLoadTiltFile: revised to load both xxx_TLT.txt
      (two columns) or xxx.rawtlt (one column).
2. Since Version 1.0.13, dark image detection takes into account Tygress
   data collection scheme.
3. Since Version 1.0.14, CTomoSart change the relaxation scheme. It depends
   on number of subsets. The more subsets are, the smaller relaxation.
4. Bug (05-30-2024)
   CTF estimation fails when pixel size is less than 1A and the defocus is
   higher than 2um. The fix is to use Fourier cropping to increase the pixel 
   size to 1A and correspondingly the Thon ring spacing. CFindCtfMain.cpp 
5. Implemented GCalcFRC.cu that calculates the FRC between a pair of 2D
   images. (05-31-2024)

AreTomo3 1.0.16: [06-17-2024]
-----------------------------
1. Bug fix: CAreTomoMain::m_fTiltOffset was not initialized, Added
   m_fTiltOffset = 0.0f in CAreTomoMain::mFindTiltOffset
2. Changes in FindCtf: Generate tiles of an entire tilt series. CTF estimation
   will be done twice, one without taking into account of focus gradient, and
   one with after tilt axis is determined. 
3. Implemented spectrum scaling based on local defocus in generating averaged
   power spectrum per tilt. [06-27-2024]
4. Implemented tile screening that excludes tiles with low standard deviation.
   This is per tilt base screening. [06-27-2024].
5. Implemented CTF based tilt angle refinement in CFindCtfMain. [07-01-2024] 
6. Bug fix: FindCtf/CFindDefocus1D::mBrutalForceSearch: incorrect calculation
   of search steps for defocus and phase. fixed on 07-09-2024 
7. Bug fix: AreTomo/Recon/CDoWbpRecon.cpp: lines 64 & 65 are debugging code.
   They should be removed. Fixed. [07-11-2024]
8. Local CTF estimation has been implemented in FindCtf/CRefineCtfMain.
8. Correct alpha offset based on local CTF estimation.

AreTomo3 1.0.17: [07-11-2024]
-----------------------------
1. Local CTF correction: CTF correction is carried out on a tile of
   which the core, the central square, is assembled into a CTF corrected
   image.
2. Local CTF correction: Rounding the edge outside the core area is done
   before CTF correction of each tile. 
3. Revised CAreTomo3Json.cpp to generate nested json file for internal
   need.
4. Revised Correct/GCorrPatchShift.cu for better randomization.
Bug fix:
1. AreTomo/Recon/CTomoBase::Setup: m_gbNoProjs is not initialized.
   Fixed on [07-15-2024].
