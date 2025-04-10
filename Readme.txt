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

AreTomo3 1.0.18: [07-22-2024]
-----------------------------
1. Implemented multi-res reconstruction based on user input -AtBin,
   which will take two floating numbers. The volume reconstructed
   with the second binning has _2ND_Vol" in its file name.
2. Added CAsyncSaveVol in DataUtil to overlap saving volumes with
   reconstruction.
3. Added tools/Remap3D folder and copied alnFile.py and remap3D.py into it.

AreTomo3 1.0.19 [07-30-2024]
----------------------------
1. Goal: implement sample thickness detection scheme (07-30-2024)
2. Added GLocalCC2D.cu in AreTomo/Util for measuring sample thickness.
3. Added 3rd AreTomo binning to reconstruction 3rd volume. This volume is
   reconstrcuted with SART. [08-01-2024]

AreTomo3 1.0.20 [08-06-2024]
----------------------------
1. Finished implementation of sample-thickness estimation.
2. Adjust the tilt range in which angular offsets are estimated in
   FindCtf/CRefineCtfMain.cpp.
3. -AlignZ is optional now. When not given, the measured thickness is
   used in projection matching. (08-07-2024).
4. Added CTsMetrics.cpp in AreTomo to generate TiltSeries_Metrics.csv file 
   containing metrics for each tilt series including sample thickness, 
   tilt axis, bad local alignments, etc. (08-08-2024).

AreTomo3 1.0.21 [08-14-2024]
----------------------------
1. Goal: implement r-factor to quantify tomo alignment accuracy.
2. Bug fix: when -AtBin is followed by zeroes, AreTomo3 crashes. The fix
   is to skip the tomogram reconstruction if the binning is less than 1.
3. Bug fix: when a tilt series contains only 1 tilt image, AreTomo3 crashes
   at calculation of sample thickness using SART reconstruction. The fix is
   to set the minimum subset to 1.
4. Added a check to number of tilt images in a tilt series. The minimum 
   number is now 7.
5. Added: when -AtBin's 2nd and 3rd binnings are zeros, the corresponding 
   volumes are not reconstructed.
6. Added: when -VolZ is not shown in the command line, the measured 
   thickness is set as VolZ.

AreTomo3 1.0.22 [08-28-2024]
----------------------------
1. Bug fix: when AreTomo3 is restarted with -resume 1, the content of
   TiltSeries_Metrics.csv is erased when it is opened.
   The fix is to open with appending if -resume is enabled.
2. Bug fix: memory leak at AreTomo/PatchAlign/CDetectFeatures.cpp::136,
   pfBinnedImg is not freed.
3. Support sm_86, sm_89, sm_90 for H100 and H200.

AreTomo3 1.0.23 [09-02-2024]
----------------------------
1.  Goal: Add -InMrc for skipping mdoc files, which starts sequential tilt
    series alignment followed by reconstruction. This behaves the same as 
    -Cmd 1.
2.  Change: Load gain reference only when -Cmd 0 is present.
3.  Change: Replaced -InMdoc with -InPrefix.
4.  Change: With -InSuffix .mrc, AreTomo3 takes tilt series as input
    rather than mdoc files. In this case, motion correction is skipped.
5.  Added: -TmpDir followed by a path enables writting sample thickness
    temporary files in that folder. Updated json file.
6.  Change: FindCtf/CRefineCtfMain: low tilt threshold changed from 30
    to 20 degree.
7.  Change: CImodUtil::CreateFolder: change the mode to 2775
8.  Added new feature for determining defocus handedness. This is done in
    AreTomo/FindCtf/CRefineCtfMain.cpp. Added m_iDfHand in CCtfResults in
    DataUtil, which is used in FindCtf/CCorrImgCtf.cpp. [09-12-2024]
9.  Added DfHand, Cs, Kv in AreTomo/CTsMetrics.cpp. [09-12-2024]
10. Implemented the goal by replacing -InMdoc with -InPrefix, which is
    used with -InSuffix to select either mdoc or mrc as input.
    -InPrefix mydir/position_ -InSuffix .mrc together will process all
    position_*.mrc files in mydir directory.
    -InPrefix mydir/position_ -InSuffix .mdoc together will process all
    position_*.mdoc files in mydir directory.
11. Renamed to 1.0.24, not pushed to Github.

AreTomo3 1.0.24 [09-12-2024]
----------------------------
1.  Renamed from 1.0.23 to 1.0.24.

AreTomo3 2.0.0 [09-14-2024]
---------------------------
1. Renamed 1.0.24 to version 2.0.0.
2. Change: AreTomo/Recon/CCalcVolThick.cpp: when the CC profile has two peaks
   with the minimum in between, use the locations of the peaks as the sample
   edges if their distance is significant.
3. Changes: expanded TiltSeries_Metrics.csv to include tilt offsets, Cs, Kv,
   etc.
4. Changes: mkdir in CImodUtil.cpp uses now S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH.
   The same change has been applied to DataUtil/CLogFile.cpp.
5. Changes: CReadMdoc.cpp now extracts ExposureDose from the mdoc file.
6. TLT.txt file now has the 3rd column for per-tilt image dose.

AreTomo3 2.0.1 [10-22-2024]
---------------------------
1. Added -ExtZ entry into Json in CAreTomo3Json.cpp. [09-17-2024)
2. Bug: when -TiltCor 0 is set, alpha and beta offsets were not saved in the
   metrics CSV file. Fix: buffer them in CCtfResult.cpp, which can pass them
   to CTsMetrics.cpp. (09-18-2024)
3. The same as AreTomo3 2.0.0_09-18-2024.

AreTomo3 2.0.2 [10-23-2024]
---------------------------
1. Added -SplitSum, giving users an option to suppress the generation of 
   odd/even tilt series and volumes.
2. CInput.cpp: pixel size is set to zero by default.

AreTomo3 2.0.3 [10-25-2024]
---------------------------
1. Changes: When using measured sample thickness for AlignZ, cap it to be at
   at least 200 pixels. During the local alignment, cap it to be no more
   than 1200 in PatchAlign/CLocalAlign.cpp.
2. Bug fix: AreTomo/MrcUtil/CRemoveDarkFrames::mRemoveDarkFrames (line 108):
   memset(pcLog, 0, sizeof(char) * iSize) strangely caused the single
   instance of CInput reset to zeros. The fix is removing this line.
3. Bug fix: AreTomo/Correct/CCorrProj.cpp: removed cudaSetDevice, which
   caused segmentation fault.
4. Bug fix: DataUtil/CStackFolder::mReadSingle (line 127):
   pInput->m_acInPrefix cannot be pushed directly into the queue.
   Used this->PushFile(pInput->m_acInPrefix) instead. (Note: this
   bug may be the root cause of bug 2 and 3.)
5. Change: AreTomo/Recon/CCalcVolThick::mDetectEdges:
   Smoothed the CC curve before detecting the edges. 
   CAtInput.cpp: The default ExtZ is changed to 300.

AreTomo3 2.0.4 [11-01-2024]
---------------------------
1. 1) Change: AreTomo/Recon/CCalcVolThick::mDetectEdges:
   Removed the two-peak cases, seem unnecessary and inaccurate.
   2) Revised the help message for tilt axis input in CAtInput.cpp.

AreTomo3 2.0.5 [12-04-2024]
---------------------------
1. 1) Revise CAreTomoMain.cpp: Set AlignZ upper ceiling to 1200 for global.
   2) Revise AreTomo/PatchAlign/CLocalAlign.cpp: Set AlignZ <=1000 for local.
2. 1) When defocus handedness is estimated to be -1, subtract 180 degree from
   the measured tilt axis. CAreTomoMain.cpp.
   2) Added FindCtf/CTiltInducedZ.cpp to calculate tilting induced z change.
      It considers both alpha and beta offset.
   3) As a result, CCorrImgCtf.cpp corrects local CTF with alpha0 and beta0
      taken into account.
3. 1) Fixed the careless mistake in calculating deltaZ due to tilting. The fix
      has been implemented in CTiltInducedZ.cpp.
   2) Bug fix in AreTomo/CTsMetrics.cpp:92: "float afS[] = {0.0f}", Changed to
      "float afS[2] = {0.0f};".
4. 1) [12-17-2024] Bug in MotionCor/DataUtil/CFmGroupParam.cpp::Setup:
      m_pfGroupCenters were based on raw frames, inconsistent with
      MotionCor/Align/CEarlyMotion.cpp that treats them relative to the
      integrated frames. Now m_pfGroupCenters in CFmGroupParam have changed
      to integrated frame based.
   2) [12-17-2024]: Saved metrics into file after tomograms are saved. When
      DenoisET finds a tomogram of interest, it knows the tomogram is
      already there for processing.
   3) [12-17-2024] Bug in MotionCor/DataUtil/CFmGroupParam.cpp: m_pfGroupCenters[g] =
      iNumRawFms + 0.5f * (iGroupRawFms - 1); is a bug that causes segmentation fault
      in MotionCor/Align/CEarlyMotion.cpp. Use m_pfGroupCenters[g] = 
      m_piGroupStart[g] + 0.5f * m_piGroupSize[g]; instead.

AreTomo3 2.0.6 [Jan-01-2025]
----------------------------
1. 1) Added -Cmd 4. This mode rotates the tilt axis by 180 degree and generates
      the updated .aln file. The files in _Imod directory including .xf,
      _st.mrc are also updated to reflect the 180 degree rotation of the tilt
      axis.
   2) -Cmd 4 also reconstructs new volumes using the rotated tilt axis.
   3) When -OutImod is enabled (its value > 0), its contents are removed if the
      corresponding directory exists. The new contents are saved afterwards.
   4) Revised the implementation of -Cmd 3. Added local CTF estimation.
2. Updated Remap3D by copying Remap3D_0.3_07dec24 into tools.
3. Updated user manuals for running -Cmd 4 and running Remap3D.

AreTomo 2.0.7 [Jan-13-2025]
---------------------------
1. 1) Bug fix (MotionCor/CFmIntParam.cpp)  If the sum of the group sizes of 
      (n-1) lines exceeds the total frames, AreTomo3 crashes. Fixed.
   2) Bug fix (AreTomo/MrcUtil/CAlignParam.cpp) Remove dark frames one by
      one can be buggy because the old implementation does not track the
      original sequence. New implementation removes them all at once.
   3) Bug fix (AreTomo/MrcUtil/CAlignParam.cpp) Section indices are 
      initialized to 1-based. Aln and CTF files contain 1-based section
      indices.
   4) Bug fix (AreTomo/FindCtf/CSaveCtfResults) Using 4 decimals for extra
      phase shift since it is in radian.
   5) Bug fix (MotionCor/DataUtil/CPatchShifts::SetRawShift) Incorrect
      determination of the starting location of m_pfPatShifts. Fixed!
      5.1) SetRawShift: loop starts from i=1, should be i=0, fixed.
   6) Bug fix (MotionCor/DataUtil/CFmGroupParam) Group center should be based
      on number of raw frames, not integrated frames.
   7) Bug fix (MotionCor/Align/CEarlyMotion) The shifts at the nodes are the
      ones of integrated frames. Do not use CFmGroupParam::m_pfGroupCenters
2. 1) FindCtf: limit the estimation of extra phase shift to within [0, 150]
      range to prevent accidental contrast flipping.

AreTomo3 2.0.8 [Jan-23-2025]
----------------------------
1. 1) Bug fix (AreTomo/FindCtf/CRefineCtfMain.cpp): Memory leak in 
      CFindCtfMain::mGenAvgSpects. When m_ppfSpects[i] is not empty, 
      the new memory pointer returned by m_pFindCtf2D->GetHalfSpect
      overrides the old one.
      Fix: Added mCleanSpects and mGenAvgSpects alway calls it.
2. In MotionCor/Align/CAlignMain.cpp, temporally diabled local motion correction
   on tilt images when their tilt angles higher than 5 degree. Need more robust
   implementation.

AreTomo3 2.0.9 [Jan-26-2025]
----------------------------
1. 1) Bug fix (AreTomo/CAreTomoMain.cpp): valgrind reported indirect memory leak
      at mFlipVol. Changes were made in mWbpRecon and mSartRecon to delete
      pVolStack in place. After this change valgrind reported no memory leak.

AreTomo3 2.0.10 [Feb-17-2025]
-----------------------------
1. 1) Bug fix: -Cmd 4 incorrectly generates .aln and .ctf file when there are
      rejected dark frames. This is because old .aln files are 0-based section
      indices and the new ones are 1-based.
      Changes haved been in AreTomo/MrcUtil/CDarkFrames, CRemoveDarkFrames,
      CLoadAlignFile, CSaveAlignFile, CAlignParam  AreTomo/CAreTomoMain, and
      AreTomo/FindCtf/CLoadCtfResults.cpp 

AreTomo3 2.1.0 [Feb-18-2025]
----------------------------
1. Bug Fix:
   1) Forgot to delete m_iUpsample = 1 (debugging code) in GCorrectPatchShift.
      Fixed (02-24-2025).
2. Improvement:
   1) Improved location motion correction. The iterative alignment stops
      it gets worse.
   2) Using fixed frame integration instead since per-tilt dose is so
      small.
   3) Using sliding window to avoid shift interpolation.

AreTomo3 2.1.1 [Mar-03-03-2025]
-------------------------------
1. Bug Fix:
2. Improvement:
   1) Added CTimeStamp.cpp in DataUtil to measure the time stamp at different
      points of execution. This is a global timer. The time stamp is relative
      to the begining of the execution.

AreTomo3 2.1.2 [Mar-03-07-2025]
-------------------------------
1. Bug fix
2. Improvement:
   1) Adjustment in DataUtil/CTimeStamp.cpp. Record and save when a thread
      starts the processing. This helps detect if the processing is stuck.

AreTomo3 2.1.2a [Mar-11-2025]
-----------------------------
1. Bug fix
2. Improvement:
   1) FindCtf/CRefineCtfMain.cpp: refine alpha offset to higher tilt range.

AreTomo3 2.1.3 [Mar-19-2025]
----------------------------
1. Renamed 2.1.2a to 2.1.3

AreTomo3 2.1.4 [Apr-04-2025]
----------------------------
1. Big fix
2. Improvement:
   1) GBackProj.cu: switched to bilinear interpolation to reduce aliasing.
   2) GForProj.cu:  the same as above
   2) DataUtil/CCalcThick.cpp: Searching is done within an interior z range by
      removing 20% on top and bottom.

AreTomo3 2.1.5 [Apr-09-2025]
----------------------------
1. Bug fix
2. Improvement
   1) Added GBackProjWbp.cu to do weighted back projection on 2x expanded
      grid and Fourier-crop it back to original size to remove aliasing.
