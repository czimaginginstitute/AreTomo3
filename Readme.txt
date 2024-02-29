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
