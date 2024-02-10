AreTomo3 1.0.0: 01/30/2024
--------------------
A fully automated preprocessing pipeline that enables real time generation of cryoET tomograms with integrated correction of beam induced motion, CTF estimation, tomographic alignment and reconstruction in a single multi-GPU accelerated application.

02/09/2024
----------
1. MotionCor/DataUtil/CFmIntFile.cpp: Bug Fix
   It has a memory access error. Reimplemented the calculation of number of
   raw frames in each integrated frames in mCalcIntFms().
2. DataUtil/CTiltSeries.cpp::Create(...): Bug Fix
   mCleanCenters() has been moved to the first line to avoid the change
   caused of m_aiStkSize by CMrcStack::Create() 
