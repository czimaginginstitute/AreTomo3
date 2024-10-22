#!/usr/bin/python

import alnFile
import argparse
import pandas
import numpy
import math
import sys

class Remap3dMain: 
    def __init__(self):
        self.alnFiles = [None, None]
        self.srcVolSize = None
        self.dstVolSize = None
        self.fD2R = math.pi / 180.0

    def setVolSizes(self, srcSize, dstSize):
        self.srcVolSize = srcSize
        self.dstVolSize = dstSize

    def setBinnings(self, fSrcBin, fDstBin):
        self.fSrcBin = fSrcBin
        self.fDstBin = fDstBin

    def setAlnFiles(self, srcAlnName, dstAlnName):
        self.readAlns = [None, None]
        self.readAlns[0] = alnFile.ReadAlnFile()
        self.readAlns[1] = alnFile.ReadAlnFile()
        self.readAlns[0].doIt(srcAlnName)
        self.readAlns[1].doIt(dstAlnName)

    def doIt(self, srcTgts):
        self.srcTgts = srcTgts
        iNumTgts = len(srcTgts)
        self.dstTgts = numpy.ones((iNumTgts, 3))
        for i in range(iNumTgts):
            self.mMapTarget(i)
        return self.dstTgts
        
    def mMapTarget(self, iNthTgt):
        tgt = self.srcTgts[iNthTgt]
        oldTgt = [0, 0, 0]
        oldTgt[0] = (tgt[0] - self.srcVolSize[0] * 0.5) * self.fSrcBin
        oldTgt[1] = (tgt[1] - self.srcVolSize[1] * 0.5) * self.fSrcBin
        oldTgt[2] = (tgt[2] - self.srcVolSize[2] * 0.5) * self.fSrcBin
        #--------------------
        oldTgt_0deg = self.mForwProj(oldTgt, 0.0)
        oldTgt_30deg = self.mForwProj(oldTgt, 30.0)
        #--------------------
        newTgt = self.mCalcNewTgt(oldTgt_0deg, oldTgt_30deg)
        #--------------------
        dstTgt = self.dstTgts[iNthTgt]
        dstTgt[0] = newTgt[0] / self.fDstBin + self.dstVolSize[0] * 0.5
        dstTgt[1] = newTgt[1] / self.fDstBin + self.dstVolSize[1] * 0.5
        dstTgt[2] = newTgt[2] / self.fDstBin + self.dstVolSize[2] * 0.5

#-----------------------------------------------------------
# Rotate the target (fX, fY) by fRotAngle degree counter
#    clockwise. The coordinate system is fixed. This
#    increases target angle relative x axis.
#-----------------------------------------------------------
    def mRotate(self, fX, fY, fRotAngle):
        fCos = math.cos(fRotAngle * self.fD2R)
        fSin = math.sin(fRotAngle * self.fD2R)
        fRx = fX * fCos - fY * fSin
        fRy = fX * fSin + fY * fCos
        return [fRx, fRy]

    def mForwProj(self, volTgt, fTilt):
        iSecId = self.readAlns[0].findSectionId(fTilt)
        [fShiftX, fShiftY] = self.readAlns[0].getGlobalShift(iSecId)
        fTilt1 = self.readAlns[0].getTiltAngle(iSecId)
        fTiltAxis = self.readAlns[0].getTiltAxis(iSecId)
        #----------------------------------------------------
        # forward project volume target to projection plane
        #----------------------------------------------------
        fCos = math.cos(fTilt1 * self.fD2R)
        fSin = math.sin(fTilt1 * self.fD2R)
        fPx = volTgt[0] * fCos + volTgt[2] * fSin
        fPy = volTgt[1]
        #------------------------------
        # uncorrect in-plane rotation
        #------------------------------
        [fX, fY] = self.mRotate(fPx, fPy, fTiltAxis)
        #-------------------------
        # uncorrect global shift
        #-------------------------
        fX += fShiftX
        fY += fShiftY
        return [fX, fY, iSecId]

    def mCalcNewTgt(self, oldTgt1, oldTgt2):
        gs1 = self.readAlns[1].getGlobalShift(oldTgt1[2])
        gs2 = self.readAlns[1].getGlobalShift(oldTgt2[2])
        #--------------------
        fTilt1 = self.readAlns[1].getTiltAngle(oldTgt1[2])
        fTilt2 = self.readAlns[1].getTiltAngle(oldTgt2[2])
        #--------------------
        fTiltAxis1 = self.readAlns[1].getTiltAxis(oldTgt1[2])
        fTiltAxis2 = self.readAlns[1].getTiltAxis(oldTgt2[2])
        #-----------------------
        # correct global shift
        #-----------------------
        fX1 = oldTgt1[0] - gs1[0]
        fY1 = oldTgt1[1] - gs1[1]
        fX2 = oldTgt2[0] - gs2[0]
        fY2 = oldTgt2[1] - gs2[1]
        #----------------------------
        # correct in-plane rotation
        #----------------------------
        [fX1, fY1] = self.mRotate(fX1, fY1, -fTiltAxis1)
        [fX2, fY2] = self.mRotate(fX2, fY2, -fTiltAxis2)
        #--------------------------------------
        # trace two rays to determine x and z
        #--------------------------------------
        fCos1 = math.cos(fTilt1 * self.fD2R)
        fSin1 = math.sin(fTilt1 * self.fD2R)
        fCos2 = math.cos(fTilt2 * self.fD2R)
        fSin2 = math.sin(fTilt2 * self.fD2R)
        fSinD = math.sin((fTilt2 - fTilt1) * self.fD2R)
        fX = (fX1 * fSin2 - fX2 * fSin1) / fSinD
        fZ = (-fX1 * fCos2 + fX2 * fCos1) / fSinD
        fY = (fY1 + fY2) * 0.5
        return [fX, fY, fZ]

#--------------------------------------------------------------------
# 1. remap3D.py maps 3D (x, y, z) targets picked from the 1st
#    tomogram into the 2nd tomogram that is reconstructed from the
#    same tilt series with a different alignment file (.aln).
# 2. The input parameters include:
#    1) vol_size1: three integers of the 1st tomogram sizes in
#       x, y, and z dimensions, respectively.
#    2) vol_size2: the same as the vol_size2 except that it is
#       for the 2nd tomogram.
#    3) bin1: binning of the 1st tomogram relative to the raw
#       tilt series.
#    4) bin2: binning of the 2nd tomogram relative to the raw
#       tilt series.
#    5) aln_file1: AreTomo3/2 generated aln file for the 1st
#       tomogram.
#    6) aln_file2: AreTomo3/2 generated aln file for the 2nd
#       tomogram.
#    7) tgt_file: a 3-column text file that lists the (x, y, z)
#       coordinates of the targets picked in the 1st tomogram.
# 3. The output is out_file, a 3-column text file that lists
#    the mapped targets for the 2nd tomogram.
# 4. Examplary command line
#       python remap3D.py -vs1 1364 1364 400 -vs2 818 818 360 \
#       -af1 Position_5_4.aln -af2 Position_5_4.aln_new \
#       -bin1 3 -bin2 5 -tf tgtFile.txt -out newTgt.txt
#
#--------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class= \
        argparse.ArgumentDefaultsHelpFormatter)
    #--------------------
    parser.add_argument("--vol_size1", "-vs1", type=int, nargs='+', \
        default=None, help="tomogram size: size_x size_y size_z")
    parser.add_argument("--vol_size2", "-vs2", type=int, nargs='+', \
        default=None, help="tomogram size: size_x size_y size_z")
    #--------------------
    parser.add_argument("--aln_file1", "-af1", type=str, default=None, \
        help="aln file generated by AreTomo2/3")
    parser.add_argument("--aln_file2", "-af2", type=str, default=None, \
        help="aln file generated by AreTomo2/3")
    #--------------------
    parser.add_argument("--bin1", "-b1", type=float, default=1.0, \
        help="binning of the 1st tomogram")
    parser.add_argument("--bin2", "-b2", type=float, default=1.0, \
        help="binning of the 2nd tomogram")
    #--------------------
    parser.add_argument("--tgt_file", "-tf", type=str, \
        default=None, help="3-column text file for target \
        coordinates relative lower left corner")
    parser.add_argument("--out_file", "-of", type=str, \
        default=None, help="3-column text file for mapped \
        coordinates in the 2nd tomogram.")
    #--------------------
    args = parser.parse_args()
    if args.vol_size1 == None or len(args.vol_size1) < 3:
        print("Error: incorrect vol size 1, quit. " + str(args.vol_size1))
        return
    if args.vol_size2 == None or len(args.vol_size2) < 3:
        print("Error: incorrect vol size 2, quit. " + str(args.vol_size1))
        return
    if args.aln_file1 == None:
        print("Error: no aln file 1 is given, quit")
        return
    if args.aln_file2 == None:
        print("Error: no aln file 2 is given, quit")
        return
    if args.bin1 == None:
        print("Error: binning of 1st tomogram not given, quit.")
        return
    if args.bin2 == None:
        print("Error: binning of 2nd tomogram not given, quit.")
        return
    if args.tgt_file == None:
        print("Error: target file not given, quit.")
    #--------------------
    remapMain = Remap3dMain()
    remapMain.setAlnFiles(args.aln_file1, args.aln_file2)
    remapMain.setVolSizes(args.vol_size1, args.vol_size2)
    remapMain.setBinnings(args.bin1, args.bin2)
    #--------------------
    dataFrame = pandas.read_csv(args.tgt_file, header=None, \
        delimiter=r"\s+", comment='#')
    srcTgt = dataFrame.to_numpy()
    dstTgt = remapMain.doIt(srcTgt)
    #dstTgt = dstTgt.astype(int)
    #--------------------
    numpy.set_printoptions(precision=1)
    print("\nOld vol targets:")
    print(srcTgt)
    print("\nNew vol targets:")
    print(dstTgt, end='\n\n')
    #--------------------
    if args.out_file != None:
        numpy.savetxt(args.out_file, dstTgt, fmt="%8.1f")        


if __name__ == "__main__":
    main()


