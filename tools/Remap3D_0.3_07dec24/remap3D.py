#!/usr/bin/python

import cmdParams
import alnFile
import tgtStarFile
#import pandas
import numpy
import math
import sys

class Remap3dMain: 
    def __init__(self):
        self.alnFiles = [None, None]
        self.oldVolSize = None
        self.newVolSize = None
        self.fD2R = math.pi / 180.0

    def setVolSizes(self, oldVolSize, newVolSize):
        self.oldVolSize = oldVolSize
        self.newVolSize = newVolSize

    def setPixSizes(self, fOldPixSize, fNewPixSize):
        self.fOldPixSize = fOldPixSize
        self.fNewPixSize = fNewPixSize

    def setAlnFiles(self, oldAlnName, newAlnName):
        self.readAlns = [None, None]
        self.readAlns[0] = alnFile.ReadAlnFile()
        self.readAlns[1] = alnFile.ReadAlnFile()
        bRead1 = self.readAlns[0].doIt(oldAlnName)
        bRead2 = self.readAlns[1].doIt(newAlnName)
        if not bRead1:
            print("Error: " + str(oldAlnName))
            print("       unable to read old aln file.\n")
            return False
        if not bRead2:
            print("Error: " + str(newAlnName))
            print("       unable to read new aln file.\n")
            return False
        return True
        
    def doIt(self, tgt):
        oldTgt = [0, 0, 0]
        oldTgt[0] = (tgt[0] - self.oldVolSize[0] * 0.5) * self.fOldPixSize
        oldTgt[1] = (tgt[1] - self.oldVolSize[1] * 0.5) * self.fOldPixSize
        oldTgt[2] = (tgt[2] - self.oldVolSize[2] * 0.5) * self.fOldPixSize
        #--------------------
        oldTgt_deg1 = self.mForwProj(oldTgt, -20.0)
        oldTgt_deg2 = self.mForwProj(oldTgt, 20.0)
        #--------------------
        newTgt = self.mCalcNewTgt(oldTgt_deg1, oldTgt_deg2)
        #--------------------
        x = newTgt[0] / self.fNewPixSize + self.newVolSize[0] * 0.5
        y = newTgt[1] / self.fNewPixSize + self.newVolSize[1] * 0.5
        z = newTgt[2] / self.fNewPixSize + self.newVolSize[2] * 0.5
        return [x, y, z]

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


def getAlnName(rlnTomoName, strPrefix):
    iStartIdx = rlnTomoName.find(strPrefix)
    if iStartIdx == -1:
        print("Warning: rlnTomoNmae does not contain the prefix.")
        print("   rlnTomoName: " + str(rlnTomoName))
        print("   prefix:      " + str(strPrefix) + "\n")
        return None
    #------------------------
    alnName = rlnTomoName[iStartIdx:]
    if alnName.find(".aln") == -1:
        return alnName + ".aln"
    else:
        return alnName

        
#--------------------------------------------------------------------
# 1. remap3D.py maps 3D (x, y, z) targets picked from the old
#    tomogram into the new tomogram that is reconstructed from the
#    same tilt series with a different alignment file (.aln).
# 2. The input parameters include:
#    1) old_vol_size: three integers of the old tomogram sizes in
#       x, y, and z dimensions, respectively.
#    2) new_vol_size: the same as the old_vol_size except that it
#       for the new tomogram.
#    3) old_star: particle star file corresponding to old_vol_size
#    4) new_star: particle star file corresponding to new_vol_size.
#       This is also new star file to be generated.
#    5) old_aln: name of the directory storing the old aln files.
#    6) new_aln: name of the directory storing the new aln files.
#    7) old_aln_prefix: prefix of the old aln file name.
#    8) old_pix_size: the pixel size associated with the old star
#       file.
#    9) new_pix_size: the pixel size associated with the new star
#       file.
# 3. Example of command line:
#    python ~/PyProjs/AreTomo3/Remap3D/remap3D.py 
#       -ovs 4096 4096 1200 
#       -nvs 4096 4096 1200
#       -ops 1.54
#       -nps 1.54
#       -os 24mar08a/stars/20240308_002_ribosome.star 
#       -ns Temp/ribo_new.star 
#       -oa 24mar08a/run002/alns 
#       -na 20240308_002_Krios1_RP_Lys6prtns/run006/alns/ 
#       -oap Position_
#    20240308_002_ribosome.star is relion generated particle star
#       file that lists picked particle 3D coordinates, which are
#       to be mapped to new alignment based tomograms.
#    Temp/ribo_new.star stores the mapped target coordinates for
#       the new tomograms.
#    24mar08a/run002/alns is the directory that contains all the
#       old aln files generated by AreTomo3 and corresponding to
#       the tomograms where 3D targets in 20240308_002_ribosome.star
#       were picked.
#    20240308_002_Krios1_RP_Lys6prtns/run006/alns/ is the directory
#       that stores the new aln files, which are used to reconstruct
#       the new tomograms to which the old targets will be mapped.
#    Positin_ is the prefix of the aln file names since tomogram
#       names listed in the star file may be appended with extra
#       information. remap3D uses this prefix to extract the
#       correct aln file names.
#    -ovs: old volume size (in pixel)
#    -nvs: new volume size (in pixel)
#    -os:  old star file
#    -ns:  new star file
#    -oa:  old aln file directory
#    -na:  new aln file directory
#    -oap: prefix of old aln file names, e.g. Position_ in Position_1.aln
#    -ops: old volume pixel size
#    -nps: new volume pixel size
# 4. remap3D.py assumes the old and new aln files are the same but
#    in different directories.
#       
#--------------------------------------------------------------------
def main():
    readInput = cmdParams.ReadInput()
    bSuccess = readInput.validate()
    if not bSuccess: 
        print("Invalid input parameter(s), quit.\n")
        return
    #--------------------
    remapMain = Remap3dMain()
    remapMain.setVolSizes(readInput.oldVolSize, readInput.newVolSize)
    remapMain.setPixSizes(readInput.oldPixSize, readInput.newPixSize)
    #--------------------
    readTgtStar = tgtStarFile.ReadTgtStar()
    readTgtStar.doIt(readInput.oldStar)
    iNumTgts = readTgtStar.getNumTgts()
    if iNumTgts == 0:
        print("Error: " + str(readInput.oldStar))
        print("       not targets found.\n")
        return
    #---------------------
    writeTgtStar = tgtStarFile.WriteTgtStar()
    writeTgtStar.setup(readTgtStar.getCopy(), readTgtStar.dfOptics)
    #---------------------
    curTomoName = None
    bReadAlns = True
    alnName = None
    #---------------------
    for i in range(iNumTgts):
        rlnTomoName = readTgtStar.getTomoName(i)
        if rlnTomoName != curTomoName:
            curTomoName = rlnTomoName
            #---------------------
            alnName = getAlnName(rlnTomoName, readInput.oldAlnPrefix)
            if alnName is None:
                continue
            #---------------------
            oldAlnName = readInput.oldAlnDir + alnName
            newAlnName = readInput.newAlnDir + alnName
            bReadAlns = remapMain.setAlnFiles(oldAlnName, newAlnName)
            #---------------------
        if not bReadAlns:
            continue
            #---------------------
        srcTgt = readTgtStar.getCoordXYZ(i)
        dstTgt = remapMain.doIt(srcTgt)
        writeTgtStar.updateCoordXYZ(i, dstTgt)
    #--------------------
    writeTgtStar.doIt(readInput.newStar)
    print("\nMapped targets are saved in: " + readInput.newStar)
    print("\n")

if __name__ == "__main__":
    main()


