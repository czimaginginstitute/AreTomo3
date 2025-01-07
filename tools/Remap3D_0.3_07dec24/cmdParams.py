#!/usr/bin/python

import argparse
import sys

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
#--------------------------------------------------------------------
class ReadInput:
    def __init__(self): 
        parser = argparse.ArgumentParser(formatter_class= \
            argparse.ArgumentDefaultsHelpFormatter)
        ovsStr = "tomogram size Lx Ly Lz of the old star file"
        nvsStr = "tomogram size Lx Ly Lz of the new star file"
        #--------------------
        parser.add_argument("--old_vol_size", "-ovs", type=int, nargs='+', \
            default=None, help=ovsStr)
        parser.add_argument("--new_vol_size", "-nvs", type=int, nargs='+', \
            default=None, help=nvsStr)
        #--------------------
        parser.add_argument("--old_star", "-os", type=str, \
            help="old target star file")
        parser.add_argument("--new_star", "-ns", type=str, \
            help="new target star file")
        #--------------------
        parser.add_argument("--old_aln", "-oa", type=str, \
            help="path of the directory of old aln files")
        parser.add_argument("--new_aln", "-na", type=str, \
            help="path of the directory of new aln files")
        #--------------------
        parser.add_argument("--old_aln_prefix", "-oap", type=str, \
            help="prefix of old aln file name")
        #--------------------
        parser.add_argument("--old_pix_size", "-ops", type=float, default=1.0, \
            help="pixel size (A) of the old star file")
        parser.add_argument("--new_pix_size", "-nps", type=float, default=1.0, \
            help="pixel size (A) of the new star file")
        #--------------------
        args = parser.parse_args()
        self.oldVolSize = args.old_vol_size
        self.newVolSize = args.new_vol_size
        self.oldStar = args.old_star
        self.newStar = args.new_star
        self.oldAlnDir = args.old_aln
        self.newAlnDir = args.new_aln
        self.oldAlnPrefix = args.old_aln_prefix
        self.oldPixSize = args.old_pix_size
        self.newPixSize = args.new_pix_size
        #--------------------
        print("\nInput parameters:")
        print("old vol size:      " + str(self.oldVolSize))
        print("new vol size:      " + str(self.newVolSize))
        print("old star file:     " + str(self.oldStar))
        print("new star file:     " + str(self.newStar))
        print("old aln directory: " + str(self.oldAlnDir))
        print("new aln directory: " + str(self.newAlnDir))
        print("old aln prefix:    " + str(self.oldAlnPrefix))
        print("old pix size:      " + str(self.oldPixSize))
        print("new pix size:      " + str(self.newPixSize) + "\n") 

    def validate(self):
        if self.oldVolSize == None or len(self.oldVolSize) < 3:
            print("Error: incorrect old vol size" + str(self.oldVolSize))
            return False
        #--------------------
        if self.newVolSize == None or len(self.newVolSize) < 3:
            print("Error: incorrect new vol size" + str(self.newVolSize))
            return False
        #--------------------
        if self.oldStar == None:
            print("Error: old star file name not given")
            return False
        #--------------------
        if self.newStar == None:
            print("Error: new star file name not given")
            return False
        #--------------------
        if self.oldAlnDir == None:
            print("Error: directory of old aln files not given")
            return False
        else:
            self.oldAlnDir = self.mAddEndSlash(self.oldAlnDir)
        #--------------------
        if self.newAlnDir == None:
            print("Error: directory of new aln files not given")
            return False
        else:
            self.newAlnDir = self.mAddEndSlash(self.newAlnDir)
        #--------------------
        if self.oldAlnPrefix == None:
            print("Error: prefix of old aln files not given")
            return False
        #--------------------
        if self.oldPixSize == None:
            print("Error: old pixel size not given")
            return False
        #--------------------
        if self.newPixSize == None:
            print("Error: new pixel size not given")
            return False
        #--------------------
        return True

    def mGetPath(self, fileName):
        if fileName is None:
            return None
        iSlash = fileName.rfind("/")
        if iSlash == -1:
            return "./"
        else:
            return fileName[:iSlash+1]

    def mAddEndSlash(self, dirName):
        if dirName is None:
            return None
        iSize = len(dirName)
        if iSize == 0:
            return None
        if dirName[iSize - 1] == "/":
            return dirName
        else:
            return dirName + "/"
