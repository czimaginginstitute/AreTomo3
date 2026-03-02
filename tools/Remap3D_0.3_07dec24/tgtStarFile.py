#!/usr/bin/python
import starfile
import pandas
import numpy
import math

class ReadTgtStar:
    def __init__(self):
        self.dfParticles = None
        self.dfOptics = None


    def doIt(self, fileName):
        try:
            star = starfile.read(fileName)
            self.dfParticles = star["particles"]
            self.dfOptics = star["optics"]
            return self.getNumTgts()
        except:
            print("Error: unable to read star file")
            print("       " + str(fileName))
            self.dfParticles = None
            return 0

    def getNumTgts(self):
        if self.dfParticles is None:
            return 0
        return self.dfParticles["rlnTomoName"].size

    def getTomoName(self, iTgt):
        return self.dfParticles["rlnTomoName"][iTgt]

    def getCoordXYZ(self, iTgt):
        x = self.dfParticles["rlnCoordinateX"][iTgt]
        y = self.dfParticles["rlnCoordinateY"][iTgt]
        z = self.dfParticles["rlnCoordinateZ"][iTgt]
        return [x, y, z]

    def getCopy(self):
        if self.dfParticles is None:
            return None
        return self.dfParticles.copy()


class WriteTgtStar:
    def __init__(self):
        self.oldArr = None
        self.newArr = None

    def setup(self, dfParticles, dfOptics):
        self.dfParticles = dfParticles
        self.dfOptics = dfOptics
        #--------------------
        self.dfArr = dfParticles.to_numpy()
        self.newList = []

    def updateCoordXYZ(self, iTgt, tgtCoord):
        self.dfArr[iTgt][1] = math.ceil(tgtCoord[0] * 100) / 100
        self.dfArr[iTgt][2] = math.ceil(tgtCoord[1] * 100) / 100
        self.dfArr[iTgt][3] = math.ceil(tgtCoord[2] * 100) / 100
        #--------------------
        tgtList = self.dfArr[iTgt].tolist()
        self.newList.append(tgtList)
       
    def doIt(self, fileName):
        newArr = numpy.array(self.newList)
        dfKeys = self.dfParticles.keys()
        #--------------------
        tgtDict = {}
        for i in range(len(dfKeys)):
            key = dfKeys[i]
            val = newArr[:, i]
            tgtDict[key] = val
        dfParticles = pandas.DataFrame(tgtDict)
        #--------------------
        starfile.write({"optics": self.dfOptics, \
            "particles": dfParticles}, fileName)
