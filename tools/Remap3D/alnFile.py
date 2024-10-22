#!/usr/bin/python
import re
import math

class ReadAlnFile:
    def __init__(self):
        self.mInit()


    def doIt(self, fileName):
        self.mInit()
        alnFile = open(fileName, 'r')
        self.lines = alnFile.readlines()
        #-----------------
        if len(self.lines) <= 0:
            alnFile.close()
            return False
        #-----------------
        self.mParseHeader()
        self.mParseGlobal()
        self.mParseLocal()
        #-----------------
        self.sectionIds = self.getSectionIds()
        self.tilts = self.getTilts()
        #-----------------
        self.secToIndex = dict()
        iCount = 0
        for secIdx in self.sectionIds:
            self.secToIndex[secIdx] = iCount
            iCount += 1

    def findSectionId(self, fTilt):
        fMinDif = 1e10
        iMin = -1
        for i in range(len(self.tilts)):
            fDif = math.fabs(self.tilts[i] - fTilt)
            if fDif < fMinDif:
                fMinDif = fDif
                iMin = i
        return self.sectionIds[iMin]

    def getGlobalShift(self, secIdx):
        i = self.secToIndex[secIdx]
        g = self.globalParams[i]
        return [g[3], g[4]]

    def getTiltAngle(self, secIdx):
        i = self.secToIndex[secIdx]
        return self.tilts[i]

    def getTiltAxis(self, secIdx):
        i = self.secToIndex[secIdx]
        g = self.globalParams[i]
        return g[1]


    def getTilts(self):
        tilts = []
        for l in self.globalParams:
            tilts.append(l[9])
        return tilts


    def getTiltAxes(self):
        tiltAxes = []
        for l in self.globalParams:
            tiltAxes.append(l[1])
        return tiltAxes


    def getGlobalShifts(self):
        shifts = []
        for l in self.globalParams:
            shift = [l[3], l[4]]
            shifts.append(shift)
        return shifts;

    
    def getLocalShifts(self):
        return self.localParams


    def getNumTilts(self):
       return self.iNumTilts


    def getRawSize(self):
        return self.rawSize


    def getSectionIds(self):
        sectionIds = []
        for l in self.globalParams:
            sectionIds.append(int(l[0] + 0.5))
        return sectionIds

    
    def getNumPatches(self):
        return self.iNumPatches


    def mInit(self):
        self.rawSize = []
        self.iNumTilts = 0
        self.iNumPatches = 0
        self.fAlphaOffset = 0.0
        self.fBetaOffset = 0.0
        self.darkFrames = []
        self.globalParams = []
        self.localParams = []


    def mParseHeader(self):
        for line in self.lines:
            if line[0] != '#' : continue
            #------------------
            if line.find('RawSize') > 0 :
                self.mParseRawSize(line)
            elif line.find('NumPatches') > 0 :
                self.mParseNumPatches(line)
            elif line.find('AlphaOffset') > 0 :
                self.mParseAlphaOffset(line)
            elif line.find('BetaOffset') > 0 :
                self.mParseBetaOffset(line)
            elif line.find('DarkFrame') > 0:
                self.mParseDarkFrame(line)
            elif line.find('# SEC') > 0:
                break


    def mParseGlobal(self):
        iStart = 0
        bFound = False
        for line in self.lines:
            if line.find('# SEC') != -1: 
                bFound = True
                break
            iStart += 1
        if bFound == False: return
        #--------------------
        for i in range(iStart+1, len(self.lines)) :
            line = self.lines[i]
            if line.find('Local Alignment') != -1: break
            #----------------
            entryList = self.mParseLine(line)
            if len(entryList) == 0 : continue
            else: self.globalParams.append(entryList)
        #-------------------
        self.iNumTilts = len(self.globalParams)


    def mParseLocal(self):
        iStart = 0
        bFound = False
        for line in self.lines:
            if line.find('# Local') != -1: 
                bFound = True
                break
            iStart += 1
        if bFound == False: return
        #--------------------
        for i in range(iStart+1, len(self.lines)) :
            line = self.lines[i]
            if len(line) < 5: break
            #----------------
            entryList = self.mParseLine(line) 
            if len(entryList) == 0 : continue
            else: self.localParams.append(entryList)


    def mParseRawSize(self, line):
        tokens = re.split('[# =\n]', line)
        for t in tokens:
            try:
                self.rawSize.append(int(t))
            except:
                continue
        #--------------------
        if len(self.rawSize) != 3:
            print("Warining: incorrect RawSize: " + str(self.rawSize))
            self.rawSize = []


    def mParseNumPatches(self, line):
        tokens = re.split('[# =\n]', line)
        for t in tokens:
            try:
                self.iNumPatches = int(t)
            except:
                continue
   

    def mParseDarkFrame(self, line):
        tokens = line.split()
        darkList = []
        for token in tokens:
            try:
                darkList.append(float(token))
            except:
                continue
        #--------------------
        if len(darkList) == 0: return
        else: self.darkFrames.append(darkList)


    def mParseAlphaOffset(self, line):
        tokens = re.split('[# =\n]', line)
        for t in tokens:
            try:
                self.fAlphaOffset = float(t)
            except:
                continue 


    def mParseBetaOffset(self, line):
        tokens = re.split('[# =\n]', line)
        for t in tokens:
            try:
                self.fBetaOffset = float(t)
            except:
                continue


    def mParseLine(self, line):
        tokens = line.split()
        valList = []
        for token in tokens:
            try:
                valList.append(float(token))
            except:
                continue
        return valList

