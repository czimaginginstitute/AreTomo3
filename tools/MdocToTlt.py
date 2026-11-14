#!/usr/bin/env python3
"""
MdocToTlt.py: generate AreTomo3 _TLT.txt files from SerialEM .mdoc files.

Each output row corresponds to one [ZValue = N] section in the .mdoc file:
    column 1: TiltAngle (rows sorted in ascending order)
    column 2: 1-based acquisition index determined from DateTime
              (1 = earliest acquired image)
    column 3: ExposureDose

Usage:
    # single mode: full name of one mdoc file
    MdocToTlt.py --serial 0 -i /data/mdoc/Position_120.mdoc -o /data/tlt

    # batch mode: prefix of mdoc file names (processes prefix*.mdoc)
    MdocToTlt.py --serial 1 -i /data/mdoc/Position_ -o /data/tlt
"""

import argparse
import datetime
import glob
import os
import sys


class MdocSection:
    def __init__(self, zValue):
        self.zValue = zValue
        self.tiltAngle = None
        self.dose = 0.0
        self.dateTime = None


class ReadMdoc:
    def __init__(self):
        self.sections = []

    def doIt(self, mdocFile):
        self.sections = []
        section = None
        with open(mdocFile, "r", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = self._parseSectionHeader(line)
                    if section is not None:
                        self.sections.append(section)
                    continue
                if section is None or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "TiltAngle":
                    section.tiltAngle = float(value.split()[0])
                elif key == "ExposureDose":
                    section.dose = float(value.split()[0])
                elif key == "DateTime":
                    section.dateTime = self._parseDateTime(value)
        #------------------
        bad = [s.zValue for s in self.sections if s.tiltAngle is None]
        if len(bad) > 0:
            raise ValueError("TiltAngle missing in ZValue section(s): %s"
                             % ", ".join(str(z) for z in bad))
        return self.sections

    def _parseSectionHeader(self, line):
        # Returns a new section for "[ZValue = N]", None for other
        # bracketed headers (e.g. "[T = ...]").
        inner = line[1:-1]
        if "=" not in inner:
            return None
        key, value = inner.split("=", 1)
        if key.strip() != "ZValue":
            return None
        return MdocSection(int(value.strip()))

    def _parseDateTime(self, value):
        text = " ".join(value.split())
        for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%y %H:%M:%S"):
            try:
                return datetime.datetime.strptime(text, fmt)
            except ValueError:
                pass
        return None


class GenTlt:
    def __init__(self):
        self.reader = ReadMdoc()

    def doIt(self, mdocFile, outDir):
        sections = self.reader.doIt(mdocFile)
        if len(sections) == 0:
            print("...... No ZValue sections found, skipped: " + mdocFile)
            return False
        #------------------
        self._calcAcqIndices(sections, mdocFile)
        sections.sort(key=lambda s: (s.tiltAngle, s.zValue))
        #------------------
        prefix = os.path.splitext(os.path.basename(mdocFile))[0]
        tltFile = os.path.join(outDir, prefix + "_TLT.txt")
        with open(tltFile, "w") as fp:
            for s in sections:
                fp.write("%8.2f%6d%10.2f\n" % (s.tiltAngle, s.acqIdx, s.dose))
        print("...... %d tilts written: %s" % (len(sections), tltFile))
        return True

    def _calcAcqIndices(self, sections, mdocFile):
        if all(s.dateTime is not None for s in sections):
            order = sorted(sections, key=lambda s: (s.dateTime, s.zValue))
        else:
            print("...... Warning: DateTime missing or unparsable in "
                  + mdocFile + ", acquisition order taken from ZValue.")
            order = sorted(sections, key=lambda s: s.zValue)
        for i, s in enumerate(order):
            s.acqIdx = i + 1


def findMdocFiles(inPath, serial):
    if serial == 0:
        if not inPath.lower().endswith(".mdoc"):
            sys.exit("Error: --serial 0 requires the full name of a "
                     ".mdoc file: " + inPath)
        if not os.path.isfile(inPath):
            sys.exit("Error: mdoc file not found: " + inPath)
        return [inPath]
    #------------------
    if os.path.isdir(inPath):
        pattern = os.path.join(inPath, "*.mdoc")
    else:
        pattern = inPath + "*.mdoc"
    mdocFiles = sorted(glob.glob(pattern))
    if len(mdocFiles) == 0:
        sys.exit("Error: no mdoc files match: " + pattern)
    return mdocFiles


def main():
    parser = argparse.ArgumentParser(
        description="Generate AreTomo3 _TLT.txt files from .mdoc files.")
    parser.add_argument("-i", "--input", required=True,
        help="--serial 0: full name of the mdoc file; "
             "--serial 1: prefix of mdoc file names (with directory).")
    parser.add_argument("-o", "--outdir", required=True,
        help="Directory where _TLT.txt files are saved.")
    parser.add_argument("--serial", type=int, choices=[0, 1], default=0,
        help="0: process a single mdoc file (default); "
             "1: batch process all mdoc files matching the prefix.")
    args = parser.parse_args()
    #------------------
    mdocFiles = findMdocFiles(args.input, args.serial)
    os.makedirs(args.outdir, exist_ok=True)
    #------------------
    genTlt = GenTlt()
    numDone, numFailed = 0, 0
    for mdocFile in mdocFiles:
        print("Processing: " + mdocFile)
        try:
            if genTlt.doIt(mdocFile, args.outdir):
                numDone += 1
            else:
                numFailed += 1
        except Exception as e:
            print("...... Error: %s" % e)
            numFailed += 1
            if args.serial == 0:
                sys.exit(1)
    print("\nDone: %d processed, %d failed." % (numDone, numFailed))
    if numFailed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
