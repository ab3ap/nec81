#!/usr/bin/env python

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def buildNec81Deck(gnType, height_m):
    return f"""CM 40m dipole height sweep GN {gnType}
CE
GW 1 21 -10.0 0.0 {height_m:.4f} 10.0 0.0 {height_m:.4f} 0.001
GE 1
GN {gnType} 0 0 0 13.0 0.005
FR 0 1 0 0 7.15 0.0
EX 0 1 11 0 1.0 0.0
RP 0 19 1 1000 0.0 0.0 5.0 0.0
EN
"""

def buildNec5Deck(height_m):
    # 20 segments: segment 11 end 1 (i4=1) is dead-center at x = 0.0 m
    return f"""CM 40m dipole height sweep NEC-5
CE
GW 1 20 -10.0 0.0 {height_m:.4f} 10.0 0.0 {height_m:.4f} 0.001
GE 0
GN 0 0 0 0 13.0 0.005 1 0
FR 0 1 0 0 7.15 0.0
EX 0 1 11 1 1.0 0.0
RP 0 19 1 1000 0.0 0.0 5.0 0.0
EN
"""

def findExe(name, required=True):
    if not name:
        return None
    p = Path(name)
    if p.is_file():
        return str(p.resolve())
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).parent/name
    if candidate.is_file():
        return str(candidate)
    if required:
        raise RuntimeError(f'Console command not found: {name}')
    return None


def findNec5(userPath):
    candidates = (
        userPath,
        os.environ.get('NEC5_EXE', ''),
        'nec5',
        'nec5d',
        'NEC5',
        'nec5.exe',
        'nec5d.exe',
    )
    for item in candidates:
        exe = findExe(item, required=False)
        if exe:
            return exe
    return None


def runNec5(nec5Exe, deckPath, outPath, workDir):
    cmd = (nec5Exe, '-i', str(deckPath), '-o', str(outPath))
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=workDir,
    )
    if proc.returncode != 0 or not outPath.is_file():
        raise RuntimeError(
            f'NEC-5 execution failed (code {proc.returncode}): {proc.stderr}'
        )

def parseReport(reportPath):
    text = Path(reportPath).read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    zIn = None
    zenithGain_dBi = None

    numPat = re.compile(r'[+-]?\d+\.\d+(?:[Ee][+-]\d+)?')

    for idx, line in enumerate(lines):
        if 'ANTENNA INPUT PARAMETERS' in line and zIn is None:
            for sub in lines[idx + 1 : idx + 12]:
                parts = sub.split()
                if len(parts) >= 2 and parts.__getitem__(0).isdigit():
                    nums = numPat.findall(sub)
                    if len(nums) >= 6:
                        zIn = complex(
                            float(nums.__getitem__(4)),
                            float(nums.__getitem__(5)),
                        )
                        break
        if 'RADIATION PATTERNS' in line and zenithGain_dBi is None:
            for sub in lines[idx + 2 : idx + 15]:
                parts = sub.split()
                if len(parts) >= 5:
                    try:
                        theta_deg = float(parts.__getitem__(0))
                        if abs(theta_deg) < 1e-3:
                            zenithGain_dBi = float(parts.__getitem__(4))
                            break
                    except ValueError:
                        continue

    return zIn, zenithGain_dBi

def fmtZ(zVal):
    if zVal is None:
        return 'N/A'
    return f'{zVal.real:5.1f}{zVal.imag:+6.1f}j'


def fmtG(gVal):
    if gVal is None:
        return 'N/A'
    return f'{gVal:6.2f} dB'


def main():
    parser = argparse.ArgumentParser(
        description='Sweep dipole height comparing NEC-81 GN 0, GN 2, and NEC-5.'
    )
    parser.add_argument(
        '--nec5',
        default='',
        help='Path or command name for NEC-5 executable',
    )
    args = parser.parse_args()

    nec81Exe = findExe('nec81', required=True)
    nec5Exe = findNec5(args.nec5)
    if nec5Exe:
        print(f'Using NEC-5 executable: {nec5Exe}\n')
    else:
        print('NEC-5 executable not found (pass --nec5 /path/to/nec5 to include).\n')

    wlam_m = 299.792458/7.15
    heights_lam = (0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50)

    hdr = (
        f'{"Height":<13} {"GN 0 Z":<14} {"GN 2 Z":<14} {"NEC-5 Z":<14} '
        f'{"GN 0 G":>9} {"GN 2 G":>9} {"NEC-5 G":>9}'
    )
    print(hdr)
    print('-'*len(hdr))

    with tempfile.TemporaryDirectory(prefix='nec81_hsweep_') as tmpDir:
        tmpPath = Path(tmpDir)
        for hLam in heights_lam:
            h_m = hLam*wlam_m
            res = {}
            for gnType in (0, 2):
                deckPath = tmpPath/f'd_{gnType}.nec'
                outPath = tmpPath/f'd_{gnType}.out'
                deckPath.write_text(buildNec81Deck(gnType, h_m), encoding='utf-8')
                subprocess.run(
                    (nec81Exe, str(deckPath), str(outPath), '--no-plots3d'),
                    input='\n',
                    text=True,
                    capture_output=True,
                    cwd=tmpDir,
                    check=True,
                )
                res[gnType] = parseReport(outPath)

            if nec5Exe:
                deck5 = tmpPath/'d_nec5.nec'
                out5 = tmpPath/'d_nec5.out'
                if out5.exists():
                    out5.unlink()
                deck5.write_text(buildNec5Deck(h_m), encoding='utf-8')
                runNec5(nec5Exe, deck5, out5, tmpDir)
                res[5] = parseReport(out5)
            else:
                res[5] = (None, None)

            z0, g0 = res.__getitem__(0)
            z2, g2 = res.__getitem__(2)
            z5, g5 = res.__getitem__(5)
            hStr = f'{hLam:4.2f}l ({h_m:4.1f}m)'
            print(
                f'{hStr:<13} {fmtZ(z0):<14} {fmtZ(z2):<14} {fmtZ(z5):<14} '
                f'{fmtG(g0):>9} {fmtG(g2):>9} {fmtG(g5):>9}'
            )


if __name__ == '__main__':
    main()

