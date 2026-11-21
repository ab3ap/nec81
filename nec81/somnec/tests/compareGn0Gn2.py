#!/usr/bin/env python

from pathlib import Path
import re, shutil
import subprocess
import sys
import tempfile


def buildDeck(gnType):
    # 40m half-wave dipole at h = 0.05 lambda (2.10 m at 7.15 MHz)
    return f"""CM Low 40m dipole (h = 0.05 lambda) GN {gnType} comparison
CE
GW 1 21 -10.0 0.0 2.10 10.0 0.0 2.10 0.001
GE 1
GN {gnType} 0 0 0 13.0 0.005
FR 0 1 0 0 7.15 0.0
EX 0 1 11 0 1.0 0.0
RP 0 19 37 1000 0.0 0.0 5.0 10.0
EN
"""


def findExe(name):
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).parent/name
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError(f'Console command not found: {name}')


def parseReport(reportPath):
    text = Path(reportPath).read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    zIn = None
    effPct = None
    zenithGain_dBi = None
    maxGain_dBi = -999.0

    for idx, line in enumerate(lines):
        if 'ANTENNA INPUT PARAMETERS' in line:
            for sub in lines[idx + 1 : idx + 10]:
                parts = sub.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    nums = re.findall(r'[+-]?\d+\.\d+E[+-]\d+', sub)
                    if len(nums) >= 6:
                        # nums[0..1] = V, nums[2..3] = I, nums[4..5] = Z (R, X)
                        zIn = complex(float(nums[4]), float(nums[5]))
                        break
        if 'EFFICIENCY    =' in line:
            effPct = float(line.split('=').__getitem__(1).split().__getitem__(0))
        if 'RADIATION PATTERNS' in line:
            for sub in lines[idx + 5 :]:
                parts = sub.split()
                if len(parts) < 5:
                    if zenithGain_dBi is not None:
                        break
                    continue
                try:
                    theta_deg = float(parts.__getitem__(0))
                    total_dBi = float(parts.__getitem__(4))
                    if abs(theta_deg) < 1e-3 and zenithGain_dBi is None:
                        zenithGain_dBi = total_dBi
                    if total_dBi > maxGain_dBi:
                        maxGain_dBi = total_dBi
                except ValueError:
                    continue

    return zIn, effPct, zenithGain_dBi, maxGain_dBi


def main():
    nec81Exe = findExe('nec81')
    results = {}

    with tempfile.TemporaryDirectory(prefix='nec81_gn_cmp_') as tmpDir:
        tmpPath = Path(tmpDir)
        for gnType in (0, 2):
            deckPath = tmpPath/f'dipole_gn{gnType}.nec'
            outPath = tmpPath/f'dipole_gn{gnType}.out'
            deckPath.write_text(buildDeck(gnType), encoding='utf-8')

            cmd = (nec81Exe, str(deckPath), str(outPath), '--no-plots3d')
            subprocess.run(
                cmd,
                input='\n',
                text=True,
                capture_output=True,
                cwd=tmpDir,
                check=True,
            )
            results[gnType] = parseReport(outPath)

    print('Low 40m Dipole (h = 0.05 lambda = 2.10 m, 7.15 MHz, Average Soil)')
    print('-'*68)
    print(f'{"Model":<26} {"Feedpoint Z (ohms)":<22} {"Zenith Gain":>12}')
    print('-'*68)

    labels = {
        0: 'GN 0 (Reflection Approx)',
        2: 'GN 2 (SOMNEC Exact)',
    }
    for gnType in (0, 2):
        zIn, effPct, zGain, maxGain = results[gnType]
        zStr = f'{zIn.real:6.2f} {zIn.imag:+6.2f}j'
        print(f'{labels[gnType]:<26} {zStr:<22} {zGain:9.2f} dBi')


if __name__ == '__main__':
    main()

