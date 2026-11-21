#!/usr/bin/env python

import re
from pathlib import Path


def parseReport(reportPath):
    text = Path(reportPath).read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()

    zIn = None
    effPct = None
    maxGain_dBi = -999.0
    maxTheta_deg = None

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
                    if maxTheta_deg is not None:
                        break
                    continue
                try:
                    theta_deg = float(parts.__getitem__(0))
                    total_dBi = float(parts.__getitem__(4))
                    if total_dBi > maxGain_dBi:
                        maxGain_dBi = total_dBi
                        maxTheta_deg = theta_deg
                except ValueError:
                    continue

    return zIn, effPct, maxGain_dBi, maxTheta_deg


def main():
    reportFile = Path('testSomnec.out')
    if not reportFile.is_file():
        raise FileNotFoundError(
            'testSomnec.out not found; run testSomnecNec81.py first.'
        )

    zIn, effPct, maxGain_dBi, maxTheta_deg = parseReport(reportFile)
    print(f'Feedpoint Z_in  : {zIn.real:.2f} {zIn.imag:+.2f}j ohms')
    print(f'Efficiency      : {effPct:.2f} %')
    print(f'Peak Gain       : {maxGain_dBi:.2f} dBi at theta = {maxTheta_deg:.1f} deg')

    okZ = zIn is not None and (40.0 <= zIn.real <= 75.0) and (-30.0 <= zIn.imag <= 30.0)
    okGain = (-3.0 <= maxGain_dBi <= 3.0)
    if okZ and okGain:
        print('PASS: Values match reference NEC-2 Sommerfeld-Norton behavior.')
    else:
        print('WARN: Check testSomnec.out values against reference.')


if __name__ == '__main__':
    main()

