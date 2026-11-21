#!/usr/bin/env python

import os
import shutil
import subprocess
import sys


DECK_TEXT = """CM Quarter-wave vertical at 14.2 MHz over Sommerfeld ground
CE
GW 1 21 0.0 0.0 0.05 0.0 0.0 5.05 0.002
GE 1
GN 2 0 0 0 13.0 0.005
FR 0 1 0 0 14.2 0.0
EX 0 1 1 0 1.0 0.0
RP 0 19 1 1000 0.0 0.0 5.0 0.0
EN
"""


def findExe(name):
    found = shutil.which(name)
    if found:
        return found
    binDir = os.path.dirname(sys.executable)
    candidate = os.path.join(binDir, name)
    if os.path.exists(candidate):
        return candidate
    raise RuntimeError(f'Console command not found: {name}')


def main():
    deckFile = 'testSomnec.nec'
    outFile = 'testSomnec.out'
    gridFile = 'fort.21'

    somnecExe = findExe('somnec')
    nec81Exe = findExe('nec81')

    # Step 1: Generate Sommerfeld-Norton ground grid
    cmd1 = (
        somnecExe,
        '-e', '13.0',
        '-s', '0.005',
        '-f', '14.2',
        '-o', gridFile,
    )
    print('Step 1:', ' '.join(cmd1))
    subprocess.run(cmd1, check=True)
    if not os.path.exists(gridFile):
        raise RuntimeError('fort.21 was not created')

    # Step 2: Write GN 2 deck and run nec81
    with open(deckFile, 'w', encoding='utf-8') as fDeck:
        fDeck.write(DECK_TEXT)

    cmd2 = (nec81Exe, deckFile, outFile, '--no-plots3d')
    print('Step 2:', ' '.join(cmd2))
    proc = subprocess.run(
        cmd2,
        input=f'{gridFile}\n',
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f'nec81 failed with code {proc.returncode}')

    # Step 3: Verify Sommerfeld ground & impedance in report
    with open(outFile, 'r', encoding='utf-8') as fOut:
        report = fOut.read()

    foundSom = False
    foundImp = False
    for line in report.splitlines():
        if 'SOMMERFELD SOLUTION' in line:
            foundSom = True
            print(' ', line.strip())
        if 'IMPEDANCE' in line or '14.200' in line:
            print(' ', line.strip())
        if 'IMPEDANCE' in line:
            foundImp = True

    if foundSom and foundImp:
        print('SUCCESS: somnec -> nec81 GN 2 pipeline verified.')
    else:
        print('WARNING: Check testSomnec.out for solver details.')


if __name__ == '__main__':
    main()

