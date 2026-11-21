#!/usr/bin/env python

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SWEEP_DECK = """CM 20m vertical 3-frequency sweep over Sommerfeld ground
CE
GW 1 21 0.0 0.0 0.05 0.0 0.0 5.05 0.002
GE 1
GN 2 0 0 0 13.0 0.005
FR 0 3 0 0 14.0 0.2
EX 0 1 1 0 1.0 0.0
XQ
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


def main():
    nec81Exe = findExe('nec81')
    with tempfile.TemporaryDirectory(prefix='nec81_som_sweep_') as tmpDir:
        tmpPath = Path(tmpDir)
        deckPath = tmpPath/'sweepSomnec.nec'
        outPath = tmpPath/'sweepSomnec.out'
        deckPath.write_text(SWEEP_DECK, encoding='utf-8')

        cmd = (nec81Exe, str(deckPath), str(outPath), '--no-plots3d')
        print('Running 3-frequency GN 2 sweep (auto-compute):')
        proc = subprocess.run(
            cmd,
            input='\n',
            text=True,
            capture_output=True,
            cwd=tmpDir,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise RuntimeError(f'Sweep failed with code {proc.returncode}')

        report = outPath.read_text(encoding='utf-8', errors='replace')
        freqsFound = [
            line.strip()
            for line in report.splitlines()
            if 'FREQUENCY=' in line
        ]
        for line in freqsFound:
            print(' ', line)

        if len(freqsFound) == 3:
            print('PASS: Multi-frequency GN 2 sweep & cache verified.')
        else:
            raise RuntimeError(f'Expected 3 frequency steps, got {len(freqsFound)}')


if __name__ == '__main__':
    main()

