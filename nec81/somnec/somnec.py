#!/usr/bin/env python

try:
    from .sommerfeldIntegrals import SommerfeldEvaluator
except ImportError:
    from sommerfeldIntegrals import SommerfeldEvaluator
import argparse
import numpy as np
import struct
import sys
import time


class SomnecGridGenerator:
    def __init__(self):
        self.eval = SommerfeldEvaluator()

        # COMMON /GGRID/ AR1(11,10,4),AR2(17,5,4),AR3(9,8,4)
        self.ar1 = np.zeros((11, 10, 4), dtype=np.complex128)
        self.ar2 = np.zeros((17, 5, 4), dtype=np.complex128)
        self.ar3 = np.zeros((9, 8, 4), dtype=np.complex128)
        self.epscf = 0j

        self.dxa = np.array((0.02, 0.05, 0.1), dtype=np.float64)
        self.dya = np.array(
            (0.1745329252, 0.0872664626, 0.1745329252), dtype=np.float64
        )
        self.xsa = np.array((0.0, 0.2, 0.2), dtype=np.float64)
        self.ysa = np.array((0.0, 0.0, 0.3490658504), dtype=np.float64)
        self.nxa = np.array((11, 17, 9), dtype=np.int32)
        self.nya = np.array((10, 5, 8), dtype=np.int32)

    def computeGrids(self, epr, sig, freq_MHz):
        # IF SIG .LT. 0. THEN COMPLEX DIELECTRIC CONSTANT = EPR + J*SIG
        # AND FMHZ IS NOT USED
        if sig < 0.0:
            self.epscf = complex(epr, sig)
        else:
            wlam = 299.8/freq_MHz
            self.epscf = complex(epr, -sig*wlam*59.96)

        ev = self.eval
        ev.ck2 = 6.283185308
        ev.ck2sq = ev.ck2*ev.ck2

        # SOMMERFELD INTEGRAL EVALUATION USES EXP(-JWT), NEC USES EXP(+JWT),
        # HENCE NEED CONJG(EPSCF). CONJUGATE OF FIELDS OCCURS IN SUBROUTINE
        # EVALUA.
        ev.ck1sq = ev.ck2sq*np.conj(self.epscf)
        ev.ck1 = np.sqrt(ev.ck1sq)
        ev.ck1r = ev.ck1.real
        ev.tkmag = 100.0*abs(ev.ck1)
        ev.tsmag = (100.0*ev.ck1*np.conj(ev.ck1)).real
        ev.cksm = ev.ck2sq/(ev.ck1sq + ev.ck2sq)
        ev.ct1 = 0.5*(ev.ck1sq - ev.ck2sq)

        erv = ev.ck1sq*ev.ck1sq
        ezv = ev.ck2sq*ev.ck2sq
        ev.ct2 = 0.125*(erv - ezv)
        erv = erv*ev.ck1sq
        ezv = ezv*ev.ck2sq
        ev.ct3 = 0.0625*(erv - ezv)

        grids = (self.ar1, self.ar2, self.ar3)

        # LOOP OVER 3 GRID REGIONS
        for k in range(3):
            nr = int(self.nxa.item(k))
            nth = int(self.nya.item(k))
            dr = float(self.dxa.item(k))
            dth = float(self.dya.item(k))
            r = float(self.xsa.item(k)) - dr
            irs = 0
            if k == 0:
                r = float(self.xsa.item(k))
                irs = 1

            # LOOP OVER R. (R=SQRT(RHO**2 + (Z+H)**2))
            for ir in range(irs, nr):
                r = r + dr
                thet = float(self.ysa.item(k)) - dth

                # LOOP OVER THETA. (THETA=ATAN((Z+H)/RHO))
                for ith in range(nth):
                    thet = thet + dth
                    ev.rho = r*np.cos(thet)
                    ev.zph = r*np.sin(thet)
                    if ev.rho < 1.0e-7:
                        ev.rho = 1.0e-8
                    if ev.zph < 1.0e-7:
                        ev.zph = 0.0

                    erv, ezv, erh, eph = ev.evlua()
                    rk = ev.ck2*r
                    denom = complex(np.cos(rk), -np.sin(rk))
                    con = -complex(0.0, 4.77147)*r/denom
                    gridK = grids.__getitem__(k)
                    gridK.__setitem__((ir, ith, 0), erv*con)
                    gridK.__setitem__((ir, ith, 1), ezv*con)
                    gridK.__setitem__((ir, ith, 2), erh*con)
                    gridK.__setitem__((ir, ith, 3), eph*con)

        # FILL GRID 1 FOR R EQUAL TO ZERO.
        cl2 = -complex(0.0, 188.370)*(self.epscf - 1.0)/(self.epscf + 1.0)
        cl1 = cl2/(self.epscf + 1.0)
        ezv = self.epscf*cl1
        dth = float(self.dya.item(0))
        thet = -dth
        nth = int(self.nya.item(0))

        for ith in range(nth):
            thet = thet + dth
            if ith == nth - 1:
                erv = 0.0 + 0j
                erh = cl2 - 0.5*cl1
                eph = -erh
            else:
                tfac2 = np.cos(thet)
                tfac1 = (1.0 - np.sin(thet))/tfac2
                tfac2 = tfac1/tfac2
                erv = self.epscf*cl1*tfac1
                erh = cl1*(tfac2 - 1.0) + cl2
                eph = cl1*tfac2 - cl2

            self.ar1.__setitem__((0, ith, 0), erv)
            self.ar1.__setitem__((0, ith, 1), ezv)
            self.ar1.__setitem__((0, ith, 2), erh)
            self.ar1.__setitem__((0, ith, 3), eph)

    def saveTape21(self, filename, fortranRecord=False):
        # WRITE GRID ON TAPE21
        # ORDER: AR1, AR2, AR3, EPSCF, DXA, DYA, XSA, YSA, NXA, NYA
        buf = bytearray()
        for arr in (self.ar1, self.ar2, self.ar3):
            flat = np.ravel(arr, order='F').astype(np.complex64)
            buf.extend(flat.tobytes())
        buf.extend(np.array((self.epscf,), dtype=np.complex64).tobytes())
        for arr in (self.dxa, self.dya, self.xsa, self.ysa):
            buf.extend(arr.astype(np.float32).tobytes())
        for arr in (self.nxa, self.nya):
            buf.extend(arr.astype(np.int32).tobytes())

        with open(filename, 'wb') as fOut:
            if fortranRecord:
                recLen = len(buf)
                fOut.write(struct.pack('<I', recLen))
                fOut.write(buf)
                fOut.write(struct.pack('<I', recLen))
            else:
                fOut.write(buf)

    def saveGrids(self, filename):
        # WRITE GRID ON TAPE21 (SAVED AS NUMPY ARCHIVE)
        np.savez(
            filename,
            ar1=self.ar1,
            ar2=self.ar2,
            ar3=self.ar3,
            epscf=self.epscf,
            dxa=self.dxa,
            dya=self.dya,
            xsa=self.xsa,
            ysa=self.ysa,
            nxa=self.nxa,
            nya=self.nya,
        )

    def printGrids(self):
        # PRINT GRID
        lcomp = ('ERV', 'EZV', 'ERH', 'EPH')
        grids = (self.ar1, self.ar2, self.ar3)
        print(
            f' NEC GROUND INTERPOLATION GRID\n'
            f' DIELECTRIC CONSTANT={self.epscf.real:12.5E}'
            f'{self.epscf.imag:12.5E}'
        )
        for k in range(3):
            nr = int(self.nxa.item(k))
            nth = int(self.nya.item(k))
            r1 = float(self.xsa.item(k))
            dr = float(self.dxa.item(k))
            th1 = float(self.ysa.item(k))
            dth = float(self.dya.item(k))
            print(
                f'\n\n\n GRID{k + 1:2d}\n'
                f'    R(1)={r1:7.4f}    DR={dr:7.4f}    NR={nr:3d}\n'
                f' THET(1)={th1:7.4f}   DTH={dth:7.4f}   NTH={nth:3d}\n\n'
            )
            gridK = grids.__getitem__(k)
            for lIdx in range(4):
                print(f'\n\n\n {lcomp.__getitem__(lIdx)}')
                for ir in range(nr):
                    print(f' IR={ir + 1:3d}')
                    vals = list()
                    for ith in range(nth):
                        val = gridK.item((ir, ith, lIdx))
                        vals.append(f'{val.real:12.5E}{val.imag:12.5E}')
                    lineStr = ''.join(vals)
                    for idx in range(0, len(lineStr), 120):
                        endIdx = idx + 120
                        chunk = ''.join(
                            lineStr.__getitem__(c)
                            for c in range(idx, min(endIdx, len(lineStr)))
                        )
                        print(' ' + chunk)

def parseArgs():
    parser = argparse.ArgumentParser(
        description='Generate NEC2 Sommerfeld-Norton ground grids.'
    )
    parser.add_argument(
        '-e', '--epr', type=float, help='Dielectric constant'
    )
    parser.add_argument(
        '-s', '--sig', type=float, help='Conductivity (S/m)'
    )
    parser.add_argument(
        '-f', '--freq', type=float, help='Frequency (MHz)'
    )
    parser.add_argument(
        '-p', '--print-grids', action='store_true', help='Print grid tables'
    )
    parser.add_argument(
        '-o', '--out', default='fort.21', help='Output file (default: fort.21)'
    )
    parser.add_argument(
        '-r', '--fortran', action='store_true',
        help='Wrap output in 4-byte Fortran record markers',
    )
    parser.add_argument(
        '-z', '--npz', action='store_true', help='Save .npz instead of fort.21'
    )
    return parser.parse_args()

def main():
    # PROGRAM SOMNEC
    # PROGRAM TO GENERATE NEC INTERPOLATION GRIDS FOR FIELDS DUE TO
    # GROUND. FIELD COMPONENTS ARE COMPUTED BY NUMERICAL EVALUATION
    # OF MODIFIED SOMMERFELD INTEGRALS.
    args = parseArgs()
    hasCli = (
        args.epr is not None
        and args.sig is not None
        and args.freq is not None
    )
    if hasCli:
        epr = args.epr
        cond_Spm = args.sig
        freq_MHz = args.freq
        ipt = 1 if args.print_grids else 0
        otfile = args.out
    else:
        print(' Program to Calculate Ground Interpolation Grid')
        print(' For NEC2 Using Sommerfeld-Norton Method')
        print(' ')
        # READ GROUND PARAMETERS - EPR = RELATIVE DIELECTRIC CONSTANT
        #                          SIG = CONDUCTIVITY (MHOS/M)
        #                          FMHZ = FREQUENCY (MHZ)
        #                          IPT = 1 TO PRINT GRIDS. =0 OTHERWISE.
        epr = float(input(' Enter Relative Dielectric Constant: '))
        cond_Spm = float(input(' Enter Conductivity (Mhos/Meter): '))
        freq_MHz = float(input(' Enter Frequency (MHz): '))
        ipt = int(input(' Enter 1 to Print Grids, 0 to Suppress Printing: '))
        otfile = input(' Enter Data Output Filename: ').strip()

    print(f' Relative Dielectric Constant = {epr}')
    print(f' Conductivity (Mhos/Meter) = {cond_Spm}')
    print(f' Frequency, MHz = {freq_MHz}')
    print(f' Printing Flag = {ipt}')
    print(f' Data Output File Name = {otfile}')

    tStart = time.perf_counter()
    gen = SomnecGridGenerator()
    gen.computeGrids(epr, cond_Spm, freq_MHz)
    elapsed_s = time.perf_counter() - tStart

    try:
        if args.npz or otfile.endswith('.npz'):
            gen.saveGrids(otfile)
        else:
            gen.saveTape21(otfile, fortranRecord=args.fortran)
    except Exception:
        print(f'ERROR CREATING OUTPUT FILE = {otfile}')
        sys.exit(1)

    if ipt != 0:
        gen.printGrids()

    print(f' TIME={elapsed_s:12.3E} SECONDS')


if __name__ == '__main__':
    main()

