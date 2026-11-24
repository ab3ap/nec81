#!/usr/bin/env python

import math, os, struct
import numpy as npy
from .nec_state import NECState, LD, LD2, LD3, LD4, LOADMX, NSMAX, NETMX, NORMF, CVEL, TA
from .main2 import cang, db20, isegno, second
from .nsmaExport import setAdfMetadata, writeAdfFile
from .plots3d import solverPatchPolygons, plotAntennaGeometry, finalizePlots, recordSweepPoint, plotSession


# Stub imports for Batches 2, 3, and 4
from .datagn import datagn
from .fbngf import fbngf, fblock, gfout
from .load import load
from .cmset import cmset, cmngf
from .factrs import factrs, facgf
from .etmns import etmns
from .netwk import netwk
from .couple import couple
from .nfpat import nfpat
from .rdpat import rdpat

def parse_nec_card(line: str):
    """Parses a NEC-2 control card supporting both strict fixed-column and free-format (comma/space)."""
    raw = line.rstrip("\r\n")
    ain = raw[:2].upper()
    rest = raw[2:].strip()

    def parse_int(s: str) -> int:
        s = s.strip()
        return int(float(s)) if s else 0

    def parse_float(s: str) -> float:
        s = s.strip().replace("D", "E").replace("d", "e")
        return float(s) if s else 0.0

    if "," in rest or "\t" in rest:
        tokens = [t.strip() for t in rest.replace(",", " ").split() if t.strip()]
    else:
        padded = raw.ljust(80)
        fields = [
            padded[2:5], padded[5:10], padded[10:15], padded[15:20],
            padded[20:30], padded[30:40], padded[40:50],
            padded[50:60], padded[60:70], padded[70:80],
        ]
        try:
            _ = [parse_int(f) for f in fields[:4]]
            _ = [parse_float(f) for f in fields[4:]]
            tokens = fields
        except ValueError:
            tokens = [t.strip() for t in rest.split() if t.strip()]

    t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = (tokens + [""]*10)[:10]
    return (
        ain,
        parse_int(t0),
        parse_int(t1),
        parse_int(t2),
        parse_int(t3),
        parse_float(t4),
        parse_float(t5),
        parse_float(t6),
        parse_float(t7),
        parse_float(t8),
        parse_float(t9),
    )


def run_nec(
    input_path: str = None,
    output_path: str = None,
    somnecPath: str = None,
):
    state = NECState()
    state.geometry_plotted = False
    hpol = ["LINEAR", "RIGHT ", "LEFT  "]
    pnet = ["        ", "STRAIGHT", "CROSSED "]

    ip = npy.zeros(LD2, dtype=npy.int32)
    ix = npy.zeros(LD2, dtype=npy.int32)
    icon1 = npy.zeros(LD, dtype=npy.int32)
    icon2 = npy.zeros(LD, dtype=npy.int32)
    itag = npy.zeros(LD, dtype=npy.int32)
    iconx = npy.zeros(LD, dtype=npy.int32)

    ldtyp = npy.zeros(LOADMX, dtype=npy.int32)
    ldtag = npy.zeros(LOADMX, dtype=npy.int32)
    ldtagf = npy.zeros(LOADMX, dtype=npy.int32)
    ldtagt = npy.zeros(LOADMX, dtype=npy.int32)
    zlr = npy.zeros(LOADMX, dtype=npy.float64)
    zli = npy.zeros(LOADMX, dtype=npy.float64)
    zlc = npy.zeros(LOADMX, dtype=npy.float64)

    fnorm = npy.zeros(NORMF, dtype=npy.float64)
    gain = npy.zeros(LD4, dtype=npy.float64)
    x = npy.zeros(LD, dtype=npy.float64)
    y = npy.zeros(LD, dtype=npy.float64)
    z = npy.zeros(LD, dtype=npy.float64)
    bi = npy.zeros(LD, dtype=npy.float64)
    salp = npy.zeros(LD, dtype=npy.float64)

    air = npy.zeros(LD, dtype=npy.float64)
    aii = npy.zeros(LD, dtype=npy.float64)
    bir = npy.zeros(LD, dtype=npy.float64)
    bii = npy.zeros(LD, dtype=npy.float64)
    cir = npy.zeros(LD, dtype=npy.float64)
    cii = npy.zeros(LD, dtype=npy.float64)

    cur = npy.zeros(LD3, dtype=npy.complex64)
    rhs = npy.zeros(LD3, dtype=npy.complex64)
    zarray = npy.zeros(LD, dtype=npy.complex128)
    scratc = npy.zeros(LD2, dtype=npy.complex128)

    xtemp = npy.zeros(LD, dtype=npy.float64)
    ytemp = npy.zeros(LD, dtype=npy.float64)
    ztemp = npy.zeros(LD, dtype=npy.float64)
    sitemp = npy.zeros(LD, dtype=npy.float64)
    bitemp = npy.zeros(LD, dtype=npy.float64)

    t1x = npy.zeros(LD, dtype=npy.float64)
    t1y = npy.zeros(LD, dtype=npy.float64)
    t1z = npy.zeros(LD, dtype=npy.float64)
    t2x = npy.zeros(LD, dtype=npy.float64)
    t2y = npy.zeros(LD, dtype=npy.float64)
    t2z = npy.zeros(LD, dtype=npy.float64)

    print("\n" + " "*26 + "N.E.C. PROGRAM NEC-81")
    print(" "*15 + "v2.2, COPYRIGHT 1989, DAVID J. PINION, P.E.\n")
    print("\n" + " "*26 + "Modified by OH3MCK 1991 for SUN-4\n")

    extim = second()
    state.nxa.fill(0)

    if input_path is None:
        input_path = input(" OPEN UNIT 23 FOR NEC INPUT  FILE: ").strip()
    if output_path is None:
        output_path = input(" OPEN UNIT 24 FOR NEC OUTPUT FILE: ").strip()

    ir = open(input_path, "r")
    iw = open(output_path, "w")
    igfl = 20

    restart_deck = True
    while restart_deck:
        state.cm.fill(0j)
        restart_deck = False
        state.kcom = 0
        ifrtmw = 0
        ifrtmp = 0
        # IFRTMW/IFRTMP must survive between executions: once set, the
        # unscaled geometry in XTEMP... is kept and never re-copied from the
        # already frequency-scaled X, Y, Z arrays.
        state.ifrtmw = 0
        state.ifrtmp = 0

        while True:
            state.kcom = min(state.kcom + 1, 5)
            line = ir.readline()
            if not line:
                raise EOFError("Unexpected EOF while reading comment cards")
            ain = line[:2]
            comment_text = line[2:].rstrip("\r\n").ljust(76)[:76]
            for i in range(19):
                state.com[i][state.kcom - 1] = comment_text[i*4 : (i + 1)*4]

            if state.kcom == 1:
                iw.write("\n\n\n" + " "*33 + "*"*36 + "\n")
                iw.write(" "*36 + "NUMERICAL ELECTROMAGNETICS CODE\n")
                iw.write(" "*33 + "*"*36 + "\n")
                iw.write("\n\n\n\n" + " "*37 + "- - - - COMMENTS - - - -\n\n")

            iw.write(" "*25 + f"{ain}{comment_text}\n")
            if ain == "CM":
                continue
            if ain == "CE":
                break
            iw.write("\n\n\n" + " "*10 + "INCORRECT LABEL FOR A COMMENT CARD\n")
            ir.close()
            iw.close()
            return

        zarray.fill(0j)
        mpcnt = 0
        state.imat = 0

        datagn(
            state, state.cm, zarray, x, y, z, bi, salp,
            t1x, t1y, t1z, t2x, t2y, t2z,
            icon1, icon2, itag, iconx, ip,
            LD, LD2, state.iresrv, ir, iw, igfl
        )
        neq = state.n + 2*state.m
        npeq = state.np + 2*state.mp
        # Symmetric sub-matrix views A(KA) with leading dimension NEQ reach
        # up to NEQ elements past NEQ*NPEQ, so keep that much slack.
        needed = max(1, neq*npeq + neq + npeq)
        # If Numerical Green's Function is used, include coupled submatrices:
        if state.imat != 0 or state.n1 > 0 or state.m1 > 0:
            neq_new = (state.n - state.n1) + 2*(state.m - state.m1) + state.nscon + 2*state.npcon
            neq_ngf = state.n1 + 2*state.m1
            needed = max(needed, state.imat + neq_new*neq_new + 2*neq_new*neq_ngf)
        state.ensure_matrix_capacity(needed)

        if state.n > 0:
            dims = (
                float(npy.ptp(x[:state.n]))*state.wlam,
                float(npy.ptp(y[:state.n]))*state.wlam,
                float(npy.ptp(z[:state.n]))*state.wlam,
            )
            setAdfMetadata(dimensions_m=dims)

        if state.imat != 0:
            state.neq = 2*(state.m - state.m1 + state.npcon)
            state.neq2 = state.n - state.n1 + state.nscon + state.neq
            state.neq = state.n1 + 2*state.m1
            ib11, ic11, id11, ix11 = fbngf(state, state.neq, state.neq2,
                state.iresrv, iw)
        else:
            state.neq = state.n + 2*state.m
            state.neq2 = 0
            ib11 = ic11 = id11 = ix11 = 1
            state.icasx = 0

        state.npeq = state.np + 2*state.mp
        iw.write("\n"*5)

        state.iplp1 = state.iplp2 = state.iplp3 = state.iplp4 = 0
        igo = 1
        iflow = 0
        ifrq = 0
        nfrq = 1
        delfrq = 0.0
        state.rkh = 1.0
        state.iexk = 0
        state.ixtyp = 0
        state.nload = 0
        state.nonet = 0
        state.near = -1
        iptflg = -2
        iptflq = -1
        state.ifar = -1
        state.zrati = 1.0 + 0j
        iped = 0
        irngf = 0
        state.ncoup = 0
        state.icoup = 0
        zpnorm = 0.0
        ypnorm = 1.0e69
        nthi = nphi = 1
        xpr1 = xpr2 = xpr3 = xpr4 = xpr5 = 0.0
        thetis = phiss = 0.0
        iptag = iptagf = iptagt = 0
        iptaq = iptaqf = iptaqt = 0

        if state.icasx == 0:
            state.fmhz = CVEL
            state.nlodf = 0
            state.ksymp = 1
            state.nradl = 0
            state.iperf = 0

        while True:
            line = ir.readline()
            if not line:
                break
            ain, itmp1, itmp2, itmp3, itmp4, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6 = parse_nec_card(line)
            mpcnt += 1
            iw.write(
                f" ***** DATA CARD NO.{mpcnt:3d}   {ain:2s} {itmp1:3d} "
                f"{itmp2:5d} {itmp3:5d} {itmp4:5d} "
                f"{tmp1:12.5E} {tmp2:12.5E} {tmp3:12.5E} "
                f"{tmp4:12.5E} {tmp5:12.5E} {tmp6:12.5E}\n"
            )

            trigger_exec = False

            if ain == "FR":
                ifrq = itmp1
                if state.icasx != 0:
                    iw.write(f"\n ERROR - {ain:2s} CARD IS NOT ALLOWED WITH N.G.F.\n")
                    break
                nfrq = itmp2 if itmp2 != 0 else 1
                state.fmhz = tmp1
                delfrq = tmp2
                if iped == 1:
                    zpnorm = 0.0
                    ypnorm = 1.0e69
                igo = 1
                iflow = 1

            elif ain == "KH":
                state.rkh = tmp1
                if igo > 2:
                    igo = 2
                iflow = 1

            elif ain == "EK":
                state.iexk = 0 if itmp1 == -1 else 1
                if igo > 2:
                    igo = 2
                iflow = 1

            elif ain == "CP":
                if iflow != 2:
                    state.ncoup = 0
                state.icoup = 0
                iflow = 2
                if itmp2 != 0:
                    state.ncoup += 1
                    if state.ncoup > 5:
                        iw.write("\n NUMBER OF SEGMENTS IN COUPLING CALCULATION (CP) EXCEEDS LIMIT\n")
                        break
                    state.nctag[state.ncoup - 1] = itmp1
                    state.ncseg[state.ncoup - 1] = itmp2
                    if itmp4 != 0:
                        state.ncoup += 1
                        if state.ncoup > 5:
                            iw.write("\n NUMBER OF SEGMENTS IN COUPLING CALCULATION (CP) EXCEEDS LIMIT\n")
                            break
                        state.nctag[state.ncoup - 1] = itmp3
                        state.ncseg[state.ncoup - 1] = itmp4

            elif ain == "LD":
                if iflow != 3:
                    state.nload = 0
                    iflow = 3
                    if igo > 2:
                        igo = 2
                    if itmp1 == -1:
                        continue
                state.nload += 1
                if state.nload > LOADMX:
                    iw.write("\n\n\n          NUMBER OF LOADING CARDS EXCEEDS STORAGE ALLOTTED\n")
                    break
                idx = state.nload - 1
                ldtyp[idx] = itmp1
                ldtag[idx] = itmp2
                if itmp4 == 0:
                    itmp4 = itmp3
                ldtagf[idx] = itmp3
                ldtagt[idx] = itmp4
                if itmp4 < itmp3:
                    iw.write(
                        f"\n\n\n          DATA FAULT ON LOADING CARD NO.={state.nload:5d}     "
                        f"ITAG STEP1={itmp3:5d}  IS GREATER THAN ITAG STEP2={itmp4:5d}\n"
                    )
                    break
                zlr[idx] = tmp1
                zli[idx] = tmp2
                zlc[idx] = tmp3
            elif ain == "GN":
                iflow = 4
                if state.icasx != 0:
                    iw.write(f"\n ERROR - {ain:2s} CARD IS NOT ALLOWED WITH N.G.F.\n")
                    break
                if igo > 2:
                    igo = 2
                if itmp1 == -1:
                    state.ksymp = 1
                    state.nradl = 0
                    state.iperf = 0
                else:
                    state.iperf = itmp1
                    state.nradl = itmp2
                    state.ksymp = 2
                    state.epsr = tmp1
                    state.sig = tmp2
                    if state.nradl != 0:
                        if state.iperf == 2:
                            iw.write(" RADIAL WIRE G. S. APPROXIMATION MAY NOT BE USED WITH SOMMERFELD GROUND OPTION\n")
                            break
                        state.scrwlt = tmp3
                        state.scrwrt = tmp4
                    else:
                        state.epsr2 = tmp3
                        state.sig2 = tmp4
                        state.clt = tmp5
                        state.cht = tmp6

            elif ain == "EX":
                if iflow != 5:
                    state.nsant = 0
                    state.nvqd = 0
                    iped = 0
                    iflow = 5
                    if igo > 3:
                        igo = 3

                if 0 < itmp1 < 5:
                    if state.ixtyp == 0 or state.ixtyp >= 5:
                        state.ntsol = 0
                    state.ixtyp = itmp1
                    nthi = itmp2
                    nphi = itmp3
                    state.masym = itmp4 // 10
                    xpr1, xpr2, xpr3, xpr4, xpr5, state.xpr6 = tmp1, tmp2, tmp3, tmp4, tmp5, tmp6
                    state.nsant = 0
                    state.nvqd = 0
                    thetis = xpr1
                    phiss = xpr2
                else:
                    state.ixtyp = itmp1
                    state.ntsol = 0
                    if state.ixtyp > 5:
                        print(" ERROR: ILLEGAL EXCITATION TYPE")
                        continue
                    if state.ixtyp == 5:
                        state.nvqd += 1
                        if (state.nsant + state.nvqd) > NSMAX:
                            iw.write("\n\n\n          NUMBER OF EXCITATION CARDS EXCEEDS STORAGE ALLOTTED\n")
                            break
                        v_idx = state.nvqd - 1
                        state.ivqd[v_idx] = isegno(state, itmp2, itmp3, itag)
                        val = complex(tmp1, tmp2)
                        state.vqd[v_idx] = complex(1.0, 0.0) if abs(val) < 1.0e-20 else val
                    else:
                        state.nsant += 1
                        if (state.nsant + state.nvqd) > NSMAX:
                            iw.write("\n\n\n          NUMBER OF EXCITATION CARDS EXCEEDS STORAGE ALLOTTED\n")
                            break
                        s_idx = state.nsant - 1
                        state.isant[s_idx] = isegno(state, itmp2, itmp3, itag)
                        val = complex(tmp1, tmp2)
                        state.vsant[s_idx] = complex(1.0, 0.0) if abs(val) < 1.0e-20 else val

                    state.masym = itmp4 // 10
                    iped = itmp4 - state.masym*10
                    zpnorm = tmp3
                    ypnorm = 1.0/zpnorm if zpnorm > 0.0 else 1.0e69
                    if iped == 1 and zpnorm > 0.0:
                        iped = 2

            elif ain in ("NT", "TL"):
                if iflow != 6:
                    state.nonet = 0
                    state.ntsol = 0
                    iflow = 6
                    if igo > 3:
                        igo = 3
                    if itmp2 == -1:
                        continue
                state.nonet += 1
                if state.nonet > NETMX:
                    iw.write("\n\n\n          NUMBER OF NETWORK CARDS EXCEEDS STORAGE ALLOTTED\n")
                    break
                n_idx = state.nonet - 1
                state.ntyp[n_idx] = 1 if ain == "NT" else 2
                state.iseg1[n_idx] = isegno(state, itmp1, itmp2, itag)
                state.iseg2[n_idx] = isegno(state, itmp3, itmp4, itag)
                state.x11r[n_idx] = tmp1
                state.x11i[n_idx] = tmp2
                state.x12r[n_idx] = tmp3
                state.x12i[n_idx] = tmp4
                state.x22r[n_idx] = tmp5
                state.x22i[n_idx] = tmp6
                if state.ntyp[n_idx] != 1 and tmp1 <= 0.0:
                    state.ntyp[n_idx] = 3
                    state.x11r[n_idx] = -tmp1

            elif ain == "PL":
                state.iplp1, state.iplp2, state.iplp3, state.iplp4 = itmp1, itmp2, itmp3, itmp4

            elif ain == "PT":
                iptflg = itmp1
                iptag = itmp2
                iptagf = itmp3
                iptagt = itmp4
                if itmp3 == 0 and iptflg != -1:
                    iptflg = -2
                if itmp4 == 0:
                    iptagt = iptagf

            elif ain == "PQ":
                iptflq = itmp1
                iptaq = itmp2
                iptaqf = itmp3
                iptaqt = itmp4
                if itmp3 == 0 and iptflq != -1:
                    iptflq = -2
                if itmp4 == 0:
                    iptaqt = iptaqf

            elif ain in ("NE", "NH"):
                state.nfeh = 1 if ain == "NH" else 0
                if iflow == 8 and nfrq != 1:
                    iw.write(
                        "\n\n\n          WHEN MULTIPLE FREQUENCIES ARE REQUESTED, ONLY ONE NEAR FIELD CARD CAN BE USED -\n"
                        "          LAST CARD READ IS USED\n"
                    )
                state.near = itmp1
                state.nrx, state.nry, state.nrz = itmp2, itmp3, itmp4
                state.xnr, state.ynr, state.znr = tmp1, tmp2, tmp3
                state.dxnr, state.dynr, state.dznr = tmp4, tmp5, tmp6
                iflow = 8
                if nfrq == 1:
                    trigger_exec = True

            elif ain == "GD":
                state.epsr2, state.sig2, state.clt, state.cht = tmp1, tmp2, tmp3, tmp4
                iflow = 9

            elif ain == "RP":
                state.ifar = itmp1
                state.nth = itmp2 if itmp2 != 0 else 1
                state.nph = itmp3 if itmp3 != 0 else 1
                ipd = itmp4 // 10
                iavp = itmp4 - ipd*10
                inor = ipd // 10
                ipd = ipd - inor*10
                iax = inor // 10
                inor = inor - iax*10
                state.iax = 1 if iax != 0 else 0
                state.ipd = 1 if ipd != 0 else 0
                state.inor = inor
                state.iavp = 0 if (state.nth < 2 or state.nph < 2 or state.ifar == 1) else iavp
                state.thets, state.phis, state.dth, state.dph = tmp1, tmp2, tmp3, tmp4
                state.rfld, state.gnor = tmp5, tmp6
                iflow = 10
                trigger_exec = True

            elif ain == "WG":
                iflow = 12
                if state.icasx != 0:
                    iw.write(" ERROR - N.G.F. IN USE.  CANNOT WRITE NEW N.G.F.\n")
                    break
                irngf = state.iresrv // 2
                trigger_exec = True

            elif ain == "XQ":
                if iflow == 10 and itmp1 == 0:
                    continue
                if nfrq == 1 and itmp1 == 0 and iflow > 7:
                    continue
                if itmp1 == 0:
                    iflow = 11 if iflow > 7 else 7
                else:
                    state.ifar = 0
                    state.rfld = 0.0
                    state.ipd = state.iavp = state.inor = state.iax = 0
                    state.nth = 91
                    state.nph = 1
                    state.thets = 0.0
                    state.phis = 0.0
                    state.dth = 1.0
                    state.dph = 0.0
                    if itmp1 == 2:
                        state.phis = 90.0
                    elif itmp1 == 3:
                        state.nph = 2
                        state.dph = 90.0
                trigger_exec = True

            elif ain == "NX":
                restart_deck = True
                break

            elif ain == "EN":
                t_end = second() - extim
                if t_end < 0.0:
                    t_end += 86400.0
                iw.write(f"\n RUN TIME ={t_end/60.0:10.3f}\n")
                break

            else:
                iw.write("\n\n\n          FAULTY DATA CARD LABEL AFTER GEOMETRY SECTION\n")
                break

            if not trigger_exec:
                continue

            zpnorm, ypnorm, fnorm = _execute_frequency_sweep(
                state, state.cm, zarray, x, y, z, bi, salp,
                t1x, t1y, t1z, t2x, t2y, t2z,
                icon1, icon2, iconx, itag, ip, ix, cur, rhs, gain, scratc,
                air, aii, bir, bii, cir, cii,
                xtemp, ytemp, ztemp, sitemp, bitemp,
                ldtyp, ldtag, ldtagf, ldtagt, zlr, zli, zlc, fnorm,
                igo, iflow, ifrq, nfrq, delfrq, iped, zpnorm, ypnorm,
                iptflg, iptag, iptagf, iptagt,
                iptflq, iptaq, iptaqf, iptaqt,
                nthi, nphi, xpr1, xpr2, xpr3, xpr4, xpr5, thetis, phiss,
                irngf, ib11, ic11, id11, ix11, ifrtmw, ifrtmp, hpol, pnet,
                iw, igfl, somnecPath
            )
            igo = 3
            nfrq = 1

    ir.close()
    iw.close()
    writeAdfFile()
    finalizePlots()

def runNec81(inputFile=None, outputFile=None, somnecFile=None):
    """Compatibility wrapper expected by nec81.py."""
    return run_nec(
        input_path=inputFile,
        output_path=outputFile,
        somnecPath=somnecFile,
    )

def _execute_frequency_sweep(
    state, cm, zarray, x, y, z, bi, salp,
    t1x, t1y, t1z, t2x, t2y, t2z,
    icon1, icon2, iconx, itag, ip, ix, cur, rhs, gain, scratc,
    air, aii, bir, bii, cir, cii,
    xtemp, ytemp, ztemp, sitemp, bitemp,
    ldtyp, ldtag, ldtagf, ldtagt, zlr, zli, zlc, fnorm,
    igo, iflow, ifrq, nfrq, delfrq, iped, zpnorm, ypnorm,
    iptflg, iptag, iptagf, iptagt,
    iptflq, iptaq, iptaqf, iptaqt,
    nthi, nphi, xpr1, xpr2, xpr3, xpr4, xpr5, thetis, phiss,
    irngf, ib11, ic11, id11, ix11, ifrtmw, ifrtmp, hpol, pnet, iw, igfl,
    somnecPath=None
):
    mhz = 1
    fmhz1 = state.fmhz

    if igo <= 1:
        if state.n != 0 and getattr(state, "ifrtmw", ifrtmw) == 0:
            state.ifrtmw = 1
            xtemp[: state.n] = x[: state.n]
            ytemp[: state.n] = y[: state.n]
            ztemp[: state.n] = z[: state.n]
            sitemp[: state.n] = t1x[: state.n]
            bitemp[: state.n] = bi[: state.n]
        if state.m != 0 and getattr(state, "ifrtmp", ifrtmp) == 0:
            state.ifrtmp = 1
            start_m = LD - state.m
            xtemp[start_m:LD] = x[start_m:LD]
            ytemp[start_m:LD] = y[start_m:LD]
            ztemp[start_m:LD] = z[start_m:LD]
            bitemp[start_m:LD] = bi[start_m:LD]
        if state.imat == 0:
            fblock(state, state.npeq, state.neq, state.iresrv, irngf, state.ipsym, iw)

    while True:
        if igo <= 1:
            if mhz > 1:
                state.fmhz = state.fmhz*delfrq if ifrq == 1 else fmhz1 + (mhz - 1)*delfrq
            fr = state.fmhz/CVEL
            state.wlam = CVEL/state.fmhz
            iw.write(
                f"\n\n\n\n                                 - - - - - - FREQUENCY - - - - - -\n\n"
                f"                                    FREQUENCY={state.fmhz:11.4E} MHZ\n"
                f"                                    WAVELENGTH={state.wlam:11.4E} METERS\n"
            )
            iw.write(
                f"\n\n\n\n                    APPROXIMATE INTEGRATION EMPLOYED FOR SEGMENTS MORE THAN"
                f"{state.rkh:8.3f} WAVELENGTHS APART\n"
            )
            if state.iexk == 1:
                iw.write("\n                    THE EXTENDED THIN WIRE KERNEL WILL BE USED\n")

            if state.n != 0:
                x[: state.n] = xtemp[: state.n]*fr
                y[: state.n] = ytemp[: state.n]*fr
                z[: state.n] = ztemp[: state.n]*fr
                t1x[: state.n] = sitemp[: state.n]*fr
                bi[: state.n] = bitemp[: state.n]*fr
            if state.m != 0:
                fr2 = fr*fr
                start_m = LD - state.m
                x[start_m:LD] = xtemp[start_m:LD]*fr
                y[start_m:LD] = ytemp[start_m:LD]*fr
                z[start_m:LD] = ztemp[start_m:LD]*fr
                bi[start_m:LD] = bitemp[start_m:LD]*fr2
            igo = 2

        if igo <= 2:
            _setup_ground_and_factor(
                state, cm, zarray, x, y, z, bi, salp,
                t1x, t1y, t1z, t2x, t2y, t2z,
                icon1, icon2, iconx, itag, ip, ix, gain, scratc,
                ldtyp, ldtag, ldtagf, ldtagt, zlr, zli, zlc,
                ib11, ic11, id11, ix11, iw, somnecPath=somnecPath
            )
            igo = 3
            state.ntsol = 0
            if iflow == 12:
                gfout(state, cm, zarray, x, y, z, t1x, bi, t1y, t1z, salp,
                icon1, icon2, itag, ip, iw, igfl, LD, LD2, state.iresrv)
                return zpnorm, ypnorm, fnorm

        zpnorm = _solve_and_output_fields(
            state, cm, ib11, ic11, id11, zarray, x, y, z, bi, salp, t1x,
            t1y, t1z, t2x, t2y, t2z, icon1, icon2, itag, ip, cur, rhs, gain,
            scratc, air, aii, bir, bii, cir, cii, fnorm, mhz, nfrq, iflow,
            iped, zpnorm, iptflg, iptag, iptagf, iptagt, iptflq, iptaq,
            iptaqf, iptaqt, nthi, nphi, xpr1, xpr2, xpr3, xpr4, xpr5,
            thetis, phiss, hpol, pnet, iw
        )
        if nfrq > 1:
            freq_key = round(float(state.fmhz), 6)
            freq_gains = [
                gain_v for f_k, _, _, gain_v in plotSession.patternSamples
                if f_k == freq_key
            ]
            peak_gain = max(freq_gains) if freq_gains else None
            z_in = state.zped if (state.nsant + state.nvqd) > 0 else None
            z0_ref = zpnorm if (iped != 0 and zpnorm > 0.0) else 50.0
            recordSweepPoint(state.fmhz, z_ohms=z_in,
                peakGain_dBi=peak_gain, z0_ohms=z0_ref)


        if mhz == nfrq:
            state.ifar = -1
        if nfrq == 1:
            iw.write("\n"*5)
            return zpnorm, ypnorm, fnorm

        mhz += 1
        if mhz <= nfrq:
            igo = 1
            continue

        _write_sweep_tables(state, fnorm, nfrq, ifrq, delfrq, iped, zpnorm, ypnorm, iw)
        return zpnorm, ypnorm, fnorm


def _setup_ground_and_factor(
    state, cm, zarray, x, y, z, bi, salp,
    t1x, t1y, t1z, t2x, t2y, t2z,
    icon1, icon2, iconx, itag, ip, ix, gain, scratc,
    ldtyp, ldtag, ldtagf, ldtagt, zlr, zli, zlc,
    ib11, ic11, id11, ix11, iw, somnecPath=None
):
    fj = complex(0.0, 1.0)
    iw.write("\n\n\n                              - - - STRUCTURE IMPEDANCE LOADING - - -\n")
    if state.nload != 0:
        load(state, zarray, zlr, zli, zlc, t1x, bi, LD, itag, ldtyp, ldtag, ldtagf, ldtagt, iw)
    if state.nload == 0 and state.nlodf == 0:
        iw.write("\n                                   THIS STRUCTURE IS NOT LOADED\n")
    if state.nload == 0 and state.nlodf != 0:
        iw.write("\n                                    LOADING ONLY IN N.G.F. SECTION\n")

    iw.write("\n\n\n                                  - - - ANTENNA ENVIRONMENT - - -\n\n")
    if state.ksymp != 1:
        state.frati = 1.0 + 0j
        if state.iperf != 1:
            if state.sig < 0.0:
                state.sig = -state.sig/(59.96*state.wlam)
            epsc = complex(state.epsr, -state.sig*state.wlam*59.96)
            state.zrati = 1.0/npy.sqrt(epsc)
            state.u = state.zrati
            state.u2 = state.u*state.u
            if state.nradl != 0:
                if state.iperf == 2:
                    msg = " RADIAL WIRE G. S. APPROXIMATION MAY NOT BE USED WITH SOMMERFELD GROUND OPTION\n"
                    iw.write(msg)
                    raise RuntimeError(msg.strip())
                state.scrwl = state.scrwlt/state.wlam
                state.scrwr = state.scrwrt/state.wlam
                state.t1 = fj*2367.067/float(state.nradl)
                state.t2 = state.scrwr*float(state.nradl)
                iw.write(
                    f"                                        RADIAL WIRE GROUND SCREEN\n"
                    f"                                        {state.nradl:5d} WIRES\n"
                    f"                                        WIRE LENGTH={state.scrwlt:8.2f} METERS\n"
                    f"                                        WIRE RADIUS={state.scrwrt:10.3E} METERS\n"
                    f"                                        MEDIUM UNDER SCREEN -\n"
                )
            if state.iperf == 2:
                som_file = somnecPath
                if som_file is None and os.path.exists("fort.21"):
                    som_file = "fort.21"

                if state.nxa[0] == 0 and som_file is None:
                    if not getattr(state, "_somnec_auto", False):
                        try:
                            entered = input(
                                " OPEN UNIT 21 FOR SOMNEC INPUT FILE "
                                "(OR ENTER TO COMPUTE): "
                            ).strip()
                        except EOFError:
                            entered = ""
                        if entered:
                            som_file = entered
                        else:
                            state._somnec_auto = True

                if som_file is not None and state.nxa[0] == 0:
                    with open(som_file, "rb") as f21:
                        raw_hdr = f21.read(4)
                        if len(raw_hdr) == 4:
                            rec_le = struct.unpack("<I", raw_hdr)[0]
                            rec_be = struct.unpack(">I", raw_hdr)[0]
                            skip = 4 if (rec_le == 8624 or rec_be == 8624) else 0
                            f21.seek(skip)
                        state.ar1 = npy.fromfile(
                            f21, dtype=npy.complex64, count=440
                        ).reshape((11, 10, 4), order="F")
                        state.ar2 = npy.fromfile(
                            f21, dtype=npy.complex64, count=340
                        ).reshape((17, 5, 4), order="F")
                        state.ar3 = npy.fromfile(
                            f21, dtype=npy.complex64, count=288
                        ).reshape((9, 8, 4), order="F")
                        state.epscf = complex(
                            npy.fromfile(f21, dtype=npy.complex64, count=1).item(0)
                        )
                        state.dxa = npy.fromfile(f21, dtype=npy.float32, count=3)
                        state.dya = npy.fromfile(f21, dtype=npy.float32, count=3)
                        state.xsa = npy.fromfile(f21, dtype=npy.float32, count=3)
                        state.ysa = npy.fromfile(f21, dtype=npy.float32, count=3)
                        state.nxa = npy.fromfile(f21, dtype=npy.int32, count=3)
                        state.nya = npy.fromfile(f21, dtype=npy.int32, count=3)
                elif getattr(state, "_somnec_auto", False):
                    from .somnec.somnec import SomnecGridGenerator
                    cache = getattr(state, "_somnec_cache", {})
                    key = (float(state.epsr), float(state.sig), float(state.fmhz))
                    if key not in cache:
                        gen = SomnecGridGenerator()
                        gen.computeGrids(state.epsr, state.sig, state.fmhz)
                        cache[key] = gen
                        state._somnec_cache = cache
                    gen = cache[key]
                    state.ar1 = gen.ar1.astype(npy.complex64)
                    state.ar2 = gen.ar2.astype(npy.complex64)
                    state.ar3 = gen.ar3.astype(npy.complex64)
                    state.epscf = complex(gen.epscf)
                    state.dxa = gen.dxa.astype(npy.float32)
                    state.dya = gen.dya.astype(npy.float32)
                    state.xsa = gen.xsa.astype(npy.float32)
                    state.ysa = gen.ysa.astype(npy.float32)
                    state.nxa = gen.nxa.astype(npy.int32)
                    state.nya = gen.nya.astype(npy.int32)

                state.frati = (epsc - 1.0)/(epsc + 1.0)
                if abs((state.epscf - epsc)/epsc) >= 1.0e-3:
                    iw.write(
                        f"\n ERROR IN GROUND PARAMETERS -\n"
                        f" COMPLEX DIELECTRIC CONSTANT FROM FILE IS{state.epscf.real:12.5E}{state.epscf.imag:12.5E}\n"
                        f"                                REQUESTED{epsc.real:12.5E}{epsc.imag:12.5E}\n"
                    )
                    raise RuntimeError("SOMNEC dielectric constant mismatch")
                iw.write("                                        FINITE GROUND.  SOMMERFELD SOLUTION\n")
            else:
                iw.write("                                        FINITE GROUND.  REFLECTION COEFFICIENT APPROXIMATION\n")

            iw.write(
                f"                                        RELATIVE DIELECTRIC CONST.={state.epsr:7.3f}\n"
                f"                                        CONDUCTIVITY={state.sig:10.3E} MHOS/METER\n"
                f"                                        COMPLEX DIELECTRIC CONSTANT={epsc.real:12.5E}{epsc.imag:12.5E}\n"
            )
        else:
            iw.write("                                          PERFECT GROUND\n")
    else:
        iw.write("                                            FREE SPACE\n")

    tim1 = second()
    if state.icasx == 0:
        cmset(
            state, cm, scratc, zarray, t1x, t1y, t1z, t2x, t2y, t2z,
            x, y, z, bi, salp, icon1, icon2, iconx, state.neq, LD, LD2,
            state.iresrv
        )
        tim2 = second()
        tim = tim2 - tim1
        factrs(state, cm, scratc, state.npeq, state.neq, ip, ix, 11, 12,
            13, 14, LD2, state.iresrv)
        tim1 = second()
        tim2 = tim1 - tim2
    elif state.neq2 != 0:
        cmngf(
            state, cm, ib11, ic11, id11, zarray, t1x, t1y, t1z, t2x, t2y, t2z,
            x, y, z, bi, salp, state.rkh, icon1, icon2, iconx,
            state.npbx, state.neq, state.neq2, state.iexk, LD
        )
        tim2 = second()
        tim = tim2 - tim1
        facgf(
            state, cm, cm[ib11 - 1:], cm[ic11 - 1:], cm[id11 - 1:],
            cm[ix11 - 1:], scratc,
            ip, ix, state.np, state.n1, state.mp, state.m1,
            state.neq, state.neq2, LD2, state.iresrv
        )
        tim1 = second()
        tim2 = tim1 - tim2
    else:
        return

    if tim < 0.0:
        tim += 86400.0
    if tim2 < 0.0:
        tim2 += 86400.0
    iw.write(
        f"\n\n\n                                - - - MATRIX TIMING - - -\n\n"
        f"                        FILL={tim/60.0:9.3f} MIN.,  FACTOR={tim2/60.0:9.3f} MIN.\n"
    )

def _solve_and_output_fields(
    state, cm, ib11, ic11, id11, zarray, x, y, z, bi, salp,
    t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, itag, ip, cur, rhs, gain,
    scratc,
    air, aii, bir, bii, cir, cii, fnorm,
    mhz, nfrq, iflow, iped, zpnorm,
    iptflg, iptag, iptagf, iptagt,
    iptflq, iptaq, iptaqf, iptaqt,
    nthi, nphi, xpr1, xpr2, xpr3, xpr4, xpr5, thetis, phiss, hpol, pnet, iw
):
    nthic = nphic = inc = 1
    state.nprint = 0

    while True:
        if 0 < state.ixtyp < 5:
            if iptflg <= 0 or state.ixtyp == 4:
                iw.write("\n\n\n                                        - - - EXCITATION - - -\n")
            tmp5, tmp4 = TA*xpr5, TA*xpr4
            if state.ixtyp == 4:
                tmp1, tmp2, tmp3 = xpr1/state.wlam, xpr2/state.wlam, xpr3/state.wlam
                tmp6 = state.xpr6/(state.wlam*state.wlam)
                iw.write(
                    f"\n                               POSITION (METERS)              ORIENTATION (DEG)=\n"
                    f"                            X            Y            Z          ALPHA     BETA    DIPOLE MOMENT\n\n"
                    f"    CURRENT SOURCE    {xpr1:10.5f}   {xpr2:10.5f}   {xpr3:10.5f} "
                    f"   {xpr4:7.2f}   {xpr5:7.2f}    {state.xpr6:8.3f}\n"
                )
            else:
                tmp1, tmp2, tmp3, tmp6 = TA*xpr1, TA*xpr2, TA*xpr3, state.xpr6
                if iptflg <= 0:
                    iw.write(
                        f"\n    PLANE WAVE    THETA={xpr1:7.2f} DEG,  PHI={xpr2:7.2f} DEG,  "
                        f"ETA={xpr3:7.2f} DEG,  TYPE -{hpol[state.ixtyp - 1]:6s}=  AXIAL RATIO={state.xpr6:6.3f}\n"
                    )
        else:
            tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = tmp6 = 0.0

        etmns(
            state, cur, zarray, x, y, z, bi, salp,
            t1x, t1y, t1z, t2x, t2y, t2z,
            tmp1, tmp2, tmp3, tmp4, tmp5, tmp6,
            icon1, icon2, LD, LD2, LD3, state.ixtyp
        )

        if state.nonet != 0 and inc <= 1 and state.namprt == 0:
            iw.write("\n\n\n                                            - - - NETWORK DATA - - -\n")
            itmp3 = 0
            itmp1 = int(state.ntyp[0])
            for _ in range(2):
                if itmp1 == 3:
                    itmp1 = 2
                if itmp1 == 2:
                    iw.write(
                        "\n      - FROM -    - TO -           TRANSMISSION LINE               "
                        "-  -  SHUNT ADMITTANCES (MHOS)  -  -              LINE\n"
                        "      TAG  SEG.   TAG  SEG.      IMPEDANCE      LENGTH            "
                        "- END ONE -                 - END TWO -            TYPE\n"
                        "      NO.   NO.   NO.   NO.         OHMS        METERS         "
                        "REAL          IMAG.         REAL          IMAG.\n"
                    )
                elif itmp1 == 1:
                    iw.write(
                        "\n      - FROM -    - TO -                          "
                        "-  -  ADMITTANCE MATRIX ELEMENTS (MHOS)  -  -\n"
                        "      TAG  SEG.   TAG  SEG.             (ONE,ONE)                   "
                        "(ONE,TWO)                   (TWO,TWO)\n"
                        "      NO.   NO.   NO.   NO.        REAL          IMAG.         "
                        "REAL          IMAG.         REAL          IMAG.\n"
                    )
                for j in range(state.nonet):
                    itmp2 = int(state.ntyp[j])
                    if (itmp2 // itmp1) != 1:
                        itmp3 = itmp2
                        continue
                    itmp4, itmp5 = int(state.iseg1[j]) - 1, int(state.iseg2[j]) - 1
                    if itmp2 >= 2 and state.x11i[j] <= 0.0:
                        dist = math.sqrt((x[itmp5] - x[itmp4])**2 + (y[itmp5] - y[itmp4])**2 + (z[itmp5] - z[itmp4])**2)
                        state.x11i[j] = state.wlam*dist
                    iw.write(
                        f"    {itag[itmp4]:5d} {itmp4 + 1:5d} {itag[itmp5]:5d} {itmp5 + 1:5d} "
                        f"   {state.x11r[j]:11.4E}   {state.x11i[j]:11.4E}   "
                        f"{state.x12r[j]:11.4E}   {state.x12i[j]:11.4E}   "
                        f"{state.x22r[j]:11.4E}   {state.x22i[j]:11.4E}   {pnet[itmp2 - 1]:8s}\n"
                    )
                if itmp3 == 0:
                    break
                itmp1 = itmp3

        if inc > 1 and iptflg > 0:
            state.nprint = 1

        netwk(
            state, cm, ib11, ic11, id11, cur, rhs, scratc,
            air, aii, bir, bii, cir, cii,
            t1x, t1y, t1z, t2x, t2y, t2z, bi,
            icon1, icon2, itag, ip, iw, LD, LD2, LD3, state.iresrv
        )
        state.ntsol = 1
        if not getattr(state, "geometry_plotted", False):
            wlam = state.wlam
            if getattr(state, "patch_polygons", None):
                patch_polys = [npy.asarray(poly, dtype=float) for poly in state.patch_polygons]
            else:
                patch_polys = solverPatchPolygons(
                    x*wlam, y*wlam, z*wlam, bi*(wlam*wlam),
                    t1x, t1y, t1z, t2x, t2y, t2z, state.m, LD
                )
            wire_currents = npy.abs(cur[: state.n])*wlam if state.n > 0 else None

            source_pts = []
            for idx_s in range(state.nsant):
                seg_i = int(state.isant[idx_s]) - 1
                if 0 <= seg_i < state.n:
                    source_pts.append((x[seg_i]*wlam, y[seg_i]*wlam, z[seg_i]*wlam))
            for idx_v in range(state.nvqd):
                seg_i = int(state.ivqd[idx_v]) - 1
                if 0 <= seg_i < state.n:
                    source_pts.append((x[seg_i]*wlam, y[seg_i]*wlam, z[seg_i]*wlam))

            load_pts = []
            if (state.nload > 0 or state.nlodf > 0) and state.n > 0:
                for seg_i in range(state.n):
                    if abs(zarray[seg_i]) > 1.0e-20:
                        load_pts.append((x[seg_i]*wlam, y[seg_i]*wlam, z[seg_i]*wlam))

            src_summary = ""
            if (state.nsant + state.nvqd) > 0:
                z_in = state.zped
                z0_ref = 50.0
                gamma = abs((z_in - z0_ref)/(z_in + z0_ref))
                vswr = min((1.0 + gamma)/max(1.0 - gamma, 1e-6), 99.9)
                sign_x = "+" if z_in.imag >= 0.0 else "-"
                p_rad = state.pin - state.pnls - state.ploss
                eff = 100.0*p_rad/state.pin if state.pin > 0.0 else 0.0
                src_seg = int(state.ivqd[0]) if state.nvqd > 0 else int(state.isant[0])
                src_tag = int(itag[src_seg - 1]) if 1 <= src_seg <= state.n else 0
                src_summary = (
                    f"Freq : {state.fmhz:.4g} MHz\n"
                    f"Feed : Tag {src_tag}, Seg {src_seg}\n"
                    f"Z in : {z_in.real:.2f} {sign_x} j{abs(z_in.imag):.2f} Ω\n"
                    f"VSWR : {vswr:.2f}:1 (50 Ω)\n"
                    f"Eff. : {eff:.1f}%"
                )

            plotAntennaGeometry(
                x[: state.n]*wlam,
                y[: state.n]*wlam,
                z[: state.n]*wlam,
                t1x[: state.n]*wlam,
                t1y[: state.n],
                t1z[: state.n],
                salp[: state.n],
                patchPolygons=patch_polys,
                currents_A=wire_currents,
                sourcePoints=source_pts,
                loadPoints=load_pts,
                sourceSummary=src_summary,
            )
            state.geometry_plotted = True

        if iped != 0:
            itmp1 = mhz + 4*(mhz - 1) - 1
            if itmp1 <= (NORMF - 4):
                fnorm[itmp1], fnorm[itmp1 + 1] = state.zped.real, state.zped.imag
                fnorm[itmp1 + 2], fnorm[itmp1 + 3] = abs(state.zped), cang(state.zped)
                if iped != 2 and fnorm[itmp1 + 2] > zpnorm:
                    zpnorm = fnorm[itmp1 + 2]
        isave = 0
        if state.n != 0:
            if iptflg != -1:
                if iptflg <= 0:
                    iw.write(
                        "\n\n\n                             - - - CURRENTS AND LOCATION - - -\n\n"
                        "                                 DISTANCES IN WAVELENGTHS\n\n\n"
                        "  SEG.  TAG    COORD. OF SEG. CENTER     SEG.            - - - CURRENT (AMPS) - - -\n"
                        "  NO.   NO.     X        Y        Z      LENGTH     REAL        IMAG.       MAG.        PHASE\n"
                    )
                elif iptflg == 9 and inc <= 1:
                    iw.write(
                        f"\n\n\n                                 - - - RECEIVING PATTERN PARAMETERS - - -\n"
                        f"                                           ETA={xpr3:7.2f} DEGREES\n"
                        f"                                           TYPE -{hpol[state.ixtyp - 1]:6s}\n"
                        f"                                           AXIAL RATIO={state.xpr6:6.3f}\n\n"
                        f"           THETA      PHI          -  CURRENT  -         SEG\n"
                        f"           (DEG)     (DEG)         REAL       IMAG.      NO.\n\n"
                    )
                elif iptflg != 3 and inc <= 1:
                    iw.write(
                        f"\n\n\n                                 - - - RECEIVING PATTERN PARAMETERS - - -\n"
                        f"                                           ETA={xpr3:7.2f} DEGREES\n"
                        f"                                           TYPE -{hpol[state.ixtyp - 1]:6s}\n"
                        f"                                           AXIAL RATIO={state.xpr6:6.3f}\n\n"
                        f"           THETA      PHI          -  CURRENT  -         SEG\n"
                        f"           (DEG)     (DEG)       MAGNITUDE    PHASE      NO.\n\n"
                    )

            state.ploss = 0.0
            itmp1 = 0
            jump = iptflg + 1
            for i in range(state.n):
                curi = complex(cur[i])*state.wlam
                cmag, ph = abs(curi), cang(curi)
                if (state.nload != 0 or state.nlodf != 0) and abs(zarray[i].real) >= 1.0e-20:
                    state.ploss += 0.5*cmag*cmag*zarray[i].real*t1x[i]
                if jump == 0:
                    continue
                if jump > 0:
                    if iptag != 0 and itag[i] != iptag:
                        continue
                    itmp1 += 1
                    if itmp1 < iptagf or itmp1 > iptagt:
                        continue
                    if iptflg != 0:
                        if not (iptflg < 2 or iptflg != 9 or inc > NORMF):
                            fnorm[inc - 1] = cmag
                            isave = i + 1
                        if iptflg != 3 and iptflg < 9:
                            iw.write(f"          {xpr1:7.2f}   {xpr2:7.2f}    {cmag:11.4E}   {ph:7.2f}    {i + 1:5d}\n")
                        if iptflg == 8:
                            with open("fort.8", "a") as f8:
                                f8.write(f" {xpr1} {xpr2} {cmag} {ph} {i + 1}\n")
                        elif iptflg == 9:
                            iw.write(f"          {xpr1:7.2f}   {xpr2:7.2f}    {curi.real:11.4E} {curi.imag:11.4E}  {i + 1:5d}\n")
                            with open("fort.8", "a") as f8:
                                f8.write(f" {xpr1} {xpr2} {curi.real} {curi.imag} {i + 1}\n")
                        continue

                iw.write(
                    f" {i + 1:5d}{itag[i]:5d}{x[i]:9.4f}{y[i]:9.4f}{z[i]:9.4f}{t1x[i]:9.5f} "
                    f"{curi.real:12.4E}{curi.imag:12.4E}{cmag:12.4E}{ph:9.3f}\n"
                )
                if state.iplp1 == 1:
                    with open("fort.8", "a") as f8:
                        if state.iplp2 == 1:
                            f8.write(f" ({curi.real},{curi.imag})\n")
                        elif state.iplp2 == 2:
                            f8.write(f" {cmag} {ph}\n")

            if iptflq != -1:
                iw.write(
                    "\n\n\n                                  - - - CHARGE DENSITIES - - -\n\n"
                    "                                    DISTANCES IN WAVELENGTHS\n\n\n"
                    "  SEG.  TAG    COORD. OF SEG. CENTER     SEG.          CHARGE DENSITY (COULOMBS/METER)\n"
                    "  NO.   NO.     X        Y        Z      LENGTH     REAL        IMAG.       MAG.        PHASE\n"
                )
                itmp1 = 0
                fr_q = 1.0e-6/state.fmhz
                for i in range(state.n):
                    if iptflq != -2:
                        if iptaq != 0 and itag[i] != iptaq:
                            continue
                        itmp1 += 1
                        if itmp1 < iptaqf or itmp1 > iptaqt:
                            continue
                    curi = fr_q*complex(-bii[i], bir[i])
                    iw.write(
                        f" {i + 1:5d}{itag[i]:5d}{x[i]:9.4f}{y[i]:9.4f}{z[i]:9.4f}{t1x[i]:9.5f} "
                        f"{curi.real:12.4E}{curi.imag:12.4E}{abs(curi):12.4E}{cang(curi):9.3f}\n"
                    )
        if state.m != 0:
            iw.write(
                "\n\n\n\n                                        - - - - SURFACE PATCH CURRENTS - - - -\n\n"
                "                                                  DISTANCE IN WAVELENGTHS\n"
                "                                                  CURRENT IN AMPS/METER\n\n"
                "                           - - SURFACE COMPONENTS - -                   - - - RECTANGULAR COMPONENTS - - -\n"
                "      PATCH CENTER      TANGENT VECTOR 1   TANGENT VECTOR 2           X                   Y                   Z\n"
                "     X      Y      Z     MAG.       PHASE   MAG.       PHASE    REAL      IMAG.     REAL      IMAG.     REAL      IMAG.\n"
            )
            j_idx, itmp1 = state.n - 3, LD
            for i in range(state.m):
                j_idx += 3
                itmp1 -= 1
                ex, ey, ez = complex(cur[j_idx]), complex(cur[j_idx + 1]), complex(cur[j_idx + 2])
                eth = ex*t1x[itmp1] + ey*t1y[itmp1] + ez*t1z[itmp1]
                eph = ex*t2x[itmp1] + ey*t2y[itmp1] + ez*t2z[itmp1]
                iw.write(
                    f" {i + 1:4d}\n"
                    f" {x[itmp1]:7.3f}{y[itmp1]:7.3f}{z[itmp1]:7.3f}"
                    f"{abs(eth):11.4E}{cang(eth):8.2f}{abs(eph):11.4E}{cang(eph):8.2f}"
                    f"{ex.real:10.2E}{ex.imag:10.2E}{ey.real:10.2E}{ey.imag:10.2E}{ez.real:10.2E}{ez.imag:10.2E}\n"
                )
                if state.iplp1 == 1 and state.iplp3 in (1, 2, 3, 4):
                    with open("fort.8", "a") as f8:
                        if state.iplp3 == 1:
                            f8.write(f" ({ex.real},{ex.imag})\n")
                        elif state.iplp3 == 2:
                            f8.write(f" ({ey.real},{ey.imag})\n")
                        elif state.iplp3 == 3:
                            f8.write(f" ({ez.real},{ez.imag})\n")
                        elif state.iplp3 == 4:
                            f8.write(f" ({ex.real},{ex.imag}) ({ey.real},{ey.imag}) ({ez.real},{ez.imag})\n")

        if state.ixtyp == 0 or state.ixtyp >= 5:
            tmp1 = state.pin - state.pnls - state.ploss
            tmp2 = 100.0*tmp1/state.pin if state.pin != 0.0 else 0.0
            iw.write(
                f"\n\n\n                                        - - - POWER BUDGET - - -\n\n"
                f"                                           INPUT POWER   ={state.pin:11.4E} WATTS\n"
                f"                                           RADIATED POWER={tmp1:11.4E} WATTS\n"
                f"                                           STRUCTURE LOSS={state.ploss:11.4E} WATTS\n"
                f"                                           NETWORK LOSS  ={state.pnls:11.4E} WATTS\n"
                f"                                           EFFICIENCY    ={tmp2:7.2f} PERCENT\n"
            )

        if state.ncoup > 0:
            couple(state, iw, cur, state.wlam, LD, LD3, itag)

        skip_near_far = (iflow == 7 and 0 < state.ixtyp < 4)
        if iflow == 7 and not skip_near_far:
            return zpnorm

        if not skip_near_far:
            if state.near != -1:
                nfpat(state, x, y, z, t1x, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, air, aii, bir, bii, cir, cii, cur, iw, LD, LD3)
                if mhz == nfrq:
                    state.near = -1
                if nfrq == 1:
                    return zpnorm
            if state.ifar != -1:
                state.pinr, state.pnlr = state.pin, state.pnls
                rdpat(state, cur, gain, air, aii, bir, bii, cir, cii, t1x, t1y, t1z, bi, salp, x, y, z, LD, LD3, LD4, iw)

        if state.ixtyp == 0 or state.ixtyp >= 4:
            return zpnorm
        nthic += 1
        inc += 1
        xpr1 += xpr4
        if nthic <= nthi:
            continue
        nthic = 1
        xpr1 = thetis
        xpr2 += xpr5
        nphic += 1
        if nphic <= nphi:
            continue
        nphic = 1
        xpr2 = phiss

        if 2 <= iptflg <= 7:
            itmp1 = min(nthi*nphi, NORMF)
            if nthi*nphi > NORMF:
                iw.write("\n\n\n    RECEIVING PATTERN STORAGE TOO SMALL,ARRAY TRUNCATED\n")
            tmp1 = float(npy.max(fnorm[:itmp1]))
            if tmp1 <= 0.0:
                tmp1 = 1.0
            iw.write(
                f"\n\n\n                                - - - NORMALIZED RECEIVING PATTERN - - -\n"
                f"                                         NORMALIZATION FACTOR={tmp1:11.4E}\n"
                f"                                         ETA={xpr3:7.2f} DEGREES\n"
                f"                                         TYPE -{hpol[state.ixtyp - 1]:6s}\n"
                f"                                         AXIAL RATIO={state.xpr6:6.3f}\n"
                f"                                         SEGMENT NO.={isave:5d}\n\n"
                f"                     THETA      PHI         -  PATTERN  -\n"
                f"                     (DEG)     (DEG)        DB        MAGNITUDE\n\n"
            )
            for jph in range(nphi):
                for i in range(nthi):
                    itmp3 = i + nthi*jph
                    if itmp3 >= itmp1:
                        break
                    tmp2 = fnorm[itmp3]/tmp1
                    iw.write(f"                    {xpr1:7.2f}   {xpr2:7.2f}    {db20(tmp2):7.2f}    {tmp2:11.4E}\n")
                    xpr1 += xpr4
                xpr1 = thetis
                xpr2 += xpr5
            xpr2 = phiss
        return zpnorm


def _write_sweep_tables(state, fnorm, nfrq, ifrq, delfrq, iped, zpnorm, ypnorm, iw):
    if iped == 0:
        return
    src_seg = int(state.ivqd[state.nvqd - 1]) if state.nvqd >= 1 else int(state.isant[state.nsant - 1])
    itmp1 = min(nfrq, NORMF // 4)
    iw.write(
        f"\n\n\n                                    - - - INPUT IMPEDANCE DATA - - -\n"
        f"                                             SOURCE SEGMENT NO.{src_seg:4d}\n"
        f"                                             NORMALIZATION FACTOR={zpnorm:12.5E}\n\n"
        f"       FREQ.             -  -  UNNORMALIZED IMPEDANCE  -  -                     -  -  NORMALIZED IMPEDANCE  -  -            VSWR\n"
        f"                   RESISTANCE    REACTANCE      MAGNITUDE    PHASE       RESISTANCE    REACTANCE      MAGNITUDE    PHASE\n"
        f"        MHZ           OHMS          OHMS           OHMS     DEGREES                                               DEGREES\n\n"
    )
    tmp1 = state.fmhz - (nfrq - 1)*delfrq if ifrq == 0 else state.fmhz/(delfrq**(nfrq - 1))
    for i in range(itmp1):
        itmp2 = i*5
        r_val, x_val, z_mag, z_ph = fnorm[itmp2], fnorm[itmp2 + 1], fnorm[itmp2 + 2], fnorm[itmp2 + 3]
        tmp2, tmp3, tmp4 = r_val/zpnorm, x_val/zpnorm, z_mag/zpnorm
        swra = math.sqrt((tmp2 + 1.0)**2 + tmp3**2)
        swrb = math.sqrt((tmp2 - 1.0)**2 + tmp3**2)
        swr = (swra + swrb)/(swra - swrb) if swra != swrb else 999.99
        iw.write(f"   {tmp1:9.3f}    {r_val:12.5E}  {x_val:12.5E}   {z_mag:12.5E}  {z_ph:7.2f}    {tmp2:12.5E}  {tmp3:12.5E}   {tmp4:12.5E}  {z_ph:7.2f}   {swr:5.2f}\n")
        if state.iplp1 == 4:
            with open("fort.8", "a") as f8:
                f8.write(f" {tmp1:11.3E}{r_val:11.3E}{x_val:11.3E}{tmp2:11.3E}{tmp3:11.3E}{swr:11.3E}\n")
        tmp1 = tmp1 + delfrq if ifrq == 0 else tmp1*delfrq

    iw.write(
        f"\n\n\n                                   - - - INPUT ADMITTANCE DATA - - -\n"
        f"                                             SOURCE SEGMENT NO.{src_seg:4d}\n"
        f"                                             NORMALIZATION FACTOR={ypnorm:12.5E}\n\n"
        f"       FREQ.            -  -  UNNORMALIZED ADMITTANCE  -  -                    -  -  NORMALIZED ADMITTANCE  -  -          VSWR\n"
        f"                   CONDUCTANCE   SUSCEPTANCE     MAGNITUDE    PHASE      CONDUCTANCE   SUSCEPTANCE   MAGNITUDE    PHASE\n"
        f"        MHZ           MHOS          MHOS           MHOS     DEGREES                                               DEGREES\n\n"
    )
    tmp1 = state.fmhz - (nfrq - 1)*delfrq if ifrq == 0 else state.fmhz/(delfrq**(nfrq - 1))
    for i in range(itmp1):
        itmp2 = i*5
        r_val, x_val, z_mag, z_ph = fnorm[itmp2], fnorm[itmp2 + 1], fnorm[itmp2 + 2], fnorm[itmp2 + 3]
        denom = r_val*r_val + x_val*x_val
        ytmp2, ytmp3, ytmp4, ytmp5 = r_val/denom, -x_val/denom, 1.0/z_mag, -z_ph
        yntmp2, yntmp3, yntmp4 = ytmp2/ypnorm, ytmp3/ypnorm, ytmp4/ypnorm
        yntmp5 = ytmp5
        swra = math.sqrt((yntmp2 + 1.0)**2 + yntmp3**2)
        swrb = math.sqrt((yntmp2 - 1.0)**2 + yntmp3**2)
        swr = (swra + swrb)/(swra - swrb) if swra != swrb else 999.99
        iw.write(f"   {tmp1:9.3f}    {ytmp2:12.5E}  {ytmp3:12.5E}   {ytmp4:12.5E}  {ytmp5:7.2f}    {yntmp2:12.5E}  {yntmp3:12.5E}   {yntmp4:12.5E}  {yntmp5:7.2f}   {swr:5.2f}\n")
        if state.iplp1 == 5:
            with open("fort.8", "a") as f8:
                f8.write(f" {tmp1:11.3E}{ytmp2:11.3E}{ytmp3:11.3E}{yntmp2:11.3E}{yntmp3:11.3E}{swr:11.3E}\n")
        tmp1 = tmp1 + delfrq if ifrq == 0 else tmp1*delfrq
    iw.write("\n"*5)


if __name__ == "__main__":
    run_nec()

