import math
import numpy as npy
from .nec_state import NECState
from .main2 import tbf
from .factr import factr, solve, solves, ltsolv


def netwk(
    state: NECState,
    cm: npy.ndarray,
    ib11: int,
    ic11: int,
    id11: int,
    einc: npy.ndarray,
    rhs: npy.ndarray,
    scratc: npy.ndarray,
    air: npy.ndarray,
    aii: npy.ndarray,
    bir: npy.ndarray,
    bii: npy.ndarray,
    cir: npy.ndarray,
    cii: npy.ndarray,
    t1x: npy.ndarray,
    t1y: npy.ndarray,
    t1z: npy.ndarray,
    t2x: npy.ndarray,
    t2y: npy.ndarray,
    t2z: npy.ndarray,
    bi: npy.ndarray,
    icon1: npy.ndarray,
    icon2: npy.ndarray,
    itag: npy.ndarray,
    ip: npy.ndarray,
    iw,
    ld: int,
    ld2: int,
    ld3: int,
    iresrv: int,
):
    """Solves for structure currents including non-radiating networks."""
    ndimn, ndimnp, tp = 100, 101, 6.283185308
    cmb = cm[ib11 - 1 :]
    cmc = cm[ic11 - 1 :]
    cmd = cm[id11 - 1 :]

    neqz2 = 1 if state.neq2 == 0 else state.neq2
    state.pin = 0.0
    state.pnls = 0.0
    neqt = state.neq + state.neq2
    ntsc = 0
    nteq = 0

    if state.ntsol == 0:
        if state.masym != 0 and state.namprt == 0:
            irow1 = 0
            for i in range(state.nonet):
                for nseg in (int(state.iseg1[i]), int(state.iseg2[i])):
                    if nseg not in state.ipnt[:irow1]:
                        state.ipnt[irow1] = nseg
                        irow1 += 1
            for i in range(state.nsant):
                nseg = int(state.isant[i])
                if nseg not in state.ipnt[:irow1]:
                    state.ipnt[irow1] = nseg
                    irow1 += 1
            if irow1 >= ndimnp:
                raise RuntimeError("ERROR - - NETWORK ARRAY DIMENSIONS TOO SMALL")
            if irow1 >= 2:
                for i in range(irow1):
                    isc1 = int(state.ipnt[i]) - 1
                    asm = t1x[isc1]
                    rhs[:neqt] = 0j
                    rhs[isc1] = 1.0 + 0j
                    solgf(state, cm, cmb, cmc, cmd, rhs, scratc, state.np, state.n1, state.n, state.mp, state.m1, state.m, state.neq, state.neq2, neqz2, ip, ld2, ld3, iresrv)
                    cabc(state, rhs, bi, t1x, t1y, t1z, t2x, t2y, t2z, air, aii, bir, bii, cir, cii, icon1, icon2, ld, ld3)
                    for j in range(irow1):
                        state.cmn[j, i] = rhs[int(state.ipnt[j]) - 1] / asm

                asm = asa = 0.0
                nteq_m = ntsc_m = 0
                for i in range(1, irow1):
                    for j in range(i):
                        cux = complex(state.cmn[i, j])
                        pwr = abs((cux - complex(state.cmn[j, i])) / cux)
                        asa += pwr * pwr
                        if pwr >= asm:
                            asm = pwr
                            nteq_m, ntsc_m = int(state.ipnt[i]), int(state.ipnt[j])
                asa = math.sqrt(asa * 2.0 / float(irow1 * (irow1 - 1)))
                iw.write(
                    f"\n\n\n   MAXIMUM RELATIVE ASYMMETRY OF THE DRIVING POINT ADMITTANCE MATRIX IS{asm:10.3E}"
                    f" FOR SEGMENTS{nteq_m:5d} AND{ntsc_m:5d}\n   RMS RELATIVE ASYMMETRY IS{asa:10.3E}\n"
                )

        if state.nonet != 0:
            state.rhnx.fill(0j)
            state.cmn.fill(0j)
            nteq = ntsc = 0
            for j in range(state.nonet):
                nseg1, nseg2 = int(state.iseg1[j]), int(state.iseg2[j])
                if state.ntyp[j] <= 1:
                    y11r, y11i = state.x11r[j], state.x11i[j]
                    y12r, y12i = state.x12r[j], state.x12i[j]
                    y22r, y22i = state.x22r[j], state.x22i[j]
                else:
                    y22r = tp * state.x11i[j] / state.wlam
                    y12r = 0.0
                    y12i = 1.0 / (state.x11r[j] * math.sin(y22r))
                    y11i = -y12i * math.cos(y22r)
                    y11r = state.x12r[j]
                    y22i = y11i + state.x22i[j]
                    y11i = y11i + state.x12i[j]
                    y22r = state.x22r[j]
                    if state.ntyp[j] != 2:
                        y12r, y12i = -y12r, -y12i

                def map_port(nseg: int):
                    nonlocal nteq, ntsc
                    for i in range(state.nsant):
                        if nseg == int(state.isant[i]):
                            for k in range(ntsc):
                                if nseg == int(state.ntsca[k]):
                                    return i + 1, ndimnp - (k + 1) - 1
                            state.ntsca[ntsc] = nseg
                            state.vsrc[ntsc] = state.vsant[i]
                            ntsc += 1
                            return i + 1, ndimnp - ntsc - 1
                    for k in range(nteq):
                        if nseg == int(state.nteqa[k]):
                            return 0, k
                    state.nteqa[nteq] = nseg
                    nteq += 1
                    return 0, nteq - 1

                isc1, irow1 = map_port(nseg1)
                isc2, irow2 = map_port(nseg2)
                if ntsc + nteq >= ndimnp:
                    raise RuntimeError("ERROR - - NETWORK ARRAY DIMENSIONS TOO SMALL")

                if isc1 == 0:
                    state.cmn[irow1, irow1] -= complex(y11r, y11i) * t1x[nseg1 - 1]
                    state.cmn[irow1, irow2] -= complex(y12r, y12i) * t1x[nseg1 - 1]
                else:
                    state.rhnx[irow1] += complex(y11r, y11i) * state.vsant[isc1 - 1] / state.wlam
                    state.rhnx[irow2] += complex(y12r, y12i) * state.vsant[isc1 - 1] / state.wlam

                if isc2 == 0:
                    state.cmn[irow2, irow2] -= complex(y22r, y22i) * t1x[nseg2 - 1]
                    state.cmn[irow2, irow1] -= complex(y12r, y12i) * t1x[nseg2 - 1]
                else:
                    state.rhnx[irow1] += complex(y12r, y12i) * state.vsant[isc2 - 1] / state.wlam
                    state.rhnx[irow2] += complex(y22r, y22i) * state.vsant[isc2 - 1] / state.wlam

            for i in range(nteq):
                rhs[:neqt] = 0j
                rhs[int(state.nteqa[i]) - 1] = 1.0 + 0j
                solgf(state, cm, cmb, cmc, cmd, rhs, scratc, state.np, state.n1, state.n, state.mp, state.m1, state.m, state.neq, state.neq2, neqz2, ip, ld2, ld3, iresrv)
                cabc(state, rhs, bi, t1x, t1y, t1z, t2x, t2y, t2z, air, aii, bir, bii, cir, cii, icon1, icon2, ld, ld3)
                for j in range(nteq):
                    state.cmn[i, j] += rhs[int(state.nteqa[j]) - 1]

            factr(state.cmn, scratc, nteq, 30, state.ipnt, ld2)
    if state.nonet != 0:
        rhs[:neqt] = einc[:neqt]
        solgf(state, cm, cmb, cmc, cmd, rhs, scratc, state.np, state.n1, state.n, state.mp, state.m1, state.m, state.neq, state.neq2, neqz2, ip, ld2, ld3, iresrv)
        cabc(state, rhs, bi, t1x, t1y, t1z, t2x, t2y, t2z, air, aii, bir, bii, cir, cii, icon1, icon2, ld, ld3)
        for i in range(nteq):
            state.rhnt[i] = state.rhnx[i] + rhs[int(state.nteqa[i]) - 1]

        solve(state.cmn, state.rhnt, scratc, nteq, 30, state.ipnt, ld2)

        for i in range(nteq):
            einc[int(state.nteqa[i]) - 1] -= state.rhnt[i]

        solgf(state, cm, cmb, cmc, cmd, einc, scratc, state.np, state.n1, state.n, state.mp, state.m1, state.m, state.neq, state.neq2, neqz2, ip, ld2, ld3, iresrv)
        cabc(state, einc, bi, t1x, t1y, t1z, t2x, t2y, t2z, air, aii, bir, bii, cir, cii, icon1, icon2, ld, ld3)

        if state.nprint == 0 and state.namprt == 0:
            iw.write(
                "\n\n\n                           - - - STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS - - -\n\n"
                "   TAG   SEG.    VOLTAGE (VOLTS)         CURRENT (AMPS)         IMPEDANCE (OHMS)        ADMITTANCE (MHOS)      POWER\n"
                "   NO.   NO.    REAL        IMAG.       REAL        IMAG.       REAL        IMAG.       REAL        IMAG.     (WATTS)\n"
            )
        for i in range(nteq):
            irow1 = int(state.nteqa[i]) - 1
            vlt = complex(state.rhnt[i]) * t1x[irow1] * state.wlam
            cux = complex(einc[irow1]) * state.wlam
            ymit, zped = cux / vlt, vlt / cux
            pwr = 0.5 * (vlt * cux.conjugate()).real
            state.pnls -= pwr
            if state.nprint == 0 and state.namprt == 0:
                iw.write(f" {itag[irow1]:5d} {irow1 + 1:5d}{vlt.real:12.5E}{vlt.imag:12.5E}{cux.real:12.5E}{cux.imag:12.5E}{zped.real:12.5E}{zped.imag:12.5E}{ymit.real:12.5E}{ymit.imag:12.5E}{pwr:12.5E}\n")

        for i in range(ntsc):
            irow1 = int(state.ntsca[i]) - 1
            vlt = state.vsrc[i]
            cux = complex(einc[irow1]) * state.wlam
            ymit, zped = cux / vlt, vlt / cux
            pwr = 0.5 * (vlt * cux.conjugate()).real
            state.pnls -= pwr
            if state.nprint == 0 and state.namprt == 0:
                iw.write(f" {itag[irow1]:5d} {irow1 + 1:5d}{vlt.real:12.5E}{vlt.imag:12.5E}{cux.real:12.5E}{cux.imag:12.5E}{zped.real:12.5E}{zped.imag:12.5E}{ymit.real:12.5E}{ymit.imag:12.5E}{pwr:12.5E}\n")
    else:
        solgf(state, cm, cmb, cmc, cmd, einc, scratc, state.np, state.n1, state.n, state.mp, state.m1, state.m, state.neq, state.neq2, neqz2, ip, ld2, ld3, iresrv)
        cabc(state, einc, bi, t1x, t1y, t1z, t2x, t2y, t2z, air, aii, bir, bii, cir, cii, icon1, icon2, ld, ld3)
        ntsc = 0

    if state.nsant + state.nvqd == 0:
        return
    if state.namprt == 0:
        iw.write(
            "\n\n\n                                          - - - ANTENNA INPUT PARAMETERS - - -\n\n"
            "   TAG   SEG.    VOLTAGE (VOLTS)         CURRENT (AMPS)         IMPEDANCE (OHMS)        ADMITTANCE (MHOS)      POWER\n"
            "   NO.   NO.    REAL        IMAG.       REAL        IMAG.       REAL        IMAG.       REAL        IMAG.     (WATTS)\n"
        )

    for i in range(state.nsant):
        isc1 = int(state.isant[i])
        vlt = state.vsant[i]
        found_sc = -1
        for j in range(ntsc):
            if int(state.ntsca[j]) == isc1:
                found_sc = j
                break
        if found_sc < 0:
            cux = complex(einc[isc1 - 1]) * state.wlam
            irow1 = 0
        else:
            irow1 = ndimnp - (found_sc + 1) - 1
            cux = state.rhnx[irow1] - npy.sum(state.cmn[:nteq, irow1] * state.rhnt[:nteq])
            cux = (complex(einc[isc1 - 1]) + cux) * state.wlam
        ymit, state.zped = cux / vlt, vlt / cux
        pwr = 0.5 * (vlt * cux.conjugate()).real
        state.pin += pwr
        if irow1 != 0:
            state.pnls += pwr
        if state.namprt == 0:
            iw.write(f" {itag[isc1 - 1]:5d} {isc1:5d}{vlt.real:12.5E}{vlt.imag:12.5E}{cux.real:12.5E}{cux.imag:12.5E}{state.zped.real:12.5E}{state.zped.imag:12.5E}{ymit.real:12.5E}{ymit.imag:12.5E}{pwr:12.5E}\n")

    for i in range(state.nvqd):
        isc1 = int(state.ivqd[i])
        isc_idx = isc1 - 1
        vlt = state.vqd[i]
        cux = complex(air[isc_idx], aii[isc_idx])
        ymit = complex(bir[isc_idx], bii[isc_idx])
        zped_c = complex(cir[isc_idx], cii[isc_idx])
        pwr_arg = t1x[isc_idx] * tp * 0.5
        cux = (cux - ymit * math.sin(pwr_arg) + zped_c * math.cos(pwr_arg)) * state.wlam
        ymit, state.zped = cux / vlt, vlt / cux
        pwr = 0.5 * (vlt * cux.conjugate()).real
        state.pin += pwr
        if state.namprt == 0:
            iw.write(f" {itag[isc_idx]:5d} *{isc1:4d}{vlt.real:12.5E}{vlt.imag:12.5E}{cux.real:12.5E}{cux.imag:12.5E}{state.zped.real:12.5E}{state.zped.imag:12.5E}{ymit.real:12.5E}{ymit.imag:12.5E}{pwr:12.5E}\n")
def cabc(state: NECState, curx, bi, t1x, t1y, t1z, t2x, t2y, t2z, air, aii, bir, bii, cir, cii, icon1, icon2, ld: int, ld3: int):
    """Computes A, B, C current interpolation coefficients and converts patch currents to X, Y, Z."""
    tp, ccj = 6.283185308, complex(0.0, -0.01666666667)
    if state.n != 0:
        for arr in (air, aii, bir, bii, cir, cii):
            arr[: state.n] = 0.0
        for i in range(state.n):
            c_val = complex(curx[i])
            ar, ai = c_val.real, c_val.imag
            tbf(state, t1x, bi, icon1, icon2, 1, i + 1)
            for jx in range(state.jsno):
                j = int(state.jco[jx]) - 1
                air[j] += state.ax[jx] * ar
                aii[j] += state.ax[jx] * ai
                bir[j] += state.bx[jx] * ar
                bii[j] += state.bx[jx] * ai
                cir[j] += state.cx[jx] * ar
                cii[j] += state.cx[jx] * ai

        for is_idx in range(state.nqds):
            i = int(state.iqds[is_idx])
            i_idx = i - 1
            jxx = int(icon1[i_idx])
            icon1[i_idx] = 0
            tbf(state, t1x, bi, icon1, icon2, 0, i)
            icon1[i_idx] = jxx
            sh = t1x[i_idx] * 0.5
            jsno_idx = state.jsno - 1
            curd = ccj * state.vqds[is_idx] / ((math.log(2.0 * sh / bi[i_idx]) - 1.0) * (state.bx[jsno_idx] * math.cos(tp * sh) + state.cx[jsno_idx] * math.sin(tp * sh)) * state.wlam)
            ar, ai = curd.real, curd.imag
            for jx in range(state.jsno):
                j = int(state.jco[jx]) - 1
                air[j] += state.ax[jx] * ar
                aii[j] += state.ax[jx] * ai
                bir[j] += state.bx[jx] * ar
                bii[j] += state.bx[jx] * ai
                cir[j] += state.cx[jx] * ar
                cii[j] += state.cx[jx] * ai

        for i in range(state.n):
            curx[i] = complex(air[i] + cir[i], aii[i] + cii[i])

    if state.m == 0:
        return
    k = ld - state.m - 1
    jco1 = state.n + 2 * state.m
    jco2 = jco1 + state.m
    for _ in range(state.m):
        k += 1
        jco1 -= 2
        jco2 -= 3
        cs1, cs2 = complex(curx[jco1]), complex(curx[jco1 + 1])
        curx[jco2] = cs1 * t1x[k] + cs2 * t2x[k]
        curx[jco2 + 1] = cs1 * t1y[k] + cs2 * t2y[k]
        curx[jco2 + 2] = cs1 * t1z[k] + cs2 * t2z[k]


def solgf(state: NECState, a, b, c, d, xy, y, np_sym, n1, n, mp_sym, m1, m, n1c, n2c, n2cz, ip, ld2, ld3, iresrv):
    """Solves partitioned N.G.F. system using Schur complement."""
    ifl = 13 if state.icasx > 0 else 14
    if n2c <= 0:
        solves(state, a, xy, y, n1c, np_sym, n, mp_sym, m, ip, 1, 13, ifl, ld2, iresrv)
        return

    npm = n + 2 * m1
    if n1 != n and m1 != 0:
        y[n1:npm] = xy[n1:npm]
        xy[n1 : n1 + 2 * m1] = y[n:npm]
        xy[n1 + 2 * m1 : npm] = y[n1:n]

    neqs = state.nscon + 2 * state.npcon
    if neqs != 0:
        neq = n1c + n2c
        xy[neq - neqs : neq] = 0j

    solves(state, a, xy, y, n1c, np_sym, n1, mp_sym, m1, ip, 1, 13, ifl, ld2, iresrv)

    c_mat = c[: n1c * state.npbl].reshape((n1c, state.npbl), order="F")
    f15 = open("fort.15", "rb") if state.icasx > 1 else None
    ni, npb = 0, state.npbl
    for jj in range(1, state.nbbl + 1):
        if jj == state.nbbl:
            npb = state.nlbl
        if state.icasx > 1:
            c_mat[:, :npb] = npy.fromfile(f15, dtype=c.dtype, count=n1c * npb).reshape((n1c, npb), order="F")
        ii = n1c + ni
        xy[ii : ii + npb] -= (c_mat[:, :npb].T @ xy[:n1c]).astype(xy.dtype)
        ni += state.npbl
    if f15 is not None:
        f15.close()

    if state.icasx <= 1:
        solve(d[: n2c * n2c].reshape((n2c, n2c), order="F"), xy[n1c : n1c + n2c], y, n2c, n2c, ip[n1c : n1c + n2c], ld2)
    elif state.icasx != 4:
        with open("fort.11", "rb") as f11:
            b[: n2c * n2c] = npy.fromfile(f11, dtype=b.dtype, count=n2c * n2c)
        solve(b[: n2c * n2c].reshape((n2c, n2c), order="F"), xy[n1c : n1c + n2c], y, n2c, n2c, ip[n1c : n1c + n2c], ld2)
    else:
        nblsys, npsys, nlsys, icass = state.nblsym, state.npsym, state.nlsym, state.icase
        state.nblsym, state.npsym, state.nlsym, state.icase = state.nbbl, state.npbl, state.nlbl, 3
        with open("fort.11", "rb") as f11, open("fort.16", "rb") as f16:
            ltsolv(state, b, xy[n1c : n1c + n2c].reshape((n2c, 1), order="F"), y, ip[n1c : n1c + n2c], n2c, n2c, 1, f11, f16, ld2)
        state.nblsym, state.npsym, state.nlsym, state.icase = nblsys, npsys, nlsys, icass

    b_mat = b[: n1c * state.npbl].reshape((n1c, state.npbl), order="F")
    f14 = open("fort.14", "rb") if state.icasx > 1 else None
    ni, npb = 0, state.npbl
    for jj in range(1, state.nbbl + 1):
        if jj == state.nbbl:
            npb = state.nlbl
        if state.icasx > 1:
            b_mat[:, :npb] = npy.fromfile(f14, dtype=b.dtype, count=n1c * npb).reshape((n1c, npb), order="F")
        ii = n1c + ni
        xy[:n1c] -= (b_mat[:, :npb] @ xy[ii : ii + npb]).astype(xy.dtype)
        ni += state.npbl
    if f14 is not None:
        f14.close()

    if n1 != n and m1 != 0:
        y[n1:npm] = xy[n1:npm]
        xy[n1:n] = y[n1 + 2 * m1 : npm]
        xy[n:npm] = y[n1 : n1 + 2 * m1]

    if state.nscon != 0:
        j = (n1c + n2c) - neqs
        for i in range(state.nscon):
            xy[int(state.iscon[i]) - 1] = xy[j + i]

