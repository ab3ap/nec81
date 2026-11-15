import numpy as npy
from .nec_state import NECState
from .main2 import tbf, blckot
from .hintg import trio, hsfld, hintg, pcint, unere, _set_ext_wire_flags
from .nfpat import efld


def cmset(
    state: NECState,
    cm: npy.ndarray,
    d: npy.ndarray,
    zarray: npy.ndarray,
    t1x: npy.ndarray,
    t1y: npy.ndarray,
    t1z: npy.ndarray,
    t2x: npy.ndarray,
    t2y: npy.ndarray,
    t2z: npy.ndarray,
    x: npy.ndarray,
    y: npy.ndarray,
    z: npy.ndarray,
    bi: npy.ndarray,
    salp: npy.ndarray,
    icon1: npy.ndarray,
    icon2: npy.ndarray,
    iconx: npy.ndarray,
    nrow: int,
    ld: int,
    ld2: int,
    iresrv: int,
):
    '''Sets up the complex interaction matrix in the array CM.'''
    mp2 = 2*state.mp
    npeq = state.np + mp2
    neq = state.n + 2*state.m
    nop = neq//npeq

    f11 = open('fort.11', 'wb') if state.icase > 2 else None
    iout = 2*state.npblk*nrow
    it = state.npblk

    for ixblk1 in range(1, state.nbloks+1):
        isv = (ixblk1 - 1)*state.npblk
        if ixblk1 == state.nbloks:
            it = state.nlast

        cm_blk = cm[: nrow*it].reshape((nrow, it), order='F')
        cm_blk.fill(0j)

        i1 = isv + 1
        i2 = isv + it
        in2 = min(i2, state.np)
        im1 = max(i1 - state.np, 1)
        im2 = i2 - state.np
        ist = (state.np - i1 + 2) if i1 <= state.np else 1

        if state.n != 0:
            for j in range(1, state.n+1):
                trio(state, t1x, bi, icon1, icon2, j, ld)
                for i in range(state.jsno):
                    ij = int(state.jco[i])
                    state.jco[i] = ((ij - 1)//state.np)*mp2 + ij

                if i1 <= in2:
                    cmww(
                        state, cm_blk, cm_blk, x, y, z, bi, salp, t1x, t1y, t1z,
                        icon1, icon2, nrow, nrow, j, i1, in2, 1, ld
                    )
                if im1 <= im2:
                    cmws(
                        state, cm_blk[:, ist-1 :], cm_blk, x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, nrow, nrow, j, im1, im2, 1, ld
                    )
                if state.nload != 0 and j <= state.np:
                    ipr = j - isv
                    if 1 <= ipr <= it:
                        zaj = zarray[j-1]
                        for i in range(state.jsno):
                            jss = int(state.jco[i]) - 1
                            cm_blk[jss, ipr-1] -= (state.ax[i] + state.cx[i])*zaj

        if state.m != 0:
            jm1 = 1 - state.mp
            jm2 = 0
            jst = 1 - mp2
            for _ in range(nop):
                jm1 += state.mp
                jm2 += state.mp
                jst += npeq
                if i1 <= in2:
                    cmsw(
                        state, cm_blk[jst-1 :, :], cm_blk, x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, iconx,
                        jm1, jm2, i1, in2, nrow, 0, 1, ld
                    )
                if im1 <= im2:
                    cmss(
                        state, cm_blk[jst-1 :, ist-1 :], x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, jm1, jm2, im1, im2, nrow, 1, ld
                    )

        if state.icase != 1:
            if state.icase != 3:
                for i in range(it):
                    for j in range(npeq):
                        for k in range(nop):
                            d[k] = cm_blk[j + k*npeq, i]
                        cm_blk[j, i] = npy.sum(d[:nop])
                        for k in range(1, nop):
                            ka = j + k*npeq
                            deter = d[0] + npy.sum(d[1:nop]*state.ssx[k, 1:nop])
                            cm_blk[ka, i] = deter
            if state.icase >= 3:
                blckot(cm, 1, iout, 1, 31, f11)

    if f11 is not None:
        f11.close()


def cmww(state: NECState, cm, cw, x, y, z, bi, salp, t1x, t1y, t1z, icon1, icon2, nr, nw, j, i1, i2, itrp, ld):
    j_idx = j - 1
    state.s, state.b = t1x[j_idx], bi[j_idx]
    state.xj, state.yj, state.zj = x[j_idx], y[j_idx], z[j_idx]
    state.cabj, state.sabj, state.salpj = t1y[j_idx], t1z[j_idx], salp[j_idx]
    _set_ext_wire_flags(state, j, t1y, t1z, salp, bi, icon1, icon2)

    for ipr, i in enumerate(range(i1, i2+1)):
        i_idx = i - 1
        efld(state, x[i_idx], y[i_idx], z[i_idx], bi[i_idx], i-j)
        cabi, sabi, salpi = t1y[i_idx], t1z[i_idx], salp[i_idx]
        etk = state.exk*cabi + state.eyk*sabi + state.ezk*salpi
        ets = state.exs*cabi + state.eys*sabi + state.ezs*salpi
        etc = state.exc*cabi + state.eyc*sabi + state.ezc*salpi

        for ij in range(state.jsno):
            jx = int(state.jco[ij]) - 1
            val = etk*state.ax[ij] + ets*state.bx[ij] + etc*state.cx[ij]
            if itrp == 0:
                cm[ipr, jx] += val
            elif itrp != 2:
                cm[jx, ipr] += val
            else:
                if jx < nr:
                    cm[jx, ipr] += val
                else:
                    cw[jx-nr, ipr] += val


def cmss(state: NECState, cm, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, j1, j2, im1, im2, nrow, itrp, ld):
    i1 = (im1 + 1)//2
    i2 = (im2 + 1)//2
    icomp = i1*2 - 3
    ii1 = -3 if (icomp + 2 < im1) else -2

    for i in range(i1, i2+1):
        il = ld - i
        icomp += 2
        ii1 += 2
        ii2 = ii1 + 1
        s_il = salp[il]
        t1xi, t1yi, t1zi = t1x[il]*s_il, t1y[il]*s_il, t1z[il]*s_il
        t2xi, t2yi, t2zi = t2x[il]*s_il, t2y[il]*s_il, t2z[il]*s_il
        xi, yi, zi = x[il], y[il], z[il]

        jj1 = -2
        for j in range(j1, j2+1):
            jl = ld - j
            jj1 += 2
            jj2 = jj1 + 1
            state.s = bi[jl]
            state.xj, state.yj, state.zj = x[jl], y[jl], z[jl]
            state.cabj, state.sabj, state.salpj = t1x[jl], t1y[jl], t1z[jl]
            state.b, state.ind1, state.ind2 = t2x[jl], t2y[jl], t2z[jl]

            hintg(state, xi, yi, zi)
            g11 = -(t2xi*state.exk + t2yi*state.eyk + t2zi*state.ezk)
            g12 = -(t2xi*state.exs + t2yi*state.eys + t2zi*state.ezs)
            g21 = -(t1xi*state.exk + t1yi*state.eyk + t1zi*state.ezk)
            g22 = -(t1xi*state.exs + t1yi*state.eys + t1zi*state.ezs)
            if i == j:
                g11 -= 0.5
                g22 += 0.5

            if itrp == 0:
                if icomp >= im1:
                    cm[ii1, jj1], cm[ii1, jj2] = g11, g12
                if icomp < im2:
                    cm[ii2, jj1], cm[ii2, jj2] = g21, g22
            else:
                if icomp >= im1:
                    cm[jj1, ii1], cm[jj2, ii1] = g11, g12
                if icomp < im2:
                    cm[jj1, ii2], cm[jj2, ii2] = g21, g22


def cmsw(state: NECState, cm, cw, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, iconx, j1, j2, i1, i2, nrow, ncw, itrp, ld):
    pi = 3.141592654
    neqs = state.n - state.n1 + 2*(state.m - state.m1)
    if itrp < 0:
        if i1 <= j1 <= i2:
            j1_idx = j1 - 1
            ipch = int(icon1[j1_idx])
            fsign = -1.0
            if ipch < 10000:
                ipch = int(icon2[j1_idx])
                if ipch < 10000:
                    return
                fsign = 1.0
            ipch -= 10000
            if ipch > state.m1:
                return
            js = ld - ipch
            state.ipgnd = 1
            state.cabj, state.sabj, state.salpj = t1x[js], t1y[js], t1z[js]
            state.b, state.ind1, state.ind2 = t2x[js], t2y[js], t2z[js]
            state.xj, state.yj, state.zj, state.s = x[js], y[js], z[js], bi[js]
            emel = pcint(state, x[j1_idx], y[j1_idx], z[j1_idx], t1y[j1_idx], t1z[j1_idx], salp[j1_idx])
            py = pi*t1x[j1_idx]*fsign
            exc = emel[8]*fsign
            il = int(state.jco[state.jsno-1]) - 1
            k = j1 - i1
            cw[k, il] += exc*(state.ax[state.jsno-1] + state.bx[state.jsno-1]*npy.sin(py) + state.cx[state.jsno-1]*npy.cos(py))
        return

    icgo = 1
    for k, i in enumerate(range(i1, i2+1)):
        i_idx = i - 1
        xi, yi, zi = x[i_idx], y[i_idx], z[i_idx]
        cabi, sabi, salpi = t1y[i_idx], t1z[i_idx], salp[i_idx]
        ipch = 0
        fsign = 1.0
        if icon1[i_idx] >= 10000:
            ipch = int(icon1[i_idx]) - 10000
            fsign = -1.0
        if icon2[i_idx] >= 10000:
            ipch = int(icon2[i_idx]) - 10000
            fsign = 1.0

        jl = 0
        for j in range(j1, j2+1):
            js = ld - j
            jl += 2
            state.cabj, state.sabj, state.salpj = t1x[js], t1y[js], t1z[js]
            state.b, state.ind1, state.ind2 = t2x[js], t2y[js], t2z[js]
            state.xj, state.yj, state.zj, state.s = x[js], y[js], z[js], bi[js]

            for ip in range(1, state.ksymp+1):
                state.ipgnd = ip
                if (ipch == j or icgo != 1) and ip != 2:
                    if icgo <= 1:
                        emel = pcint(state, xi, yi, zi, cabi, sabi, salpi)
                        py = pi*t1x[i_idx]*fsign
                        exc = emel[8]*fsign
                        trio(state, t1x, bi, icon1, icon2, i, ld)
                        if i <= state.n1:
                            il = neqs + int(iconx[i_idx]) - 1
                        else:
                            il = i - ncw
                            if i <= state.np:
                                il = ((il - 1)//state.np)*2*state.mp + il
                            il -= 1
                        term = exc*(state.ax[state.jsno-1] + state.bx[state.jsno-1]*npy.sin(py) + state.cx[state.jsno-1]*npy.cos(py))
                        if itrp == 0:
                            cw[k, il] += term
                        else:
                            cw[il, k] += term
                    if itrp == 0:
                        cm[k, jl-2], cm[k, jl-1] = emel[icgo-1], emel[icgo+3]
                    else:
                        cm[jl-2, k], cm[jl-1, k] = emel[icgo-1], emel[icgo+3]
                    icgo = 1 if icgo == 4 else icgo + 1
                else:
                    unere(state, xi, yi, zi)
                    v1 = state.exk*cabi + state.eyk*sabi + state.ezk*salpi
                    v2 = state.exs*cabi + state.eys*sabi + state.ezs*salpi
                    if itrp == 0:
                        cm[k, jl-2] += v1
                        cm[k, jl-1] += v2
                    else:
                        cm[jl-2, k] += v1
                        cm[jl-1, k] += v2


def cmws(state: NECState, cm, cw, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, nr, nw, j, i1, i2, itrp, ld):
    j_idx = j - 1
    state.s, state.b = t1x[j_idx], bi[j_idx]
    state.xj, state.yj, state.zj = x[j_idx], y[j_idx], z[j_idx]
    state.cabj, state.sabj, state.salpj = t1y[j_idx], t1z[j_idx], salp[j_idx]

    for ipr, i in enumerate(range(i1, i2+1)):
        ipatch = (i + 1)//2
        ik = i - (i//2)*2
        js = ld - ipatch
        if not (ik == 0 and ipr != 0):
            hsfld(state, x[js], y[js], z[js], 0.0)
        if ik != 0:
            tx, ty, tz = t2x[js], t2y[js], t2z[js]
        else:
            tx, ty, tz = t1x[js], t1y[js], t1z[js]
        etk = -(state.exk*tx + state.eyk*ty + state.ezk*tz)*salp[js]
        ets = -(state.exs*tx + state.eys*ty + state.ezs*tz)*salp[js]
        etc = -(state.exc*tx + state.eyc*ty + state.ezc*tz)*salp[js]

        for ij in range(state.jsno):
            jx = int(state.jco[ij]) - 1
            val = etk*state.ax[ij] + ets*state.bx[ij] + etc*state.cx[ij]
            if itrp == 0:
                cm[ipr, jx] += val
            elif itrp != 2:
                cm[jx, ipr] += val
            else:
                if jx < nr:
                    cm[jx, ipr] += val
                else:
                    cw[jx-nr, ipr] += val

def cmngf(
    state: NECState,
    cm: npy.ndarray,
    ib11: int,
    ic11: int,
    id11: int,
    zarray: npy.ndarray,
    t1x: npy.ndarray,
    t1y: npy.ndarray,
    t1z: npy.ndarray,
    t2x: npy.ndarray,
    t2y: npy.ndarray,
    t2z: npy.ndarray,
    x: npy.ndarray,
    y: npy.ndarray,
    z: npy.ndarray,
    bi: npy.ndarray,
    salp: npy.ndarray,
    rkhx: float,
    icon1: npy.ndarray,
    icon2: npy.ndarray,
    iconx: npy.ndarray,
    nb: int,
    nc: int,
    nd: int,
    iexkx: int,
    ld: int,
):
    '''Fills interaction matrices B, C, and D for Numerical Green's Function (N.G.F.) solution.'''
    state.rkh = rkhx
    state.iexk = iexkx
    m1eq = 2*state.m1
    m2eq = m1eq + 1
    meq = 2*state.m
    neqp = nd - state.npcon*2
    neqs = neqp - state.nscon
    neqsp = neqs + nc
    neqn = nc + state.n - state.n1
    itx = 2 if state.nscon > 0 else 1

    cb_flat = cm[ib11-1 :]
    cc_flat = cm[ic11-1 :]
    cd_flat = cm[id11-1 :]

    cb = cb_flat[: nb*nd].reshape((nb, nd), order='F')
    cc = cc_flat[: nc*nd].reshape((nc, nd), order='F')
    cd = cd_flat[: nd*nd].reshape((nd, nd), order='F')

    f12 = f14 = f15 = None
    if state.icasx != 1:
        f12 = open('fort.12', 'wb')
        f14 = open('fort.14', 'wb')
        f15 = open('fort.15', 'wb')

    if state.icasx <= 2:
        cd.fill(0j)
        cb.fill(0j)
        cc.fill(0j)

    ist = state.n - state.n1
    it = state.npbx
    isv = -state.npbx

    # Fill B (and D(WW), D(WS) when ICASX <= 2)
    for iblk in range(1, state.nbbx+1):
        isv += state.npbx
        if iblk == state.nbbx:
            it = state.nlbx
        if state.icasx >= 3:
            cb[:it, :nd].fill(0j)

        i1, i2 = isv+1, isv+it
        in2 = min(i2, state.n1)
        im1 = max(i1 - state.n1, 1)
        im2 = i2 - state.n1
        imx = (state.n1 - i1+1) if i1 <= state.n1 else 0

        if state.n2 <= state.n:
            for j in range(state.n2, state.n+1):
                trio(state, t1x, bi, icon1, icon2, j, ld)
                for i in range(state.jsno):
                    jss = int(state.jco[i])
                    state.jco[i] = (jss - state.n1) if jss >= state.n2 else (neqs + int(iconx[jss-1]))

                if i1 <= in2:
                    cmww(state, cb, cb, x, y, z, bi, salp, t1x, t1y, t1z, icon1, icon2, nb, nb, j, i1, in2, 0, ld)
                if im1 <= im2:
                    cmws(state, cb[imx:, :], cb, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, nb, nb, j, im1, im2, 0, ld)

                if state.icasx <= 2:
                    cmww(state, cd, cd, x, y, z, bi, salp, t1x, t1y, t1z, icon1, icon2, nd, nd, j, state.n2, state.n, 1, ld)
                    if state.m2 <= state.m:
                        cmws(state, cd[:, ist:], cd, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, nd, nd, j, m2eq, meq, 1, ld)
                    if state.nload != 0:
                        ir_idx = j - state.n1 - 1
                        exk = zarray[j-1]
                        for i in range(state.jsno):
                            jss = int(state.jco[i]) - 1
                            cd[jss, ir_idx] -= (state.ax[i] + state.cx[i])*exk

        if state.nscon != 0:
            for i in range(1, state.nscon+1):
                j = int(state.iscon[i-1])
                trio(state, t1x, bi, icon1, icon2, j, ld)
                jss = 0
                for ix_i in range(state.jsno):
                    ir_v = int(state.jco[ix_i])
                    if ir_v >= state.n2:
                        ir_v -= state.n1
                    else:
                        ir_v = int(iconx[ir_v-1])
                        if ir_v == 0:
                            continue
                        ir_v += neqs
                    state.jco[jss] = ir_v
                    state.ax[jss], state.bx[jss], state.cx[jss] = state.ax[ix_i], state.bx[ix_i], state.cx[ix_i]
                    jss += 1
                state.jsno = jss

                if i1 <= in2:
                    cmww(state, cb, cb, x, y, z, bi, salp, t1x, t1y, t1z, icon1, icon2, nb, nb, j, i1, in2, 0, ld)
                if im1 <= im2:
                    cmws(state, cb[imx:, :], cb, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, nb, nb, j, im1, im2, 0, ld)
                if i1 <= in2:
                    cmsw(state, cb, cb, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, iconx, j, i, i1, in2, nb, 0, -1, ld)

                if state.nlodf != 0:
                    jx = j - isv
                    if 1 <= jx <= it:
                        exk = zarray[j-1]
                        for ix_i in range(state.jsno):
                            cb[jx-1, int(state.jco[ix_i])-1] -= (state.ax[ix_i] + state.cx[ix_i])*exk

                tbf(state, t1x, bi, icon1, icon2, 1, j)
                jsx = state.jsno
                state.jsno = 1
                jco_saved = state.jco[:jsx].copy()
                ax_saved = state.ax[:jsx].copy()
                bx_saved = state.bx[:jsx].copy()
                cx_saved = state.cx[:jsx].copy()
                state.jco = neqs + i

                for ix_i in range(jsx):
                    ir_v = int(jco_saved[ix_i])
                    state.ax, state.bx, state.cx = ax_saved[ix_i], bx_saved[ix_i], cx_saved[ix_i]
                    if ir_v > state.n1 or iconx[ir_v-1] != 0:
                        continue
                    if i1 <= in2:
                        cmww(state, cb, cb, x, y, z, bi, salp, t1x, t1y, t1z, icon1, icon2, nb, nb, ir_v, i1, in2, 0, ld)
                    if im1 <= im2:
                        cmws(state, cb[imx:, :], cb, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, nb, nb, ir_v, im1, im2, 0, ld)
                    if state.nlodf != 0:
                        jx = ir_v - isv
                        if 1 <= jx <= it:
                            cb[jx-1, neqs+i-1] -= (state.ax + state.cx)*zarray[ir_v-1]

        if state.npcon != 0:
            jss = neqp
            for i in range(state.npcon):
                ix_p = int(state.ipcon[i])*2 + state.n1 - isv
                ir_p = ix_p - 1
                if 1 <= ir_p <= it:
                    cb[ir_p-1, jss] = 1.0 + 0j
                jss += 1
                if 1 <= ix_p <= it:
                    cb[ix_p-1, jss] = 1.0 + 0j
                jss += 1

        if state.m2 <= state.m:
            if i1 <= in2:
                cmsw(state, cb[:, ist:], cb, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, iconx, state.m2, state.m, i1, in2, nb, state.n1, 0, ld)
            if im1 <= im2:
                cmss(state, cb[imx:, ist:], x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, state.m2, state.m, im1, im2, nb, 0, ld)

        if state.icasx != 1:
            cb[:it, :nd].flatten(order='F').tofile(f14)
    # Fill C and D
    it = state.npbl
    isv = -state.npbl
    for iblk in range(1, state.nbbl+1):
        isv += state.npbl
        isvv = isv + nc
        if iblk == state.nbbl:
            it = state.nlbl
        if state.icasx >= 3:
            cc[:nc, :it].fill(0j)
            cd[:nd, :it].fill(0j)

        i1, i2 = isvv+1, isvv+it
        in1 = i1 - m1eq
        in2 = min(i2 - m1eq, state.n)
        im1 = max(i1 - state.n, m2eq)
        im2 = min(i2 - state.n, meq)
        imx = (neqn - i1+1) if in1 <= in2 else 0

        if state.icasx >= 3 and state.n2 <= state.n:
            for j in range(state.n2, state.n+1):
                trio(state, t1x, bi, icon1, icon2, j, ld)
                for i in range(state.jsno):
                    jss = int(state.jco[i])
                    state.jco[i] = (jss - state.n1) if jss >= state.n2 else (neqs + int(iconx[jss-1]))
                if in1 <= in2:
                    cmww(state, cd, cd, x, y, z, bi, salp, t1x, t1y, t1z, icon1, icon2, nd, nd, j, in1, in2, 1, ld)
                if im1 <= im2:
                    cmws(state, cd[:, imx:], cd, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, nd, nd, j, im1, im2, 1, ld)
                if state.nload != 0:
                    ir_v = j - state.n1 - isv
                    if 1 <= ir_v <= it:
                        exk = zarray[j-1]
                        for i in range(state.jsno):
                            cd[int(state.jco[i])-1, ir_v-1] -= (state.ax[i] + state.cx[i])*exk

        if state.m2 <= state.m:
            if in1 <= in2:
                cmsw(state, cd[ist:, :], cd, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, iconx, state.m2, state.m, in1, in2, nd, state.n1, 1, ld)
            if im1 <= im2:
                cmss(state, cd[ist:, imx:], x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, state.m2, state.m, im1, im2, nd, 1, ld)

        if state.n1 >= 1:
            for j in range(1, state.n1+1):
                trio(state, t1x, bi, icon1, icon2, j, ld)
                if state.nscon != 0:
                    for ix_i in range(state.jsno):
                        jss = int(state.jco[ix_i])
                        if jss >= state.n2:
                            state.jco[ix_i] = jss + m1eq
                        else:
                            ir_v = int(iconx[jss-1])
                            if ir_v != 0:
                                state.jco[ix_i] = neqsp + ir_v
                if in1 <= in2:
                    cmww(state, cc, cd, x, y, z, bi, salp, t1x, t1y, t1z, icon1, icon2, nc, nd, j, in1, in2, itx, ld)
                if im1 <= im2:
                    cmws(state, cc[:, imx:], cd[:, imx:], x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, nc, nd, j, im1, im2, itx, ld)

            if state.nscon != 0:
                for ix_i in range(state.nscon):
                    ir_idx = int(state.iscon[ix_i]) - 1
                    jss = neqs + (ix_i+1) - isv
                    if 1 <= jss <= it:
                        cc[ir_idx, jss-1] = 1.0 + 0j

        if state.npcon != 0:
            jss = neqp - isv
            for i in range(state.npcon):
                ix_p = int(state.ipcon[i])*2 + state.n1
                ir_p = ix_p - 1
                jss += 1
                if 1 <= jss <= it:
                    cc[ir_p-1, jss-1] = 1.0 + 0j
                jss += 1
                if 1 <= jss <= it:
                    cc[ix_p-1, jss-1] = 1.0 + 0j

        if state.m1 >= 1:
            if in1 <= in2:
                cmsw(state, cc[state.n2-1 :, :], cc, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, iconx, 1, state.m1, in1, in2, nc, 0, 1, ld)
            if im1 <= im2:
                cmss(state, cc[state.n2-1 :, imx:], x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, 1, state.m1, im1, im2, nc, 1, ld)

        if state.icasx != 1:
            cd[:nd, :it].flatten(order='F').tofile(f12)
            cc[:nc, :it].flatten(order='F').tofile(f15)

    for fp in (f12, f14, f15):
        if fp is not None:
            fp.close()

