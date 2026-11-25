import math
import numpy as npy
from .necState import NECState
from .main2 import tbf, blckot
from .hintg import trio, hsfld, hintg, pcint, unere, _set_ext_wire_flags
from .eksc import ekscVec, ekscxVec, rom2, sflds, rom2Vec, sfldsVec


def efldVec(
    state: NECState, xi: npy.ndarray, yi: npy.ndarray, zi: npy.ndarray,
    ai: npy.ndarray, ijDiff: npy.ndarray
):
    """Vectorized near E fields across observation points (ground included)."""
    eta, pi, tp = 376.73, 3.141592654, 6.283185308
    nObs = len(xi)
    xij, yij = xi - state.xj, yi - state.yj
    rfl = -1.0

    exk = npy.zeros(nObs, dtype=npy.complex128)
    eyk = npy.zeros(nObs, dtype=npy.complex128)
    ezk = npy.zeros(nObs, dtype=npy.complex128)
    exs = npy.zeros(nObs, dtype=npy.complex128)
    eys = npy.zeros(nObs, dtype=npy.complex128)
    ezs = npy.zeros(nObs, dtype=npy.complex128)
    exc = npy.zeros(nObs, dtype=npy.complex128)
    eyc = npy.zeros(nObs, dtype=npy.complex128)
    ezc = npy.zeros(nObs, dtype=npy.complex128)

    for ip in range(1, state.ksymp + 1):
        ijZero = (ijDiff == 0) if ip == 1 else npy.zeros(nObs, dtype=bool)
        rfl = -rfl
        salpr = state.salpj*rfl
        zij = zi - rfl*state.zj
        zp = xij*state.cabj + yij*state.sabj + zij*salpr
        rhox = xij - state.cabj*zp
        rhoy = yij - state.sabj*zp
        rhoz = zij - salpr*zp
        rh = npy.sqrt(rhox*rhox + rhoy*rhoy + rhoz*rhoz + ai*ai)

        mRh = (rh > 1.0e-10)
        rhox = npy.where(mRh, rhox/npy.where(mRh, rh, 1.0), 0.0)
        rhoy = npy.where(mRh, rhoy/npy.where(mRh, rh, 1.0), 0.0)
        rhoz = npy.where(mRh, rhoz/npy.where(mRh, rh, 1.0), 0.0)

        r = npy.sqrt(zp*zp + rh*rh)
        farMask = (r >= state.rkh)
        nearMask = ~farMask

        tezs = npy.zeros(nObs, dtype=npy.complex128)
        ters = npy.zeros(nObs, dtype=npy.complex128)
        tezc = npy.zeros(nObs, dtype=npy.complex128)
        terc = npy.zeros(nObs, dtype=npy.complex128)
        tezk = npy.zeros(nObs, dtype=npy.complex128)
        terk = npy.zeros(nObs, dtype=npy.complex128)
        txs = npy.zeros(nObs, dtype=npy.complex128)
        tys = npy.zeros(nObs, dtype=npy.complex128)
        tzs = npy.zeros(nObs, dtype=npy.complex128)

        if npy.any(farMask):
            rf = r[farMask]
            zpf = zp[farMask]
            rhf = rh[farMask]
            rmag = tp*rf
            cth, px = zpf/rf, rhf/rf
            txkF = npy.exp(-1j*rmag)
            py = tp*rf*rf
            tyk = eta*cth*txkF*(1.0 - 1j/rmag)/py
            tzk = eta*px*txkF*(1.0 + 1j*(rmag - 1.0/rmag))/(2.0*py)
            tezkF = tyk*cth - tzk*px
            terkF = tyk*px + tzk*cth
            rmagS = math.sin(pi*state.s)/pi
            tezc[farMask] = tezkF*rmagS
            terc[farMask] = terkF*rmagS
            tezk[farMask] = tezkF*state.s
            terk[farMask] = terkF*state.s

        if npy.any(nearMask):
            if state.iexk == 1:
                res = ekscxVec(
                    state.b, state.s, zp[nearMask], rh[nearMask],
                    tp, ijZero[nearMask], state.ind1, state.ind2
                )
            else:
                res = ekscVec(
                    state.s, zp[nearMask], rh[nearMask],
                    tp, ijZero[nearMask]
                )
            tezs[nearMask], ters[nearMask] = res[0], res[1]
            tezc[nearMask], terc[nearMask] = res[2], res[3]
            tezk[nearMask], terk[nearMask] = res[4], res[5]
            txs[nearMask] = (
                tezs[nearMask]*state.cabj + ters[nearMask]*rhox[nearMask]
            )
            tys[nearMask] = (
                tezs[nearMask]*state.sabj + ters[nearMask]*rhoy[nearMask]
            )
            tzs[nearMask] = (
                tezs[nearMask]*salpr + ters[nearMask]*rhoz[nearMask]
            )

        txk = tezk*state.cabj + terk*rhox
        tyk = tezk*state.sabj + terk*rhoy
        tzk = tezk*salpr + terk*rhoz
        txc = tezc*state.cabj + terc*rhox
        tyc = tezc*state.sabj + terc*rhoy
        tzc = tezc*salpr + terc*rhoz
        if ip != 2:
            exk, eyk, ezk = txk, tyk, tzk
            exs, eys, ezs = txs, tys, tzs
            exc, eyc, ezc = txc, tyc, tzc
        else:
            if state.iperf <= 0:
                zratx = npy.full(nObs, state.zrati, dtype=npy.complex128)
                xymag = npy.sqrt(xij*xij + yij*yij)
                if state.nradl != 0:
                    denom = zi + state.zj
                    xspec = (xi*state.zj + zi*state.xj)/denom
                    yspec = (yi*state.zj + zi*state.yj)/denom
                    rhospc = npy.sqrt(
                        xspec*xspec + yspec*yspec + state.t2*state.t2
                    )
                    mScrn = (rhospc <= state.scrwl)
                    if npy.any(mScrn):
                        rs = rhospc[mScrn]
                        zscrn = state.t1*rs*npy.log(rs/state.t2)
                        zratx[mScrn] = (
                            (zscrn*state.zrati)
                            /(eta*state.zrati + zscrn)
                        )
                mXy = (xymag > 1.0e-6)
                safeXy = npy.where(mXy, xymag, 1.0)
                px = npy.where(mXy, -yij/safeXy, 0.0)
                py = npy.where(mXy, xij/safeXy, 0.0)
                cth = npy.where(mXy, zij/r, 1.0)
                zrsin = npy.where(
                    mXy,
                    npy.sqrt(1.0 - zratx*zratx*(1.0 - cth*cth)),
                    1.0 + 0j,
                )
                refs = (cth - zratx*zrsin)/(cth + zratx*zrsin)
                refps = -(zratx*cth - zrsin)/(zratx*cth + zrsin) - refs

                epy = px*txk + py*tyk
                txk = refs*txk + refps*px*epy
                tyk = refs*tyk + refps*py*epy
                tzk = refs*tzk

                epy = px*txs + py*tys
                txs = refs*txs + refps*px*epy
                tys = refs*tys + refps*py*epy
                tzs = refs*tzs

                epy = px*txc + py*tyc
                txc = refs*txc + refps*px*epy
                tyc = refs*tyc + refps*py*epy
                tzc = refs*tzc

            exk -= txk*state.frati
            eyk -= tyk*state.frati
            ezk -= tzk*state.frati
            exs -= txs*state.frati
            eys -= tys*state.frati
            ezs -= tzs*state.frati
            exc -= txc*state.frati
            eyc -= tyc*state.frati
            ezc -= tzc*state.frati

    if state.iperf != 2:
        return exk, eyk, ezk, exs, eys, ezs, exc, eyc, ezc

    sn = math.sqrt(state.cabj*state.cabj + state.sabj*state.sabj)
    if sn < 1.0e-5:
        state.sn, state.xsn, state.ysn = 0.0, 1.0, 0.0
    else:
        state.sn, state.xsn, state.ysn = sn, state.cabj/sn, state.sabj/sn

    salpr = -state.salpj
    zijG = zi + state.zj
    rhoxG = state.sabj*zijG - salpr*yij
    rhoyG = salpr*xij - state.cabj*zijG
    rhozG = state.cabj*yij - state.sabj*xij
    rhSq = rhoxG*rhoxG + rhoyG*rhoyG + rhozG*rhozG

    mSmall = (rhSq <= 1.0e-10)
    safeRh = npy.sqrt(npy.where(mSmall, 1.0, rhSq))
    rhScale = npy.where(rhozG < 0.0, -ai/safeRh, ai/safeRh)
    xo = npy.where(mSmall, xi - ai*state.ysn, xi + rhScale*rhoxG)
    yo = npy.where(mSmall, yi + ai*state.xsn, yi + rhScale*rhoyG)
    zo = npy.where(mSmall, zi, zi + rhScale*rhozG)

    rSq = xij*xij + yij*yij + zijG*zijG
    mClose = (rSq <= 0.95)
    mFar = ~mClose
    egndAll = npy.zeros((9, nObs), dtype=npy.complex128)

    if npy.any(mClose):
        shaf = 0.5*state.s
        dmin = 0.01*npy.sqrt(
            npy.abs(exk[mClose])**2 + npy.abs(eyk[mClose])**2
            + npy.abs(ezk[mClose])**2
        )
        egC = rom2Vec(state, -shaf, shaf, dmin, xo[mClose], yo[mClose], zo[mClose])
        zpC = xij[mClose]*state.cabj + yij[mClose]*state.sabj + zijG[mClose]*salpr
        rhP = rSq[mClose] - zpC*zpC
        aiC = ai[mClose]
        dminP = npy.where(
            rhP <= 1.0e-10, 0.0,
            npy.sqrt(npy.maximum(rhP, 0.0)/(npy.maximum(rhP, 0.0) + aiC*aiC))
        )
        mProj = (dminP <= 0.95)
        if npy.any(mProj):
            pxV = 1.0 - dminP[mProj]
            dProj = dminP[mProj]
            for base in (0, 3, 6):
                eX = egC[base, mProj]
                eY = egC[base + 1, mProj]
                eZ = egC[base + 2, mProj]
                ter = (eX*state.cabj + eY*state.sabj + eZ*salpr)*pxV
                egC[base, mProj] = dProj*eX + ter*state.cabj
                egC[base + 1, mProj] = dProj*eY + ter*state.sabj
                egC[base + 2, mProj] = dProj*eZ + ter*salpr
        egndAll[:, mClose] = egC

    if npy.any(mFar):
        tZero = npy.zeros(int(npy.sum(mFar)), dtype=npy.float64)
        egndAll[:, mFar] = sfldsVec(
            state, tZero, xo[mFar], yo[mFar], zo[mFar], 2
        )

    exk0, eyk0, ezk0, exs0, eys0, ezs0, exc0, eyc0, ezc0 = egndAll
    exk += exk0
    eyk += eyk0
    ezk += ezk0
    exs += exs0
    eys += eys0
    ezs += ezs0
    exc += exc0
    eyc += eyc0
    ezc += ezc0
    return exk, eyk, ezk, exs, eys, ezs, exc, eyc, ezc

def cmset(
    state: NECState, cm: npy.ndarray, d: npy.ndarray,
    zarray: npy.ndarray, t1x: npy.ndarray, t1y: npy.ndarray,
    t1z: npy.ndarray, t2x: npy.ndarray, t2y: npy.ndarray,
    t2z: npy.ndarray, x: npy.ndarray, y: npy.ndarray, z: npy.ndarray,
    bi: npy.ndarray, salp: npy.ndarray, icon1: npy.ndarray,
    icon2: npy.ndarray, iconx: npy.ndarray, nrow: int, ld: int,
    ld2: int, iresrv: int,
):
    """Sets up the complex interaction matrix in the array CM."""
    mp2 = 2*state.mp
    npeq = state.np + mp2
    neq = state.n + 2*state.m
    nop = neq//npeq

    f11 = open("fort.11", "wb") if state.icase > 2 else None
    iout = 2*state.npblk*nrow
    it = state.npblk

    for ixblk1 in range(1, state.nbloks + 1):
        isv = (ixblk1 - 1)*state.npblk
        if ixblk1 == state.nbloks:
            it = state.nlast

        cmBlk = cm[:nrow*it].reshape((nrow, it), order="F")
        cmBlk.fill(0j)

        i1 = isv + 1
        i2 = isv + it
        in2 = min(i2, state.np)
        im1 = max(i1 - state.np, 1)
        im2 = i2 - state.np
        ist = (state.np - i1 + 2) if i1 <= state.np else 1

        if state.n != 0:
            for j in range(1, state.n + 1):
                trio(state, t1x, bi, icon1, icon2, j, ld)
                jsno = state.jsno
                jcoSlice = state.jco[:jsno]
                state.jco[:jsno] = ((jcoSlice - 1)//state.np)*mp2 + jcoSlice

                if i1 <= in2:
                    cmww(
                        state, cmBlk, cmBlk, x, y, z, bi, salp,
                        t1x, t1y, t1z, icon1, icon2, nrow, nrow,
                        j, i1, in2, 1, ld
                    )
                if im1 <= im2:
                    cmws(
                        state, cmBlk[:, ist - 1:], cmBlk, x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, nrow, nrow,
                        j, im1, im2, 1, ld
                    )
                if state.nload != 0 and j <= state.np:
                    ipr = j - isv
                    if 1 <= ipr <= it:
                        zaj = zarray[j - 1]
                        jss = state.jco[:jsno].astype(npy.intp) - 1
                        cmBlk[jss, ipr - 1] -= (
                            state.ax[:jsno] + state.cx[:jsno]
                        )*zaj

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
                        state, cmBlk[jst - 1:, :], cmBlk, x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, iconx,
                        jm1, jm2, i1, in2, nrow, 0, 1, ld
                    )
                if im1 <= im2:
                    cmss(
                        state, cmBlk[jst - 1:, ist - 1:], x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, jm1, jm2,
                        im1, im2, nrow, 1, ld
                    )
        if state.icase != 1:
            if state.icase != 3:
                # j + k*npeq in Fortran order -> shape (npeq, nop, it), order='F'
                blkModes = cmBlk[:nop*npeq, :it].reshape((npeq, nop, it), order="F")
                ssxFwd = state.ssx[:nop, :nop].copy()
                ssxFwd[0, :] = 1.0
                ssxFwd[:, 0] = 1.0
                blkModes[:] = npy.einsum("kl,jli->jki", ssxFwd, blkModes)

    if f11 is not None:
        f11.close()


def cmww(
    state: NECState, cm, cw, x, y, z, bi, salp, t1x, t1y, t1z,
    icon1, icon2, nr, nw, j, i1, i2, itrp, ld
):
    """Vectorized wire-to-wire interaction matrix fill for source j."""
    jIdx = j - 1
    state.s, state.b = t1x[jIdx], bi[jIdx]
    state.xj, state.yj, state.zj = x[jIdx], y[jIdx], z[jIdx]
    state.cabj, state.sabj, state.salpj = t1y[jIdx], t1z[jIdx], salp[jIdx]
    _set_ext_wire_flags(state, j, t1y, t1z, salp, bi, icon1, icon2)

    sIdx = slice(i1 - 1, i2)
    ijDiff = npy.arange(i1, i2 + 1) - j
    exk, eyk, ezk, exs, eys, ezs, exc, eyc, ezc = efldVec(
        state, x[sIdx], y[sIdx], z[sIdx], bi[sIdx], ijDiff
    )

    cabi, sabi, salpi = t1y[sIdx], t1z[sIdx], salp[sIdx]
    etk = exk*cabi + eyk*sabi + ezk*salpi
    ets = exs*cabi + eys*sabi + ezs*salpi
    etc = exc*cabi + eyc*sabi + ezc*salpi

    jsno = state.jsno
    jxArr = state.jco[:jsno].astype(npy.intp) - 1
    # Outer product: shape (jsno, nObs)
    vals = (
        state.ax[:jsno, None]*etk[None, :]
        + state.bx[:jsno, None]*ets[None, :]
        + state.cx[:jsno, None]*etc[None, :]
    )
    nObs = i2 - i1 + 1
    if itrp == 0:
        npy.add.at(cm[:, :nObs].T, jxArr, vals)
    elif itrp != 2:
        npy.add.at(cm[:, :nObs], jxArr, vals)
    else:
        mNr = (jxArr < nr)
        if npy.any(mNr):
            npy.add.at(cm[:, :nObs], jxArr[mNr], vals[mNr])
        if npy.any(~mNr):
            npy.add.at(cw[:, :nObs], jxArr[~mNr] - nr, vals[~mNr])

def cmss(
    state: NECState, cm, x, y, z, bi, salp, t1x, t1y, t1z,
    t2x, t2y, t2z, j1, j2, im1, im2, nrow, itrp, ld
):
    """Fills surface-patch to surface-patch interaction submatrix."""
    i1 = (im1 + 1)//2
    i2 = (im2 + 1)//2
    icomp = i1*2 - 3
    ii1 = -3 if (icomp + 2 < im1) else -2

    for i in range(i1, i2 + 1):
        il = ld - i
        icomp += 2
        ii1 += 2
        ii2 = ii1 + 1
        sIl = salp[il]
        t1xi, t1yi, t1zi = t1x[il]*sIl, t1y[il]*sIl, t1z[il]*sIl
        t2xi, t2yi, t2zi = t2x[il]*sIl, t2y[il]*sIl, t2z[il]*sIl
        xi, yi, zi = x[il], y[il], z[il]

        jj1 = -2
        for j in range(j1, j2 + 1):
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


def cmsw(
    state: NECState, cm, cw, x, y, z, bi, salp, t1x, t1y, t1z,
    t2x, t2y, t2z, icon1, icon2, iconx, j1, j2, i1, i2,
    nrow, ncw, itrp, ld
):
    """Fills surface-patch to wire interaction submatrix."""
    pi = 3.141592654
    neqs = state.n - state.n1 + 2*(state.m - state.m1)
    if itrp < 0:
        if i1 <= j1 <= i2:
            j1Idx = j1 - 1
            ipch = int(icon1[j1Idx])
            fsign = -1.0
            if ipch < 10000:
                ipch = int(icon2[j1Idx])
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
            emel = pcint(
                state, x[j1Idx], y[j1Idx], z[j1Idx],
                t1y[j1Idx], t1z[j1Idx], salp[j1Idx]
            )
            py = pi*t1x[j1Idx]*fsign
            exc = emel[8]*fsign
            il = int(state.jco[state.jsno - 1]) - 1
            k = j1 - i1
            jLast = state.jsno - 1
            cw[k, il] += exc*(
                state.ax[jLast] + state.bx[jLast]*math.sin(py)
                + state.cx[jLast]*math.cos(py)
            )
        return

    icgo = 1
    for k, i in enumerate(range(i1, i2 + 1)):
        iIdx = i - 1
        xi, yi, zi = x[iIdx], y[iIdx], z[iIdx]
        cabi, sabi, salpi = t1y[iIdx], t1z[iIdx], salp[iIdx]
        ipch = 0
        fsign = 1.0
        if icon1[iIdx] >= 10000:
            ipch = int(icon1[iIdx]) - 10000
            fsign = -1.0
        if icon2[iIdx] >= 10000:
            ipch = int(icon2[iIdx]) - 10000
            fsign = 1.0

        jl = 0
        for j in range(j1, j2 + 1):
            js = ld - j
            jl += 2
            state.cabj, state.sabj, state.salpj = t1x[js], t1y[js], t1z[js]
            state.b, state.ind1, state.ind2 = t2x[js], t2y[js], t2z[js]
            state.xj, state.yj, state.zj, state.s = x[js], y[js], z[js], bi[js]

            for ip in range(1, state.ksymp + 1):
                state.ipgnd = ip
                if (ipch == j or icgo != 1) and ip != 2:
                    if icgo <= 1:
                        emel = pcint(state, xi, yi, zi, cabi, sabi, salpi)
                        py = pi*t1x[iIdx]*fsign
                        exc = emel[8]*fsign
                        trio(state, t1x, bi, icon1, icon2, i, ld)
                        if i <= state.n1:
                            il = neqs + int(iconx[iIdx]) - 1
                        else:
                            il = i - ncw
                            if i <= state.np:
                                il = ((il - 1)//state.np)*2*state.mp + il
                            il -= 1
                        jLast = state.jsno - 1
                        term = exc*(
                            state.ax[jLast] + state.bx[jLast]*math.sin(py)
                            + state.cx[jLast]*math.cos(py)
                        )
                        if itrp == 0:
                            cw[k, il] += term
                        else:
                            cw[il, k] += term
                    if itrp == 0:
                        cm[k, jl - 2] = emel[icgo - 1]
                        cm[k, jl - 1] = emel[icgo + 3]
                    else:
                        cm[jl - 2, k] = emel[icgo - 1]
                        cm[jl - 1, k] = emel[icgo + 3]
                    icgo = 1 if icgo == 4 else icgo + 1
                else:
                    unere(state, xi, yi, zi)
                    v1 = state.exk*cabi + state.eyk*sabi + state.ezk*salpi
                    v2 = state.exs*cabi + state.eys*sabi + state.ezs*salpi
                    if itrp == 0:
                        cm[k, jl - 2] += v1
                        cm[k, jl - 1] += v2
                    else:
                        cm[jl - 2, k] += v1
                        cm[jl - 1, k] += v2
def cmws(
    state: NECState, cm, cw, x, y, z, bi, salp, t1x, t1y, t1z,
    t2x, t2y, t2z, nr, nw, j, i1, i2, itrp, ld
):
    """Fills wire-to-surface-patch interaction submatrix."""
    jIdx = j - 1
    state.s, state.b = t1x[jIdx], bi[jIdx]
    state.xj, state.yj, state.zj = x[jIdx], y[jIdx], z[jIdx]
    state.cabj, state.sabj, state.salpj = t1y[jIdx], t1z[jIdx], salp[jIdx]

    jsno = state.jsno
    jxArr = state.jco[:jsno].astype(npy.intp) - 1
    axS, bxS, cxS = state.ax[:jsno], state.bx[:jsno], state.cx[:jsno]

    for ipr, i in enumerate(range(i1, i2 + 1)):
        ipatch = (i + 1)//2
        ik = i - (i//2)*2
        js = ld - ipatch
        if not (ik == 0 and ipr != 0):
            hsfld(state, x[js], y[js], z[js], 0.0)
        if ik != 0:
            tx, ty, tz = t2x[js], t2y[js], t2z[js]
        else:
            tx, ty, tz = t1x[js], t1y[js], t1z[js]
        salpJs = salp[js]
        etk = -(state.exk*tx + state.eyk*ty + state.ezk*tz)*salpJs
        ets = -(state.exs*tx + state.eys*ty + state.ezs*tz)*salpJs
        etc = -(state.exc*tx + state.eyc*ty + state.ezc*tz)*salpJs

        vals = etk*axS + ets*bxS + etc*cxS
        if itrp == 0:
            npy.add.at(cm[ipr], jxArr, vals)
        elif itrp != 2:
            npy.add.at(cm[:, ipr], jxArr, vals)
        else:
            mNr = (jxArr < nr)
            if npy.any(mNr):
                npy.add.at(cm[:, ipr], jxArr[mNr], vals[mNr])
            if npy.any(~mNr):
                npy.add.at(cw[:, ipr], jxArr[~mNr] - nr, vals[~mNr])


def cmngf(
    state: NECState, cm: npy.ndarray, ib11: int, ic11: int, id11: int,
    zarray: npy.ndarray, t1x: npy.ndarray, t1y: npy.ndarray,
    t1z: npy.ndarray, t2x: npy.ndarray, t2y: npy.ndarray,
    t2z: npy.ndarray, x: npy.ndarray, y: npy.ndarray, z: npy.ndarray,
    bi: npy.ndarray, salp: npy.ndarray, rkhx: float,
    icon1: npy.ndarray, icon2: npy.ndarray, iconx: npy.ndarray,
    nb: int, nc: int, nd: int, iexkx: int, ld: int,
):
    """Fills interaction matrices B, C, and D for N.G.F. solution."""
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

    cbFlat = cm[ib11 - 1:]
    ccFlat = cm[ic11 - 1:]
    cdFlat = cm[id11 - 1:]

    cb = cbFlat[:nb*nd].reshape((nb, nd), order="F")
    cc = ccFlat[:nc*nd].reshape((nc, nd), order="F")
    cd = cdFlat[:nd*nd].reshape((nd, nd), order="F")

    f12 = f14 = f15 = None
    if state.icasx != 1:
        f12 = open("fort.12", "wb")
        f14 = open("fort.14", "wb")
        f15 = open("fort.15", "wb")

    if state.icasx <= 2:
        cd.fill(0j)
        cb.fill(0j)
        cc.fill(0j)

    ist = state.n - state.n1
    it = state.npbx
    isv = -state.npbx

    # Fill B (and D(WW), D(WS) when ICASX <= 2)
    for iblk in range(1, state.nbbx + 1):
        isv += state.npbx
        if iblk == state.nbbx:
            it = state.nlbx
        if state.icasx >= 3:
            cb[:it, :nd].fill(0j)

        i1, i2 = isv + 1, isv + it
        in2 = min(i2, state.n1)
        im1 = max(i1 - state.n1, 1)
        im2 = i2 - state.n1
        imx = (state.n1 - i1 + 1) if i1 <= state.n1 else 0
        if state.n2 <= state.n:
            for j in range(state.n2, state.n + 1):
                trio(state, t1x, bi, icon1, icon2, j, ld)
                jsno = state.jsno
                for i in range(jsno):
                    jss = int(state.jco[i])
                    state.jco[i] = (
                        (jss - state.n1) if jss >= state.n2
                        else (neqs + int(iconx[jss - 1]))
                    )

                if i1 <= in2:
                    cmww(
                        state, cb, cb, x, y, z, bi, salp, t1x, t1y, t1z,
                        icon1, icon2, nb, nb, j, i1, in2, 0, ld
                    )
                if im1 <= im2:
                    cmws(
                        state, cb[imx:, :], cb, x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, nb, nb,
                        j, im1, im2, 0, ld
                    )

                if state.icasx <= 2:
                    cmww(
                        state, cd, cd, x, y, z, bi, salp, t1x, t1y, t1z,
                        icon1, icon2, nd, nd, j, state.n2, state.n, 1, ld
                    )
                    if state.m2 <= state.m:
                        cmws(
                            state, cd[:, ist:], cd, x, y, z, bi, salp,
                            t1x, t1y, t1z, t2x, t2y, t2z, nd, nd,
                            j, m2eq, meq, 1, ld
                        )
                    if state.nload != 0:
                        irIdx = j - state.n1 - 1
                        exk = zarray[j - 1]
                        jssArr = state.jco[:jsno].astype(npy.intp) - 1
                        cd[jssArr, irIdx] -= (
                            state.ax[:jsno] + state.cx[:jsno]
                        )*exk

        if state.nscon != 0:
            for i in range(1, state.nscon + 1):
                j = int(state.iscon[i - 1])
                trio(state, t1x, bi, icon1, icon2, j, ld)
                jss = 0
                for ixI in range(state.jsno):
                    irV = int(state.jco[ixI])
                    if irV >= state.n2:
                        irV -= state.n1
                    else:
                        irV = int(iconx[irV - 1])
                        if irV == 0:
                            continue
                        irV += neqs
                    state.jco[jss] = irV
                    state.ax[jss] = state.ax[ixI]
                    state.bx[jss] = state.bx[ixI]
                    state.cx[jss] = state.cx[ixI]
                    jss += 1
                state.jsno = jss

                if i1 <= in2:
                    cmww(
                        state, cb, cb, x, y, z, bi, salp, t1x, t1y, t1z,
                        icon1, icon2, nb, nb, j, i1, in2, 0, ld
                    )
                if im1 <= im2:
                    cmws(
                        state, cb[imx:, :], cb, x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, nb, nb,
                        j, im1, im2, 0, ld
                    )
                if i1 <= in2:
                    cmsw(
                        state, cb, cb, x, y, z, bi, salp, t1x, t1y, t1z,
                        t2x, t2y, t2z, icon1, icon2, iconx,
                        j, i, i1, in2, nb, 0, -1, ld
                    )

                if state.nlodf != 0:
                    jx = j - isv
                    if 1 <= jx <= it:
                        exk = zarray[j - 1]
                        for ixI in range(state.jsno):
                            colIdx = int(state.jco[ixI]) - 1
                            cb[jx - 1, colIdx] -= (
                                state.ax[ixI] + state.cx[ixI]
                            )*exk

                tbf(state, t1x, bi, icon1, icon2, 1, j)
                jsx = state.jsno
                state.jsno = 1
                jcoSaved = state.jco[:jsx].copy()
                axSaved = state.ax[:jsx].copy()
                bxSaved = state.bx[:jsx].copy()
                cxSaved = state.cx[:jsx].copy()
                state.jco = neqs + i

                for ixI in range(jsx):
                    irV = int(jcoSaved[ixI])
                    state.ax = axSaved[ixI]
                    state.bx = bxSaved[ixI]
                    state.cx = cxSaved[ixI]
                    if irV > state.n1 or iconx[irV - 1] != 0:
                        continue
                    if i1 <= in2:
                        cmww(
                            state, cb, cb, x, y, z, bi, salp, t1x, t1y, t1z,
                            icon1, icon2, nb, nb, irV, i1, in2, 0, ld
                        )
                    if im1 <= im2:
                        cmws(
                            state, cb[imx:, :], cb, x, y, z, bi, salp,
                            t1x, t1y, t1z, t2x, t2y, t2z, nb, nb,
                            irV, im1, im2, 0, ld
                        )
                    if state.nlodf != 0:
                        jx = irV - isv
                        if 1 <= jx <= it:
                            cb[jx - 1, neqs + i - 1] -= (
                                state.ax + state.cx
                            )*zarray[irV - 1]
        if state.npcon != 0:
            jss = neqp
            for i in range(state.npcon):
                ixP = int(state.ipcon[i])*2 + state.n1 - isv
                irP = ixP - 1
                if 1 <= irP <= it:
                    cb[irP - 1, jss] = 1.0 + 0j
                jss += 1
                if 1 <= ixP <= it:
                    cb[ixP - 1, jss] = 1.0 + 0j
                jss += 1

        if state.m2 <= state.m:
            if i1 <= in2:
                cmsw(
                    state, cb[:, ist:], cb, x, y, z, bi, salp, t1x, t1y, t1z,
                    t2x, t2y, t2z, icon1, icon2, iconx,
                    state.m2, state.m, i1, in2, nb, state.n1, 0, ld
                )
            if im1 <= im2:
                cmss(
                    state, cb[imx:, ist:], x, y, z, bi, salp, t1x, t1y, t1z,
                    t2x, t2y, t2z, state.m2, state.m, im1, im2, nb, 0, ld
                )

        if state.icasx != 1:
            cb[:it, :nd].flatten(order="F").tofile(f14)

    # Fill C and D
    it = state.npbl
    isv = -state.npbl
    for iblk in range(1, state.nbbl + 1):
        isv += state.npbl
        isvv = isv + nc
        if iblk == state.nbbl:
            it = state.nlbl
        if state.icasx >= 3:
            cc[:nc, :it].fill(0j)
            cd[:nd, :it].fill(0j)

        i1, i2 = isvv + 1, isvv + it
        in1 = i1 - m1eq
        in2 = min(i2 - m1eq, state.n)
        im1 = max(i1 - state.n, m2eq)
        im2 = min(i2 - state.n, meq)
        imx = (neqn - i1 + 1) if in1 <= in2 else 0

        if state.icasx >= 3 and state.n2 <= state.n:
            for j in range(state.n2, state.n + 1):
                trio(state, t1x, bi, icon1, icon2, j, ld)
                for i in range(state.jsno):
                    jss = int(state.jco[i])
                    state.jco[i] = (
                        (jss - state.n1) if jss >= state.n2
                        else (neqs + int(iconx[jss - 1]))
                    )
                if in1 <= in2:
                    cmww(
                        state, cd, cd, x, y, z, bi, salp, t1x, t1y, t1z,
                        icon1, icon2, nd, nd, j, in1, in2, 1, ld
                    )
                if im1 <= im2:
                    cmws(
                        state, cd[:, imx:], cd, x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, nd, nd,
                        j, im1, im2, 1, ld
                    )
                if state.nload != 0:
                    irV = j - state.n1 - isv
                    if 1 <= irV <= it:
                        exkV = zarray[j - 1]
                        for i in range(state.jsno):
                            rowIdx = int(state.jco[i]) - 1
                            cd[rowIdx, irV - 1] -= (
                                state.ax[i] + state.cx[i]
                            )*exkV
        if state.m2 <= state.m:
            if in1 <= in2:
                cmsw(
                    state, cd[ist:, :], cd, x, y, z, bi, salp, t1x, t1y, t1z,
                    t2x, t2y, t2z, icon1, icon2, iconx,
                    state.m2, state.m, in1, in2, nd, state.n1, 1, ld
                )
            if im1 <= im2:
                cmss(
                    state, cd[ist:, imx:], x, y, z, bi, salp, t1x, t1y, t1z,
                    t2x, t2y, t2z, state.m2, state.m, im1, im2, nd, 1, ld
                )

        if state.n1 >= 1:
            for j in range(1, state.n1 + 1):
                trio(state, t1x, bi, icon1, icon2, j, ld)
                if state.nscon != 0:
                    for ixI in range(state.jsno):
                        jss = int(state.jco[ixI])
                        if jss >= state.n2:
                            state.jco[ixI] = jss + m1eq
                        else:
                            irV = int(iconx[jss - 1])
                            if irV != 0:
                                state.jco[ixI] = neqsp + irV
                if in1 <= in2:
                    cmww(
                        state, cc, cd, x, y, z, bi, salp, t1x, t1y, t1z,
                        icon1, icon2, nc, nd, j, in1, in2, itx, ld
                    )
                if im1 <= im2:
                    cmws(
                        state, cc[:, imx:], cd[:, imx:], x, y, z, bi, salp,
                        t1x, t1y, t1z, t2x, t2y, t2z, nc, nd,
                        j, im1, im2, itx, ld
                    )

            if state.nscon != 0:
                for ixI in range(state.nscon):
                    irIdx = int(state.iscon[ixI]) - 1
                    jss = neqs + (ixI + 1) - isv
                    if 1 <= jss <= it:
                        cc[irIdx, jss - 1] = 1.0 + 0j

        if state.npcon != 0:
            jss = neqp - isv
            for i in range(state.npcon):
                ixP = int(state.ipcon[i])*2 + state.n1
                irP = ixP - 1
                jss += 1
                if 1 <= jss <= it:
                    cc[irP - 1, jss - 1] = 1.0 + 0j
                jss += 1
                if 1 <= jss <= it:
                    cc[ixP - 1, jss - 1] = 1.0 + 0j

        if state.m1 >= 1:
            if in1 <= in2:
                cmsw(
                    state, cc[state.n2 - 1:, :], cc, x, y, z, bi, salp,
                    t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, iconx,
                    1, state.m1, in1, in2, nc, 0, 1, ld
                )
            if im1 <= im2:
                cmss(
                    state, cc[state.n2 - 1:, imx:], x, y, z, bi, salp,
                    t1x, t1y, t1z, t2x, t2y, t2z, 1, state.m1,
                    im1, im2, nc, 1, ld
                )

        if state.icasx != 1:
            cd[:nd, :it].flatten(order="F").tofile(f12)
            cc[:nc, :it].flatten(order="F").tofile(f15)

    for fp in (f12, f14, f15):
        if fp is not None:
            fp.close()

