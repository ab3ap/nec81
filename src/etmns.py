import math
import cmath
import numpy as npy
from .necState import NECState
from .main2 import tbf
from .hintg import _set_ext_wire_flags, hsfld
from .cmset import efldVec


def etmns(
    state: NECState, e: npy.ndarray, zarray: npy.ndarray,
    x: npy.ndarray, y: npy.ndarray, z: npy.ndarray, bi: npy.ndarray,
    salp: npy.ndarray, t1x: npy.ndarray, t1y: npy.ndarray, t1z: npy.ndarray,
    t2x: npy.ndarray, t2y: npy.ndarray, t2z: npy.ndarray,
    p1: float, p2: float, p3: float, p4: float, p5: float, p6: float,
    icon1: npy.ndarray, icon2: npy.ndarray, ld: int, ld2: int,
    ld3: int, ipr: int,
):
    """Vectorized fill of E with negative of incident electric field."""
    tp, reta = 6.283185308, 2.654420938e-3
    nSeg, mPat = state.n, state.m
    neq = nSeg + 2*mPat
    state.nqds = 0

    if ipr <= 0 or ipr == 5:
        e[:neq] = 0j
        for i in range(state.nsant):
            isIdx = int(state.isant[i]) - 1
            e[isIdx] = -state.vsant[i]/(t1x[isIdx]*state.wlam)
        for i in range(state.nvqd):
            isSeg = int(state.ivqd[i])
            qdsrc(
                state, e, zarray, state.vqd[i], x, y, z, bi, salp,
                t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, isSeg, ld
            )
        return

    if ipr <= 3:
        cth, sth = math.cos(p1), math.sin(p1)
        cph, sph = math.cos(p2), math.sin(p2)
        cet, set_ = math.cos(p3), math.sin(p3)
        px = cth*cph*cet - sph*set_
        py = cth*sph*cet + cph*set_
        pz = -sth*cet
        wx, wy, wz = -sth*cph, -sth*sph, -cth
        qx, qy, qz = wy*pz - wz*py, wz*px - wx*pz, wx*py - wy*px

        if state.ksymp != 1:
            if state.iperf == 1:
                rrv = rrh = complex(-1.0, 0.0)
            else:
                rrv = cmath.sqrt(1.0 - state.zrati*state.zrati*sth*sth)
                rrh = state.zrati*cth
                rrh = (rrh - rrv)/(rrh + rrv)
                rrv = state.zrati*rrv
                rrv = -(cth - rrv)/(cth + rrv)

        if ipr <= 1:
            if nSeg != 0:
                xS, yS, zS = x[:nSeg], y[:nSeg], z[:nSeg]
                cS, sS, slS = t1y[:nSeg], t1z[:nSeg], salp[:nSeg]
                arg = -tp*(wx*xS + wy*yS + wz*zS)
                e[:nSeg] = -(px*cS + py*sS + pz*slS)*npy.exp(1j*arg)
                if state.ksymp != 1:
                    tt1 = (py*cph - px*sph)*(rrh - rrv)
                    cx, cy, cz = rrv*px - tt1*sph, rrv*py + tt1*cph, -rrv*pz
                    argG = -tp*(wx*xS + wy*yS - wz*zS)
                    e[:nSeg] -= (cx*cS + cy*sS + cz*slS)*npy.exp(1j*argG)

            if mPat != 0:
                pIdx = npy.arange(ld - 1, ld - 1 - mPat, -1)
                xP, yP, zP = x[pIdx], y[pIdx], z[pIdx]
                t1xP, t1yP, t1zP = t1x[pIdx], t1y[pIdx], t1z[pIdx]
                t2xP, t2yP, t2zP = t2x[pIdx], t2y[pIdx], t2z[pIdx]
                salpP = salp[pIdx]
                arg = -tp*(wx*xP + wy*yP + wz*zP)
                tt1 = npy.exp(1j*arg)*salpP*reta
                pView = e[nSeg:nSeg + 2*mPat].reshape((mPat, 2))
                pView[:, 1] = (qx*t1xP + qy*t1yP + qz*t1zP)*tt1
                pView[:, 0] = (qx*t2xP + qy*t2yP + qz*t2zP)*tt1
                if state.ksymp != 1:
                    tt1G = (qy*cph - qx*sph)*(rrv - rrh)
                    cx = -(rrh*qx - tt1G*sph)
                    cy = -(rrh*qy + tt1G*cph)
                    cz = rrh*qz
                    argG = -tp*(wx*xP + wy*yP - wz*zP)
                    tt2 = npy.exp(1j*argG)*salpP*reta
                    pView[:, 1] += (cx*t1xP + cy*t1yP + cz*t1zP)*tt2
                    pView[:, 0] += (cx*t2xP + cy*t2yP + cz*t2zP)*tt2
            return

        # Elliptic polarization (IPR = 2 or 3)
        tt1 = complex(0.0, -p6) if ipr != 3 else complex(0.0, p6)
        if nSeg != 0:
            xS, yS, zS = x[:nSeg], y[:nSeg], z[:nSeg]
            cS, sS, slS = t1y[:nSeg], t1z[:nSeg], salp[:nSeg]
            cx, cy, cz = px + tt1*qx, py + tt1*qy, pz + tt1*qz
            arg = -tp*(wx*xS + wy*yS + wz*zS)
            e[:nSeg] = -(cx*cS + cy*sS + cz*slS)*npy.exp(1j*arg)
            if state.ksymp != 1:
                tt2 = (cy*cph - cx*sph)*(rrh - rrv)
                cx, cy, cz = rrv*cx - tt2*sph, rrv*cy + tt2*cph, -rrv*cz
                argG = -tp*(wx*xS + wy*yS - wz*zS)
                e[:nSeg] -= (cx*cS + cy*sS + cz*slS)*npy.exp(1j*argG)

        if mPat != 0:
            pIdx = npy.arange(ld - 1, ld - 1 - mPat, -1)
            xP, yP, zP = x[pIdx], y[pIdx], z[pIdx]
            t1xP, t1yP, t1zP = t1x[pIdx], t1y[pIdx], t1z[pIdx]
            t2xP, t2yP, t2zP = t2x[pIdx], t2y[pIdx], t2z[pIdx]
            salpP = salp[pIdx]
            cx, cy, cz = qx - tt1*px, qy - tt1*py, qz - tt1*pz
            arg = -tp*(wx*xP + wy*yP + wz*zP)
            tt2 = npy.exp(1j*arg)*salpP*reta
            pView = e[nSeg:nSeg + 2*mPat].reshape((mPat, 2))
            pView[:, 1] = (cx*t1xP + cy*t1yP + cz*t1zP)*tt2
            pView[:, 0] = (cx*t2xP + cy*t2yP + cz*t2zP)*tt2
            if state.ksymp != 1:
                tt1G = (cy*cph - cx*sph)*(rrv - rrh)
                cx = -(rrh*cx - tt1G*sph)
                cy = -(rrh*cy + tt1G*cph)
                cz = rrh*cz
                argG = -tp*(wx*xP + wy*yP - wz*zP)
                tt2G = npy.exp(1j*argG)*salpP*reta
                pView[:, 1] += (cx*t1xP + cy*t1yP + cz*t1zP)*tt2G
                pView[:, 0] += (cx*t2xP + cy*t2yP + cz*t2zP)*tt2G
        return

    # Elementary current source (IPR = 4)
    wz = math.cos(p4)
    wx, wy, wz = wz*math.cos(p5), wz*math.sin(p5), math.sin(p4)
    ds, dsh = p6*59.958, p6/(2.0*tp)

    if nSeg != 0:
        pxA = x[:nSeg] - p1
        pyA = y[:nSeg] - p2
        pzA = z[:nSeg] - p3
        rs = pxA*pxA + pyA*pyA + pzA*pzA
        mV = (rs >= 1.0e-30)
        safeRs = npy.where(mV, rs, 1.0)
        r = npy.sqrt(safeRs)
        pxA, pyA, pzA = pxA/r, pyA/r, pzA/r
        cth = pxA*wx + pyA*wy + pzA*wz
        sth = npy.sqrt(npy.maximum(0.0, 1.0 - cth*cth))
        qxA, qyA, qzA = pxA - wx*cth, pyA - wy*cth, pzA - wz*cth
        qNorm = npy.sqrt(qxA*qxA + qyA*qyA + qzA*qzA)
        mQ = (qNorm >= 1.0e-30)
        safeQ = npy.where(mQ, qNorm, 1.0)
        qxA = npy.where(mQ, qxA/safeQ, 1.0)
        qyA = npy.where(mQ, qyA/safeQ, 0.0)
        qzA = npy.where(mQ, qzA/safeQ, 0.0)

        tt1 = npy.exp(-1j*tp*r)
        tt2 = (1.0 - 1j/(r*tp))/safeRs
        er = ds*tt1*tt2*cth
        et = 0.5*ds*tt1*(1j*tp/r + tt2)*sth
        ezh = er*cth - et*sth
        erh = er*sth + et*cth
        cx = ezh*wx + erh*qxA
        cy = ezh*wy + erh*qyA
        cz = ezh*wz + erh*qzA
        e[:nSeg] = npy.where(
            mV, -(cx*t1y[:nSeg] + cy*t1z[:nSeg] + cz*salp[:nSeg]), 0j
        )

    if mPat != 0:
        pIdx = npy.arange(ld - 1, ld - 1 - mPat, -1)
        pxA = x[pIdx] - p1
        pyA = y[pIdx] - p2
        pzA = z[pIdx] - p3
        rs = pxA*pxA + pyA*pyA + pzA*pzA
        mV = (rs >= 1.0e-30)
        safeRs = npy.where(mV, rs, 1.0)
        r = npy.sqrt(safeRs)
        pxA, pyA, pzA = pxA/r, pyA/r, pzA/r
        cth = pxA*wx + pyA*wy + pzA*wz
        sth = npy.sqrt(npy.maximum(0.0, 1.0 - cth*cth))
        qxA, qyA, qzA = pxA - wx*cth, pyA - wy*cth, pzA - wz*cth
        qNorm = npy.sqrt(qxA*qxA + qyA*qyA + qzA*qzA)
        mQ = (qNorm >= 1.0e-30)
        safeQ = npy.where(mQ, qNorm, 1.0)
        qxA = npy.where(mQ, qxA/safeQ, 1.0)
        qyA = npy.where(mQ, qyA/safeQ, 0.0)
        qzA = npy.where(mQ, qzA/safeQ, 0.0)

        pxC = wy*qzA - wz*qyA
        pyC = wz*qxA - wx*qzA
        pzC = wx*qyA - wy*qxA
        tt1 = npy.exp(-1j*tp*r)
        tt2 = dsh*tt1*(1.0/r + 1j*tp)/r*sth*salp[pIdx]
        cx, cy, cz = tt2*pxC, tt2*pyC, tt2*pzC
        pView = e[nSeg:nSeg + 2*mPat].reshape((mPat, 2))
        pView[:, 1] = npy.where(
            mV, cx*t1x[pIdx] + cy*t1y[pIdx] + cz*t1z[pIdx], 0j
        )
        pView[:, 0] = npy.where(
            mV, cx*t2x[pIdx] + cy*t2y[pIdx] + cz*t2z[pIdx], 0j
        )

def qdsrc(
    state: NECState, e, zarray, v: complex, x, y, z, bi, salp,
    t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, isSeg: int, ld: int
):
    """Fills incident field array for charge-discontinuity voltage source."""
    tp = 6.283185308
    ccj = complex(0.0, -0.01666666667)
    isIdx = isSeg - 1
    savedIcon1 = int(icon1[isIdx])
    icon1[isIdx] = 0
    tbf(state, t1x, bi, icon1, icon2, 0, isSeg)
    icon1[isIdx] = savedIcon1

    s = t1x[isIdx]*0.5
    jsnoIdx = state.jsno - 1
    curd = ccj*v/(
        (math.log(2.0*s/bi[isIdx]) - 1.0)
        *(state.bx[jsnoIdx]*math.cos(tp*s) + state.cx[jsnoIdx]*math.sin(tp*s))
        *state.wlam
    )
    state.nqds += 1
    state.vqds[state.nqds - 1] = v
    state.iqds[state.nqds - 1] = isSeg

    nSeg = state.n
    obsIdx = npy.arange(1, nSeg + 1)
    for jx in range(state.jsno):
        j = int(state.jco[jx])
        jIdx = j - 1
        state.s, state.b = t1x[jIdx], bi[jIdx]
        state.xj, state.yj, state.zj = x[jIdx], y[jIdx], z[jIdx]
        state.cabj, state.sabj, state.salpj = t1y[jIdx], t1z[jIdx], salp[jIdx]
        _set_ext_wire_flags(state, j, t1y, t1z, salp, bi, icon1, icon2)

        exk, eyk, ezk, exs, eys, ezs, exc, eyc, ezc = efldVec(
            state, x[:nSeg], y[:nSeg], z[:nSeg], bi[:nSeg], obsIdx - j
        )
        cabi, sabi, salpi = t1y[:nSeg], t1z[:nSeg], salp[:nSeg]
        etk = exk*cabi + eyk*sabi + ezk*salpi
        ets = exs*cabi + eys*sabi + ezs*salpi
        etc = exc*cabi + eyc*sabi + ezc*salpi
        e[:nSeg] -= (
            etk*state.ax[jx] + ets*state.bx[jx] + etc*state.cx[jx]
        )*curd

        if state.m != 0:
            ij, i1 = ld, nSeg - 1
            for _ in range(state.m):
                ij -= 1
                hsfld(state, x[ij], y[ij], z[ij], 0.0)
                tx, ty, tz = t2x[ij], t2y[ij], t2z[ij]
                e[i1] += (
                    (state.exk*tx + state.eyk*ty + state.ezk*tz)*state.ax[jx]
                    + (state.exs*tx + state.eys*ty + state.ezs*tz)*state.bx[jx]
                    + (state.exc*tx + state.eyc*ty + state.ezc*tz)*state.cx[jx]
                )*curd*salp[ij]
                i1 += 1
                tx, ty, tz = t1x[ij], t1y[ij], t1z[ij]
                e[i1] += (
                    (state.exk*tx + state.eyk*ty + state.ezk*tz)*state.ax[jx]
                    + (state.exs*tx + state.eys*ty + state.ezs*tz)*state.bx[jx]
                    + (state.exc*tx + state.eyc*ty + state.ezc*tz)*state.cx[jx]
                )*curd*salp[ij]
                i1 += 1

