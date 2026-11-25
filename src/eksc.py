import math
import numpy as npy
from .necState import NECState
from .main2 import gwave
from .intrp import intrp, intrpVec


def gxVec(zz: npy.ndarray, rh: npy.ndarray, xk: float):
    """Vectorized segment end contributions for thin wire approximation."""
    r2 = zz*zz + rh*rh
    r = npy.sqrt(r2)
    rk = xk*r
    gz = npy.exp(-1j*rk)/r
    gzp = -(1.0 + 1j*rk)*gz/r2
    return gz, gzp


def gxxVec(
    zz: npy.ndarray, rh: npy.ndarray, a: npy.ndarray,
    a2: npy.ndarray, xk: float, ira: npy.ndarray
):
    """Vectorized segment end contributions for extended thin wire."""
    r2 = zz*zz + rh*rh
    r = npy.sqrt(r2)
    r4 = r2*r2
    rk = xk*r
    rk2 = rk*rk
    rh2 = rh*rh
    t1 = 0.25*a2*rh2/r4
    t2 = 0.5*a2/r2
    c1 = 1.0 + 1j*rk
    c2 = 3.0*c1 - rk2
    c3 = (6.0 + 1j*rk)*rk2 - 15.0*c1
    gz = npy.exp(-1j*rk)/r
    g2 = gz*(1.0 + t1*c2)
    g1 = g2 - t2*c1*gz
    gz = gz/r2
    g2p = gz*(t1*c3 - c1)
    gzp = t2*c2*gz
    g3 = g2p + gzp
    g1p = g3*zz

    outG2 = npy.zeros_like(gz)
    outG2p = npy.zeros_like(gz)
    outG3 = npy.zeros_like(gz)
    outGzp = -zz*c1*gz

    m0 = (ira != 1)
    if npy.any(m0):
        outG3[m0] = (g3[m0] + gzp[m0])*rh[m0]
        mRh = m0 & (rh > 1.0e-10)
        outG2[mRh] = g2[mRh]/rh[mRh]
        outG2p[mRh] = g2p[mRh]*zz[mRh]/rh[mRh]

    m1 = ~m0
    if npy.any(m1):
        t2a = 0.5*a[m1]
        g2a = -t2a*c1[m1]*gz[m1]
        g2pa = t2a*gz[m1]*c2[m1]/r2[m1]
        outG2[m1] = g2a
        outG2p[m1] = g2pa*zz[m1]
        outG3[m1] = rh2[m1]*g2pa - a[m1]*gz[m1]*c1[m1]

    return g1, g1p, outG2, outG2p, outG3, outGzp


def intxVec(
    el1: float, el2: float, b: npy.ndarray, zpk: npy.ndarray,
    rkb2: npy.ndarray, ijZero: npy.ndarray
):
    """Vectorized Romberg integration of exp(jkr)/r across observations."""
    nx, nm, nts, rx = 1, 65536, 4, 1.0e-4
    nPts = len(b)
    sgr = npy.zeros(nPts, dtype=npy.float64)
    sgi = npy.zeros(nPts, dtype=npy.float64)
    if nPts == 0:
        return sgr, sgi

    z = npy.full(nPts, el1, dtype=npy.float64)
    ze = npy.where(ijZero, 0.0, el2)
    s = ze - z
    ep = s/(10.0*nm)
    zend = ze - ep
    ns = npy.full(nPts, nx, dtype=npy.int32)
    nt = npy.zeros(nPts, dtype=npy.int32)

    def _gfVec(zkV: npy.ndarray, rkb2V: npy.ndarray, zpkV: npy.ndarray, ijZ: npy.ndarray):
        zdk = zkV - zpkV
        rk = npy.sqrt(rkb2V + zdk*zdk)
        si = npy.sin(rk)/rk
        co = npy.empty_like(rk)
        mNonZero = ~ijZ
        if npy.any(mNonZero):
            co[mNonZero] = npy.cos(rk[mNonZero])/rk[mNonZero]
        mZeroLarge = ijZ & (rk >= 0.2)
        if npy.any(mZeroLarge):
            co[mZeroLarge] = (npy.cos(rk[mZeroLarge]) - 1.0)/rk[mZeroLarge]
        mZeroSmall = ijZ & (rk < 0.2)
        if npy.any(mZeroSmall):
            rks = rk[mZeroSmall]*rk[mZeroSmall]
            co[mZeroSmall] = (
                (-1.38888889e-3*rks + 4.16666667e-2)*rks - 0.5
            )*rk[mZeroSmall]
        return co, si

    active = npy.arange(nPts, dtype=npy.intp)
    g1r, g1i = _gfVec(z, rkb2, zpk, ijZero)

    while len(active) > 0:
        dz = s[active]/ns.astype(npy.float64)
        over = (z + dz > ze[active])
        dz = npy.where(over, ze[active] - z, dz)
        keep = (npy.abs(dz) > ep[active])
        if not npy.all(keep):
            active = active[keep]
            z = z[keep]
            ns = ns[keep]
            nt = nt[keep]
            dz = dz[keep]
            g1r, g1i = g1r[keep], g1i[keep]
            if len(active) == 0:
                break

        dzot = dz*0.5
        rkbA, zpkA, ijA = rkb2[active], zpk[active], ijZero[active]
        g3r, g3i = _gfVec(z + dzot, rkbA, zpkA, ijA)
        g5r, g5i = _gfVec(z + dz, rkbA, zpkA, ijA)

        subIdx = npy.arange(len(active), dtype=npy.intp)
        while len(subIdx) > 0:
            dzS, dzotS = dz[subIdx], dzot[subIdx]
            g1rS, g1iS = g1r[subIdx], g1i[subIdx]
            g3rS, g3iS = g3r[subIdx], g3i[subIdx]
            g5rS, g5iS = g5r[subIdx], g5i[subIdx]

            t00r, t00i = (g1rS + g5rS)*dzotS, (g1iS + g5iS)*dzotS
            t01r, t01i = (t00r + dzS*g3rS)*0.5, (t00i + dzS*g3iS)*0.5
            t10r, t10i = (4.0*t01r - t00r)/3.0, (4.0*t01i - t00i)/3.0
            den1 = npy.maximum(npy.abs(t10r), npy.abs(t10i))
            safeD1 = npy.where(den1 < 1.0e-37, 1.0, den1)
            te1r = npy.where(den1 < 1.0e-37, 0.0, npy.abs(t01r - t10r)/safeD1)
            te1i = npy.where(den1 < 1.0e-37, 0.0, npy.abs(t01i - t10i)/safeD1)
            c1 = (te1i <= rx) & (te1r <= rx)

            if npy.any(c1):
                done1 = subIdx[c1]
                sgr[active[done1]] += t10r[c1]
                sgi[active[done1]] += t10i[c1]
                nt[done1] += 2
                subIdx = subIdx[~c1]
                if len(subIdx) == 0:
                    break

            dzS, dzotS = dz[subIdx], dzot[subIdx]
            t01rS, t01iS = t01r[~c1], t01i[~c1]
            t10rS, t10iS = t10r[~c1], t10i[~c1]
            actS = active[subIdx]
            g2r, g2i = _gfVec(
                z[subIdx] + dzS*0.25, rkb2[actS], zpk[actS], ijZero[actS]
            )
            g4r, g4i = _gfVec(
                z[subIdx] + dzS*0.75, rkb2[actS], zpk[actS], ijZero[actS]
            )
            t02r = (t01rS + dzotS*(g2r + g4r))*0.5
            t02i = (t01iS + dzotS*(g2i + g4i))*0.5
            t11r, t11i = (4.0*t02r - t01rS)/3.0, (4.0*t02i - t01iS)/3.0
            t20r, t20i = (16.0*t11r - t10rS)/15.0, (16.0*t11i - t10iS)/15.0
            den2 = npy.maximum(npy.abs(t20r), npy.abs(t20i))
            safeD2 = npy.where(den2 < 1.0e-37, 1.0, den2)
            te2r = npy.where(den2 < 1.0e-37, 0.0, npy.abs(t11r - t20r)/safeD2)
            te2i = npy.where(den2 < 1.0e-37, 0.0, npy.abs(t11i - t20i)/safeD2)
            c2 = ((te2i <= rx) & (te2r <= rx)) | (ns[subIdx] >= nm)

            if npy.any(c2):
                done2 = subIdx[c2]
                sgr[active[done2]] += t20r[c2]
                sgi[active[done2]] += t20i[c2]
                nt[done2] += 1

            rem = ~c2
            if not npy.any(rem):
                break
            remSub = subIdx[rem]
            nt[remSub] = 0
            ns[remSub] *= 2
            dz[remSub] = s[active[remSub]]/ns[remSub].astype(npy.float64)
            dzot[remSub] = dz[remSub]*0.5
            g5r[remSub], g5i[remSub] = g3r[remSub], g3i[remSub]
            g3r[remSub], g3i[remSub] = g2r[~c2], g2i[~c2]
            subIdx = remSub

        z += dz
        notEnd = (z < zend[active])
        if not npy.all(notEnd):
            active = active[notEnd]
            z = z[notEnd]
            ns = ns[notEnd]
            nt = nt[notEnd]
            g5r, g5i = g5r[notEnd], g5i[notEnd]
            if len(active) == 0:
                break
        g1r, g1i = g5r, g5i
        halfMask = (nt >= nts) & (ns > nx)
        ns = npy.where(halfMask, ns//2, ns)
        nt = npy.where(halfMask, 1, nt)

    if npy.any(ijZero):
        bZ = b[ijZero]
        sZ = s[ijZero]
        sgr[ijZero] = 2.0*(sgr[ijZero] + npy.log((npy.sqrt(bZ*bZ + sZ*sZ) + sZ)/bZ))
        sgi[ijZero] = 2.0*sgi[ijZero]
    return sgr, sgi


def ekscVec(
    s: float, z: npy.ndarray, rh: npy.ndarray, xk: float,
    ijZero: npy.ndarray
):
    """Vectorized thin-wire E field for sine, cosine, and const currents."""
    con = 4.771341189j
    zpk = xk*z
    rhk = xk*rh
    rkb2 = rhk*rhk
    sh = 0.5*s
    shk = xk*sh
    ss = math.sin(shk)
    cs = math.cos(shk)
    z2 = sh - z
    z1 = -(sh + z)

    gz1, gp1 = gxVec(z1, rh, xk)
    gz2, gp2 = gxVec(z2, rh, xk)

    gzp1 = gp1*z1
    gzp2 = gp2*z2
    ezs = con*((gz2 - gz1)*cs*xk - (gzp2 + gzp1)*ss)
    ezc = -con*((gz2 + gz1)*ss*xk + (gzp2 - gzp1)*cs)
    erk = con*(gp2 - gp1)*rh
    cint, sint = intxVec(-shk, shk, rhk, zpk, rkb2, ijZero)
    ezk = -con*(gzp2 - gzp1 + xk*xk*(cint - 1j*sint))

    gzp1 = gzp1*z1
    gzp2 = gzp2*z2
    ers = npy.zeros_like(ezs)
    erc = npy.zeros_like(ezc)
    mRh = (rh >= 1.0e-10)
    if npy.any(mRh):
        numS = (
            (gzp2[mRh] + gzp1[mRh] + gz2[mRh] + gz1[mRh])*ss
            - (z2[mRh]*gz2[mRh] - z1[mRh]*gz1[mRh])*cs*xk
        )
        numC = (
            (gzp2[mRh] - gzp1[mRh] + gz2[mRh] - gz1[mRh])*cs
            + (z2[mRh]*gz2[mRh] + z1[mRh]*gz1[mRh])*ss*xk
        )
        ers[mRh] = -con*numS/rh[mRh]
        erc[mRh] = -con*numC/rh[mRh]
    return ezs, ers, ezc, erc, ezk, erk

def ekscxVec(
    bx: float, s: float, z: npy.ndarray, rhx: npy.ndarray,
    xk: float, ijZero: npy.ndarray, inx1: int, inx2: int
):
    """Vectorized extended thin-wire E field across observation points."""
    con = 4.771341189j
    nPts = len(z)
    geBx = (rhx >= bx)
    rh = npy.where(geBx, rhx, bx)
    b = npy.where(geBx, bx, rhx)
    ira = npy.where(geBx, 0, 1)

    sh = 0.5*s
    zpk = xk*z
    rhk = xk*rh
    rkb2 = rhk*rhk
    shk = xk*sh
    ss = math.sin(shk)
    cs = math.cos(shk)
    z2 = sh - z
    z1 = -(sh + z)
    a2 = b*b

    if inx1 != 2:
        gz1, gzp1, gr1, grp1, grk1, gzz1 = gxxVec(
            z1, rh, b, a2, xk, ira
        )
    else:
        gz1, grk1 = gxVec(z1, rhx, xk)
        gzp1 = grk1*z1
        gr1, grp1, grk1 = gz1/rhx, gzp1/rhx, grk1*rhx
        gzz1 = npy.zeros(nPts, dtype=npy.complex128)

    if inx2 != 2:
        gz2, gzp2, gr2, grp2, grk2, gzz2 = gxxVec(
            z2, rh, b, a2, xk, ira
        )
    else:
        gz2, grk2 = gxVec(z2, rhx, xk)
        gzp2 = grk2*z2
        gr2, grp2, grk2 = gz2/rhx, gzp2/rhx, grk2*rhx
        gzz2 = npy.zeros(nPts, dtype=npy.complex128)

    ezs = con*((gz2 - gz1)*cs*xk - (gzp2 + gzp1)*ss)
    ezc = -con*((gz2 + gz1)*ss*xk + (gzp2 - gzp1)*cs)
    ers = -con*(
        (z2*grp2 + z1*grp1 + gr2 + gr1)*ss - (z2*gr2 - z1*gr1)*cs*xk
    )
    erc = -con*(
        (z2*grp2 - z1*grp1 + gr2 - gr1)*cs + (z2*gr2 + z1*gr1)*ss*xk
    )
    erk = con*(grk2 - grk1)
    cint, sint = intxVec(-shk, shk, rhk, zpk, rkb2, ijZero)
    bk = b*xk
    bk2 = bk*bk*0.25
    ezk = -con*(
        gzp2 - gzp1 + xk*xk*(1.0 - bk2)*(cint - 1j*sint)
        - bk2*(gzz2 - gzz1)
    )
    return ezs, ers, ezc, erc, ezk, erk


def gx(zz: float, rh: float, xk: float):
    """Segment end contributions for thin wire approximation."""
    r2 = zz*zz + rh*rh
    r = math.sqrt(r2)
    rk = xk*r
    gz = complex(math.cos(rk), -math.sin(rk))/r
    gzp = -complex(1.0, rk)*gz/r2
    return gz, gzp


def gxx(zz: float, rh: float, a: float, a2: float, xk: float, ira: int):
    """Segment end contributions for extended thin wire approximation."""
    r2 = zz*zz + rh*rh
    r = math.sqrt(r2)
    r4 = r2*r2
    rk = xk*r
    rk2 = rk*rk
    rh2 = rh*rh
    t1 = 0.25*a2*rh2/r4
    t2 = 0.5*a2/r2
    c1 = complex(1.0, rk)
    c2 = 3.0*c1 - rk2
    c3 = complex(6.0, rk)*rk2 - 15.0*c1
    gz = complex(math.cos(rk), -math.sin(rk))/r
    g2 = gz*(1.0 + t1*c2)
    g1 = g2 - t2*c1*gz
    gz = gz/r2
    g2p = gz*(t1*c3 - c1)
    gzp = t2*c2*gz
    g3 = g2p + gzp
    g1p = g3*zz
    if ira != 1:
        g3 = (g3 + gzp)*rh
        gzp = -zz*c1*gz
        if rh <= 1.0e-10:
            return g1, g1p, 0j, 0j, g3, gzp
        return g1, g1p, g2/rh, g2p*zz/rh, g3, gzp

    t2 = 0.5*a
    g2 = -t2*c1*gz
    g2p = t2*gz*c2/r2
    g3 = rh2*g2p - a*gz*c1
    g2p = g2p*zz
    gzp = -zz*c1*gz
    return g1, g1p, g2, g2p, g3, gzp

def eksc(state: NECState, s: float, z: float, rh: float, xk: float, ij: int):
    """Computes E field of sine, cosine, and constant currents (thin wire)."""
    con = complex(0.0, 4.771341189)
    state.ijx = ij
    state.zpk = xk*z
    rhk = xk*rh
    state.rkb2 = rhk*rhk
    sh = 0.5*s
    shk = xk*sh
    ss = math.sin(shk)
    cs = math.cos(shk)
    z2 = sh - z
    z1 = -(sh + z)

    gz1, gp1 = gx(z1, rh, xk)
    gz2, gp2 = gx(z2, rh, xk)

    gzp1 = gp1*z1
    gzp2 = gp2*z2
    ezs = con*((gz2 - gz1)*cs*xk - (gzp2 + gzp1)*ss)
    ezc = -con*((gz2 + gz1)*ss*xk + (gzp2 - gzp1)*cs)
    erk = con*(gp2 - gp1)*rh
    cint, sint = intx(state, -shk, shk, rhk, ij)
    ezk = -con*(gzp2 - gzp1 + xk*xk*complex(cint, -sint))
    gzp1 = gzp1*z1
    gzp2 = gzp2*z2
    if rh < 1.0e-10:
        return ezs, 0j, ezc, 0j, ezk, erk
    ers = -con*(
        (gzp2 + gzp1 + gz2 + gz1)*ss - (z2*gz2 - z1*gz1)*cs*xk
    )/rh
    erc = -con*(
        (gzp2 - gzp1 + gz2 - gz1)*cs + (z2*gz2 + z1*gz1)*ss*xk
    )/rh
    return ezs, ers, ezc, erc, ezk, erk


def ekscx(
    state: NECState, bx: float, s: float, z: float, rhx: float,
    xk: float, ij: int, inx1: int, inx2: int
):
    """Computes E field of sine, cosine, and const currents (extended)."""
    con = complex(0.0, 4.771341189)
    if rhx >= bx:
        rh, b, ira = rhx, bx, 0
    else:
        rh, b, ira = bx, rhx, 1

    sh = 0.5*s
    state.ijx = ij
    state.zpk = xk*z
    rhk = xk*rh
    state.rkb2 = rhk*rhk
    shk = xk*sh
    ss = math.sin(shk)
    cs = math.cos(shk)
    z2 = sh - z
    z1 = -(sh + z)
    a2 = b*b

    if inx1 != 2:
        gz1, gzp1, gr1, grp1, grk1, gzz1 = gxx(z1, rh, b, a2, xk, ira)
    else:
        gz1, grk1 = gx(z1, rhx, xk)
        gzp1 = grk1*z1
        gr1, grp1, grk1, gzz1 = gz1/rhx, gzp1/rhx, grk1*rhx, 0j

    if inx2 != 2:
        gz2, gzp2, gr2, grp2, grk2, gzz2 = gxx(z2, rh, b, a2, xk, ira)
    else:
        gz2, grk2 = gx(z2, rhx, xk)
        gzp2 = grk2*z2
        gr2, grp2, grk2, gzz2 = gz2/rhx, gzp2/rhx, grk2*rhx, 0j

    ezs = con*((gz2 - gz1)*cs*xk - (gzp2 + gzp1)*ss)
    ezc = -con*((gz2 + gz1)*ss*xk + (gzp2 - gzp1)*cs)
    ers = -con*(
        (z2*grp2 + z1*grp1 + gr2 + gr1)*ss - (z2*gr2 - z1*gr1)*cs*xk
    )
    erc = -con*(
        (z2*grp2 - z1*grp1 + gr2 - gr1)*cs + (z2*gr2 + z1*gr1)*ss*xk
    )
    erk = con*(grk2 - grk1)
    cint, sint = intx(state, -shk, shk, rhk, ij)
    bk = b*xk
    bk2 = bk*bk*0.25
    ezk = -con*(
        gzp2 - gzp1 + xk*xk*(1.0 - bk2)*complex(cint, -sint)
        - bk2*(gzz2 - gzz1)
    )
    return ezs, ers, ezc, erc, ezk, erk


def test(f1r: float, f2r: float, f1i: float, f2i: float, dmin: float):
    """Convergence test for Romberg integration. Returns (tr, ti)."""
    den = max(abs(f2r), abs(f2i), dmin)
    if den < 1.0e-37:
        return 0.0, 0.0
    return abs((f1r - f2r)/den), abs((f1i - f2i)/den)

def gf(state: NECState, zk: float):
    """Integrand exp(jkr)/(kr) for numerical integration. Returns (co, si)."""
    zdk = zk - state.zpk
    rk = math.sqrt(state.rkb2 + zdk*zdk)
    si = math.sin(rk)/rk
    if state.ijx != 0:
        return math.cos(rk)/rk, si
    if rk >= 0.2:
        return (math.cos(rk) - 1.0)/rk, si
    rks = rk*rk
    return ((-1.38888889e-3*rks + 4.16666667e-2)*rks - 0.5)*rk, si


def intx(state: NECState, el1: float, el2: float, b: float, ij: int):
    """Variable-interval-width Romberg integration of exp(jkr)/r."""
    nx, nm, nts, rx = 1, 65536, 4, 1.0e-4
    z = el1
    ze = 0.0 if ij == 0 else el2
    s = ze - z
    ep = s/(10.0*nm)
    zend = ze - ep
    sgr = sgi = 0.0
    ns, nt = nx, 0
    g1r, g1i = gf(state, z)

    while True:
        dz = s/float(ns)
        if z + dz > ze:
            dz = ze - z
            if abs(dz) <= ep:
                break
        dzot = dz*0.5
        g3r, g3i = gf(state, z + dzot)
        g5r, g5i = gf(state, z + dz)

        while True:
            t00r, t00i = (g1r + g5r)*dzot, (g1i + g5i)*dzot
            t01r, t01i = (t00r + dz*g3r)*0.5, (t00i + dz*g3i)*0.5
            t10r, t10i = (4.0*t01r - t00r)/3.0, (4.0*t01i - t00i)/3.0
            te1r, te1i = test(t01r, t10r, t01i, t10i, 0.0)
            if te1i <= rx and te1r <= rx:
                sgr += t10r
                sgi += t10i
                nt += 2
                break

            g2r, g2i = gf(state, z + dz*0.25)
            g4r, g4i = gf(state, z + dz*0.75)
            t02r = (t01r + dzot*(g2r + g4r))*0.5
            t02i = (t01i + dzot*(g2i + g4i))*0.5
            t11r, t11i = (4.0*t02r - t01r)/3.0, (4.0*t02i - t01i)/3.0
            t20r, t20i = (16.0*t11r - t10r)/15.0, (16.0*t11i - t10i)/15.0
            te2r, te2i = test(t11r, t20r, t11i, t20i, 0.0)
            if (te2i <= rx and te2r <= rx) or ns >= nm:
                sgr += t20r
                sgi += t20i
                nt += 1
                break

            nt = 0
            ns *= 2
            dz = s/float(ns)
            dzot = dz*0.5
            g5r, g5i = g3r, g3i
            g3r, g3i = g2r, g2i

        z += dz
        if z >= zend:
            break
        g1r, g1i = g5r, g5i
        if nt >= nts and ns > nx:
            ns //= 2
            nt = 1

    if ij == 0:
        sgr = 2.0*(sgr + math.log((math.sqrt(b*b + s*s) + s)/b))
        sgi = 2.0*sgi
    return sgr, sgi


def rom2(state: NECState, a: float, b: float, dmin: float) -> npy.ndarray:
    """Romberg integration over source segment for Sommerfeld ground."""
    nm, nts, nx, rx = 65536, 4, 1, 1.0e-4
    s = b - a
    if s < 0.0:
        raise ValueError(" ERROR - B LESS THAN A IN ROM2")
    ep = s/(1.0e4*nm)
    zend = b - ep
    total = npy.zeros(9, dtype=npy.complex128)
    z = a
    ns, nt = nx, 0
    g1 = sflds(state, z)

    while True:
        dz = s/float(ns)
        if z + dz > b:
            dz = b - z
            if dz <= ep:
                break
        dzot = dz*0.5
        g3 = sflds(state, z + dzot)
        g5 = sflds(state, z + dz)

        while True:
            t00 = (g1 + g5)*dzot
            t01 = (t00 + dz*g3)*0.5
            t10 = (4.0*t01 - t00)/3.0
            tmag1 = math.sqrt(float(npy.sum(npy.abs(t01[:3])**2)))
            tmag2 = math.sqrt(float(npy.sum(npy.abs(t10[:3])**2)))
            tr, _ = test(tmag1, tmag2, 0.0, 0.0, dmin)
            if tr <= rx:
                total += t10
                nt += 2
                break

            g2 = sflds(state, z + dz*0.25)
            g4 = sflds(state, z + dz*0.75)
            t02 = (t01 + dzot*(g2 + g4))*0.5
            t11 = (4.0*t02 - t01)/3.0
            t20 = (16.0*t11 - t10)/15.0
            tmag1 = math.sqrt(float(npy.sum(npy.abs(t11[:3])**2)))
            tmag2 = math.sqrt(float(npy.sum(npy.abs(t20[:3])**2)))
            tr, _ = test(tmag1, tmag2, 0.0, 0.0, dmin)
            if tr <= rx or ns >= nm:
                total += t20
                nt += 1
                break

            nt = 0
            ns *= 2
            dz = s/float(ns)
            dzot = dz*0.5
            g5 = g3.copy()
            g3 = g2.copy()

        z += dz
        if z > zend:
            break
        g1 = g5.copy()
        if nt >= nts and ns > nx:
            ns //= 2
            nt = 1

    return total


def rom2Vec(
    state: NECState, a: float, b: float, dmin: npy.ndarray,
    xo: npy.ndarray, yo: npy.ndarray, zo: npy.ndarray
) -> npy.ndarray:
    """Vectorized Romberg integration dropping converged points early."""
    nm, nts, nx, rx = 65536, 4, 1, 1.0e-4
    nPts = len(xo)
    total = npy.zeros((9, nPts), dtype=npy.complex128)
    if nPts == 0:
        return total

    s = b - a
    if s < 0.0:
        raise ValueError(" ERROR - B LESS THAN A IN ROM2")
    ep = s/(1.0e4*nm)
    zend = b - ep

    # Track per-point integration state so converged points exit early
    active = npy.arange(nPts, dtype=npy.intp)
    z = npy.full(nPts, a, dtype=npy.float64)
    ns = npy.full(nPts, nx, dtype=npy.int32)
    nt = npy.zeros(nPts, dtype=npy.int32)
    g1 = sfldsVec(state, z, xo, yo, zo, 1)

    while len(active) > 0:
        dz = s/ns.astype(npy.float64)
        over = (z + dz > b)
        dz = npy.where(over, b - z, dz)
        keep = (dz > ep)
        if not npy.all(keep):
            active = active[keep]
            z = z[keep]
            ns = ns[keep]
            nt = nt[keep]
            dz = dz[keep]
            g1 = g1[:, keep]
            if len(active) == 0:
                break

        dzot = dz*0.5
        xoA, yoA, zoA = xo[active], yo[active], zo[active]
        dminA = dmin[active]
        g3 = sfldsVec(state, z + dzot, xoA, yoA, zoA, 1)
        g5 = sfldsVec(state, z + dz, xoA, yoA, zoA, 1)

        subIdx = npy.arange(len(active), dtype=npy.intp)
        while len(subIdx) > 0:
            dzS = dz[subIdx]
            dzotS = dzot[subIdx]
            g1S, g3S, g5S = g1[:, subIdx], g3[:, subIdx], g5[:, subIdx]

            t00 = (g1S + g5S)*dzotS[None, :]
            t01 = (t00 + dzS[None, :]*g3S)*0.5
            t10 = (4.0*t01 - t00)/3.0
            tmag1 = npy.sqrt(npy.sum(npy.abs(t01[:3])**2, axis=0))
            tmag2 = npy.sqrt(npy.sum(npy.abs(t10[:3])**2, axis=0))
            den = npy.maximum(npy.maximum(tmag2, dminA[subIdx]), 1.0e-37)
            c1 = (npy.abs(tmag1 - tmag2)/den <= rx)

            if npy.any(c1):
                done1 = subIdx[c1]
                total[:, active[done1]] += t10[:, c1]
                nt[done1] += 2
                subIdx = subIdx[~c1]
                if len(subIdx) == 0:
                    break

            dzS = dz[subIdx]
            dzotS = dzot[subIdx]
            t01S, t10S = t01[:, ~c1], t10[:, ~c1]
            g3S = g3[:, subIdx]
            actS = active[subIdx]
            g2S = sfldsVec(
                state, z[subIdx] + dzS*0.25, xo[actS], yo[actS], zo[actS], 1
            )
            g4S = sfldsVec(
                state, z[subIdx] + dzS*0.75, xo[actS], yo[actS], zo[actS], 1
            )
            t02 = (t01S + dzotS[None, :]*(g2S + g4S))*0.5
            t11 = (4.0*t02 - t01S)/3.0
            t20 = (16.0*t11 - t10S)/15.0
            tmag1 = npy.sqrt(npy.sum(npy.abs(t11[:3])**2, axis=0))
            tmag2 = npy.sqrt(npy.sum(npy.abs(t20[:3])**2, axis=0))
            den = npy.maximum(npy.maximum(tmag2, dminA[subIdx]), 1.0e-37)
            c2 = (npy.abs(tmag1 - tmag2)/den <= rx) | (ns[subIdx] >= nm)

            if npy.any(c2):
                done2 = subIdx[c2]
                total[:, active[done2]] += t20[:, c2]
                nt[done2] += 1

            rem = ~c2
            if not npy.any(rem):
                break
            remSub = subIdx[rem]
            nt[remSub] = 0
            ns[remSub] *= 2
            dz[remSub] = s/ns[remSub].astype(npy.float64)
            dzot[remSub] = dz[remSub]*0.5
            g5[:, remSub] = g3S[:, rem]
            g3[:, remSub] = g2S[:, rem]
            subIdx = remSub

        z += dz
        notEnd = (z <= zend)
        if not npy.all(notEnd):
            active = active[notEnd]
            z = z[notEnd]
            ns = ns[notEnd]
            nt = nt[notEnd]
            g5 = g5[:, notEnd]
            if len(active) == 0:
                break
        g1 = g5
        halfMask = (nt >= nts) & (ns > nx)
        ns = npy.where(halfMask, ns//2, ns)
        nt = npy.where(halfMask, 1, nt)

    return total


def sflds(state: NECState, t: float) -> npy.ndarray:
    """Field due to ground for current element on source segment."""
    pi, tp, pot = 3.141592654, 6.283185308, 1.570796327
    e = npy.zeros(9, dtype=npy.complex128)
    xt = state.xj + t*state.cabj
    yt = state.yj + t*state.sabj
    zt = state.zj + t*state.salpj
    rhx, rhy = state.xo - xt, state.yo - yt
    rhs = rhx*rhx + rhy*rhy
    rho = math.sqrt(rhs)
    if rho <= 0.0:
        rhx, rhy, phx, phy = 1.0, 0.0, 0.0, 1.0
    else:
        rhx, rhy = rhx/rho, rhy/rho
        phx, phy = -rhy, rhx

    cph = rhx*state.xsn + rhy*state.ysn
    sph = rhy*state.xsn - rhx*state.ysn
    if abs(cph) < 1.0e-10:
        cph = 0.0
    if abs(sph) < 1.0e-10:
        sph = 0.0

    state.zph = state.zo + zt
    zphs = state.zph*state.zph
    r2s = rhs + zphs
    state.r2 = math.sqrt(r2s)
    rk = state.r2*tp
    state.xx2 = complex(math.cos(rk), -math.sin(rk))

    if state.isnor != 1:
        state.zmh = 1.0
        state.r1 = 1.0
        state.xx1 = 0j
        erv, ezv, erh, ezh, eph = gwave(state)
        et = -4.77134j*state.frati*state.xx2/(r2s*state.r2)
        er = 2.0*et*(1.0 + 1j*rk)
        et = et*(1.0 - rk*rk + 1j*rk)
        hrv = (er + et)*rho*state.zph/r2s
        hzv = (zphs*er - rhs*et)/r2s
        hrh = (rhs*er - zphs*et)/r2s
        erv -= hrv
        ezv -= hzv
        erh += hrh
        ezh += hrv
        eph += et

        erv *= state.salpj
        ezv *= state.salpj
        erh = erv + erh*state.sn*cph
        ezh = ezh*state.sn*cph
        eph = eph*state.sn*sph
        e[0] = (erh*rhx + eph*phx)*state.s
        e[1] = (erh*rhy + eph*phy)*state.s
        e[2] = (ezv + ezh)*state.s
        sfac = math.sin(pi*state.s)/(pi*state.s)
        e[6], e[7], e[8] = e[0]*sfac, e[1]*sfac, e[2]*sfac
        return e

    thet = pot if rho < 1.0e-12 else math.atan(state.zph/rho)
    erv, ezv, erh, eph = intrp(state, state.r2, thet)
    xx2 = state.xx2/state.r2
    sfac = state.sn*cph
    erh = xx2*(state.salpj*erv + sfac*erh)
    ezh = xx2*(state.salpj*ezv - sfac*erv)
    eph = state.sn*sph*xx2*eph
    e[0] = erh*rhx + eph*phx
    e[1] = erh*rhy + eph*phy
    e[2] = ezh
    rk = tp*t
    sfacS, sfacC = math.sin(rk), math.cos(rk)
    e[3], e[4], e[5] = e[0]*sfacS, e[1]*sfacS, e[2]*sfacS
    e[6], e[7], e[8] = e[0]*sfacC, e[1]*sfacC, e[2]*sfacC
    return e

def sfldsVec(
    state: NECState, t: npy.ndarray, xo: npy.ndarray, yo: npy.ndarray,
    zo: npy.ndarray, isnor: int
) -> npy.ndarray:
    """Vectorized ground field for current elements at offsets t."""
    pi, tp, pot = 3.141592654, 6.283185308, 1.570796327
    nPts = len(xo)
    e = npy.zeros((9, nPts), dtype=npy.complex128)
    if nPts == 0:
        return e

    xt = state.xj + t*state.cabj
    yt = state.yj + t*state.sabj
    zt = state.zj + t*state.salpj
    rhx, rhy = xo - xt, yo - yt
    rhs = rhx*rhx + rhy*rhy
    rho = npy.sqrt(rhs)

    mPos = (rho > 0.0)
    safeRho = npy.where(mPos, rho, 1.0)
    rhx = npy.where(mPos, rhx/safeRho, 1.0)
    rhy = npy.where(mPos, rhy/safeRho, 0.0)
    phx = npy.where(mPos, -rhy, 0.0)
    phy = npy.where(mPos, rhx, 1.0)

    cph = rhx*state.xsn + rhy*state.ysn
    sph = rhy*state.xsn - rhx*state.ysn
    cph = npy.where(npy.abs(cph) < 1.0e-10, 0.0, cph)
    sph = npy.where(npy.abs(sph) < 1.0e-10, 0.0, sph)

    zph = zo + zt
    zphs = zph*zph
    r2s = rhs + zphs
    r2 = npy.sqrt(r2s)
    rk = r2*tp
    xx2 = npy.exp(-1j*rk)

    if isnor != 1:
        erv = npy.zeros(nPts, dtype=npy.complex128)
        ezv = npy.zeros(nPts, dtype=npy.complex128)
        erh = npy.zeros(nPts, dtype=npy.complex128)
        ezh = npy.zeros(nPts, dtype=npy.complex128)
        eph = npy.zeros(nPts, dtype=npy.complex128)
        for idx in range(nPts):
            state.zmh, state.r1, state.xx1 = 1.0, 1.0, 0j
            state.r2, state.zph, state.xx2 = r2[idx], zph[idx], xx2[idx]
            erv[idx], ezv[idx], erh[idx], ezh[idx], eph[idx] = gwave(state)

        et = -4.77134j*state.frati*xx2/(r2s*r2)
        er = 2.0*et*(1.0 + 1j*rk)
        et = et*(1.0 - rk*rk + 1j*rk)
        hrv = (er + et)*rho*zph/r2s
        hzv = (zphs*er - rhs*et)/r2s
        hrh = (rhs*er - zphs*et)/r2s
        erv -= hrv
        ezv -= hzv
        erh += hrh
        ezh += hrv
        eph += et

        erv *= state.salpj
        ezv *= state.salpj
        erh = erv + erh*state.sn*cph
        ezh = ezh*state.sn*cph
        eph = eph*state.sn*sph
        e0 = (erh*rhx + eph*phx)*state.s
        e1 = (erh*rhy + eph*phy)*state.s
        e2 = (ezv + ezh)*state.s
        sfac = math.sin(pi*state.s)/(pi*state.s)
        e[0], e[1], e[2] = e0, e1, e2
        e[6], e[7], e[8] = e0*sfac, e1*sfac, e2*sfac
        return e

    thet = npy.where(rho < 1.0e-12, pot, npy.arctan(zph/npy.where(rho < 1.0e-12, 1.0, rho)))
    erv, ezv, erh, eph = intrpVec(state, r2, thet)
    xx2R = xx2/r2
    sfac = state.sn*cph
    erh = xx2R*(state.salpj*erv + sfac*erh)
    ezh = xx2R*(state.salpj*ezv - sfac*erv)
    eph = state.sn*sph*xx2R*eph
    e0 = erh*rhx + eph*phx
    e1 = erh*rhy + eph*phy
    e2 = ezh
    rkT = tp*t
    sfacS, sfacC = npy.sin(rkT), npy.cos(rkT)
    e[0], e[1], e[2] = e0, e1, e2
    e[3], e[4], e[5] = e0*sfacS, e1*sfacS, e2*sfacS
    e[6], e[7], e[8] = e0*sfacC, e1*sfacC, e2*sfacC
    return e


