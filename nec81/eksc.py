import math
import numpy as npy
from .nec_state import NECState
from .main2 import gwave
from .intrp import intrp


def gx(zz: float, rh: float, xk: float):
    """Segment end contributions for thin wire approximation. Returns (gz, gzp)."""
    r2 = zz * zz + rh * rh
    r = math.sqrt(r2)
    rk = xk * r
    gz = complex(math.cos(rk), -math.sin(rk)) / r
    gzp = -complex(1.0, rk) * gz / r2
    return gz, gzp


def gxx(zz: float, rh: float, a: float, a2: float, xk: float, ira: int):
    """Segment end contributions for extended thin wire approximation."""
    r2 = zz * zz + rh * rh
    r = math.sqrt(r2)
    r4 = r2 * r2
    rk = xk * r
    rk2 = rk * rk
    rh2 = rh * rh
    t1 = 0.25 * a2 * rh2 / r4
    t2 = 0.5 * a2 / r2
    c1 = complex(1.0, rk)
    c2 = 3.0 * c1 - rk2
    c3 = complex(6.0, rk) * rk2 - 15.0 * c1
    gz = complex(math.cos(rk), -math.sin(rk)) / r
    g2 = gz * (1.0 + t1 * c2)
    g1 = g2 - t2 * c1 * gz
    gz = gz / r2
    g2p = gz * (t1 * c3 - c1)
    gzp = t2 * c2 * gz
    g3 = g2p + gzp
    g1p = g3 * zz
    if ira != 1:
        g3 = (g3 + gzp) * rh
        gzp = -zz * c1 * gz
        if rh <= 1.0e-10:
            return g1, g1p, 0j, 0j, g3, gzp
        return g1, g1p, g2 / rh, g2p * zz / rh, g3, gzp

    t2 = 0.5 * a
    g2 = -t2 * c1 * gz
    g2p = t2 * gz * c2 / r2
    g3 = rh2 * g2p - a * gz * c1
    g2p = g2p * zz
    gzp = -zz * c1 * gz
    return g1, g1p, g2, g2p, g3, gzp


def eksc(state: NECState, s: float, z: float, rh: float, xk: float, ij: int):
    """Computes E field of sine, cosine, and constant current filaments (thin wire)."""
    con = complex(0.0, 4.771341189)
    state.ijx = ij
    state.zpk = xk * z
    rhk = xk * rh
    state.rkb2 = rhk * rhk
    sh = 0.5 * s
    shk = xk * sh
    ss = math.sin(shk)
    cs = math.cos(shk)
    z2 = sh - z
    z1 = -(sh + z)

    gz1, gp1 = gx(z1, rh, xk)
    gz2, gp2 = gx(z2, rh, xk)

    gzp1 = gp1 * z1
    gzp2 = gp2 * z2
    ezs = con * ((gz2 - gz1) * cs * xk - (gzp2 + gzp1) * ss)
    ezc = -con * ((gz2 + gz1) * ss * xk + (gzp2 - gzp1) * cs)
    erk = con * (gp2 - gp1) * rh
    cint, sint = intx(state, -shk, shk, rhk, ij)
    ezk = -con * (gzp2 - gzp1 + xk * xk * complex(cint, -sint))
    gzp1 = gzp1 * z1
    gzp2 = gzp2 * z2
    if rh < 1.0e-10:
        return ezs, 0j, ezc, 0j, ezk, erk
    ers = -con * ((gzp2 + gzp1 + gz2 + gz1) * ss - (z2 * gz2 - z1 * gz1) * cs * xk) / rh
    erc = -con * ((gzp2 - gzp1 + gz2 - gz1) * cs + (z2 * gz2 + z1 * gz1) * ss * xk) / rh
    return ezs, ers, ezc, erc, ezk, erk


def ekscx(state: NECState, bx: float, s: float, z: float, rhx: float, xk: float, ij: int, inx1: int, inx2: int):
    """Computes E field of sine, cosine, and constant current filaments (extended thin wire)."""
    con = complex(0.0, 4.771341189)
    if rhx >= bx:
        rh, b, ira = rhx, bx, 0
    else:
        rh, b, ira = bx, rhx, 1

    sh = 0.5 * s
    state.ijx = ij
    state.zpk = xk * z
    rhk = xk * rh
    state.rkb2 = rhk * rhk
    shk = xk * sh
    ss = math.sin(shk)
    cs = math.cos(shk)
    z2 = sh - z
    z1 = -(sh + z)
    a2 = b * b

    if inx1 != 2:
        gz1, gzp1, gr1, grp1, grk1, gzz1 = gxx(z1, rh, b, a2, xk, ira)
    else:
        gz1, grk1 = gx(z1, rhx, xk)
        gzp1 = grk1 * z1
        gr1, grp1, grk1, gzz1 = gz1 / rhx, gzp1 / rhx, grk1 * rhx, 0j

    if inx2 != 2:
        gz2, gzp2, gr2, grp2, grk2, gzz2 = gxx(z2, rh, b, a2, xk, ira)
    else:
        gz2, grk2 = gx(z2, rhx, xk)
        gzp2 = grk2 * z2
        gr2, grp2, grk2, gzz2 = gz2 / rhx, gzp2 / rhx, grk2 * rhx, 0j

    ezs = con * ((gz2 - gz1) * cs * xk - (gzp2 + gzp1) * ss)
    ezc = -con * ((gz2 + gz1) * ss * xk + (gzp2 - gzp1) * cs)
    ers = -con * ((z2 * grp2 + z1 * grp1 + gr2 + gr1) * ss - (z2 * gr2 - z1 * gr1) * cs * xk)
    erc = -con * ((z2 * grp2 - z1 * grp1 + gr2 - gr1) * cs + (z2 * gr2 + z1 * gr1) * ss * xk)
    erk = con * (grk2 - grk1)
    cint, sint = intx(state, -shk, shk, rhk, ij)
    bk = b * xk
    bk2 = bk * bk * 0.25
    ezk = -con * (gzp2 - gzp1 + xk * xk * (1.0 - bk2) * complex(cint, -sint) - bk2 * (gzz2 - gzz1))
    return ezs, ers, ezc, erc, ezk, erk

def test(f1r: float, f2r: float, f1i: float, f2i: float, dmin: float):
    """Convergence test for Romberg numerical integration. Returns (tr, ti)."""
    den = max(abs(f2r), abs(f2i), dmin)
    if den < 1.0e-37:
        return 0.0, 0.0
    return abs((f1r - f2r) / den), abs((f1i - f2i) / den)


def gf(state: NECState, zk: float):
    """Integrand exp(jkr)/(kr) for numerical integration. Returns (co, si)."""
    zdk = zk - state.zpk
    rk = math.sqrt(state.rkb2 + zdk * zdk)
    si = math.sin(rk) / rk
    if state.ijx != 0:
        return math.cos(rk) / rk, si
    if rk >= 0.2:
        return (math.cos(rk) - 1.0) / rk, si
    rks = rk * rk
    return ((-1.38888889e-3 * rks + 4.16666667e-2) * rks - 0.5) * rk, si


def intx(state: NECState, el1: float, el2: float, b: float, ij: int):
    """Variable-interval-width Romberg integration of exp(jkr)/r. Returns (sgr, sgi)."""
    nx, nm, nts, rx = 1, 65536, 4, 1.0e-4
    z = el1
    ze = 0.0 if ij == 0 else el2
    s = ze - z
    ep = s / (10.0 * nm)
    zend = ze - ep
    sgr = sgi = 0.0
    ns, nt = nx, 0
    g1r, g1i = gf(state, z)

    while True:
        dz = s / float(ns)
        if z + dz > ze:
            dz = ze - z
            if abs(dz) <= ep:
                break
        dzot = dz * 0.5
        g3r, g3i = gf(state, z + dzot)
        g5r, g5i = gf(state, z + dz)

        while True:
            t00r, t00i = (g1r + g5r) * dzot, (g1i + g5i) * dzot
            t01r, t01i = (t00r + dz * g3r) * 0.5, (t00i + dz * g3i) * 0.5
            t10r, t10i = (4.0 * t01r - t00r) / 3.0, (4.0 * t01i - t00i) / 3.0
            te1r, te1i = test(t01r, t10r, t01i, t10i, 0.0)
            if te1i <= rx and te1r <= rx:
                sgr += t10r
                sgi += t10i
                nt += 2
                break

            g2r, g2i = gf(state, z + dz * 0.25)
            g4r, g4i = gf(state, z + dz * 0.75)
            t02r, t02i = (t01r + dzot * (g2r + g4r)) * 0.5, (t01i + dzot * (g2i + g4i)) * 0.5
            t11r, t11i = (4.0 * t02r - t01r) / 3.0, (4.0 * t02i - t01i) / 3.0
            t20r, t20i = (16.0 * t11r - t10r) / 15.0, (16.0 * t11i - t10i) / 15.0
            te2r, te2i = test(t11r, t20r, t11i, t20i, 0.0)
            if (te2i <= rx and te2r <= rx) or ns >= nm:
                sgr += t20r
                sgi += t20i
                nt += 1
                break

            nt = 0
            ns *= 2
            dz = s / float(ns)
            dzot = dz * 0.5
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
        sgr = 2.0 * (sgr + math.log((math.sqrt(b * b + s * s) + s) / b))
        sgi = 2.0 * sgi
    return sgr, sgi


def rom2(state: NECState, a: float, b: float, dmin: float) -> npy.ndarray:
    """Romberg integration over source segment for Sommerfeld/Norton ground field."""
    nm, nts, nx, rx = 65536, 4, 1, 1.0e-4
    s = b - a
    if s < 0.0:
        raise ValueError(" ERROR - B LESS THAN A IN ROM2")
    ep = s / (1.0e4 * nm)
    zend = b - ep
    total = npy.zeros(9, dtype=npy.complex128)
    z = a
    ns, nt = nx, 0
    g1 = sflds(state, z)

    while True:
        dz = s / float(ns)
        if z + dz > b:
            dz = b - z
            if dz <= ep:
                break
        dzot = dz * 0.5
        g3 = sflds(state, z + dzot)
        g5 = sflds(state, z + dz)

        while True:
            t00 = (g1 + g5) * dzot
            t01 = (t00 + dz * g3) * 0.5
            t10 = (4.0 * t01 - t00) / 3.0
            tmag1 = math.sqrt(float(npy.sum(npy.abs(t01[:3]) ** 2)))
            tmag2 = math.sqrt(float(npy.sum(npy.abs(t10[:3]) ** 2)))
            tr, _ = test(tmag1, tmag2, 0.0, 0.0, dmin)
            if tr <= rx:
                total += t10
                nt += 2
                break

            g2 = sflds(state, z + dz * 0.25)
            g4 = sflds(state, z + dz * 0.75)
            t02 = (t01 + dzot * (g2 + g4)) * 0.5
            t11 = (4.0 * t02 - t01) / 3.0
            t20 = (16.0 * t11 - t10) / 15.0
            tmag1 = math.sqrt(float(npy.sum(npy.abs(t11[:3]) ** 2)))
            tmag2 = math.sqrt(float(npy.sum(npy.abs(t20[:3]) ** 2)))
            tr, _ = test(tmag1, tmag2, 0.0, 0.0, dmin)
            if tr <= rx or ns >= nm:
                total += t20
                nt += 1
                break

            nt = 0
            ns *= 2
            dz = s / float(ns)
            dzot = dz * 0.5
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


def sflds(state: NECState, t: float) -> npy.ndarray:
    """Field due to ground for current element on source segment at offset T."""
    pi, tp, pot = 3.141592654, 6.283185308, 1.570796327
    e = npy.zeros(9, dtype=npy.complex128)
    xt = state.xj + t * state.cabj
    yt = state.yj + t * state.sabj
    zt = state.zj + t * state.salpj
    rhx, rhy = state.xo - xt, state.yo - yt
    rhs = rhx * rhx + rhy * rhy
    rho = math.sqrt(rhs)
    if rho <= 0.0:
        rhx, rhy, phx, phy = 1.0, 0.0, 0.0, 1.0
    else:
        rhx, rhy = rhx / rho, rhy / rho
        phx, phy = -rhy, rhx

    cph = rhx * state.xsn + rhy * state.ysn
    sph = rhy * state.xsn - rhx * state.ysn
    if abs(cph) < 1.0e-10:
        cph = 0.0
    if abs(sph) < 1.0e-10:
        sph = 0.0

    state.zph = state.zo + zt
    zphs = state.zph * state.zph
    r2s = rhs + zphs
    state.r2 = math.sqrt(r2s)
    rk = state.r2 * tp
    state.xx2 = complex(math.cos(rk), -math.sin(rk))

    if state.isnor != 1:
        state.zmh = 1.0
        state.r1 = 1.0
        state.xx1 = 0j
        erv, ezv, erh, ezh, eph = gwave(state)
        et = -complex(0.0, 4.77134) * state.frati * state.xx2 / (r2s * state.r2)
        er = 2.0 * et * complex(1.0, rk)
        et = et * complex(1.0 - rk * rk, rk)
        hrv = (er + et) * rho * state.zph / r2s
        hzv = (zphs * er - rhs * et) / r2s
        hrh = (rhs * er - zphs * et) / r2s
        erv -= hrv
        ezv -= hzv
        erh += hrh
        ezh += hrv
        eph += et

        erv *= state.salpj
        ezv *= state.salpj
        erh = erv + erh * state.sn * cph
        ezh = ezh * state.sn * cph
        eph = eph * state.sn * sph
        e[0] = (erh * rhx + eph * phx) * state.s
        e[1] = (erh * rhy + eph * phy) * state.s
        e[2] = (ezv + ezh) * state.s
        sfac = math.sin(pi * state.s) / (pi * state.s)
        e[6], e[7], e[8] = e[0] * sfac, e[1] * sfac, e[2] * sfac
        return e

    thet = pot if rho < 1.0e-12 else math.atan(state.zph / rho)
    erv, ezv, erh, eph = intrp(state, state.r2, thet)
    xx2 = state.xx2 / state.r2
    sfac = state.sn * cph
    erh = xx2 * (state.salpj * erv + sfac * erh)
    ezh = xx2 * (state.salpj * ezv - sfac * erv)
    eph = state.sn * sph * xx2 * eph
    e[0] = erh * rhx + eph * phx
    e[1] = erh * rhy + eph * phy
    e[2] = ezh
    rk = tp * t
    sfac_s, sfac_c = math.sin(rk), math.cos(rk)
    e[3], e[4], e[5] = e[0] * sfac_s, e[1] * sfac_s, e[2] * sfac_s
    e[6], e[7], e[8] = e[0] * sfac_c, e[1] * sfac_c, e[2] * sfac_c
    return e

