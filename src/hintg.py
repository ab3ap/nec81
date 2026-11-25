import math
import cmath
import numpy as npy
from .necState import NECState
from .eksc import test


def hintg(state: NECState, xi: float, yi: float, zi: float):
    """Computes H field of a surface patch current."""
    fpi, tp = 12.56637062, 6.283185308
    t1xj, t1yj, t1zj = state.cabj, state.sabj, state.salpj
    t2xj, t2yj, t2zj = state.b, state.ind1, state.ind2

    rx, ry = xi - state.xj, yi - state.yj
    rfl = -1.0
    state.exk = state.eyk = state.ezk = 0j
    state.exs = state.eys = state.ezs = 0j

    for ip in range(1, state.ksymp + 1):
        rfl = -rfl
        rz = zi - state.zj * rfl
        rsq = rx * rx + ry * ry + rz * rz
        if rsq < 1.0e-20:
            continue
        r = math.sqrt(rsq)
        rk = tp * r
        cr, sr = math.cos(rk), math.sin(rk)
        gam = -(complex(cr, -sr) + rk * complex(sr, cr)) / (fpi * rsq * r) * state.s
        exc, eyc, ezc = gam * rx, gam * ry, gam * rz
        t1zr, t2zr = t1zj * rfl, t2zj * rfl
        f1x = eyc * t1zr - ezc * t1yj
        f1y = ezc * t1xj - exc * t1zr
        f1z = exc * t1yj - eyc * t1xj
        f2x = eyc * t2zr - ezc * t2yj
        f2y = ezc * t2xj - exc * t2zr
        f2z = exc * t2yj - eyc * t2xj

        if ip != 1:
            if state.iperf == 1:
                f1x, f1y, f1z = -f1x, -f1y, -f1z
                f2x, f2y, f2z = -f2x, -f2y, -f2z
            else:
                xymag = math.sqrt(rx * rx + ry * ry)
                if xymag <= 1.0e-6:
                    px, py, cth, rrv = 0.0, 0.0, 1.0, 1.0 + 0j
                else:
                    px, py = -ry / xymag, rx / xymag
                    cth = rz / r
                    rrv = cmath.sqrt(1.0 - state.zrati * state.zrati * (1.0 - cth * cth))
                rrh = state.zrati * cth
                rrh = (rrh - rrv) / (rrh + rrv)
                rrv = state.zrati * rrv
                rrv = -(cth - rrv) / (cth + rrv)
                gam = (f1x * px + f1y * py) * (rrv - rrh)
                f1x, f1y, f1z = f1x * rrh + gam * px, f1y * rrh + gam * py, f1z * rrh
                gam = (f2x * px + f2y * py) * (rrv - rrh)
                f2x, f2y, f2z = f2x * rrh + gam * px, f2y * rrh + gam * py, f2z * rrh

        state.exk += f1x
        state.eyk += f1y
        state.ezk += f1z
        state.exs += f2x
        state.eys += f2y
        state.ezs += f2z


def trio(state: NECState, si: npy.ndarray, bi: npy.ndarray, icon1: npy.ndarray, icon2: npy.ndarray, j: int, ld: int):
    """Computes components of all basis functions overlapping segment J (1-based)."""
    jmax = 30
    state.jsno = 0
    j_idx = j - 1

    def walk(start_jcox: int, init_jend: int):
        jcox = start_jcox
        jend = init_jend
        if jcox == 0 or jcox > 10000:
            return
        if jcox < 0:
            jcox = -jcox
        else:
            jend = -jend
        while jcox != j:
            state.jsno += 1
            if state.jsno >= jmax:
                raise RuntimeError(f" TRIO - SEGMENT CONNENTION ERROR FOR SEGMENT{j:5d}")
            idx = state.jsno - 1
            a, b, c = sbf(si, bi, icon1, icon2, jcox, j, ld)
            state.ax[idx], state.bx[idx], state.cx[idx] = a, b, c
            state.jco[idx] = jcox
            jcox_idx = jcox - 1
            jcox = int(icon2[jcox_idx]) if jend == 1 else int(icon1[jcox_idx])
            if jcox == 0:
                raise RuntimeError(f" TRIO - SEGMENT CONNENTION ERROR FOR SEGMENT{j:5d}")
            if jcox < 0:
                jcox = -jcox
            else:
                jend = -jend

    walk(int(icon1[j_idx]), -1)
    walk(int(icon2[j_idx]), 1)

    state.jsno += 1
    idx = state.jsno - 1
    a, b, c = sbf(si, bi, icon1, icon2, j, j, ld)
    state.ax[idx], state.bx[idx], state.cx[idx] = a, b, c
    state.jco[idx] = j
def sbf(si: npy.ndarray, bi: npy.ndarray, icon1: npy.ndarray, icon2: npy.ndarray, i_seg: int, is_seg: int, ld: int):
    """Computes component of basis function I on segment IS. Returns (aa, bb, cc)."""
    pi, jmax = 3.141592654, 30
    aa = bb = cc = 0.0
    june = jsno = 0
    pp = 0.0

    def trace(start_jcox: int, init_jend: int, iend_val: int):
        nonlocal aa, bb, cc, june, jsno, pp
        jcox, jend, sig = start_jcox, init_jend, -1.0
        if jcox == 0:
            return
        if jcox < 0:
            jcox = -jcox
        else:
            sig, jend = -sig, -jend

        while True:
            jsno += 1
            if jsno >= jmax:
                raise RuntimeError(f" SBF - SEGMENT CONNECTION ERROR FOR SEGMENT{i_seg:5d}")
            j_idx = jcox - 1
            d = pi * si[j_idx]
            sdh, cdh = math.sin(d), math.cos(d)
            sd = 2.0 * sdh * cdh
            omc = ((1.3888889e-3 * (4.0 * d * d) - 4.1666666667e-2) * (4.0 * d * d) + 0.5) * (4.0 * d * d) if d <= 0.015 else (1.0 - cdh * cdh + sdh * sdh)
            aj = 1.0 / (math.log(1.0 / (pi * bi[j_idx])) - 0.577215664)
            pp -= omc / sd * aj
            if jcox == is_seg:
                aa, bb, cc, june = aj / sd * sig, aj / (2.0 * cdh), -aj / (2.0 * sdh) * sig, iend_val
            if jcox == i_seg:
                if jcox == is_seg:
                    bb = -bb
                return
            jcox = int(icon2[j_idx]) if jend == 1 else int(icon1[j_idx])
            if abs(jcox) == i_seg:
                return
            if jcox == 0:
                raise RuntimeError(f" SBF - SEGMENT CONNECTION ERROR FOR SEGMENT{i_seg:5d}")
            if jcox < 0:
                jcox = -jcox
            else:
                sig, jend = -sig, -jend

    i_idx = i_seg - 1
    jc1 = int(icon1[i_idx])
    trace(i_seg if jc1 > 10000 else jc1, -1, -1)
    pm, pp, njun1 = -pp, 0.0, jsno
    jc2 = int(icon2[i_idx])
    trace(i_seg if jc2 > 10000 else jc2, 1, 1)
    njun2 = jsno - njun1

    d = pi * si[i_idx]
    sdh, cdh = math.sin(d), math.cos(d)
    sd, cd = 2.0 * sdh * cdh, cdh * cdh - sdh * sdh
    omc = ((1.3888889e-3 * (4.0 * d * d) - 4.1666666667e-2) * (4.0 * d * d) + 0.5) * (4.0 * d * d) if d <= 0.015 else (1.0 - cd)
    ap = aj = 1.0 / (math.log(1.0 / (pi * bi[i_idx])) - 0.577215664)

    if njun1 != 0 and njun2 != 0:
        qp = sd * (pm * pp + aj * ap) + cd * (pm * ap - pp * aj)
        qm, qp = (ap * omc - pp * sd) / qp, -(aj * omc + pm * sd) / qp
        if june < 0:
            aa, bb, cc = aa * qm, bb * qm, cc * qm
        elif june > 0:
            aa, bb, cc = -aa * qp, bb * qp, -cc * qp
        if i_seg == is_seg:
            aa -= 1.0
            bb += (aj * qm + ap * qp) * sdh / sd
            cc += (aj * qm - ap * qp) * cdh / sd
    elif njun1 == 0 and njun2 != 0:
        qp = pi * bi[i_idx]
        xxi = qp * (1.0 - 0.5 * qp * qp) / (1.0 - qp * qp)
        qp = -(omc + xxi * sd) / (sd * (ap + xxi * pp) + cd * (xxi * ap - pp))
        if june == 1:
            aa, bb, cc = -aa * qp, bb * qp, -cc * qp
        if june != 1 or i_seg == is_seg:
            aa -= 1.0
            d_den = cd - xxi * sd
            bb += (sdh + ap * qp * (cdh - xxi * sdh)) / d_den
            cc += (cdh + ap * qp * (sdh + xxi * cdh)) / d_den
    elif njun1 != 0 and njun2 == 0:
        qm = pi * bi[i_idx]
        xxi = qm * (1.0 - 0.5 * qm * qm) / (1.0 - qm * qm)
        qm = (omc + xxi * sd) / (sd * (aj - xxi * pm) + cd * (pm + xxi * aj))
        if june == -1:
            aa, bb, cc = aa * qm, bb * qm, cc * qm
        if june != -1 or i_seg == is_seg:
            aa -= 1.0
            d_den = cd - xxi * sd
            bb += (aj * qm * (cdh - xxi * sdh) - sdh) / d_den
            cc += (cdh - aj * qm * (sdh + xxi * cdh)) / d_den
    else:
        aa = -1.0
        qp = pi * bi[i_idx]
        xxi = qp * (1.0 - 0.5 * qp * qp) / (1.0 - qp * qp)
        cc = 1.0 / (cdh - xxi * sdh)
    return aa, bb, cc


def unere(state: NECState, xob: float, yob: float, zob: float):
    """Calculates E field due to unit current in T1 and T2 directions on a patch."""
    tpi, const = 6.283185308, 4.771341188
    t1xj, t1yj, t1zj = state.cabj, state.sabj, state.salpj
    t2xj, t2yj, t2zj = state.b, state.ind1, state.ind2
    zr, t1zr, t2zr = (-state.zj, -t1zj, -t2zj) if state.ipgnd == 2 else (state.zj, t1zj, t2zj)

    rx, ry, rz = xob - state.xj, yob - state.yj, zob - zr
    r2 = rx * rx + ry * ry + rz * rz
    if r2 <= 1.0e-20:
        state.exk = state.eyk = state.ezk = state.exs = state.eys = state.ezs = 0j
        return

    r = math.sqrt(r2)
    tt1 = -tpi * r
    rt = r2 * r
    er = complex(math.sin(tt1), -math.cos(tt1)) * (const * state.s)
    q1 = complex(tt1 * tt1 - 1.0, tt1) * er / rt
    q2 = complex(3.0 - tt1 * tt1, -3.0 * tt1) * er / (rt * r2)

    er = q2 * (t1xj * rx + t1yj * ry + t1zr * rz)
    state.exk, state.eyk, state.ezk = q1 * t1xj + er * rx, q1 * t1yj + er * ry, q1 * t1zr + er * rz
    er = q2 * (t2xj * rx + t2yj * ry + t2zr * rz)
    state.exs, state.eys, state.ezs = q1 * t2xj + er * rx, q1 * t2yj + er * ry, q1 * t2zr + er * rz

    if state.ipgnd == 1:
        return
    if state.iperf == 1:
        state.exk, state.eyk, state.ezk = -state.exk, -state.eyk, -state.ezk
        state.exs, state.eys, state.ezs = -state.exs, -state.eys, -state.ezs
        return

    xymag = math.sqrt(rx * rx + ry * ry)
    if xymag <= 1.0e-6:
        px, py, cth, rrv = 0.0, 0.0, 1.0, 1.0 + 0j
    else:
        px, py = -ry / xymag, rx / xymag
        cth = rz / math.sqrt(xymag * xymag + rz * rz)
        rrv = cmath.sqrt(1.0 - state.zrati * state.zrati * (1.0 - cth * cth))
    rrh = (state.zrati * cth - rrv) / (state.zrati * cth + rrv)
    rrv = -(cth - state.zrati * rrv) / (cth + state.zrati * rrv)
    edp = (state.exk * px + state.eyk * py) * (rrh - rrv)
    state.exk, state.eyk, state.ezk = state.exk * rrv + edp * px, state.eyk * rrv + edp * py, state.ezk * rrv
    edp = (state.exs * px + state.eys * py) * (rrh - rrv)
    state.exs, state.eys, state.ezs = state.exs * rrv + edp * px, state.eys * rrv + edp * py, state.ezs * rrv

def pcint(
    state: NECState, xi: float, yi: float, zi: float,
    cabi: float, sabi: float, salpi: float
) -> npy.ndarray:
    """Vectorized 10x10 integration over surface patch at wire junction."""
    tpi, nint, const = 6.283185308, 10, 4.771341188
    t1xj, t1yj, t1zj = state.cabj, state.sabj, state.salpj
    t2xj, t2yj, t2zj = state.b, state.ind1, state.ind2
    d = math.sqrt(state.s)*0.5
    ds = 4.0*d/float(nint)
    gcon, fcon = 1.0/state.s, 1.0/(2.0*tpi*d)
    patchArea = ds*ds

    # 10x10 quadrature grid offsets s1, s2 across [-d + ds/2 .. d - ds/2]
    steps = npy.arange(1, nint + 1, dtype=npy.float64)
    sVals = (d + ds*0.5) - steps*ds
    s1G, s2G = npy.meshgrid(sVals, sVals, indexing="ij")
    s1 = s1G.ravel()
    s2 = s2G.ravel()

    xj = state.xj + s1*t1xj + s2*t2xj
    yj = state.yj + s1*t1yj + s2*t2yj
    zj = state.zj + s1*t1zj + s2*t2zj

    if state.ipgnd == 2:
        zr, t1zr, t2zr = -zj, -t1zj, -t2zj
    else:
        zr, t1zr, t2zr = zj, t1zj, t2zj

    rx, ry, rz = xi - xj, yi - yj, zi - zr
    r2 = rx*rx + ry*ry + rz*rz
    mValid = (r2 > 1.0e-20)
    safeR2 = npy.where(mValid, r2, 1.0)
    r = npy.sqrt(safeR2)
    tt1 = -tpi*r
    rt = safeR2*r

    er = npy.exp(1j*(tt1 - 1.570796327))*(const*patchArea)
    q1 = (tt1*tt1 - 1.0 + 1j*tt1)*er/rt
    q2 = (3.0 - tt1*tt1 - 3.0j*tt1)*er/(rt*safeR2)

    dot1 = q2*(t1xj*rx + t1yj*ry + t1zr*rz)
    exkV = npy.where(mValid, q1*t1xj + dot1*rx, 0j)
    eykV = npy.where(mValid, q1*t1yj + dot1*ry, 0j)
    ezkV = npy.where(mValid, q1*t1zr + dot1*rz, 0j)

    dot2 = q2*(t2xj*rx + t2yj*ry + t2zr*rz)
    exsV = npy.where(mValid, q1*t2xj + dot2*rx, 0j)
    eysV = npy.where(mValid, q1*t2yj + dot2*ry, 0j)
    ezsV = npy.where(mValid, q1*t2zr + dot2*rz, 0j)

    if state.ipgnd != 1:
        if state.iperf == 1:
            exkV, eykV, ezkV = -exkV, -eykV, -ezkV
            exsV, eysV, ezsV = -exsV, -eysV, -ezsV
        else:
            xymag = npy.sqrt(rx*rx + ry*ry)
            mXy = (xymag > 1.0e-6)
            safeXy = npy.where(mXy, xymag, 1.0)
            px = npy.where(mXy, -ry/safeXy, 0.0)
            py = npy.where(mXy, rx/safeXy, 0.0)
            cth = npy.where(mXy, rz/npy.sqrt(xymag*xymag + rz*rz), 1.0)
            rrv = npy.where(
                mXy,
                npy.sqrt(1.0 - state.zrati*state.zrati*(1.0 - cth*cth)),
                1.0 + 0j,
            )
            rrh = (state.zrati*cth - rrv)/(state.zrati*cth + rrv)
            rrv = -(cth - state.zrati*rrv)/(cth + state.zrati*rrv)
            edp1 = (exkV*px + eykV*py)*(rrh - rrv)
            exkV, eykV, ezkV = exkV*rrv + edp1*px, eykV*rrv + edp1*py, ezkV*rrv
            edp2 = (exsV*px + eysV*py)*(rrh - rrv)
            exsV, eysV, ezsV = exsV*rrv + edp2*px, eysV*rrv + edp2*py, ezsV*rrv

    exk = exkV*cabi + eykV*sabi + ezkV*salpi
    exs = exsV*cabi + eysV*sabi + ezsV*salpi

    g1 = (d + s1)*(d + s2)*gcon
    g2 = (d - s1)*(d + s2)*gcon
    g3 = (d - s1)*(d - s2)*gcon
    g4 = (d + s1)*(d - s2)*gcon
    fDen = (s1*s1 + s2*s2)*tpi
    f1 = s1/fDen - (g1 - g2 - g3 + g4)*fcon
    f2 = s2/fDen - (g1 + g2 - g3 - g4)*fcon

    e = npy.zeros(9, dtype=npy.complex128)
    e[0] = npy.sum(exk*g1)
    e[1] = npy.sum(exk*g2)
    e[2] = npy.sum(exk*g3)
    e[3] = npy.sum(exk*g4)
    e[4] = npy.sum(exs*g1)
    e[5] = npy.sum(exs*g2)
    e[6] = npy.sum(exs*g3)
    e[7] = npy.sum(exs*g4)
    e[8] = npy.sum(exk*f1 + exs*f2)
    return e


def hsfld(state: NECState, xi: float, yi: float, zi: float, ai: float):
    """Computes H field for constant, sine, and cosine current on a segment including ground."""
    eta = 376.73
    xij, yij = xi - state.xj, yi - state.yj
    rfl = -1.0
    for ip in range(1, state.ksymp + 1):
        rfl = -rfl
        salpr = state.salpj * rfl
        zij = zi - rfl * state.zj
        zp = xij * state.cabj + yij * state.sabj + zij * salpr
        rhox, rhoy, rhoz = xij - state.cabj * zp, yij - state.sabj * zp, zij - salpr * zp
        rh = math.sqrt(rhox * rhox + rhoy * rhoy + rhoz * rhoz + ai * ai)
        if rh <= 1.0e-10:
            state.exk = state.eyk = state.ezk = state.exs = state.eys = state.ezs = state.exc = state.eyc = state.ezc = 0j
            continue
        rhox, rhoy, rhoz = rhox / rh, rhoy / rh, rhoz / rh
        phx = state.sabj * rhoz - salpr * rhoy
        phy = salpr * rhox - state.cabj * rhoz
        phz = state.cabj * rhoy - state.sabj * rhox
        hpk, hps, hpc = hsflx(state, state.s, rh, zp)

        if ip != 2:
            state.exk, state.eyk, state.ezk = hpk * phx, hpk * phy, hpk * phz
            state.exs, state.eys, state.ezs = hps * phx, hps * phy, hps * phz
            state.exc, state.eyc, state.ezc = hpc * phx, hpc * phy, hpc * phz
        elif state.iperf == 1:
            state.exk, state.eyk, state.ezk = state.exk - hpk * phx, state.eyk - hpk * phy, state.ezk - hpk * phz
            state.exs, state.eys, state.ezs = state.exs - hps * phx, state.eys - hps * phy, state.ezs - hps * phz
            state.exc, state.eyc, state.ezc = state.exc - hpc * phx, state.eyc - hpc * phy, state.ezc - hpc * phz
        else:
            zratx = state.zrati
            rmag, xymag = math.sqrt(zp * zp + rh * rh), math.sqrt(xij * xij + yij * yij)
            if state.nradl != 0:
                xspec, yspec = (xi * state.zj + zi * state.xj) / (zi + state.zj), (yi * state.zj + zi * state.yj) / (zi + state.zj)
                rhospc = math.sqrt(xspec * xspec + yspec * yspec + state.t2 * state.t2)
                if rhospc <= state.scrwl:
                    rrv = state.t1 * rhospc * math.log(rhospc / state.t2)
                    zratx = (rrv * state.zrati) / (eta * state.zrati + rrv)
            if xymag <= 1.0e-6:
                px, py, cth, rrv = 0.0, 0.0, 1.0, 1.0 + 0j
            else:
                px, py, cth = -yij / xymag, xij / xymag, zij / rmag
                rrv = cmath.sqrt(1.0 - zratx * zratx * (1.0 - cth * cth))
            rrh = -(zratx * cth - rrv) / (zratx * cth + rrv)
            rrv = (cth - zratx * rrv) / (cth + zratx * rrv)
            qy = (phx * px + phy * py) * (rrv - rrh)
            qx, qy, qz = qy * px + phx * rrh, qy * py + phy * rrh, phz * rrh
            state.exk, state.eyk, state.ezk = state.exk - hpk * qx, state.eyk - hpk * qy, state.ezk - hpk * qz
            state.exs, state.eys, state.ezs = state.exs - hps * qx, state.eys - hps * qy, state.ezs - hps * qz
            state.exc, state.eyc, state.ezc = state.exc - hpc * qx, state.eyc - hpc * qy, state.ezc - hpc * qz


def hsflx(state: NECState, s: float, rh: float, zpx: float):
    tp, pi8, fj, fjk = 6.283185308, 25.13274123, complex(0.0, 1.0), complex(0.0, -6.283185308)
    if rh < 1.0e-10:
        return 0j, 0j, 0j
    zp, hss = (zpx, 1.0) if zpx >= 0.0 else (-zpx, -1.0)
    dh = 0.5 * s
    z1, z2 = zp + dh, zp - dh
    rhz = 1.0 if z2 < 1.0e-7 else rh / z2
    dk = tp * dh
    cdk, sdk = math.cos(dk), math.sin(dk)
    hkr, hki = hfk(state, -dk, dk, rh * tp, zp * tp)
    hpk = complex(hkr, hki)
    if rhz >= 1.0e-3:
        rh2 = rh * rh
        r1, r2 = math.sqrt(rh2 + z1 * z1), math.sqrt(rh2 + z2 * z2)
        ekr1, ekr2 = cmath.exp(fjk * r1), cmath.exp(fjk * r2)
        t1, t2 = z1 * ekr1 / r1, z2 * ekr2 / r2
        cons = -fj / (2.0 * tp * rh)
        return hpk, cons * (cdk * (ekr2 - ekr1) - fj * sdk * (t2 + t1)) * hss, cons * (-sdk * (ekr2 + ekr1) - fj * cdk * (t2 - t1))
    ekr1, ekr2 = complex(cdk, sdk) / (z2 * z2), complex(cdk, -sdk) / (z1 * z1)
    t1 = tp * (1.0 / z1 - 1.0 / z2)
    t2 = cmath.exp(fjk * zp) * rh / pi8
    return hpk, t2 * (t1 + (ekr1 + ekr2) * sdk) * hss, t2 * (-fj * t1 + (ekr1 - ekr2) * cdk)


def gh(state: NECState, zk: float):
    rs = state.rhks + (zk - state.zpk_h) ** 2
    r = math.sqrt(rs)
    ckr, skr = math.cos(r), math.sin(r)
    rr2 = 1.0 / rs
    rr3 = rr2 / r
    return skr * rr2 + ckr * rr3, ckr * rr2 - skr * rr3


def hfk(state: NECState, el1: float, el2: float, rhk: float, zpkx: float):
    nx, nm, nts, rx = 1, 65536, 4, 1.0e-4
    state.zpk_h, state.rhks = zpkx, rhk * rhk
    z, ze = el1, el2
    s, ep = ze - z, (ze - z) / (10.0 * nm)
    zend = ze - ep
    sgr = sgi = 0.0
    ns, nt = nx, 0
    g1r, g1i = gh(state, z)
    while True:
        dz = s / float(ns)
        if z + dz > ze:
            dz = ze - z
            if abs(dz) <= ep:
                break
        dzot = dz * 0.5
        g3r, g3i = gh(state, z + dzot)
        g5r, g5i = gh(state, z + dz)
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
            g2r, g2i = gh(state, z + dz * 0.25)
            g4r, g4i = gh(state, z + dz * 0.75)
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
            g5r, g5i, g3r, g3i = g3r, g3i, g2r, g2i
        z += dz
        if z >= zend:
            break
        g1r, g1i = g5r, g5i
        if nt >= nts and ns > nx:
            ns //= 2
            nt = 1
    return sgr * rhk * 0.5, sgi * rhk * 0.5

def _set_ext_wire_flags(state: NECState, j: int, t1y, t1z, salp, bi, icon1, icon2):
    j_idx = j - 1
    b = bi[j_idx]
    cabj, sabj, salpj = t1y[j_idx], t1z[j_idx], salp[j_idx]
    if state.iexk == 0:
        return

    ipr = int(icon1[j_idx])
    if ipr == 0:
        state.ind1 = 1
    else:
        ok = False
        if ipr < 0:
            ipr = -ipr
            ok = (-int(icon1[ipr - 1]) == j)
        elif ipr == j:
            ok = (cabj * cabj + sabj * sabj <= 1.0e-8)
        else:
            ok = (int(icon2[ipr - 1]) == j)
        if ok and ipr != j:
            xi = abs(cabj * t1y[ipr - 1] + sabj * t1z[ipr - 1] + salpj * salp[ipr - 1])
            ok = (xi >= 0.999999 and abs(bi[ipr - 1] / b - 1.0) <= 1.0e-6)
        state.ind1 = 0 if ok else 2

    ipr = int(icon2[j_idx])
    if ipr == 0:
        state.ind2 = 1
    else:
        ok = False
        if ipr < 0:
            ipr = -ipr
            ok = (-int(icon2[ipr - 1]) == j)
        elif ipr == j:
            ok = (cabj * cabj + sabj * sabj <= 1.0e-8)
        else:
            ok = (int(icon1[ipr - 1]) == j)
        if ok and ipr != j:
            xi = abs(cabj * t1y[ipr - 1] + sabj * t1z[ipr - 1] + salpj * salp[ipr - 1])
            ok = (xi >= 0.999999 and abs(bi[ipr - 1] / b - 1.0) <= 1.0e-6)
        state.ind2 = 0 if ok else 2

