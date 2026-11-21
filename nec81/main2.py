import math
import cmath
import time
import numpy as npy
from .nec_state import NECState


def gwave(state: NECState):
    """
    Computes the electric field, including ground wave, of a current element
    over a ground plane using formulas of K.A. Norton (Proc. IRE, Sept. 1937).
    Returns (erv, ezv, erh, ezh, eph).
    """
    tpj = complex(0.0, 6.283185308)
    econ = complex(0.0, -188.367)

    sppp = state.zmh / state.r1
    sppp2 = sppp * sppp
    cppp2 = max(1.0 - sppp2, 1.0e-20)
    cppp = math.sqrt(cppp2)

    spp = state.zph / state.r2
    spp2 = spp * spp
    cpp2 = max(1.0 - spp2, 1.0e-20)
    cpp = math.sqrt(cpp2)

    rk1 = -tpj * state.r1
    rk2 = -tpj * state.r2
    t1 = 1.0 - state.u2 * cpp2
    t2 = cmath.sqrt(t1)
    t3 = (1.0 - 1.0 / rk1) / rk1
    t4 = (1.0 - 1.0 / rk2) / rk2

    p1 = rk2 * state.u2 * t1 / (2.0 * cpp2)
    rv = (spp - state.u * t2) / (spp + state.u * t2)
    omr = 1.0 - rv
    w = 1.0 / omr
    w = 4.0 * p1 * w * w
    f = fbar(w)

    q1 = rk2 * t1 / (2.0 * state.u2 * cpp2)
    rh = (t2 - state.u * spp) / (t2 + state.u * spp)
    v = 1.0 / (1.0 + rh)
    v = 4.0 * q1 * v * v
    g = fbar(v)

    xr1 = state.xx1 / state.r1
    xr2 = state.xx2 / state.r2

    x1 = cppp2 * xr1
    x2 = rv * cpp2 * xr2
    x3 = omr * cpp2 * f * xr2
    x4 = state.u * t2 * spp * 2.0 * xr2 / rk2
    x5 = xr1 * t3 * (1.0 - 3.0 * sppp2)
    x6 = xr2 * t4 * (1.0 - 3.0 * spp2)
    ezv = (x1 + x2 + x3 - x4 - x5 - x6) * econ

    x1 = sppp * cppp * xr1
    x2 = rv * spp * cpp * xr2
    x3 = cpp * omr * state.u * t2 * f * xr2
    x4 = spp * cpp * omr * xr2 / rk2
    x5 = 3.0 * sppp * cppp * t3 * xr1
    x6 = cpp * state.u * t2 * omr * xr2 / rk2 * 0.5
    x7 = 3.0 * spp * cpp * t4 * xr2
    erv = -(x1 + x2 - x3 + x4 - x5 + x6 - x7) * econ
    ezh = -(x1 - x2 + x3 - x4 - x5 - x6 + x7) * econ

    x1 = sppp2 * xr1
    x2 = rv * spp2 * xr2
    x4 = state.u2 * t1 * omr * f * xr2
    x5 = t3 * (1.0 - 3.0 * cppp2) * xr1
    x6 = t4 * (1.0 - 3.0 * spp2) * (1.0 - state.u2 * (1.0 + rv) - state.u2 * omr * f) * xr2
    x7 = (
        state.u2
        * cpp2
        * omr
        * (1.0 - 1.0 / rk2)
        * (f * (state.u2 * t1 - spp2 - 1.0 / rk2) + 1.0 / rk2)
        * xr2
    )
    erh = (x1 - x2 - x4 - x5 + x6 + x7) * econ

    x1 = xr1
    x2 = rh * xr2
    x3 = (rh + 1.0) * g * xr2
    x4 = t3 * xr1
    x5 = t4 * (1.0 - state.u2 * (1.0 + rv) - state.u2 * omr * f) * xr2
    x6 = 0.5 * state.u2 * omr * (f * (state.u2 * t1 - spp2 - 1.0 / rk2) + 1.0 / rk2) * xr2 / rk2
    eph = -(x1 - x2 + x3 - x4 + x5 + x6) * econ

    return erv, ezv, erh, ezh, eph


def tbf(state: NECState, si: npy.ndarray, bi: npy.ndarray, icon1: npy.ndarray, icon2: npy.ndarray, icap: int, i_seg: int):
    """
    Compute basis function I (1-based segment index `i_seg`).
    Populates state.ax, state.bx, state.cx, state.jco, and state.jsno.
    """
    pi = 3.141592654
    jmax = 30

    jsno = 0
    pp = 0.0

    def trace_junction(start_jcox: int, init_jend: int, iend_flag: int):
        nonlocal jsno, pp
        jcox = start_jcox
        jend = init_jend
        sig = -1.0

        if jcox == 0:
            return
        if jcox < 0:
            jcox = -jcox
        else:
            sig = -sig
            jend = -jend

        while True:
            jsno += 1
            if jsno >= jmax:
                raise RuntimeError(f" TBF - SEGMENT CONNECTION ERROR FOR SEGMENT{i_seg:5d}")

            idx = jsno - 1
            j_idx = jcox - 1
            state.jco[idx] = jcox
            d = pi * si[j_idx]
            sdh = math.sin(d)
            cdh = math.cos(d)
            sd = 2.0 * sdh * cdh
            if d <= 0.015:
                omc = 4.0 * d * d
                omc = ((1.3888889e-3 * omc - 4.1666666667e-2) * omc + 0.5) * omc
            else:
                omc = 1.0 - cdh * cdh + sdh * sdh

            aj = 1.0 / (math.log(1.0 / (pi * bi[j_idx])) - 0.577215664)
            pp = pp - omc / sd * aj
            state.ax[idx] = aj / sd * sig
            state.bx[idx] = aj / (2.0 * cdh)
            state.cx[idx] = -aj / (2.0 * sdh) * sig

            if jcox == i_seg:
                state.bx[idx] = -state.bx[idx]
                return

            if jend == 1:
                jcox = icon2[j_idx]
            else:
                jcox = icon1[j_idx]

            if abs(jcox) == i_seg:
                return
            if jcox == 0:
                raise RuntimeError(f" TBF - SEGMENT CONNECTION ERROR FOR SEGMENT{i_seg:5d}")
            if jcox < 0:
                jcox = -jcox
            else:
                sig = -sig
                jend = -jend

    # End 1
    jcox1 = int(icon1[i_seg - 1])
    if jcox1 > 10000:
        jcox1 = i_seg
    trace_junction(jcox1, -1, -1)

    pm = -pp
    pp = 0.0
    njun1 = jsno

    # End 2
    jcox2 = int(icon2[i_seg - 1])
    if jcox2 > 10000:
        jcox2 = i_seg
    trace_junction(jcox2, 1, 1)

    njun2 = jsno - njun1
    jsnop = jsno + 1
    i_idx = i_seg - 1
    p_idx = jsnop - 1

    state.jco[p_idx] = i_seg
    d = pi * si[i_idx]
    sdh = math.sin(d)
    cdh = math.cos(d)
    sd = 2.0 * sdh * cdh
    cd = cdh * cdh - sdh * sdh
    if d <= 0.015:
        omc = 4.0 * d * d
        omc = ((1.3888889e-3 * omc - 4.1666666667e-2) * omc + 0.5) * omc
    else:
        omc = 1.0 - cd

    ap = 1.0 / (math.log(1.0 / (pi * bi[i_idx])) - 0.577215664)
    aj = ap

    if njun1 != 0 and njun2 != 0:
        qp = sd * (pm * pp + aj * ap) + cd * (pm * ap - pp * aj)
        qm = (ap * omc - pp * sd) / qp
        qp = -(aj * omc + pm * sd) / qp
        state.bx[p_idx] = (aj * qm + ap * qp) * sdh / sd
        state.cx[p_idx] = (aj * qm - ap * qp) * cdh / sd
        state.ax[:njun1] *= qm
        state.bx[:njun1] *= qm
        state.cx[:njun1] *= qm
        state.ax[njun1:jsno] *= -qp
        state.bx[njun1:jsno] *= qp
        state.cx[njun1:jsno] *= -qp
    elif njun1 == 0 and njun2 != 0:
        if icap == 0:
            xxi = 0.0
        else:
            qp = pi * bi[i_idx]
            xxi = qp * qp
            xxi = qp * (1.0 - 0.5 * xxi) / (1.0 - xxi)
        qp = -(omc + xxi * sd) / (sd * (ap + xxi * pp) + cd * (xxi * ap - pp))
        d_den = cd - xxi * sd
        state.bx[p_idx] = (sdh + ap * qp * (cdh - xxi * sdh)) / d_den
        state.cx[p_idx] = (cdh + ap * qp * (sdh + xxi * cdh)) / d_den
        state.ax[:njun2] *= -qp
        state.bx[:njun2] *= qp
        state.cx[:njun2] *= -qp
    elif njun1 != 0 and njun2 == 0:
        if icap == 0:
            xxi = 0.0
        else:
            qm = pi * bi[i_idx]
            xxi = qm * qm
            xxi = qm * (1.0 - 0.5 * xxi) / (1.0 - xxi)
        qm = (omc + xxi * sd) / (sd * (aj - xxi * pm) + cd * (pm + xxi * aj))
        d_den = cd - xxi * sd
        state.bx[p_idx] = (aj * qm * (cdh - xxi * sdh) - sdh) / d_den
        state.cx[p_idx] = (cdh - aj * qm * (sdh + xxi * cdh)) / d_den
        state.ax[:njun1] *= qm
        state.bx[:njun1] *= qm
        state.cx[:njun1] *= qm
    else:
        state.bx[p_idx] = 0.0
        if icap == 0:
            xxi = 0.0
        else:
            qp = pi * bi[i_idx]
            xxi = qp * qp
            xxi = qp * (1.0 - 0.5 * xxi) / (1.0 - xxi)
        state.cx[p_idx] = 1.0 / (cdh - xxi * sdh)

    state.jsno = jsnop
    state.ax[jsnop - 1] = -1.0


def atgn2(x: float, y: float) -> float:
    """Modified arctangent returning 0.0 when x == 0 and y == 0."""
    if x == 0.0 and y == 0.0:
        return 0.0
    return math.atan2(x, y)


def cang(z: complex) -> float:
    """Returns the phase angle of a complex number in degrees."""
    return atgn2(z.imag, z.real) * 57.29577951


def db10(x: float) -> float:
    """Returns dB for power magnitude (10 * log10(x)), floored at -999.99."""
    if x < 1.0e-20:
        return -999.99
    return 10.0 * math.log10(x)


def db20(x: float) -> float:
    """Returns dB for field magnitude (20 * log10(x)), floored at -999.99."""
    if x < 1.0e-20:
        return -999.99
    return 20.0 * math.log10(x)


def fbar(p: complex) -> complex:
    """Sommerfeld attenuation function for numerical distance P."""
    tosp = 1.128379167
    accs = 1.0e-12
    sp = 1.772453851
    fj = complex(0.0, 1.0)

    z = fj * cmath.sqrt(p)
    if abs(z) <= 3.0:
        # Series expansion
        zs = z * z
        s_sum = z
        pow_term = z
        for i in range(1, 101):
            pow_term = -pow_term * zs / float(i)
            term = pow_term / (2.0 * i + 1.0)
            s_sum = s_sum + term
            tms = (term * term.conjugate()).real
            sms = (s_sum * s_sum.conjugate()).real
            if tms / sms < accs:
                break
        return 1.0 - (1.0 - s_sum * tosp) * sp * z * cmath.exp(zs)

    # Asymptotic expansion
    minus = 0
    if z.real < 0.0:
        minus = 1
        z = -z
    zs = 0.5 / (z * z)
    s_sum = complex(0.0, 0.0)
    term = complex(1.0, 0.0)
    for i in range(1, 7):
        term = -term * (2.0 * i - 1.0) * zs
        s_sum = s_sum + term
    if minus == 1:
        s_sum = s_sum - 2.0 * sp * z * cmath.exp(z * z)
    return -s_sum


def isegno(state: NECState, itagi: int, mx: int, itag: npy.ndarray) -> int:
    """
    Returns the 1-based segment number of the MX-th segment having tag ITAGI.
    If ITAGI == 0, segment number MX is returned.
    """
    if mx <= 0:
        raise ValueError("CHECK DATA, PARAMETER SPECIFYING SEGMENT POSITION IN A GROUP OF EQUAL TAGS MUST NOT BE ZERO")
    if itagi == 0:
        return mx
    icnt = 0
    for i in range(state.n):
        if itag[i] == itagi:
            icnt += 1
            if icnt == mx:
                return i + 1
    raise ValueError(f"NO SEGMENT HAS AN ITAG OF {itagi:5d}")


def blckot(ar: npy.ndarray, ix1: int, ix2: int, nblks: int, neof: int, file_obj):
    """Writes out-of-core matrix block slice (1-based complex indices ix1..ix2) to binary file."""
    i1 = (ix1 + 1) // 2 - 1
    i2 = (ix2 + 1) // 2
    ar[i1:i2].tofile(file_obj)


def blckin(ar: npy.ndarray, ix1: int, ix2: int, nblks: int, neof: int, file_obj) -> int:
    """Reads out-of-core matrix block slice from binary file."""
    i1 = (ix1 + 1) // 2 - 1
    count = (ix2 + 1) // 2 - i1
    for _ in range(nblks):
        chunk = npy.fromfile(file_obj, dtype=ar.dtype, count=count)
        if chunk.size < count:
            if neof != 777:
                raise EOFError(f"EOF ON NBLKS={nblks} NEOF={neof}")
            return 0
        ar[i1 : i1 + count] = chunk
    return neof


def second() -> float:
    """Returns elapsed wall-clock seconds."""
    return time.perf_counter()

