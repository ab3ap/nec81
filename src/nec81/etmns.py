import math
import cmath
import numpy as npy
from .nec_state import NECState
from .main2 import tbf
from .hintg import _set_ext_wire_flags
from .hintg import hsfld
from .nfpat import efld


def etmns(
    state: NECState,
    e: npy.ndarray,
    zarray: npy.ndarray,
    x: npy.ndarray,
    y: npy.ndarray,
    z: npy.ndarray,
    bi: npy.ndarray,
    salp: npy.ndarray,
    t1x: npy.ndarray,
    t1y: npy.ndarray,
    t1z: npy.ndarray,
    t2x: npy.ndarray,
    t2y: npy.ndarray,
    t2z: npy.ndarray,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    p5: float,
    p6: float,
    icon1: npy.ndarray,
    icon2: npy.ndarray,
    ld: int,
    ld2: int,
    ld3: int,
    ipr: int,
):
    """Fills array E with negative of incident electric field (RHS of matrix equation)."""
    tp, reta = 6.283185308, 2.654420938e-3
    neq = state.n + 2 * state.m
    state.nqds = 0

    if ipr <= 0 or ipr == 5:
        e[:neq] = 0j
        for i in range(state.nsant):
            is_idx = int(state.isant[i]) - 1
            e[is_idx] = -state.vsant[i] / (t1x[is_idx] * state.wlam)
        for i in range(state.nvqd):
            is_seg = int(state.ivqd[i])
            qdsrc(state, e, zarray, state.vqd[i], x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, is_seg, ld)
        return

    if ipr <= 3:
        cth, sth = math.cos(p1), math.sin(p1)
        cph, sph = math.cos(p2), math.sin(p2)
        cet, set_ = math.cos(p3), math.sin(p3)
        px = cth * cph * cet - sph * set_
        py = cth * sph * cet + cph * set_
        pz = -sth * cet
        wx, wy, wz = -sth * cph, -sth * sph, -cth
        qx, qy, qz = wy * pz - wz * py, wz * px - wx * pz, wx * py - wy * px

        if state.ksymp != 1:
            if state.iperf == 1:
                rrv = rrh = complex(-1.0, 0.0)
            else:
                rrv = cmath.sqrt(1.0 - state.zrati * state.zrati * sth * sth)
                rrh = state.zrati * cth
                rrh = (rrh - rrv) / (rrh + rrv)
                rrv = state.zrati * rrv
                rrv = -(cth - rrv) / (cth + rrv)

        if ipr <= 1:
            for i in range(state.n):
                arg = -tp * (wx * x[i] + wy * y[i] + wz * z[i])
                e[i] = -(px * t1y[i] + py * t1z[i] + pz * salp[i]) * complex(math.cos(arg), math.sin(arg))
            if state.ksymp != 1 and state.n != 0:
                tt1 = (py * cph - px * sph) * (rrh - rrv)
                cx, cy, cz = rrv * px - tt1 * sph, rrv * py + tt1 * cph, -rrv * pz
                for i in range(state.n):
                    arg = -tp * (wx * x[i] + wy * y[i] - wz * z[i])
                    e[i] -= (cx * t1y[i] + cy * t1z[i] + cz * salp[i]) * complex(math.cos(arg), math.sin(arg))

            if state.m != 0:
                i, i1 = ld, state.n - 2
                for _ in range(state.m):
                    i -= 1
                    i1 += 2
                    arg = -tp * (wx * x[i] + wy * y[i] + wz * z[i])
                    tt1 = complex(math.cos(arg), math.sin(arg)) * salp[i] * reta
                    e[i1 + 1] = (qx * t1x[i] + qy * t1y[i] + qz * t1z[i]) * tt1
                    e[i1] = (qx * t2x[i] + qy * t2y[i] + qz * t2z[i]) * tt1
                if state.ksymp != 1:
                    tt1 = (qy * cph - qx * sph) * (rrv - rrh)
                    cx, cy, cz = -(rrh * qx - tt1 * sph), -(rrh * qy + tt1 * cph), rrh * qz
                    i, i1 = ld, state.n - 2
                    for _ in range(state.m):
                        i -= 1
                        i1 += 2
                        arg = -tp * (wx * x[i] + wy * y[i] - wz * z[i])
                        tt1 = complex(math.cos(arg), math.sin(arg)) * salp[i] * reta
                        e[i1 + 1] += (cx * t1x[i] + cy * t1y[i] + cz * t1z[i]) * tt1
                        e[i1] += (cx * t2x[i] + cy * t2y[i] + cz * t2z[i]) * tt1
            return

        # Elliptic polarization (IPR = 2 or 3)
        tt1 = complex(0.0, -p6) if ipr != 3 else complex(0.0, p6)
        if state.n != 0:
            cx, cy, cz = px + tt1 * qx, py + tt1 * qy, pz + tt1 * qz
            for i in range(state.n):
                arg = -tp * (wx * x[i] + wy * y[i] + wz * z[i])
                e[i] = -(cx * t1y[i] + cy * t1z[i] + cz * salp[i]) * complex(math.cos(arg), math.sin(arg))
            if state.ksymp != 1:
                tt2 = (cy * cph - cx * sph) * (rrh - rrv)
                cx, cy, cz = rrv * cx - tt2 * sph, rrv * cy + tt2 * cph, -rrv * cz
                for i in range(state.n):
                    arg = -tp * (wx * x[i] + wy * y[i] - wz * z[i])
                    e[i] -= (cx * t1y[i] + cy * t1z[i] + cz * salp[i]) * complex(math.cos(arg), math.sin(arg))

        if state.m != 0:
            cx, cy, cz = qx - tt1 * px, qy - tt1 * py, qz - tt1 * pz
            i, i1 = ld, state.n - 2
            for _ in range(state.m):
                i -= 1
                i1 += 2
                arg = -tp * (wx * x[i] + wy * y[i] + wz * z[i])
                tt2 = complex(math.cos(arg), math.sin(arg)) * salp[i] * reta
                e[i1 + 1] = (cx * t1x[i] + cy * t1y[i] + cz * t1z[i]) * tt2
                e[i1] = (cx * t2x[i] + cy * t2y[i] + cz * t2z[i]) * tt2
            if state.ksymp != 1:
                tt1 = (cy * cph - cx * sph) * (rrv - rrh)
                cx, cy, cz = -(rrh * cx - tt1 * sph), -(rrh * cy + tt1 * cph), rrh * cz
                i, i1 = ld, state.n - 2
                for _ in range(state.m):
                    i -= 1
                    i1 += 2
                    arg = -tp * (wx * x[i] + wy * y[i] - wz * z[i])
                    tt1 = complex(math.cos(arg), math.sin(arg)) * salp[i] * reta
                    e[i1 + 1] += (cx * t1x[i] + cy * t1y[i] + cz * t1z[i]) * tt1
                    e[i1] += (cx * t2x[i] + cy * t2y[i] + cz * t2z[i]) * tt1
        return

    # Elementary current source (IPR = 4)
    wz = math.cos(p4)
    wx, wy, wz = wz * math.cos(p5), wz * math.sin(p5), math.sin(p4)
    ds, dsh = p6 * 59.958, p6 / (2.0 * tp)
    is_idx, i1 = ld, state.n - 2
    for i in range(state.n + state.m):
        if i < state.n:
            ii = i
        else:
            is_idx -= 1
            ii = is_idx
            i1 += 2
        px, py, pz = x[ii] - p1, y[ii] - p2, z[ii] - p3
        rs = px * px + py * py + pz * pz
        if rs < 1.0e-30:
            continue
        r = math.sqrt(rs)
        px, py, pz = px / r, py / r, pz / r
        cth = px * wx + py * wy + pz * wz
        sth = math.sqrt(max(0.0, 1.0 - cth * cth))
        qx, qy, qz = px - wx * cth, py - wy * cth, pz - wz * cth
        arg = math.sqrt(qx * qx + qy * qy + qz * qz)
        if arg < 1.0e-30:
            qx, qy, qz = 1.0, 0.0, 0.0
        else:
            qx, qy, qz = qx / arg, qy / arg, qz / arg

        tt1 = complex(math.cos(-tp * r), math.sin(-tp * r))
        if i < state.n:
            tt2 = complex(1.0, -1.0 / (r * tp)) / rs
            er = ds * tt1 * tt2 * cth
            et = 0.5 * ds * tt1 * (complex(0.0, tp / r) + tt2) * sth
            ezh = er * cth - et * sth
            erh = er * sth + et * cth
            cx, cy, cz = ezh * wx + erh * qx, ezh * wy + erh * qy, ezh * wz + erh * qz
            e[i] = -(cx * t1y[i] + cy * t1z[i] + cz * salp[i])
        else:
            px_c, py_c, pz_c = wy * qz - wz * qy, wz * qx - wx * qz, wx * qy - wy * qx
            tt2 = dsh * tt1 * complex(1.0 / r, tp) / r * sth * salp[ii]
            cx, cy, cz = tt2 * px_c, tt2 * py_c, tt2 * pz_c
            e[i1 + 1] = cx * t1x[ii] + cy * t1y[ii] + cz * t1z[ii]
            e[i1] = cx * t2x[ii] + cy * t2y[ii] + cz * t2z[ii]


def qdsrc(state: NECState, e, zarray, v: complex, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, is_seg: int, ld: int):
    """Fills incident field array for charge-discontinuity voltage source."""
    tp = 6.283185308
    ccj = complex(0.0, -0.01666666667)
    is_idx = is_seg - 1
    saved_icon1 = int(icon1[is_idx])
    icon1[is_idx] = 0
    tbf(state, t1x, bi, icon1, icon2, 0, is_seg)
    icon1[is_idx] = saved_icon1

    s = t1x[is_idx] * 0.5
    jsno_idx = state.jsno - 1
    curd = ccj * v / ((math.log(2.0 * s / bi[is_idx]) - 1.0) * (state.bx[jsno_idx] * math.cos(tp * s) + state.cx[jsno_idx] * math.sin(tp * s)) * state.wlam)
    state.nqds += 1
    state.vqds[state.nqds - 1] = v
    state.iqds[state.nqds - 1] = is_seg

    for jx in range(state.jsno):
        j = int(state.jco[jx])
        j_idx = j - 1
        state.s, state.b = t1x[j_idx], bi[j_idx]
        state.xj, state.yj, state.zj = x[j_idx], y[j_idx], z[j_idx]
        state.cabj, state.sabj, state.salpj = t1y[j_idx], t1z[j_idx], salp[j_idx]
        _set_ext_wire_flags(state, j, t1y, t1z, salp, bi, icon1, icon2)

        for i in range(state.n):
            efld(state, x[i], y[i], z[i], bi[i], (i + 1) - j)
            etk = state.exk * t1y[i] + state.eyk * t1z[i] + state.ezk * salp[i]
            ets = state.exs * t1y[i] + state.eys * t1z[i] + state.ezs * salp[i]
            etc = state.exc * t1y[i] + state.eyc * t1z[i] + state.ezc * salp[i]
            e[i] -= (etk * state.ax[jx] + ets * state.bx[jx] + etc * state.cx[jx]) * curd

        if state.m != 0:
            ij, i1 = ld, state.n - 1
            for _ in range(state.m):
                ij -= 1
                hsfld(state, x[ij], y[ij], z[ij], 0.0)
                tx, ty, tz = t2x[ij], t2y[ij], t2z[ij]
                e[i1] += ((state.exk * tx + state.eyk * ty + state.ezk * tz) * state.ax[jx]
                          + (state.exs * tx + state.eys * ty + state.ezs * tz) * state.bx[jx]
                          + (state.exc * tx + state.eyc * ty + state.ezc * tz) * state.cx[jx]) * curd * salp[ij]
                i1 += 1
                tx, ty, tz = t1x[ij], t1y[ij], t1z[ij]
                e[i1] += ((state.exk * tx + state.eyk * ty + state.ezk * tz) * state.ax[jx]
                          + (state.exs * tx + state.eys * ty + state.ezs * tz) * state.bx[jx]
                          + (state.exc * tx + state.eyc * ty + state.ezc * tz) * state.cx[jx]) * curd * salp[ij]
                i1 += 1

        if state.nload > 0 or state.nlodf > 0:
            xku = 0.0
            e[j_idx] += zarray[j_idx] * curd * (state.ax[jx] + state.cx[jx]) * xku

