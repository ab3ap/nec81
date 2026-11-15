import math
import numpy as npy
from .nec_state import NECState, TA


def patch(
    state: NECState,
    nx: int,
    ny: int,
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    x3: float,
    y3: float,
    z3: float,
    x4: float,
    y4: float,
    z4: float,
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
    ld: int,
):
    """Generates and modifies surface patch geometry data."""
    state.m += 1
    mi = ld - state.m
    ntp = 2 if nx > 0 else ny

    if ntp <= 1:
        x[mi], y[mi], z[mi] = x1, y1, z1
        bi[mi] = z2
        znv = math.cos(x2)
        xnv = znv * math.cos(y2)
        ynv = znv * math.sin(y2)
        znv = math.sin(x2)
        xa = math.sqrt(xnv * xnv + ynv * ynv)
        if xa < 1.0e-6:
            t1x[mi], t1y[mi], t1z[mi] = 1.0, 0.0, 0.0
        else:
            t1x[mi], t1y[mi], t1z[mi] = -ynv / xa, xnv / xa, 0.0
    else:
        s1x, s1y, s1z = x2 - x1, y2 - y1, z2 - z1
        s2x, s2y, s2z = x3 - x2, y3 - y2, z3 - z2
        if nx != 0:
            s1x /= nx
            s1y /= nx
            s1z /= nx
            s2x /= ny
            s2y /= ny
            s2z /= ny

        xnv = s1y * s2z - s1z * s2y
        ynv = s1z * s2x - s1x * s2z
        znv = s1x * s2y - s1y * s2x
        xa = math.sqrt(xnv * xnv + ynv * ynv + znv * znv)
        xnv /= xa
        ynv /= xa
        znv /= xa

        xst = math.sqrt(s1x * s1x + s1y * s1y + s1z * s1z)
        t1x[mi] = s1x / xst
        t1y[mi] = s1y / xst
        t1z[mi] = s1z / xst

        if ntp <= 2:
            x[mi] = x1 + 0.5 * (s1x + s2x)
            y[mi] = y1 + 0.5 * (s1y + s2y)
            z[mi] = z1 + 0.5 * (s1z + s2z)
            bi[mi] = xa
        elif ntp != 4:
            x[mi] = (x1 + x2 + x3) / 3.0
            y[mi] = (y1 + y2 + y3) / 3.0
            z[mi] = (z1 + z2 + z3) / 3.0
            bi[mi] = 0.5 * xa
        else:
            s1x, s1y, s1z = x3 - x1, y3 - y1, z3 - z1
            s2x, s2y, s2z = x4 - x1, y4 - y1, z4 - z1
            xn2 = s1y * s2z - s1z * s2y
            yn2 = s1z * s2x - s1x * s2z
            zn2 = s1x * s2y - s1y * s2x
            xst = math.sqrt(xn2 * xn2 + yn2 * yn2 + zn2 * zn2)
            salpn = 1.0 / (3.0 * (xa + xst))
            x[mi] = (xa * (x1 + x2 + x3) + xst * (x1 + x3 + x4)) * salpn
            y[mi] = (xa * (y1 + y2 + y3) + xst * (y1 + y3 + y4)) * salpn
            z[mi] = (xa * (z1 + z2 + z3) + xst * (z1 + z3 + z4)) * salpn
            bi[mi] = 0.5 * (xa + xst)
            dot = (xnv * xn2 + ynv * yn2 + znv * zn2) / xst
            if dot <= 0.9998:
                raise RuntimeError(" ERROR -- CORNERS OF QUADRILATERAL PATCH DO NOT LIE IN A PLANE")

    t2x[mi] = ynv * t1z[mi] - znv * t1y[mi]
    t2y[mi] = znv * t1x[mi] - xnv * t1z[mi]
    t2z[mi] = xnv * t1y[mi] - ynv * t1x[mi]
    salp[mi] = 1.0

    if nx != 0:
        state.m += nx * ny - 1
        xn2 = x[mi] - s1x - s2x
        yn2 = y[mi] - s1y - s2y
        zn2 = z[mi] - s1z - s2z
        xs, ys, zs = t1x[mi], t1y[mi], t1z[mi]
        xt, yt, zt = t2x[mi], t2y[mi], t2z[mi]
        mi += 1
        for _ in range(ny):
            xn2 += s2x
            yn2 += s2y
            zn2 += s2z
            for ix in range(1, nx + 1):
                xst = float(ix)
                mi -= 1
                cx = xn2 + xst * s1x
                cy = yn2 + xst * s1y
                cz = zn2 + xst * s1z
                x[mi], y[mi], z[mi] = cx, cy, cz
                bi[mi] = xa
                salp[mi] = 1.0
                t1x[mi], t1y[mi], t1z[mi] = xs, ys, zs
                t2x[mi], t2y[mi], t2z[mi] = xt, yt, zt
                state.patch_polygons.append([
                    [cx - 0.5 * (s1x + s2x), cy - 0.5 * (s1y + s2y), cz - 0.5 * (s1z + s2z)],
                    [cx + 0.5 * (s1x - s2x), cy + 0.5 * (s1y - s2y), cz + 0.5 * (s1z - s2z)],
                    [cx + 0.5 * (s1x + s2x), cy + 0.5 * (s1y + s2y), cz + 0.5 * (s1z + s2z)],
                    [cx - 0.5 * (s1x - s2x), cy - 0.5 * (s1y - s2y), cz - 0.5 * (s1z - s2z)],
                ])
    else:
        if ntp == 2:
            p4x, p4y, p4z = x1 + x3 - x2, y1 + y3 - y2, z1 + z3 - z2
            state.patch_polygons.append([
                [x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [p4x, p4y, p4z],
            ])
        elif ntp == 3:
            state.patch_polygons.append([
                [x1, y1, z1], [x2, y2, z2], [x3, y3, z3],
            ])
        elif ntp == 4:
            state.patch_polygons.append([
                [x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4],
            ])
        else:
            half_m = 0.5 * math.sqrt(abs(bi[mi]))
            cx, cy, cz = x[mi], y[mi], z[mi]
            ux, uy, uz = t1x[mi], t1y[mi], t1z[mi]
            vx, vy, vz = t2x[mi], t2y[mi], t2z[mi]
            state.patch_polygons.append([
                [cx + half_m * (s1 * ux + s2 * vx),
                 cy + half_m * (s1 * uy + s2 * vy),
                 cz + half_m * (s1 * uz + s2 * vz)]
                for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ])

    state.ipsym = 0
    state.np = state.n
    state.mp = state.m


def subph(
    state: NECState,
    nx: int,
    ny: int,
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
    ld: int,
):
    """Divides a patch into 4 sub-patches at a wire-surface connection."""
    if ny <= 0 and nx != state.m:
        ixx = ld - state.m - 1
        for _ in range(nx + 1, state.m + 1):
            ixx += 1
            nyp = ixx - 3
            x[nyp], y[nyp], z[nyp] = x[ixx], y[ixx], z[ixx]
            bi[nyp], salp[nyp] = bi[ixx], salp[ixx]
            t1x[nyp], t1y[nyp], t1z[nyp] = t1x[ixx], t1y[ixx], t1z[ixx]
            t2x[nyp], t2y[nyp], t2z[nyp] = t2x[ixx], t2y[ixx], t2z[ixx]

    mi = ld - nx
    xs, ys, zs = x[mi], y[mi], z[mi]
    xa = bi[mi] * 0.25
    xst = math.sqrt(xa) * 0.5
    s1x, s1y, s1z = t1x[mi], t1y[mi], t1z[mi]
    s2x, s2y, s2z = t2x[mi], t2y[mi], t2z[mi]
    saln = salp[mi]
    xt = xst
    yt = xst

    if ny <= 0:
        mia = mi
    else:
        state.m += 1
        state.mp += 1
        mia = ld - state.m

    for ixx in range(1, 5):
        x[mia] = xs + xt * s1x + yt * s2x
        y[mia] = ys + xt * s1y + yt * s2y
        z[mia] = zs + xt * s1z + yt * s2z
        bi[mia] = xa
        t1x[mia], t1y[mia], t1z[mia] = s1x, s1y, s1z
        t2x[mia], t2y[mia], t2z[mia] = s2x, s2y, s2z
        salp[mia] = saln
        if ixx == 2:
            yt = -yt
        if ixx in (1, 3):
            xt = -xt
        mia -= 1

    state.m += 3
    if nx <= state.mp:
        state.mp += 3
    if ny > 0:
        z[mi] = 10000.0


def arc(
    state: NECState,
    itg: int,
    ns: int,
    rada: float,
    ang1: float,
    ang2: float,
    rad: float,
    x: npy.ndarray,
    y: npy.ndarray,
    z: npy.ndarray,
    bi: npy.ndarray,
    itag: npy.ndarray,
    x2: npy.ndarray,
    y2: npy.ndarray,
    z2: npy.ndarray,
):
    """Generates segment geometry data for a circular arc of NS segments in the X-Z plane."""
    ist = state.n
    state.n += ns
    state.np = state.n
    state.mp = state.m
    state.ipsym = 0
    if ns < 1:
        return
    if abs(ang2 - ang1) >= 360.00001:
        raise RuntimeError(" ERROR -- ARC ANGLE EXCEEDS 360. DEGREES")

    ang = ang1 * TA
    dang = (ang2 - ang1) * TA / float(ns)
    xs1 = rada * math.cos(ang)
    zs1 = rada * math.sin(ang)
    for i in range(ist, state.n):
        ang += dang
        xs2 = rada * math.cos(ang)
        zs2 = rada * math.sin(ang)
        x[i], y[i], z[i] = xs1, 0.0, zs1
        x2[i], y2[i], z2[i] = xs2, 0.0, zs2
        xs1, zs1 = xs2, zs2
        bi[i] = rad
        itag[i] = itg


def conect(
    state: NECState,
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
    icon1: npy.ndarray,
    icon2: npy.ndarray,
    iconx: npy.ndarray,
    ignd: int,
    iw,
    ld: int,
):
    """Sets up segment connection data in ICON1 and ICON2 by finding contacting segment ends and patches."""
    jmax = 30
    smin = 1.0e-3
    nsmax = 50
    npmax = 10

    state.nscon = 0
    state.npcon = 0

    if ignd != 0:
        iw.write("\n   GROUND PLANE SPECIFIED.\n")
        if ignd > 0:
            iw.write("\n   WHERE WIRE ENDS TOUCH GROUND, CURRENT WILL BE INTERPOLATED TO IMAGE IN GROUND PLANE.\n\n")
        if state.ipsym == 2:
            state.np *= 2
            state.mp *= 2
        if abs(state.ipsym) > 2:
            state.np = state.n
            state.mp = state.m
        if state.np > state.n:
            raise RuntimeError("CONECT: NP > N")
        if state.np == state.n and state.mp == state.m:
            state.ipsym = 0

    if state.n != 0:
        for i in range(state.n):
            i_seg = i + 1
            iconx[i] = 0
            xi1, yi1, zi1 = x[i], y[i], z[i]
            xi2, yi2, zi2 = t1x[i], t1y[i], t1z[i]
            slen = math.sqrt((xi2 - xi1) ** 2 + (yi2 - yi1) ** 2 + (zi2 - zi1) ** 2) * smin

            # End 1 connection
            skip_end1_search = False
            if ignd >= 1:
                if zi1 <= -slen:
                    raise RuntimeError(f" GEOMETRY DATA ERROR-- SEGMENT{i_seg:5d} EXTENDS BELOW GROUND")
                if zi1 <= slen:
                    icon1[i] = i_seg
                    z[i] = 0.0
                    skip_end1_search = True

            if not skip_end1_search:
                ic = i_seg
                found = False
                for _ in range(2, state.n + 1):
                    ic += 1
                    if ic > state.n:
                        ic = 1
                    ic_idx = ic - 1
                    sep = abs(xi1 - x[ic_idx]) + abs(yi1 - y[ic_idx]) + abs(zi1 - z[ic_idx])
                    if sep <= slen:
                        icon1[i] = -ic
                        found = True
                        break
                    sep = abs(xi1 - t1x[ic_idx]) + abs(yi1 - t1y[ic_idx]) + abs(zi1 - t1z[ic_idx])
                    if sep <= slen:
                        icon1[i] = ic
                        found = True
                        break
                if not found:
                    if not (i_seg < state.n2 and icon1[i] > 10000):
                        icon1[i] = 0

            # End 2 connection
            skip_end2_search = False
            if ignd >= 1:
                if zi2 <= -slen:
                    raise RuntimeError(f" GEOMETRY DATA ERROR-- SEGMENT{i_seg:5d} EXTENDS BELOW GROUND")
                if zi2 <= slen:
                    if icon1[i] == i_seg:
                        raise RuntimeError(f" GEOMETRY DATA ERROR--SEGMENT{i_seg:5d} LIES IN GROUND PLANE.")
                    icon2[i] = i_seg
                    t1z[i] = 0.0
                    skip_end2_search = True
            if not skip_end2_search:
                ic = i_seg
                found = False
                for _ in range(2, state.n + 1):
                    ic += 1
                    if ic > state.n:
                        ic = 1
                    ic_idx = ic - 1
                    sep = abs(xi2 - x[ic_idx]) + abs(yi2 - y[ic_idx]) + abs(zi2 - z[ic_idx])
                    if sep <= slen:
                        icon2[i] = ic
                        found = True
                        break
                    sep = abs(xi2 - t1x[ic_idx]) + abs(yi2 - t1y[ic_idx]) + abs(zi2 - t1z[ic_idx])
                    if sep <= slen:
                        icon2[i] = -ic
                        found = True
                        break
                if not found:
                    if not (i_seg < state.n2 and icon2[i] > 10000):
                        icon2[i] = 0

        # Wire-surface connections for new patches
        if state.m != 0:
            ix = ld - state.m1
            i_pat = state.m2
            while i_pat <= state.m:
                ix -= 1
                xs, ys, zs = x[ix], y[ix], z[ix]
                for iseg in range(state.n):
                    xi1, yi1, zi1 = x[iseg], y[iseg], z[iseg]
                    xi2, yi2, zi2 = t1x[iseg], t1y[iseg], t1z[iseg]
                    slen = (abs(xi2 - xi1) + abs(yi2 - yi1) + abs(zi2 - zi1)) * smin

                    sep = abs(xi1 - xs) + abs(yi1 - ys) + abs(zi1 - zs)
                    if sep <= slen:
                        icon1[iseg] = 10000 + i_pat
                        subph(state, i_pat, 0, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, ld)
                        break

                    sep = abs(xi2 - xs) + abs(yi2 - ys) + abs(zi2 - zs)
                    if sep <= slen:
                        icon2[iseg] = 10000 + i_pat
                        subph(state, i_pat, 0, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, ld)
                        break
                i_pat += 1

            # New segments connected to NGF patches
            if state.m1 != 0 and state.n2 <= state.n:
                ix = ld
                i_pat = 1
                while i_pat <= state.m1:
                    ix -= 1
                    xs, ys, zs = x[ix], y[ix], z[ix]
                    for iseg in range(state.n2 - 1, state.n):
                        xi1, yi1, zi1 = x[iseg], y[iseg], z[iseg]
                        xi2, yi2, zi2 = t1x[iseg], t1y[iseg], t1z[iseg]
                        slen = (abs(xi2 - xi1) + abs(yi2 - yi1) + abs(zi2 - zi1)) * smin

                        sep = abs(xi1 - xs) + abs(yi1 - ys) + abs(zi1 - zs)
                        if sep <= slen:
                            icon1[iseg] = 10001 + state.m
                            state.npcon += 1
                            if state.npcon > npmax:
                                raise RuntimeError(f" ERROR - NO. NEW SEGMENTS CONNECTED TO N.G.F. SEGMENTS OR PATCHES EXCEEDS LIMIT OF{npmax:5d}")
                            state.ipcon[state.npcon - 1] = i_pat
                            subph(state, i_pat, 1, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, ld)
                            break

                        sep = abs(xi2 - xs) + abs(yi2 - ys) + abs(zi2 - zs)
                        if sep <= slen:
                            icon2[iseg] = 10001 + state.m
                            state.npcon += 1
                            if state.npcon > npmax:
                                raise RuntimeError(f" ERROR - NO. NEW SEGMENTS CONNECTED TO N.G.F. SEGMENTS OR PATCHES EXCEEDS LIMIT OF{npmax:5d}")
                            state.ipcon[state.npcon - 1] = i_pat
                            subph(state, i_pat, 1, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, ld)
                            break
                    i_pat += 1

    iw.write(f"\n   TOTAL SEGMENTS USED={state.n:5d}     NO. SEG. IN A SYMMETRIC CELL={state.np:5d}     SYMMETRY FLAG={state.ipsym:3d}\n")
    if state.m > 0:
        iw.write(f"   TOTAL PATCHES USED={state.m:5d}      NO. PATCHES IN A SYMMETRIC CELL={state.mp:5d}\n")

    iseg_sym = (state.n + state.m) // (state.np + state.mp)
    if iseg_sym != 1:
        if state.ipsym == 0:
            raise RuntimeError("CONECT: Symmetry mismatch with IPSYM == 0")
        elif state.ipsym < 0:
            iw.write(f" STRUCTURE HAS{iseg_sym:4d} FOLD ROTATIONAL SYMMETRY\n\n")
        else:
            ic = 3 if iseg_sym == 8 else iseg_sym // 2
            iw.write(f" STRUCTURE HAS{ic:2d} PLANES OF SYMMETRY\n\n")

    if state.n == 0:
        return

    iw.write("\n\n         - MULTIPLE WIRE JUNCTIONS -\n JUNCTION    SEGMENTS  (- FOR END 1, + FOR END 2)\n")
    junc_cnt = 0

    for j in range(state.n):
        j_seg = j + 1
        for end_pass in (1, 2):
            if end_pass == 1:
                iend = -1
                jend = -1
                ix = int(icon1[j])
                ic = 1
                state.jco[0] = -j_seg
                xa, ya, za = x[j], y[j], z[j]
            else:
                iend = 1
                jend = 1
                ix = int(icon2[j])
                ic = 1
                state.jco[0] = j_seg
                xa, ya, za = t1x[j], t1y[j], t1z[j]

            if ix == 0 or ix == j_seg or ix > 10000:
                continue

            nsflg = 0
            aborted = False
            while True:
                if ix == 0:
                    raise RuntimeError(f" CONNECT - SEGMENT CONNECTION ERROR FOR SEGMENT{ix:5d}")
                if ix < 0:
                    ix = -ix
                else:
                    jend = -jend

                if ix == j_seg:
                    break
                if ix < j_seg:
                    aborted = True
                    break

                ic += 1
                if ic > jmax:
                    raise RuntimeError(f" CONNECT - SEGMENT CONNECTION ERROR FOR SEGMENT{ix:5d}")
                state.jco[ic - 1] = ix * jend
                if ix > state.n1:
                    nsflg = 1

                ix_idx = ix - 1
                if jend == 1:
                    xa += t1x[ix_idx]
                    ya += t1y[ix_idx]
                    za += t1z[ix_idx]
                    ix = int(icon2[ix_idx])
                else:
                    xa += x[ix_idx]
                    ya += y[ix_idx]
                    za += z[ix_idx]
                    ix = int(icon1[ix_idx])

            if aborted:
                continue

            sep = float(ic)
            xa /= sep
            ya /= sep
            za /= sep
            for i in range(ic):
                seg_signed = int(state.jco[i])
                if seg_signed > 0:
                    s_idx = seg_signed - 1
                    t1x[s_idx], t1y[s_idx], t1z[s_idx] = xa, ya, za
                else:
                    s_idx = -seg_signed - 1
                    x[s_idx], y[s_idx], z[s_idx] = xa, ya, za

            if state.n1 != 0 and nsflg != 0:
                for i in range(ic):
                    ix_abs = abs(int(state.jco[i]))
                    if ix_abs <= state.n1 and iconx[ix_abs - 1] == 0:
                        state.nscon += 1
                        if state.nscon > nsmax:
                            raise RuntimeError(f" ERROR - NO. NEW SEGMENTS CONNECTED TO N.G.F. SEGMENTS OR PATCHES EXCEEDS LIMIT OF{nsmax:5d}")
                        state.iscon[state.nscon - 1] = ix_abs
                        iconx[ix_abs - 1] = state.nscon

            if ic >= 3:
                junc_cnt += 1
                segs_str = "".join(f"{int(state.jco[k]):5d}" for k in range(min(ic, 20)))
                iw.write(f" {junc_cnt:5d}     {segs_str}\n")
                for offset in range(20, ic, 20):
                    extra_str = "".join(f"{int(state.jco[k]):5d}" for k in range(offset, min(ic, offset + 20)))
                    iw.write(f"           {extra_str}\n")

    if junc_cnt == 0:
        iw.write("  NONE\n")

    if state.n1 != 0 and state.m1 != state.m:
        for j in range(state.n1):
            connects_new_patch = False
            ix1 = int(icon1[j])
            if ix1 >= 10000 and (ix1 - 10000) > state.m1:
                connects_new_patch = True
            ix2 = int(icon2[j])
            if ix2 >= 10000 and (ix2 - 10000) >= state.m2:
                connects_new_patch = True
            if connects_new_patch and iconx[j] == 0:
                state.nscon += 1
                state.iscon[state.nscon - 1] = j + 1
                iconx[j] = state.nscon


def helix(
    state: NECState,
    iw,
    s: float,
    hl: float,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
    rad: float,
    ns: int,
    itg: int,
    x: npy.ndarray,
    y: npy.ndarray,
    z: npy.ndarray,
    bi: npy.ndarray,
    itag: npy.ndarray,
    x2: npy.ndarray,
    y2: npy.ndarray,
    z2: npy.ndarray,
):
    """Generates segment geometry data for a helix or spiral of NS segments."""
    pi = 3.141592654
    ist = state.n
    state.n += ns
    state.np = state.n
    state.mp = state.m
    state.ipsym = 0
    if ns < 1:
        return

    zinc = abs(hl / float(ns))
    z[ist] = 0.0
    for i in range(ist, state.n):
        bi[i] = rad
        itag[i] = itg
        if i != ist:
            z[i] = z[i - 1] + zinc
        z2[i] = z[i] + zinc

        if a2 == a1:
            if b1 == 0.0:
                b1 = a1
            x[i] = a1 * math.cos(2.0 * pi * z[i] / s)
            y[i] = b1 * math.sin(2.0 * pi * z[i] / s)
            x2[i] = a1 * math.cos(2.0 * pi * z2[i] / s)
            y2[i] = b1 * math.sin(2.0 * pi * z2[i] / s)
        else:
            if b2 == 0.0:
                b2 = a2
            abs_hl = abs(hl)
            x[i] = (a1 + (a2 - a1) * z[i] / abs_hl) * math.cos(2.0 * pi * z[i] / s)
            y[i] = (b1 + (b2 - b1) * z[i] / abs_hl) * math.sin(2.0 * pi * z[i] / s)
            x2[i] = (a1 + (a2 - a1) * z2[i] / abs_hl) * math.cos(2.0 * pi * z2[i] / s)
            y2[i] = (b1 + (b2 - b1) * z2[i] / abs_hl) * math.sin(2.0 * pi * z2[i] / s)

        if hl <= 0.0:
            x[i], y[i] = y[i], x[i]
            x2[i], y2[i] = y2[i], x2[i]


