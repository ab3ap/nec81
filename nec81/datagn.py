import math
import numpy as npy
from .nec_state import NECState, TA
from .main2 import atgn2, blckin, blckot
from .datagn2 import patch, arc, conect, helix

TD = 57.29577951

def parse_geom_card(line: str):
    """Parses a NEC-2 geometry card supporting both strict fixed-column and free-format (comma/space)."""
    raw = line.rstrip("\r\n")
    agm = raw[:2].upper()
    rest = raw[2:].strip()

    def parse_int(s: str) -> int:
        s = s.strip()
        return int(float(s)) if s else 0

    def parse_float(s: str) -> float:
        s = s.strip().replace("D", "E").replace("d", "e")
        return float(s) if s else 0.0

    if "," in rest or "\t" in rest:
        tokens = [t.strip() for t in rest.replace(",", " ").split() if t.strip()]
    else:
        padded = raw.ljust(80)
        fields = [
            padded[2:5], padded[5:10],
            padded[10:20], padded[20:30], padded[30:40],
            padded[40:50], padded[50:60], padded[60:70],
            padded[70:80],
        ]
        try:
            _ = [parse_int(f) for f in fields[:2]]
            _ = [parse_float(f) for f in fields[2:]]
            tokens = fields
        except ValueError:
            tokens = [t.strip() for t in rest.split() if t.strip()]

    t0, t1, t2, t3, t4, t5, t6, t7, t8 = (tokens + [""] * 9)[:9]
    return (
        agm,
        parse_int(t0),
        parse_int(t1),
        parse_float(t2),
        parse_float(t3),
        parse_float(t4),
        parse_float(t5),
        parse_float(t6),
        parse_float(t7),
        parse_float(t8),
    )


def datagn(
    state: NECState,
    cm: npy.ndarray,
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
    icon1: npy.ndarray,
    icon2: npy.ndarray,
    itag: npy.ndarray,
    iconx: npy.ndarray,
    ip: npy.ndarray,
    ld: int,
    ld2: int,
    iresrv: int,
    ir,
    iw,
    igfl: int,
) -> int:
    """Main routine for input and generation of structure geometry data. Returns updated iresrv."""
    afx = [" ", "X"]
    afy = [" ", "Y"]
    afz = [" ", "Z"]
    apt = ["P", "R", "T", "Q"]

    state.patch_polygons = []
    igpflg = 0
    state.ipsym = 0
    nwire = 0
    state.n = 0
    state.np = 0
    state.m = 0
    state.mp = 0
    state.n1 = 0
    state.n2 = 1
    state.m1 = 0
    state.m2 = 1
    isct = 0
    iphd = 0

    npsav = 0
    mpsav = 0
    ipsav = 0
    x3 = y3 = z3 = x4 = y4 = z4 = 0.0

    while True:
        line = ir.readline()
        if not line:
            raise EOFError("Unexpected EOF in geometry input section")
        agm, itg, ns, xw1, yw1, zw1, xw2, yw2, zw2, rad = parse_geom_card(line)

        if state.n + state.m > ld:
            raise RuntimeError(" NUMBER OF WIRE SEGMENTS AND SURFACE PATCHES EXCEEDS DIMENSION LIMIT.")

        if agm == "GF":
            if state.n + state.m != 0:
                raise RuntimeError(" ERROR - GF MUST BE FIRST GEOMETRY DATA CARD")
            gfil(
                state, cm, zarray, x, y, z, t1x, bi, t1y, t1z, salp,
                t2x, t2y, itag, ip, iw, igfl, itg, ld, ld2, iresrv
            )
            npsav = state.np
            mpsav = state.mp
            ipsav = state.ipsym
            continue

        if iphd == 0:
            iw.write(
                "\n\n\n\n                                 - - - STRUCTURE SPECIFICATION - - -\n\n"
                "                                     COORDINATES MUST BE INPUT IN\n"
                "                                     METERS OR BE SCALED TO METERS\n"
                "                                     BEFORE STRUCTURE INPUT IS ENDED\n\n\n"
            )
            iw.write(
                "  WIRE                                                                               NO. OF    FIRST  LAST     TAG\n"
                "  NO.        X1         Y1         Z1          X2         Y2         Z2      RADIUS   SEG.     SEG.   SEG.     NO.\n"
            )
            iphd = 1

        if agm == "SC":
            if isct == 0:
                raise RuntimeError(" PATCH DATA ERROR")
            i1 = state.m + 1
            ns += 1
            if itg != 0 or (ns != 2 and ns != 4):
                raise RuntimeError(" PATCH DATA ERROR")
            xs1, ys1, zs1 = x4, y4, z4
            xs2, ys2, zs2 = x3, y3, z3
            x3, y3, z3 = xw1, yw1, zw1
            if ns == 4:
                x4, y4, z4 = xw2, yw2, zw2
            xw1, yw1, zw1 = xs1, ys1, zs1
            xw2, yw2, zw2 = xs2, ys2, zs2
            if ns != 4:
                x4 = xw1 + x3 - xw2
                y4 = yw1 + y3 - yw2
                z4 = zw1 + z3 - zw2
            iw.write(f" {i1:5d}{apt[ns - 1]:1s}{xw1:10.5f}{yw1:11.5f}{zw1:11.5f} {xw2:11.5f}{yw2:11.5f}{zw2:11.5f}\n")
            iw.write(f"      {x3:11.5f}{y3:11.5f}{z3:11.5f} {x4:11.5f}{y4:11.5f}{z4:11.5f}\n")
            patch(
                state, itg, ns, xw1, yw1, zw1, xw2, yw2, zw2, x3, y3, z3,
                x4, y4, z4, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z,
                ld
            )
            continue

        isct = 0

        if agm == "GP":
            igpflg = 1
            continue

        elif agm == "GW":
            nwire += 1
            i1 = state.n + 1
            i2 = state.n + ns
            iw.write(
                f" {nwire:5d}{xw1:11.5f}{yw1:11.5f}{zw1:11.5f} "
                f"{xw2:11.5f}{yw2:11.5f}{zw2:11.5f}{rad:11.5f}  {ns:5d}    {i1:5d} {i2:5d}   {itg:5d}\n"
            )
            if rad != 0.0:
                xs1 = 1.0
                ys1 = 1.0
            else:
                line2 = ir.readline()
                agm2, _, _, xs1, ys1, zs1, _, _, _, _ = parse_geom_card(line2)
                if agm2 != "GC" or ys1 == 0.0 or zs1 == 0.0:
                    raise RuntimeError(" GEOMETRY DATA CARD ERROR")
                iw.write(
                    f"         ABOVE WIRE IS TAPERED.  SEG. LENGTH RATIO ={xs1:9.5f}\n"
                    f"                                 RADIUS FROM{ys1:9.5f} TO{zs1:9.5f}\n"
                )
                rad = ys1
                ys1 = (zs1 / ys1) ** (1.0 / (ns - 1.0))

            wire(state, x, y, z, bi, t1x, t1y, t1z, xw1, yw1, zw1, xw2, yw2,
                zw2, rad, xs1, ys1, itag, ns, itg)

        elif agm == "GA":
            nwire += 1
            i1 = state.n + 1
            i2 = state.n + ns
            iw.write(
                f" {nwire:5d}  ARC RADIUS ={xw1:9.5f}  FROM{yw1:8.3f} TO{zw1:8.3f} DEGREES"
                f"           {xw2:11.5f}  {ns:5d}    {i1:5d} {i2:5d}   {itg:5d}\n"
            )
            arc(state, itg, ns, xw1, yw1, zw1, xw2, x, y, z, bi, itag, t1x, t1y, t1z)

        elif agm == "GH":
            nwire += 1
            i1 = state.n + 1
            i2 = state.n + ns
            iw.write(
                f"     HELIX STRUCTURE-   AXIAL SPACING BETWEEN TURNS ={xw1:8.3f} TOTAL AXIAL LENGTH ={yw1:8.3f}\n"
                f" {nwire:5d}  RADIUS OF HELIX =  {zw1:8.3f}  {xw2:8.3f}  {yw2:8.3f}  {zw2:8.3f}"
                f"       {rad:11.5f}{ns:8d}    {i1:5d} {i2:5d}   {itg:5d}\n"
            )
            helix(state, iw, xw1, yw1, zw1, xw2, yw2, zw2, rad, ns, itg, x, y, z, bi, itag, t1x, t1y, t1z)

        elif agm == "SP":
            i1 = state.m + 1
            ns += 1
            if itg != 0:
                raise RuntimeError(" PATCH DATA ERROR")
            iw.write(f" {i1:5d}{apt[ns - 1]:1s}{xw1:10.5f}{yw1:11.5f}{zw1:11.5f} {xw2:11.5f}{yw2:11.5f}{zw2:11.5f}\n")
            if ns in (2, 4):
                isct = 1
            if ns <= 1:
                xw2 *= TA
                yw2 *= TA
                patch(
                    state, itg, ns, xw1, yw1, zw1, xw2, yw2, zw2, x3, y3,
                    z3, x4, y4, z4, x, y, z, bi, salp, t1x, t1y, t1z, t2x,
                    t2y, t2z, ld
                )
            else:
                line2 = ir.readline()
                agm2, _, _, x3, y3, z3, x4, y4, z4, _ = parse_geom_card(line2)
                if ns != 2 and itg < 1:
                    pass
                else:
                    x4 = xw1 + x3 - xw2
                    y4 = yw1 + y3 - yw2
                    z4 = zw1 + z3 - zw2
                iw.write(f"      {x3:11.5f}{y3:11.5f}{z3:11.5f} {x4:11.5f}{y4:11.5f}{z4:11.5f}\n")
                if agm2 != "SC":
                    raise RuntimeError(" PATCH DATA ERROR")
                patch(
                    state, itg, ns, xw1, yw1, zw1, xw2, yw2, zw2, x3, y3, z3, x4, y4, z4,
                    x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, ld
                )

        elif agm == "SM":
            i1 = state.m + 1
            iw.write(
                f" {i1:5d}{apt[1]:1s}{xw1:10.5f}{yw1:11.5f}{zw1:11.5f} "
                f"{xw2:11.5f}{yw2:11.5f}{zw2:11.5f}     SURFACE -{itg:4d} BY{ns:3d} PATCHES\n"
            )
            if itg < 1 or ns < 1:
                raise RuntimeError(" PATCH DATA ERROR")
            line2 = ir.readline()
            agm2, _, _, x3, y3, z3, x4, y4, z4, _ = parse_geom_card(line2)
            x4 = xw1 + x3 - xw2
            y4 = yw1 + y3 - yw2
            z4 = zw1 + z3 - zw2
            iw.write(f"      {x3:11.5f}{y3:11.5f}{z3:11.5f} {x4:11.5f}{y4:11.5f}{z4:11.5f}\n")
            if agm2 != "SC":
                raise RuntimeError(" PATCH DATA ERROR")
            patch(
                state, itg, ns, xw1, yw1, zw1, xw2, yw2, zw2, x3, y3, z3, x4, y4, z4,
                x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z, ld
            )

        elif agm in ("GX", "GR"):
            if agm == "GX":
                iy = ns // 10
                iz = ns - iy * 10
                ix = iy // 10
                iy = iy - ix * 10
                ix = 1 if ix != 0 else 0
                iy = 1 if iy != 0 else 0
                iz = 1 if iz != 0 else 0
                iw.write(
                    f"      STRUCTURE REFLECTED ALONG THE AXES {afx[ix]} {afy[iy]} {afz[iz]}"
                    f".TAGS INCREMENTED BY{itg:5d}\n"
                )
            else:
                iw.write(f"      STRUCTURE ROTATED ABOUT Z-AXIS{ns:3d} TIMES.  LABELS INCREMENTED BY{itg:5d}\n")
                ix = -1
                iy = 0
                iz = 0
            reflc(
                state, ix, iy, iz, itg, ns, ld, x, y, z, bi,
                itag, salp, t1x, t1y, t1z, t2x, t2y, t2z, t1x, t1y, t1z
            )

        elif agm == "GS":
            if itg == 1:
                xw1 = 0.3048
            elif itg == 2:
                xw1 = 0.0254
            if state.n >= state.n2:
                s_idx = slice(state.n2 - 1, state.n)
                x[s_idx] *= xw1
                y[s_idx] *= xw1
                z[s_idx] *= xw1
                t1x[s_idx] *= xw1
                t1y[s_idx] *= xw1
                t1z[s_idx] *= xw1
                bi[s_idx] *= xw1
            if state.m >= state.m2:
                yw1 = xw1 * xw1
                ix_m = ld - state.m
                iy_m = ld - state.m1
                x[ix_m:iy_m] *= xw1
                y[ix_m:iy_m] *= xw1
                z[ix_m:iy_m] *= xw1
                bi[ix_m:iy_m] *= yw1
            iw.write(f"      STRUCTURE SCALED BY FACTOR{xw1:10.5f}\n")

        elif agm == "GM":
            iw.write(
                f"      THE STRUCTURE HAS BEEN MOVED, MOVE DATA CARD IS -\n"
                f"      {itg:3d}{ns:5d}{xw1:10.5f}{yw1:10.5f}{zw1:10.5f}"
                f"{xw2:10.5f}{yw2:10.5f}{zw2:10.5f}{rad:10.5f}\n"
            )
            xw1 *= TA
            yw1 *= TA
            zw1 *= TA
            rad_adj = rad + 0.5e-3
            imov1 = int(rad_adj)
            imov2 = int((rad_adj - imov1) * 1.0e3)
            move(
                state, xw1, yw1, zw1, xw2, yw2, zw2, imov1, imov2, ns, itg, ld,
                x, y, z, bi, itag, salp, t1x, t1y, t1z, t2x, t2y, t2z, t1x, t1y, t1z
            )

        elif agm == "GE":
            if xw1 > 0.0:
                state.iresrv = int(xw1)
            if ns != 0:
                state.iplp1 = 1
                state.iplp2 = 2 if ns == 2 else 1

            ix_ngf = state.n1 + state.m1
            if ix_ngf != 0:
                state.np = state.n
                state.mp = state.m
                state.ipsym = 0

            conect(
                state, x, y, z, bi, salp, t1x, t1y, t1z, t2x, t2y, t2z,
                icon1, icon2, iconx, itg, iw, ld
            )

            if ix_ngf != 0:
                state.np = npsav
                state.mp = mpsav
                state.ipsym = ipsav

            if state.n + state.m > ld:
                raise RuntimeError(" NUMBER OF WIRE SEGMENTS AND SURFACE PATCHES EXCEEDS DIMENSION LIMIT.")

            if state.n != 0:
                iw.write(
                    "\n\n\n\n                                - - - - SEGMENTATION DATA - - - -\n\n"
                    "                                        COORDINATES IN METERS\n\n"
                    "                         I+ AND I- INDICATE THE SEGMENTS BEFORE AND AFTER I\n\n"
                )
                iw.write(
                    "  SEG.   COORDINATES OF SEG. CENTER     SEG.     ORIENTATION ANGLES    WIRE    CONNECTION DATA   TAG\n"
                    "  NO.       X         Y         Z       LENGTH     ALPHA     BETA      RADIUS    I-   I    I+    NO.\n"
                )
                for i in range(state.n):
                    dx = t1x[i] - x[i]
                    dy = t1y[i] - y[i]
                    dz = t1z[i] - z[i]
                    x[i] = (x[i] + t1x[i]) * 0.5
                    y[i] = (y[i] + t1y[i]) * 0.5
                    z[i] = (z[i] + t1z[i]) * 0.5
                    slen2 = dx * dx + dy * dy + dz * dz
                    slen = math.sqrt(slen2)
                    slen = (slen2 / slen + slen) * 0.5
                    t1x[i] = slen
                    t1y[i] = dx / slen
                    t1z[i] = dy / slen
                    calp = max(-1.0, min(1.0, dz / slen))
                    salp[i] = calp
                    alpha_deg = math.asin(calp) * TD
                    beta_deg = atgn2(dy, dx) * TD

                    if igpflg != 1:
                        iw.write(
                            f" {i + 1:5d}{x[i]:10.5f}{y[i]:10.5f}{z[i]:10.5f}{t1x[i]:10.5f} "
                            f"{alpha_deg:10.5f}{beta_deg:10.5f}{bi[i]:10.5f} "
                            f"{icon1[i]:5d}{i + 1:5d}{icon2[i]:5d}  {itag[i]:5d}\n"
                        )
                    if state.iplp1 == 1:
                        with open("fort.8", "a") as f8:
                            if state.iplp2 == 2:
                                f8.write(f" {x[i]} {y[i]} {z[i]} {t1x[i]} {i + 1} {itag[i]}\n")
                            else:
                                f8.write(
                                    f" {x[i]} {y[i]} {z[i]} {t1x[i]} {alpha_deg} {beta_deg} "
                                    f"{bi[i]} {icon1[i]} {i + 1} {icon2[i]}\n"
                                )
                    if t1x[i] <= 1.0e-20 or bi[i] <= 0.0:
                        raise RuntimeError(" SEGMENT DATA ERROR")

            if state.m != 0:
                iw.write(
                    "\n\n\n\n                                            - - - SURFACE PATCH DATA - - -\n\n"
                    "                                                 COORDINATES IN METERS\n\n"
                    " PATCH     COORD. OF PATCH CENTER       UNIT NORMAL VECTOR      PATCH            COMPONENTS OF UNIT TANGENT VECTORS\n"
                    "  NO.      X         Y         Z         X       Y       Z       AREA            X1      Y1      Z1       X2      Y2      Z2\n"
                )
                j = ld
                for i in range(state.m):
                    j -= 1
                    nxv = (t1y[j] * t2z[j] - t1z[j] * t2y[j]) * salp[j]
                    nyv = (t1z[j] * t2x[j] - t1x[j] * t2z[j]) * salp[j]
                    nzv = (t1x[j] * t2y[j] - t1y[j] * t2x[j]) * salp[j]
                    if igpflg != 1:
                        iw.write(
                            f" {i + 1:4d}{x[j]:10.5f}{y[j]:10.5f}{z[j]:10.5f} "
                            f"{nxv:8.4f}{nyv:8.4f}{nzv:8.4f}{bi[j]:10.5f} "
                            f"{t1x[j]:8.4f}{t1y[j]:8.4f}{t1z[j]:8.4f} "
                            f"{t2x[j]:8.4f}{t2y[j]:8.4f}{t2z[j]:8.4f}\n"
                        )
            return state.iresrv

        else:
            iw.write(" GEOMETRY DATA CARD ERROR\n")
            raise RuntimeError(f" {agm:2s}{itg:3d}{ns:5d}{xw1:10.5f}{yw1:10.5f}{zw1:10.5f}{xw2:10.5f}{yw2:10.5f}{zw2:10.5f}{rad:10.5f}")
def move(
    state: NECState,
    rox: float,
    roy: float,
    roz: float,
    xs: float,
    ys: float,
    zs: float,
    ixt1: int,
    ixt2: int,
    nrpt: int,
    itgi: int,
    ld: int,
    x: npy.ndarray,
    y: npy.ndarray,
    z: npy.ndarray,
    bi: npy.ndarray,
    itag: npy.ndarray,
    salp: npy.ndarray,
    t1x: npy.ndarray,
    t1y: npy.ndarray,
    t1z: npy.ndarray,
    t2x: npy.ndarray,
    t2y: npy.ndarray,
    t2z: npy.ndarray,
    x2: npy.ndarray,
    y2: npy.ndarray,
    z2: npy.ndarray,
):
    """Moves or reproduces the structure via Euler rotations (rox, roy, roz) and translation (xs, ys, zs)."""
    if abs(rox) + abs(roy) > 1.0e-10:
        state.ipsym *= 3

    sps, cps = math.sin(rox), math.cos(rox)
    sth, cth = math.sin(roy), math.cos(roy)
    sph, cph = math.sin(roz), math.cos(roz)

    xx = cph * cth
    xy = cph * sth * sps - sph * cps
    xz = cph * sth * cps + sph * sps
    yx = sph * cth
    yy = sph * sth * sps + cph * cps
    yz = sph * sth * cps - cph * sps
    zx = -sth
    zy = cth * sps
    zz = cth * cps

    nrp = 1 if nrpt == 0 else nrpt

    if state.n >= state.n2:
        imt1 = ixt1
        imt2 = imt1 if ixt2 == 0 else ixt2
        if imt1 == 0:
            imt2 = 0

        i1 = state.n2
        k = state.n
        for _ in range(nrp):
            for i in range(i1 - 1, state.n):
                if imt1 != 0 and (itag[i] < imt1 or itag[i] > imt2):
                    continue
                k += 1
                k_idx = i if nrpt == 0 else k - 1

                xi, yi, zi = x[i], y[i], z[i]
                x[k_idx] = xi * xx + yi * xy + zi * xz + xs
                y[k_idx] = xi * yx + yi * yy + zi * yz + ys
                z[k_idx] = xi * zx + yi * zy + zi * zz + zs

                xi, yi, zi = x2[i], y2[i], z2[i]
                x2[k_idx] = xi * xx + yi * xy + zi * xz + xs
                y2[k_idx] = xi * yx + yi * yy + zi * yz + ys
                z2[k_idx] = xi * zx + yi * zy + zi * zz + zs

                bi[k_idx] = bi[i]
                itag[k_idx] = itag[i] + itgi if itag[i] != 0 else 0

            i1 = state.n + 1
            if nrpt > 0:
                state.n = k
            imt1 = 0

    if state.m >= state.m2:
        i1 = state.m2
        k = state.m1 if nrpt == 0 else state.m
        for _ in range(nrp):
            for i in range(i1, state.m + 1):
                k += 1
                ir_idx = ld - i
                kr_idx = ld - k

                xi, yi, zi = x[ir_idx], y[ir_idx], z[ir_idx]
                x[kr_idx] = xi * xx + yi * xy + zi * xz + xs
                y[kr_idx] = xi * yx + yi * yy + zi * yz + ys
                z[kr_idx] = xi * zx + yi * zy + zi * zz + zs

                xi, yi, zi = t1x[ir_idx], t1y[ir_idx], t1z[ir_idx]
                t1x[kr_idx] = xi * xx + yi * xy + zi * xz
                t1y[kr_idx] = xi * yx + yi * yy + zi * yz
                t1z[kr_idx] = xi * zx + yi * zy + zi * zz

                xi, yi, zi = t2x[ir_idx], t2y[ir_idx], t2z[ir_idx]
                t2x[kr_idx] = xi * xx + yi * xy + zi * xz
                t2y[kr_idx] = xi * yx + yi * yy + zi * yz
                t2z[kr_idx] = xi * zx + yi * zy + zi * zz

                salp[kr_idx] = salp[ir_idx]
                bi[kr_idx] = bi[ir_idx]
            i1 = state.m + 1
            state.m = k

    if nrpt == 0 and ixt1 == 0:
        return
    state.np = state.n
    state.mp = state.m
    state.ipsym = 0


def wire(
    state: NECState,
    x: npy.ndarray,
    y: npy.ndarray,
    z: npy.ndarray,
    bi: npy.ndarray,
    x2: npy.ndarray,
    y2: npy.ndarray,
    z2: npy.ndarray,
    xw1: float,
    yw1: float,
    zw1: float,
    xw2: float,
    yw2: float,
    zw2: float,
    rad: float,
    rdel: float,
    rrad: float,
    itag: npy.ndarray,
    ns: int,
    itg: int,
):
    """Generates segment geometry data for a straight wire of NS segments."""
    ist = state.n
    state.n += ns
    state.np = state.n
    state.mp = state.m
    state.ipsym = 0
    if ns < 1:
        return

    xd, yd, zd = xw2 - xw1, yw2 - yw1, zw2 - zw1
    if abs(rdel - 1.0) >= 1.0e-6:
        delz = math.sqrt(xd * xd + yd * yd + zd * zd)
        xd /= delz
        yd /= delz
        zd /= delz
        delz = delz * (1.0 - rdel) / (1.0 - rdel ** ns)
        rd = rdel
    else:
        fns = float(ns)
        xd /= fns
        yd /= fns
        zd /= fns
        delz = 1.0
        rd = 1.0

    radz = rad
    xs1, ys1, zs1 = xw1, yw1, zw1
    for i in range(ist, state.n):
        itag[i] = itg
        xs2 = xs1 + xd * delz
        ys2 = ys1 + yd * delz
        zs2 = zs1 + zd * delz
        x[i], y[i], z[i] = xs1, ys1, zs1
        x2[i], y2[i], z2[i] = xs2, ys2, zs2
        bi[i] = radz
        delz *= rd
        radz *= rrad
        xs1, ys1, zs1 = xs2, ys2, zs2

    last = state.n - 1
    x2[last] = xw2
    y2[last] = yw2
    z2[last] = zw2
def reflc(
    state: NECState,
    ix: int,
    iy: int,
    iz: int,
    itx: int,
    nop: int,
    ld: int,
    x: npy.ndarray,
    y: npy.ndarray,
    z: npy.ndarray,
    bi: npy.ndarray,
    itag: npy.ndarray,
    salp: npy.ndarray,
    t1x: npy.ndarray,
    t1y: npy.ndarray,
    t1z: npy.ndarray,
    t2x: npy.ndarray,
    t2y: npy.ndarray,
    t2z: npy.ndarray,
    x2: npy.ndarray,
    y2: npy.ndarray,
    z2: npy.ndarray,
):
    """Reflects structure along X, Y, Z axes or rotates around Z-axis for cylindrical symmetry."""
    state.np = state.n
    state.mp = state.m
    state.ipsym = 0
    iti = itx

    if ix >= 0:
        if nop == 0:
            return
        state.ipsym = 1

        if iz != 0:
            state.ipsym = 2
            if state.n >= state.n2:
                for i in range(state.n2 - 1, state.n):
                    nx = i + state.n - state.n1
                    e1, e2 = z[i], z2[i]
                    if (abs(e1) + abs(e2)) <= 1.0e-5 and (e1 * e2) < -1.0e-6:
                        raise RuntimeError(f" GEOMETRY DATA ERROR--SEGMENT{i + 1:5d} LIES IN PLANE OF SYMMETRY")
                    x[nx], y[nx], z[nx] = x[i], y[i], -e1
                    x2[nx], y2[nx], z2[nx] = x2[i], y2[i], -e2
                    itag[nx] = itag[i] + iti if itag[i] != 0 else 0
                    bi[nx] = bi[i]
                state.n = state.n * 2 - state.n1
                iti *= 2

            if state.m >= state.m2:
                nxx = ld - state.m1
                for i in range(state.m2, state.m + 1):
                    nxx -= 1
                    nx = nxx - state.m + state.m1
                    if abs(z[nxx]) <= 1.0e-10:
                        raise RuntimeError(f" GEOMETRY DATA ERROR--PATCH{i:4d} LIES IN PLANE OF SYMMETRY")
                    x[nx], y[nx], z[nx] = x[nxx], y[nxx], -z[nxx]
                    t1x[nx], t1y[nx], t1z[nx] = t1x[nxx], t1y[nxx], -t1z[nxx]
                    t2x[nx], t2y[nx], t2z[nx] = t2x[nxx], t2y[nxx], -t2z[nxx]
                    salp[nx] = -salp[nxx]
                    bi[nx] = bi[nxx]
                state.m = state.m * 2 - state.m1

        if iy != 0:
            if state.n >= state.n2:
                for i in range(state.n2 - 1, state.n):
                    nx = i + state.n - state.n1
                    e1, e2 = y[i], y2[i]
                    if (abs(e1) + abs(e2)) <= 1.0e-5 and (e1 * e2) < -1.0e-6:
                        raise RuntimeError(f" GEOMETRY DATA ERROR--SEGMENT{i + 1:5d} LIES IN PLANE OF SYMMETRY")
                    x[nx], y[nx], z[nx] = x[i], -e1, z[i]
                    x2[nx], y2[nx], z2[nx] = x2[i], -e2, z2[i]
                    itag[nx] = itag[i] + iti if itag[i] != 0 else 0
                    bi[nx] = bi[i]
                state.n = state.n * 2 - state.n1
                iti *= 2

            if state.m >= state.m2:
                nxx = ld - state.m1
                for i in range(state.m2, state.m + 1):
                    nxx -= 1
                    nx = nxx - state.m + state.m1
                    if abs(y[nxx]) <= 1.0e-10:
                        raise RuntimeError(f" GEOMETRY DATA ERROR--PATCH{i:4d} LIES IN PLANE OF SYMMETRY")
                    x[nx], y[nx], z[nx] = x[nxx], -y[nxx], z[nxx]
                    t1x[nx], t1y[nx], t1z[nx] = t1x[nxx], -t1y[nxx], t1z[nxx]
                    t2x[nx], t2y[nx], t2z[nx] = t2x[nxx], -t2y[nxx], t2z[nxx]
                    salp[nx] = -salp[nxx]
                    bi[nx] = bi[nxx]
                state.m = state.m * 2 - state.m1

        if ix != 0:
            if state.n >= state.n2:
                for i in range(state.n2 - 1, state.n):
                    nx = i + state.n - state.n1
                    e1, e2 = x[i], x2[i]
                    if (abs(e1) + abs(e2)) <= 1.0e-5 and (e1 * e2) < -1.0e-6:
                        raise RuntimeError(f" GEOMETRY DATA ERROR--SEGMENT{i + 1:5d} LIES IN PLANE OF SYMMETRY")
                    x[nx], y[nx], z[nx] = -e1, y[i], z[i]
                    x2[nx], y2[nx], z2[nx] = -e2, y2[i], z2[i]
                    itag[nx] = itag[i] + iti if itag[i] != 0 else 0
                    bi[nx] = bi[i]
                state.n = state.n * 2 - state.n1

            if state.m >= state.m2:
                nxx = ld - state.m1
                for i in range(state.m2, state.m + 1):
                    nxx -= 1
                    nx = nxx - state.m + state.m1
                    if abs(x[nxx]) <= 1.0e-10:
                        raise RuntimeError(f" GEOMETRY DATA ERROR--PATCH{i:4d} LIES IN PLANE OF SYMMETRY")
                    x[nx], y[nx], z[nx] = -x[nxx], y[nxx], z[nxx]
                    t1x[nx], t1y[nx], t1z[nx] = -t1x[nxx], t1y[nxx], t1z[nxx]
                    t2x[nx], t2y[nx], t2z[nx] = -t2x[nxx], t2y[nxx], t2z[nxx]
                    salp[nx] = -salp[nxx]
                    bi[nx] = bi[nxx]
                state.m = state.m * 2 - state.m1
        return

    state.ipsym = -1
    sam = 6.283185308 / float(nop)
    cs, ss = math.cos(sam), math.sin(sam)

    if state.n >= state.n2:
        state.n = state.n1 + (state.n - state.n1) * nop
        for i in range(state.np, state.n):
            k = i - state.np + state.n1
            xk, yk = x[k], y[k]
            x[i], y[i], z[i] = xk * cs - yk * ss, xk * ss + yk * cs, z[k]
            xk, yk = x2[k], y2[k]
            x2[i], y2[i], z2[i] = xk * cs - yk * ss, xk * ss + yk * cs, z2[k]
            itag[i] = itag[k] + iti if itag[k] != 0 else 0
            bi[i] = bi[k]

    if state.m >= state.m2:
        state.m = state.m1 + (state.m - state.m1) * nop
        k = ld - state.m1
        for _ in range(state.mp, state.m):
            k -= 1
            j = k - state.mp + state.m1
            xk, yk = x[k], y[k]
            x[j], y[j], z[j] = xk * cs - yk * ss, xk * ss + yk * cs, z[k]
            xk, yk = t1x[k], t1y[k]
            t1x[j], t1y[j], t1z[j] = xk * cs - yk * ss, xk * ss + yk * cs, t1z[k]
            xk, yk = t2x[k], t2y[k]
            t2x[j], t2y[j], t2z[j] = xk * cs - yk * ss, xk * ss + yk * cs, t2z[k]
            salp[j] = salp[k]
            bi[j] = bi[k]


def gfil(
    state: NECState,
    cm: npy.ndarray,
    zarray: npy.ndarray,
    x: npy.ndarray,
    y: npy.ndarray,
    z: npy.ndarray,
    si: npy.ndarray,
    bi: npy.ndarray,
    alp: npy.ndarray,
    bet: npy.ndarray,
    salp: npy.ndarray,
    icon1: npy.ndarray,
    icon2: npy.ndarray,
    itag: npy.ndarray,
    ip: npy.ndarray,
    iw,
    igfl: int,
    iprt: int,
    ld: int,
    ld2: int,
    iresrv: int,
):
    """Reads Numerical Green's Function (N.G.F.) binary file."""
    ngf_path = input(f" OPEN UNIT {igfl:2d} FOR N.G.F. INPUT  FILE: ").strip()
    with open(ngf_path, "rb") as f_ngf:
        state.n1, state.np, state.m1, state.mp = map(int, npy.fromfile(f_ngf, dtype=npy.int32, count=4))
        state.wlam, state.fmhz = map(float, npy.fromfile(f_ngf, dtype=npy.float32, count=2))
        state.ipsym, state.ksymp, state.iperf, state.nradl = map(int, npy.fromfile(f_ngf, dtype=npy.int32, count=4))
        state.epsr, state.sig, state.scrwlt, state.scrwrt = map(float, npy.fromfile(f_ngf, dtype=npy.float32, count=4))
        state.nlodf, state.kcom = map(int, npy.fromfile(f_ngf, dtype=npy.int32, count=2))

        state.n, state.m = state.n1, state.m1
        state.n2, state.m2 = state.n1 + 1, state.m1 + 1

        if state.n1 != 0:
            n1 = state.n1
            x[:n1] = npy.fromfile(f_ngf, dtype=npy.float64, count=n1)
            y[:n1] = npy.fromfile(f_ngf, dtype=npy.float64, count=n1)
            z[:n1] = npy.fromfile(f_ngf, dtype=npy.float64, count=n1)
            si[:n1] = npy.fromfile(f_ngf, dtype=npy.float64, count=n1)
            bi[:n1] = npy.fromfile(f_ngf, dtype=npy.float64, count=n1)
            alp[:n1] = npy.fromfile(f_ngf, dtype=npy.float64, count=n1)
            bet[:n1] = npy.fromfile(f_ngf, dtype=npy.float64, count=n1)
            salp[:n1] = npy.fromfile(f_ngf, dtype=npy.float64, count=n1)
            icon1[:n1] = npy.fromfile(f_ngf, dtype=npy.int32, count=n1)
            icon2[:n1] = npy.fromfile(f_ngf, dtype=npy.int32, count=n1)
            itag[:n1] = npy.fromfile(f_ngf, dtype=npy.int32, count=n1)
            if state.nlodf != 0:
                zarray[:n1] = npy.fromfile(f_ngf, dtype=npy.complex128, count=n1)

            for i in range(n1):
                xi, yi, zi = x[i] * state.wlam, y[i] * state.wlam, z[i] * state.wlam
                dx = si[i] * 0.5 * state.wlam
                x[i], y[i], z[i] = xi - alp[i] * dx, yi - bet[i] * dx, zi - salp[i] * dx
                si[i], alp[i], bet[i] = xi + alp[i] * dx, yi + bet[i] * dx, zi + salp[i] * dx
                bi[i] *= state.wlam

        if state.m1 != 0:
            j = ld - state.m1
            m1 = state.m1
            for arr in (x, y, z, si, bi, alp, bet, salp):
                arr[j:ld] = npy.fromfile(f_ngf, dtype=npy.float64, count=m1)
            for arr in (icon1, icon2, itag):
                arr[j:ld] = npy.fromfile(f_ngf, dtype=npy.int32, count=m1)
            x[j:ld] *= state.wlam
            y[j:ld] *= state.wlam
            z[j:ld] *= state.wlam
            bi[j:ld] *= state.wlam * state.wlam

        (
            state.icase, state.nbloks, state.npblk, state.nlast,
            state.nblsym, state.npsym, state.nlsym, state.imat
        ) = map(int, npy.fromfile(f_ngf, dtype=npy.int32, count=8))

        if state.iperf == 2:
            state.ar1 = npy.fromfile(f_ngf, dtype=npy.complex64, count=440).reshape((11, 10, 4), order="F")
            state.ar2 = npy.fromfile(f_ngf, dtype=npy.complex64, count=340).reshape((17, 5, 4), order="F")
            state.ar3 = npy.fromfile(f_ngf, dtype=npy.complex64, count=288).reshape((9, 8, 4), order="F")
            state.epscf = complex(npy.fromfile(f_ngf, dtype=npy.complex64, count=1)[0])
            state.dxa = npy.fromfile(f_ngf, dtype=npy.float32, count=3)
            state.dya = npy.fromfile(f_ngf, dtype=npy.float32, count=3)
            state.xsa = npy.fromfile(f_ngf, dtype=npy.float32, count=3)
            state.ysa = npy.fromfile(f_ngf, dtype=npy.float32, count=3)
            state.nxa = npy.fromfile(f_ngf, dtype=npy.int32, count=3)
            state.nya = npy.fromfile(f_ngf, dtype=npy.int32, count=3)

        neq = state.n1 + 2 * state.m1
        npeq = state.np + 2 * state.mp
        nop = neq // npeq
        if nop > 1:
            state.ssx[:nop, :nop] = npy.fromfile(f_ngf, dtype=npy.complex128, count=nop * nop).reshape((nop, nop), order="F")

        ip[:neq] = npy.fromfile(f_ngf, dtype=npy.int32, count=neq)
        com_bytes = f_ngf.read(19 * 5 * 4)
        for col in range(5):
            for row in range(19):
                offset = (col * 19 + row) * 4
                state.com[row][col] = com_bytes[offset : offset + 4].decode("latin1")

        if state.icase <= 2:
            iout = neq * npeq
            state.ensure_matrix_capacity(iout)
            cm[:iout] = npy.fromfile(f_ngf, dtype=npy.complex64, count=iout)
        else:
            with open("fort.13", "wb") as f13:
                if state.icase == 4:
                    iout = npeq * npeq
                    for _ in range(nop):
                        chunk = npy.fromfile(f_ngf, dtype=npy.complex64, count=iout)
                        cm[:iout] = chunk
                        chunk.tofile(f13)
                else:
                    iout = state.npsym * npeq * 2
                    nbl2 = 2 * state.nblsym
                    for _ in range(nop):
                        for _ in range(nbl2):
                            blckin(cm, 1, iout, 1, 206, f_ngf)
                            blckot(cm, 1, iout, 1, 205, f13)

    star_bar = "     " + "*" * 84 + "\n"
    blank_box = "     **" + " " * 80 + "**\n"
    iw.write("\n" * 16 + star_bar * 2 + blank_box)
    iw.write(f"     ** NUMERICAL GREEN'S FUNCTION{' ' * 53}**\n")
    iw.write(f"     ** NO. SEGMENTS ={state.n1:4d}          NO. PATCHES ={state.m1:4d}{' ' * 34}**\n")
    if nop > 1:
        iw.write(f"     ** NO. SYMMETRIC SECTIONS ={nop:4d}{' ' * 51}**\n")
    iw.write(f"     ** N.G.F. MATRIX -  CORE STORAGE ={state.imat:7d} COMPLEX NUMBERS,  CASE{state.icase:2d}{' ' * 16}**\n")
    if state.icase >= 3:
        iw.write(f"     **{' ' * 19}MATRIX SIZE ={neq * npeq:7d} COMPLEX NUMBERS{' ' * 25}**\n")
    iw.write(f"     ** FREQUENCY ={state.fmhz:12.5E} MHZ.{' ' * 51}**\n")
    if state.ksymp == 2 and state.iperf == 1:
        iw.write(f"     ** PERFECT GROUND{' ' * 65}**\n")
    if state.ksymp == 2 and state.iperf == 0:
        iw.write(f"     ** FINITE GROUND.  REFLECTION COEFFICIENT APPROXIMATION{' ' * 27}**\n")
    if state.ksymp == 2 and state.iperf == 2:
        iw.write(f"     ** FINITE GROUND.  SOMMERFELD SOLUTION{' ' * 44}**\n")
    if state.ksymp == 2 and state.iperf != 1:
        iw.write(f"     ** GROUND PARAMETERS - DIELECTRIC CONSTANT ={state.epsr:12.5E}{' ' * 26}**\n")
        iw.write(f"     **{' ' * 21}CONDUCTIVITY ={state.sig:12.5E} MHOS/M.{' ' * 25}**\n")
    iw.write(blank_box)
    for j in range(state.kcom):
        line_str = "".join(state.com[i][j] for i in range(19))
        iw.write(f"     ** {line_str:<76s}   **\n")
    iw.write(blank_box + star_bar * 2 + "\n" * 4)

    if iprt != 0:
        iw.write(
            "                                       NUMERICAL GREEN'S FUNCTION DATA\n"
            "                                         COORDINATES OF SEGMENT ENDS\n"
            "                                                   (METERS)\n"
            "     SEG.           - - - END ONE - - -                          - - - END TWO - - -\n"
            "      NO.      X              Y              Z              X              Y              Z\n"
        )
        for i in range(state.n1):
            iw.write(f" {i + 1:7d}{x[i]:15.6E}{y[i]:15.6E}{z[i]:15.6E}{si[i]:15.6E}{alp[i]:15.6E}{bet[i]:15.6E}\n")

