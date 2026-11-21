import math
import cmath
import numpy as npy
from .nec_state import NECState


def zint(sigl: float, rmu: float, rolam: float) -> complex:
    """Computes internal impedance of a circular wire with relative permeability RMU."""
    pi, pot, tp, tpcmu, cmotp = 3.1415926, 1.5707963, 6.2831853, 2.368705e3, 60.0
    fj = complex(0.0, 1.0)
    cn = complex(0.70710678, 0.70710678)

    cc = [
        complex(6.0e-7, 1.9e-6), complex(-3.4e-6, 5.1e-6), complex(-2.52e-5, 0.0),
        complex(-9.06e-5, -9.01e-5), complex(0.0, -9.765e-4), complex(0.0110486, -0.0110485),
        complex(0.0, -0.3926991), complex(1.6e-6, -3.2e-6), complex(1.17e-5, -2.4e-6),
        complex(3.46e-5, 3.38e-5), complex(5.0e-7, 2.452e-4), complex(-1.3813e-3, 1.3811e-3),
        complex(-6.25001e-2, -1.0e-7), complex(0.7071068, 0.7071068),
    ]

    def th(d: float) -> complex:
        res = cc[0]
        for c in cc[1:7]:
            res = res * d + c
        return res

    def ph(d: float) -> complex:
        res = cc[7]
        for c in cc[8:14]:
            res = res * d + c
        return res

    x = math.sqrt(tpcmu * rmu * sigl) * rolam
    if x > 110.0:
        br1 = complex(0.70710678, -0.70710678)
    elif x > 8.0:
        f_x = math.sqrt(pot / x) * cmath.exp(-cn * x + th(-8.0 / x))
        g_x = cmath.exp(cn * x + th(8.0 / x)) / math.sqrt(tp * x)
        br2 = fj * f_x / pi
        br1 = (g_x + br2) / (g_x * ph(8.0 / x) - br2 * ph(-8.0 / x))
    else:
        y = (x / 8.0) ** 2
        s = y * y
        ber1 = ((((((-9.01e-6 * s + 1.22552e-3) * s - 0.08349609) * s + 2.6419140) * s - 32.363456) * s + 113.77778) * s - 64.0) * s + 1.0
        bei1 = ((((((1.1346e-4 * s - 0.01103667) * s + 0.52185615) * s - 10.567658) * s + 72.817777) * s - 113.77778) * s + 16.0) * y
        ber2 = (((((((-3.94e-6 * s + 4.5957e-4) * s - 0.02609253) * s + 0.66047849) * s - 6.0681481) * s + 14.222222) * s - 4.0) * y) * x
        bei2 = ((((((4.609e-5 * s - 3.79386e-3) * s + 0.14677204) * s - 2.3116751) * s + 11.377778) * s - 10.666667) * s + 0.5) * x
        br1 = complex(ber1, bei1) / complex(ber2, bei2)

    return fj * math.sqrt(cmotp * rmu / sigl) * br1 / rolam


def prnt(fl1: float, fl2: float, fl3: float, fl4: float, fl5: float, fl6: float, in1: int, in2: int, in3: int, iw, label: str):
    """Formats and prints a single impedance loading table row."""
    if in1 == 0 and in2 == 0 and in3 == 0:
        int_cols = f"{' ALL':>5s}{' ' * 10}{' ' * 10}"
    else:
        parts = [" " * 10 if val == 0 else f"{val:5d}{' ' * 5}" for val in (in1, in2, in3)]
        int_cols = "".join(parts)

    flt_parts = [" " * 13 if abs(val) < 1.0e-20 else f"{val:13.4E}" for val in (fl1, fl2, fl3, fl4, fl5, fl6)]
    iw.write(f"\n   {int_cols}   {''.join(flt_parts)}   {label:<20s}\n")


def load(state: NECState, zarray, zlr, zli, zlc, si, bi, ld: int, itag, ldtyp, ldtag, ldtagf, ldtagt, iw):
    """Calculates segment impedance loading for series, parallel, fixed Z, and wire conductivity/permeability."""
    tpcj = complex(0.0, 1.883698955e9)
    iw.write(
        "\n\n       LOCATION          RESISTANCE   INDUCTANCE  CAPACITANCE       IMPEDANCE (OHMS)     CONDUCTIVITY    TYPE\n"
        "    ITAG FROM THRU          OHMS        HENRYS       FARADS        REAL      IMAGINARY    MHOS/METER\n"
    )

    zarray[state.n2 - 1 : state.n] = 0j
    iwarn = 0

    for istep in range(state.nload):
        ltype = int(ldtyp[istep])
        if ltype > 5:
            iw.write(f"\n          IMPROPER LOAD TYPE CHOOSEN, REQUESTED TYPE IS {ltype:3d}\n")
            raise ValueError(f"Improper load type {ltype}")

        ldtags = int(ldtag[istep])
        ichk = 0
        l1, l2 = state.n2, state.n
        if ldtags == 0 and not (ldtagf[istep] == 0 and ldtagt[istep] == 0):
            l1, l2 = int(ldtagf[istep]), int(ldtagt[istep])
            if l1 <= state.n1:
                iw.write(" ERROR - LOADING MAY NOT BE ADDED TO SEGMENTS IN N.G.F. SECTION\n")
                raise ValueError("Cannot add loading to N.G.F. section")

        for i in range(l1 - 1, l2):
            if ldtags != 0:
                if ldtags != itag[i]:
                    continue
                if ldtagf[istep] != 0:
                    ichk += 1
                    if ichk < ldtagf[istep] or ichk > ldtagt[istep]:
                        continue
                else:
                    ichk = 1
            else:
                ichk = 1

            r_v, l_v, c_v = zlr[istep], zli[istep], zlc[istep]
            if ltype == 0:
                zt = r_v / si[i] + tpcj * l_v / (si[i] * state.wlam)
                if abs(c_v) > 1.0e-20:
                    zt += state.wlam / (tpcj * si[i] * c_v)
            elif ltype == 1:
                zt = tpcj * si[i] * c_v / state.wlam
                if abs(l_v) > 1.0e-20:
                    zt += si[i] * state.wlam / (tpcj * l_v)
                if abs(r_v) > 1.0e-20:
                    zt += si[i] / r_v
                zt = 1.0 / zt
            elif ltype == 2:
                zt = r_v * state.wlam + tpcj * l_v
                if abs(c_v) > 1.0e-20:
                    zt += 1.0 / (tpcj * si[i] * si[i] * c_v)
            elif ltype == 3:
                zt = tpcj * si[i] * si[i] * c_v
                if abs(l_v) > 1.0e-20:
                    zt += 1.0 / (tpcj * l_v)
                if abs(r_v) > 1.0e-20:
                    zt += 1.0 / (r_v * state.wlam)
                zt = 1.0 / zt
            elif ltype == 4:
                zt = complex(r_v, l_v) / si[i]
            else:
                rmu = 1.0 if l_v == 0.0 else l_v
                zt = zint(r_v * state.wlam, rmu, bi[i])

            if (abs(zarray[i].real) + abs(zarray[i].imag)) > 1.0e-20:
                iwarn = 1
            zarray[i] += zt

        if ichk == 0:
            iw.write(f"\n          LOADING DATA CARD ERROR, NO SEGMENT HAS AN ITAG {ldtags:5d}\n")
            raise ValueError(f"No segment has ITAG {ldtags}")

        f_tag, t_tag = int(ldtagf[istep]), int(ldtagt[istep])
        labels = [
            " SERIES             ",
            "PARALLEL            ",
            "SERIES (PER METER)  ",
            "PARALLEL (PER METER)",
            "FIXED IMPEDANCE     ",
            "  WIRE              ",
        ]
        if ltype < 4:
            prnt(zlr[istep], zli[istep], zlc[istep], 0.0, 0.0, 0.0, ldtags, f_tag, t_tag, iw, labels[ltype])
        elif ltype == 4:
            prnt(0.0, 0.0, 0.0, zlr[istep], zli[istep], 0.0, ldtags, f_tag, t_tag, iw, labels[ltype])
        else:
            prnt(0.0, 0.0, 0.0, 0.0, 0.0, zlr[istep], ldtags, f_tag, t_tag, iw, labels[ltype])

    if iwarn == 1:
        iw.write("\n          NOTE, SOME OF THE ABOVE SEGMENTS HAVE BEEN LOADED TWICE - IMPEDANCES ADDED\n")
    if state.n1 + 2 * state.m1 <= 0:
        nop = state.n // state.np
        if nop > 1:
            for i in range(state.np):
                zt = zarray[i]
                for l2_idx in range(1, nop):
                    zarray[i + l2_idx * state.np] = zt

