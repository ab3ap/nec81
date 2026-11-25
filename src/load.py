import math
import cmath
import numpy as npy
from .necState import NECState

def zint(sigl: float, rmu: float, rolam: float) -> complex:
    """Scalar wrapper for internal circular wire impedance."""
    res = zintVec(sigl, rmu, npy.array([rolam], dtype=npy.float64))
    return complex(res.item(0))


def prnt(
    fl1: float, fl2: float, fl3: float, fl4: float, fl5: float, fl6: float,
    in1: int, in2: int, in3: int, iw, label: str
):
    """Formats and prints a single impedance loading table row."""
    if in1 == 0 and in2 == 0 and in3 == 0:
        intCols = f"{' ALL':>5s}{' '*10}{' '*10}"
    else:
        parts = [
            " "*10 if val == 0 else f"{val:5d}{' '*5}"
            for val in (in1, in2, in3)
        ]
        intCols = "".join(parts)

    fltParts = [
        " "*13 if abs(val) < 1.0e-20 else f"{val:13.4E}"
        for val in (fl1, fl2, fl3, fl4, fl5, fl6)
    ]
    iw.write(f"\n   {intCols}   {''.join(fltParts)}   {label:<20s}\n")


def load(
    state: NECState, zarray, zlr, zli, zlc, si, bi, ld: int,
    itag, ldtyp, ldtag, ldtagf, ldtagt, iw
):
    """Vectorized segment impedance loading calculation."""
    tpcj = complex(0.0, 1.883698955e9)
    iw.write(
        "\n\n       LOCATION          RESISTANCE   INDUCTANCE  CAPACITANCE"
        "       IMPEDANCE (OHMS)     CONDUCTIVITY    TYPE\n"
        "    ITAG FROM THRU          OHMS        HENRYS       FARADS"
        "        REAL      IMAGINARY    MHOS/METER\n"
    )

    zarray[state.n2 - 1:state.n] = 0j
    iwarn = 0

    for istep in range(state.nload):
        ltype = int(ldtyp[istep])
        if ltype > 5:
            iw.write(
                f"\n          IMPROPER LOAD TYPE CHOOSEN,"
                f" REQUESTED TYPE IS {ltype:3d}\n"
            )
            raise ValueError(f"Improper load type {ltype}")

        ldtags = int(ldtag[istep])
        l1, l2 = state.n2, state.n
        if ldtags == 0 and not (ldtagf[istep] == 0 and ldtagt[istep] == 0):
            l1, l2 = int(ldtagf[istep]), int(ldtagt[istep])
            if l1 <= state.n1:
                iw.write(
                    " ERROR - LOADING MAY NOT BE ADDED TO SEGMENTS IN"
                    " N.G.F. SECTION\n"
                )
                raise ValueError("Cannot add loading to N.G.F. section")

        cand = npy.arange(l1 - 1, l2, dtype=npy.intp)
        if ldtags != 0:
            cand = cand[itag[cand] == ldtags]
            if ldtagf[istep] != 0 and len(cand) > 0:
                fIdx = max(int(ldtagf[istep]) - 1, 0)
                tIdx = int(ldtagt[istep])
                cand = cand[fIdx:tIdx]

        if len(cand) == 0:
            iw.write(
                f"\n          LOADING DATA CARD ERROR,"
                f" NO SEGMENT HAS AN ITAG {ldtags:5d}\n"
            )
            raise ValueError(f"No segment has ITAG {ldtags}")

        rV, lV, cV = float(zlr[istep]), float(zli[istep]), float(zlc[istep])
        siS = si[cand]
        biS = bi[cand]

        if ltype == 0:
            zt = rV/siS + tpcj*lV/(siS*state.wlam)
            if abs(cV) > 1.0e-20:
                zt = zt + state.wlam/(tpcj*siS*cV)
        elif ltype == 1:
            zt = tpcj*siS*cV/state.wlam
            if abs(lV) > 1.0e-20:
                zt = zt + siS*state.wlam/(tpcj*lV)
            if abs(rV) > 1.0e-20:
                zt = zt + siS/rV
            zt = 1.0/zt
        elif ltype == 2:
            zt = rV*state.wlam + tpcj*lV
            if abs(cV) > 1.0e-20:
                zt = zt + 1.0/(tpcj*siS*siS*cV)
        elif ltype == 3:
            zt = tpcj*siS*siS*cV
            if abs(lV) > 1.0e-20:
                zt = zt + 1.0/(tpcj*lV)
            if abs(rV) > 1.0e-20:
                zt = zt + 1.0/(rV*state.wlam)
            zt = 1.0/zt
        elif ltype == 4:
            zt = complex(rV, lV)/siS
        else:
            rmu = 1.0 if lV == 0.0 else lV
            zt = zintVec(rV*state.wlam, rmu, biS)

        if npy.any((npy.abs(zarray[cand].real) + npy.abs(zarray[cand].imag)) > 1.0e-20):
            iwarn = 1
        zarray[cand] += zt

        fTag, tTag = int(ldtagf[istep]), int(ldtagt[istep])
        labels = (
            " SERIES             ",
            "PARALLEL            ",
            "SERIES (PER METER)  ",
            "PARALLEL (PER METER)",
            "FIXED IMPEDANCE     ",
            "  WIRE              ",
        )
        if ltype < 4:
            prnt(rV, lV, cV, 0.0, 0.0, 0.0, ldtags, fTag, tTag, iw, labels[ltype])
        elif ltype == 4:
            prnt(0.0, 0.0, 0.0, rV, lV, 0.0, ldtags, fTag, tTag, iw, labels[ltype])
        else:
            prnt(0.0, 0.0, 0.0, 0.0, 0.0, rV, ldtags, fTag, tTag, iw, labels[ltype])

    if iwarn == 1:
        iw.write(
            "\n          NOTE, SOME OF THE ABOVE SEGMENTS HAVE BEEN LOADED"
            " TWICE - IMPEDANCES ADDED\n"
        )
    if state.n1 + 2*state.m1 <= 0:
        nop = state.n//state.np
        if nop > 1:
            zGrid = zarray[:nop*state.np].reshape((nop, state.np))
            zGrid[1:, :] = zGrid[0:1, :]

