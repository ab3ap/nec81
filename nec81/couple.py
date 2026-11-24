import math
import numpy as npy
from .nec_state import NECState
from .main2 import db10, isegno


def couple(state: NECState, iw, cur: npy.ndarray, wlam: float, ld: int, ld3: int, itag: npy.ndarray):
    """Computes maximum coupling between pairs of segments."""
    if state.nsant != 1 or state.nvqd != 0:
        return
    if state.icoup >= state.ncoup:
        return
    j = isegno(state, int(state.nctag[state.icoup]), int(state.ncseg[state.icoup]), itag)
    if j != int(state.isant[0]):
        return

    state.icoup += 1
    zin = complex(state.vsant[0])
    state.y11a[state.icoup - 1] = complex(cur[j - 1]) * wlam / zin
    l1 = (state.icoup - 1) * (state.ncoup - 1)
    for i in range(1, state.ncoup + 1):
        if i == state.icoup:
            continue
        k = isegno(state, int(state.nctag[i - 1]), int(state.ncseg[i - 1]), itag)
        l1 += 1
        state.y12a[l1 - 1] = complex(cur[k - 1]) * wlam / zin

    if state.icoup < state.ncoup:
        return

    iw.write(
        "\n\n\n                                    - - - ISOLATION DATA - - -\n\n"
        "      - - COUPLING BETWEEN - -        MAXIMUM               - - - FOR MAXIMUM COUPLING - - -\n"
        "            SEG.              SEG.   COUPLING    LOAD IMPEDANCE (2ND SEG.)       INPUT IMPEDANCE\n"
        "  TAG/SEG.   NO.    TAG/SEG.   NO.      (DB)        REAL         IMAG.         REAL         IMAG.\n"
    )
    npm1 = state.ncoup - 1
    for i in range(1, npm1 + 1):
        itt1, its1 = int(state.nctag[i - 1]), int(state.ncseg[i - 1])
        isg1 = isegno(state, itt1, its1, itag)
        for j in range(i + 1, state.ncoup + 1):
            itt2, its2 = int(state.nctag[j - 1]), int(state.ncseg[j - 1])
            isg2 = isegno(state, itt2, its2, itag)
            j1 = j + (i - 1) * npm1 - 2
            j2 = i + (j - 1) * npm1 - 1
            y11 = state.y11a[i - 1]
            y22 = state.y11a[j - 1]
            y12 = 0.5 * (state.y12a[j1] + state.y12a[j2])
            yin = y12 * y12
            dbc = abs(yin)
            den = 2.0 * y11.real * y22.real - yin.real
            c = dbc / den if den != 0.0 else -1.0
            if c < 0.0 or c > 1.0 or dbc == 0.0:
                iw.write(
                    f" {itt1:4d} {its1:4d} {isg1:5d}   {itt2:4d} {its2:4d} {isg2:5d}  "
                    f"**ERROR** COUPLING IS NOT BETWEEN 0 AND 1. (={c:12.5E})\n"
                )
                continue
            gmax = 0.5 * (c + 0.25 * c ** 3) if c < 0.01 else (1.0 - math.sqrt(1.0 - c * c)) / c
            rho = gmax * yin.conjugate() / dbc
            yl = ((1.0 - rho) / (1.0 + rho) + 1.0) * y22.real - y22
            zl = 1.0 / yl
            yin = y11 - yin / (y22 + yl)
            zin = 1.0 / yin
            dbc_db = db10(gmax)
            iw.write(
                f" {itt1:4d} {its1:4d} {isg1:5d}   {itt2:4d} {its2:4d} {isg2:5d}  "
                f"{dbc_db:9.3f}    {zl.real:12.5E} {zl.imag:12.5E}  {zin.real:12.5E} {zin.imag:12.5E}\n"
            )

