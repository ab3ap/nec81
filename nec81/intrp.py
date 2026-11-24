import numpy as npy
from .nec_state import NECState

_INTRP_CACHE = {
    "ixs": -10,
    "iys": -10,
    "igrs": -10,
    "dx": 1.0,
    "dy": 1.0,
    "xs": 0.0,
    "ys": 0.0,
    "ixeg": 0,
    "iyeg": 0,
    "a": npy.zeros((4, 4), dtype=npy.complex128),
    "b": npy.zeros((4, 4), dtype=npy.complex128),
    "c": npy.zeros((4, 4), dtype=npy.complex128),
    "d": npy.zeros((4, 4), dtype=npy.complex128),
}


def intrp(state: NECState, x: float, y: float):
    """Bivariate cubic interpolation in Sommerfeld ground tables. Returns (f1, f2, f3, f4)."""
    c = _INTRP_CACHE
    recompute = True

    if x >= c["xs"] and y >= c["ys"]:
        ix = int((x - c["xs"]) / c["dx"]) + 1
        iy = int((y - c["ys"]) / c["dy"]) + 1
        if ix >= c["ixeg"] and iy >= c["iyeg"]:
            if abs(ix - c["ixs"]) < 2 and abs(iy - c["iys"]) < 2:
                recompute = False

    if recompute:
        # Fortran: IF (X.GT.XS2) where XS2 is EQUIVALENCEd to XSA(2), the
        # start of grid 2 (R=0.2 wavelength).  Using XSA(1)=0 sent every
        # near-ground point to grids 2/3 and extrapolated them.
        if x <= float(state.xsa[1]):
            igr = 1
        else:
            igr = 3 if y > float(state.ysa[2]) else 2

        g_idx = igr - 1
        if igr != c["igrs"]:
            c["igrs"] = igr
            c["dx"] = float(state.dxa[g_idx])
            c["dy"] = float(state.dya[g_idx])
            c["xs"] = float(state.xsa[g_idx])
            c["ys"] = float(state.ysa[g_idx])

        nxm2 = int(state.nxa[g_idx]) - 2
        nym2 = int(state.nya[g_idx]) - 2
        nxms = ((nxm2 + 1) // 3) * 3 + 1
        nyms = ((nym2 + 1) // 3) * 3 + 1
        ix = int((x - c["xs"]) / c["dx"]) + 1
        iy = int((y - c["ys"]) / c["dy"]) + 1

        ixs = max(((ix - 1) // 3) * 3 + 2, 2)
        c["ixeg"] = -10000
        if ixs > nxm2:
            ixs = nxm2
            c["ixeg"] = nxms
        c["ixs"] = ixs

        iys = max(((iy - 1) // 3) * 3 + 2, 2)
        c["iyeg"] = -10000
        if iys > nym2:
            iys = nym2
            c["iyeg"] = nyms
        c["iys"] = iys

        grid = state.ar1 if igr == 1 else (state.ar2 if igr == 2 else state.ar3)
        for k in range(4):
            for i in range(4):
                row_idx = ixs - 2
                col_idx = iys - 2 + i
                p1 = complex(grid[row_idx, col_idx, k])
                p2 = complex(grid[row_idx + 1, col_idx, k])
                p3 = complex(grid[row_idx + 2, col_idx, k])
                p4 = complex(grid[row_idx + 3, col_idx, k])
                c["a"][i, k] = (p4 - p1 + 3.0 * (p2 - p3)) * 0.1666666667
                c["b"][i, k] = (p1 - 2.0 * p2 + p3) * 0.5
                c["c"][i, k] = p3 - (2.0 * p1 + 3.0 * p2 + p4) * 0.1666666667
                c["d"][i, k] = p2

    xz = (c["ixs"] - 1) * c["dx"] + c["xs"]
    yz = (c["iys"] - 1) * c["dy"] + c["ys"]
    xx = (x - xz) / c["dx"]
    yy = (y - yz) / c["dy"]

    out = []
    for k in range(4):
        fx = ((c["a"][:, k] * xx + c["b"][:, k]) * xx + c["c"][:, k]) * xx + c["d"][:, k]
        p1 = fx[3] - fx[0] + 3.0 * (fx[1] - fx[2])
        p2 = 3.0 * (fx[0] - 2.0 * fx[1] + fx[2])
        p3 = 6.0 * fx[2] - 2.0 * fx[0] - 3.0 * fx[1] - fx[3]
        out.append(((p1 * yy + p2) * yy + p3) * yy * 0.1666666667 + fx[1])

    return out[0], out[1], out[2], out[3]

