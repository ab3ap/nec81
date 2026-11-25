import numpy as npy
from .necState import NECState

_BICUBIC_CACHE = {
    "gridId": None,
    "tensors": None,
}

# Catmull-Rom / cubic Lagrange basis matrix M such that
# [x^3, x^2, x, 1] @ M @ [p1, p2, p3, p4].
_M_CUBIC = npy.array(
    [
        [-1.0/6.0, 0.5, -0.5, 1.0/6.0],
        [0.5, -1.0, 0.5, 0.0],
        [-1.0/3.0, -0.5, 1.0, -1.0/6.0],
        [0.0, 1.0, 0.0, 0.0],
    ],
    dtype=npy.float64,
)

def _buildBicubicTensors(state: NECState):
    """Precomputes 4x4 bicubic tensors and grid scalars for ar1..ar3."""
    grids = (state.ar1, state.ar2, state.ar3)
    tensors = []
    params = []
    for gIdx, rawGrid in enumerate(grids):
        grid = rawGrid.astype(npy.complex128)
        nx = int(state.nxa.item(gIdx))
        ny = int(state.nya.item(gIdx))
        nCellX = max(nx - 3, 1)
        nCellY = max(ny - 3, 1)

        patches = npy.zeros((nCellX, nCellY, 4, 4, 4), dtype=npy.complex128)
        for ixOff in range(4):
            for iyOff in range(4):
                patches[:, :, ixOff, iyOff, :] = grid[
                    ixOff:ixOff + nCellX, iyOff:iyOff + nCellY, :
                ]

        cTensor = npy.einsum(
            "pi,xyijk,qj->xypqk", _M_CUBIC, patches, _M_CUBIC
        )
        tensors.append(cTensor)
        params.append((
            float(state.dxa.item(gIdx)),
            float(state.dya.item(gIdx)),
            float(state.xsa.item(gIdx)),
            float(state.ysa.item(gIdx)),
            nx - 2,
            ny - 2,
        ))
    bounds = (float(state.xsa.item(0)), float(state.ysa.item(2)))
    return tensors, params, bounds


def intrpVec(state: NECState, x: npy.ndarray, y: npy.ndarray):
    """Fast bicubic tensor interpolation in Sommerfeld ground tables."""
    nPts = len(x)
    f1 = npy.zeros(nPts, dtype=npy.complex128)
    f2 = npy.zeros(nPts, dtype=npy.complex128)
    f3 = npy.zeros(nPts, dtype=npy.complex128)
    f4 = npy.zeros(nPts, dtype=npy.complex128)
    if nPts == 0:
        return f1, f2, f3, f4

    if _BICUBIC_CACHE["gridId"] != id(state.ar1):
        _BICUBIC_CACHE["data"] = _buildBicubicTensors(state)
        _BICUBIC_CACHE["gridId"] = id(state.ar1)

    tensors, params, (xsa0, ysa2) = _BICUBIC_CACHE["data"]
    m1 = (x <= xsa0)
    m3 = (~m1) & (y > ysa2)
    m2 = (~m1) & (~m3)

    for gIdx, mask in enumerate((m1, m2, m3)):
        if not npy.any(mask):
            continue
        xm = x[mask]
        ym = y[mask]
        dx, dy, xs, ys, nxm2, nym2 = params[gIdx]

        ix = ((xm - xs)/dx).astype(npy.intp) + 1
        iy = ((ym - ys)/dy).astype(npy.intp) + 1
        ixs = npy.minimum(npy.maximum(((ix - 1)//3)*3 + 2, 2), nxm2)
        iys = npy.minimum(npy.maximum(((iy - 1)//3)*3 + 2, 2), nym2)

        xx = (xm - ((ixs - 1)*dx + xs))/dx
        yy = (ym - ((iys - 1)*dy + ys))/dy

        cCell = tensors[gIdx][ixs - 2, iys - 2]
        xxB = xx[:, None, None]
        cy = (
            ((cCell[:, 0]*xxB + cCell[:, 1])*xxB + cCell[:, 2])*xxB
            + cCell[:, 3]
        )
        yyB = yy[:, None]
        res = ((cy[:, 0]*yyB + cy[:, 1])*yyB + cy[:, 2])*yyB + cy[:, 3]

        f1[mask], f2[mask], f3[mask], f4[mask] = (
            res[:, 0], res[:, 1], res[:, 2], res[:, 3]
        )

    return f1, f2, f3, f4

def intrp(state: NECState, x: float, y: float):
    """Scalar bivariate cubic interpolation wrapper."""
    xa = npy.array([x], dtype=npy.float64)
    ya = npy.array([y], dtype=npy.float64)
    f1, f2, f3, f4 = intrpVec(state, xa, ya)
    return (
        complex(f1.item(0)), complex(f2.item(0)),
        complex(f3.item(0)), complex(f4.item(0))
    )

