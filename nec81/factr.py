import numpy as npy
from .nec_state import NECState
from .main2 import blckin


def factr(a: npy.ndarray, d: npy.ndarray, n: int, ndim: int, ip: npy.ndarray, ld2: int):
    """Factors matrix A(ndim, n) into unit lower and upper triangular matrices (Gauss-Doolittle, transposed)."""
    for r in range(n):
        d[:n] = a[r, :n]

        for j in range(r):
            pj = int(ip[j]) - 1
            arj = d[pj]
            a[r, j] = arj
            d[pj] = d[j]
            if j + 1 < n:
                d[j + 1 : n] -= a[j, j + 1 : n] * arj

        dmax = (d[r] * d[r].conjugate()).real
        ip[r] = r + 1
        for i in range(r + 1, n):
            elmag = (d[i] * d[i].conjugate()).real
            if elmag >= dmax:
                dmax = elmag
                ip[r] = i + 1

        if dmax < 1.0e-10:
            print(f" PIVOT({r + 1:3d})={dmax:16.8E}")

        pr = int(ip[r]) - 1
        a[r, r] = d[pr]
        d[pr] = d[r]

        if r + 1 < n:
            arj = 1.0 / a[r, r]
            a[r, r + 1 : n] = d[r + 1 : n] * arj


def solve(a: npy.ndarray, b: npy.ndarray, y: npy.ndarray, n: int, ndim: int, ip: npy.ndarray, ld2: int):
    """Solves LU * X = B where L and U are stored transposed in A(ndim, n). Solution returned in B."""
    for i in range(n):
        pi = int(ip[i]) - 1
        y[i] = b[pi]
        b[pi] = b[i]
        if i + 1 < n:
            b[i + 1 : n] -= a[i, i + 1 : n] * y[i]

    for i in range(n - 1, -1, -1):
        s_val = npy.sum(a[i + 1 : n, i] * b[i + 1 : n]) if i + 1 < n else 0j
        b[i] = (y[i] - s_val) / a[i, i]
def solves(
    state: NECState,
    a: npy.ndarray,
    b: npy.ndarray,
    y: npy.ndarray,
    neq: int,
    np_sym: int,
    n: int,
    mp_sym: int,
    m: int,
    ip: npy.ndarray,
    nrh: int,
    ifl1: int,
    ifl2: int,
    ld2: int,
    iresrv: int,
):
    """Handles symmetry mode transformations and solves the matrix equation A * X = B."""
    b_mat = b[: neq * nrh].reshape((neq, nrh), order="F")
    npeq = np_sym + 2 * mp_sym
    nop = neq // npeq
    fnorm = 1.0 / float(nop)
    nrow = npeq if state.icase > 3 else neq

    if nop != 1:
        for ic in range(nrh):
            if n != 0 and m != 0:
                y[:neq] = b_mat[:neq, ic]
                kk = 2 * mp_sym
                ia, ib, j = np_sym, n, np_sym
                for k in range(1, nop + 1):
                    if k != 1:
                        b_mat[j : j + np_sym, ic] = y[ia : ia + np_sym]
                        ia += np_sym
                        j += np_sym
                    if k != nop:
                        b_mat[j : j + kk, ic] = y[ib : ib + kk]
                        ib += kk
                        j += kk

            for i in range(npeq):
                for k in range(nop):
                    y[k] = b_mat[i + k * npeq, ic]
                b_mat[i, ic] = npy.sum(y[:nop]) * fnorm
                for k in range(1, nop):
                    ia = i + k * npeq
                    s_val = y[0] + npy.sum(y[1:nop] * state.ssx[k, 1:nop].conjugate())
                    b_mat[ia, ic] = s_val * fnorm

    f1 = open(f"fort.{ifl1}", "rb") if state.icase >= 3 else None
    f2 = open(f"fort.{ifl2}", "rb") if state.icase >= 3 else None

    for kk in range(nop):
        ia = kk * npeq
        if state.icase == 4:
            cnt = npeq * npeq
            a[:cnt] = npy.fromfile(f1, dtype=a.dtype, count=cnt)
            a_sub = a[:cnt].reshape((nrow, npeq), order="F")
        else:
            a_sub = a[ia : ia + nrow * npeq].reshape((nrow, npeq), order="F")

        if state.icase in (3, 5):
            ltsolv(state, a, b_mat[ia : ia + npeq, :], y, ip[ia : ia + npeq], npeq, neq, nrh, f1, f2, ld2)
        else:
            for ic in range(nrh):
                solve(a_sub, b_mat[ia : ia + npeq, ic], y, npeq, nrow, ip[ia : ia + npeq], ld2)

    if f1 is not None:
        f1.close()
    if f2 is not None:
        f2.close()

    if nop == 1:
        return

    for ic in range(nrh):
        for i in range(npeq):
            for k in range(nop):
                y[k] = b_mat[i + k * npeq, ic]
            b_mat[i, ic] = npy.sum(y[:nop])
            for k in range(1, nop):
                ia = i + k * npeq
                b_mat[ia, ic] = y[0] + npy.sum(y[1:nop] * state.ssx[k, 1:nop])

        if n != 0 and m != 0:
            y[:neq] = b_mat[:neq, ic]
            kk = 2 * mp_sym
            ia, ib, j = np_sym, n, np_sym
            for k in range(1, nop + 1):
                if k != 1:
                    b_mat[ia : ia + np_sym, ic] = y[j : j + np_sym]
                    ia += np_sym
                    j += np_sym
                if k != nop:
                    b_mat[ib : ib + kk, ic] = y[j : j + kk]
                    ib += kk
                    j += kk


def ltsolv(state: NECState, a: npy.ndarray, b: npy.ndarray, y: npy.ndarray, ix: npy.ndarray, nrow: int, neq: int, nrh: int, f1, f2, ld2: int):
    """Out-of-core forward and backward substitution for LU(T) stored in blocks."""
    i2 = 2 * state.npsym * nrow
    a_mat = a[: nrow * state.npsym].reshape((nrow, state.npsym), order="F")

    for ixblk1 in range(1, state.nblsym + 1):
        blckin(a, 1, i2, 1, 121, f1)
        k2 = state.nlsym if ixblk1 == state.nblsym else state.npsym
        jst = (ixblk1 - 1) * state.npsym
        for ic in range(nrh):
            j = jst
            for k in range(k2):
                jm1 = j
                s_val = npy.sum(a_mat[:jm1, k] * b[:jm1, ic]) if jm1 >= 1 else 0j
                b[j, ic] = (b[j, ic] - s_val) / a_mat[j, k]
                j += 1

    jst = nrow
    for ixblk1 in range(1, state.nblsym + 1):
        blckin(a, 1, i2, 1, 122, f2)
        k2 = state.nlsym if ixblk1 == 1 else state.npsym
        for ic in range(nrh):
            kp = k2
            j = jst
            for _ in range(k2):
                kp -= 1
                jp1 = j
                j -= 1
                if nrow >= jp1 + 1:
                    s_val = npy.sum(a_mat[jp1:nrow, kp] * b[jp1:nrow, ic])
                    b[j, ic] -= s_val
        jst -= k2

    for ic in range(nrh):
        for i in range(nrow):
            y[int(ix[i]) - 1] = b[i, ic]
        b[:nrow, ic] = y[:nrow]

