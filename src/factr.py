import numpy as npy
from scipy.linalg import lu_factor, lu_solve
from .necState import NECState
from .main2 import blckin


def factr(
    a: npy.ndarray, d: npy.ndarray, n: int, ndim: int,
    ip: npy.ndarray, ld2: int
):
    """Factors matrix A(ndim, n) using LAPACK LU decomposition."""
    if n <= 0:
        return
    lu, piv = lu_factor(
        a[:n, :n].astype(npy.complex128),
        overwrite_a=False,
        check_finite=False,
    )
    a[:n, :n] = lu
    ip[:n] = piv + 1

    diagMag = (npy.diag(lu)*npy.diag(lu).conjugate()).real
    smallIdx = npy.where(diagMag < 1.0e-10)[0]
    for idx in smallIdx:
        print(f" PIVOT({idx + 1:3d})={diagMag[idx]:16.8E}")

def solve(
    a: npy.ndarray, b: npy.ndarray, y: npy.ndarray, n: int,
    ndim: int, ip: npy.ndarray, ld2: int
):
    """Solves LU * X = B using double-precision LAPACK lu_solve."""
    if n <= 0:
        return
    piv = ip[:n].astype(npy.int32) - 1
    lu64 = a[:n, :n].astype(npy.complex128)
    if b.ndim == 1:
        b[:n] = lu_solve(
            (lu64, piv), b[:n].astype(npy.complex128),
            trans=1, overwrite_b=False, check_finite=False
        )
    else:
        b[:n, :] = lu_solve(
            (lu64, piv), b[:n, :].astype(npy.complex128),
            trans=1, overwrite_b=False, check_finite=False
        )

def solves(
    state: NECState, a: npy.ndarray, b: npy.ndarray, y: npy.ndarray,
    neq: int, npSym: int, n: int, mpSym: int, m: int, ip: npy.ndarray,
    nrh: int, ifl1: int, ifl2: int, ld2: int, iresrv: int,
):
    """Handles symmetry mode transforms and solves A * X = B."""
    bMat = b[:neq*nrh].reshape((neq, nrh), order="F")
    npeq = npSym + 2*mpSym
    nop = neq//npeq
    fnorm = 1.0/float(nop)
    nrow = npeq if state.icase > 3 else neq

    if nop != 1:
        if n != 0 and m != 0:
            pass
        # j + k*npeq in Fortran order -> shape (npeq, nop, nrh), order='F'
        bModes = bMat[:neq, :].reshape((npeq, nop, nrh), order="F")
        ssxConj = state.ssx[:nop, :nop].conjugate().copy()
        ssxConj[0, :] = 1.0
        ssxConj[:, 0] = 1.0
        bModes[:] = npy.einsum("kl,jlc->jkc", ssxConj, bModes)*fnorm

    f1 = open(f"fort.{ifl1}", "rb") if state.icase >= 3 else None
    f2 = open(f"fort.{ifl2}", "rb") if state.icase >= 3 else None
    aMat = (
        a[:nrow*npeq].reshape((nrow, npeq), order="F")
        if state.icase != 4 else None
    )

    for kk in range(nop):
        ia = kk*npeq
        if state.icase == 4:
            cnt = npeq*npeq
            a[:cnt] = npy.fromfile(f1, dtype=a.dtype, count=cnt)
            aSub = a[:cnt].reshape((nrow, npeq), order="F")
        else:
            aSub = aMat[ia:ia + npeq, :npeq]

        if state.icase in (3, 5):
            ltsolv(
                state, a, bMat[ia:ia + npeq, :], y, ip[ia:ia + npeq],
                npeq, neq, nrh, f1, f2, ld2
            )
        else:
            solve(
                aSub, bMat[ia:ia + npeq, :nrh], y, npeq, nrow,
                ip[ia:ia + npeq], ld2
            )

    if f1 is not None:
        f1.close()
    if f2 is not None:
        f2.close()

    if nop == 1:
        return

    bModes = bMat[:neq, :].reshape((npeq, nop, nrh), order="F")
    ssxFwd = state.ssx[:nop, :nop].copy()
    ssxFwd[0, :] = 1.0
    ssxFwd[:, 0] = 1.0
    bModes[:] = npy.einsum("kl,jlc->jkc", ssxFwd, bModes)

    if n != 0 and m != 0:
        kk = 2*mpSym
        temp = bMat[:neq, :].copy()
        ia, ib, j = npSym, n, npSym
        for k in range(1, nop + 1):
            if k != 1:
                bMat[ia:ia + npSym, :] = temp[j:j + npSym, :]
                ia += npSym
                j += npSym
            if k != nop:
                bMat[ib:ib + kk, :] = temp[j:j + kk, :]
                ib += kk
                j += kk

def ltsolv(
    state: NECState, a: npy.ndarray, b: npy.ndarray, y: npy.ndarray,
    ix: npy.ndarray, nrow: int, neq: int, nrh: int, f1, f2, ld2: int
):
    """Out-of-core forward and backward substitution for LU(T) in blocks."""
    i2 = 2*state.npsym*nrow
    aMat = a[:nrow*state.npsym].reshape((nrow, state.npsym), order="F")

    for ixblk1 in range(1, state.nblsym + 1):
        blckin(a, 1, i2, 1, 121, f1)
        k2 = state.nlsym if ixblk1 == state.nblsym else state.npsym
        jst = (ixblk1 - 1)*state.npsym
        for k in range(k2):
            j = jst + k
            if j >= 1:
                sVal = aMat[:j, k]@b[:j, :nrh]
                b[j, :nrh] = (b[j, :nrh] - sVal)/aMat[j, k]
            else:
                b[j, :nrh] = b[j, :nrh]/aMat[j, k]

    jst = nrow
    for ixblk1 in range(1, state.nblsym + 1):
        blckin(a, 1, i2, 1, 122, f2)
        k2 = state.nlsym if ixblk1 == 1 else state.npsym
        kp = k2
        j = jst
        for _ in range(k2):
            kp -= 1
            jp1 = j
            j -= 1
            if nrow >= jp1 + 1:
                sVal = aMat[jp1:nrow, kp]@b[jp1:nrow, :nrh]
                b[j, :nrh] -= sVal
        jst -= k2

    perm = ix[:nrow].astype(npy.intp) - 1
    b[perm, :nrh] = b[:nrow, :nrh].copy()

