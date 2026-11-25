import numpy as npy
from .necState import NECState
from .main2 import blckin, blckot, second
from .factr import factr, solves


def factrs(
    state: NECState, a: npy.ndarray, scratc: npy.ndarray, npSym: int,
    nrow: int, ip: npy.ndarray, ix: npy.ndarray, iu1: int, iu2: int,
    iu3: int, iu4: int, ld2: int, iresrv: int,
):
    """Factors primary interaction matrix (in-core or out-of-core)."""
    nop = nrow//npSym
    if state.icase <= 2:
        aMat = a[:nrow*npSym].reshape((nrow, npSym), order="F")
        for kk in range(nop):
            ka = kk*npSym
            factr(
                aMat[ka:ka + npSym, :npSym], scratc, npSym, nrow,
                ip[ka:ka + npSym], ld2
            )
        return

    if state.icase <= 3:
        facio(state, a, scratc, nrow, nop, ix, iu1, iu2, iu3, iu4, ld2)
        lunscr(state, a, nrow, nop, ip, ix, iu2, iu3, iu4)
        return

    # Rewrite matrices by columns onto unit IU2
    i2 = 2*state.npblk*nrow
    aBlk = a[:nrow*state.npblk].reshape((nrow, state.npblk), order="F")
    with open(f"fort.{iu2}", "wb") as f2:
        for k in range(nop):
            with open(f"fort.{iu1}", "rb") as f1:
                icols = state.npblk
                ir1 = k*npSym
                ir2 = ir1 + npSym
                for lBlk in range(1, state.nbloks + 1):
                    if not (state.nbloks == 1 and k > 0):
                        blckin(a, 1, i2, 1, 602, f1)
                    if lBlk == state.nbloks:
                        icols = state.nlast
                    aBlk[ir1:ir2, :icols].flatten(order="F").tofile(f2)

    if state.icase != 5:
        cnt = npSym*npSym
        aSub = a[:cnt].reshape((npSym, npSym), order="F")
        with open(f"fort.{iu2}", "rb") as f2, open(f"fort.{iu3}", "wb") as f3:
            for kk in range(nop):
                a[:cnt] = npy.fromfile(f2, dtype=a.dtype, count=cnt)
                ka = kk*npSym
                factr(aSub, scratc, npSym, npSym, ip[ka:ka + npSym], ld2)
                a[:cnt].tofile(f3)
        return

    i2 = 2*state.npsym*npSym
    aSym = a[:npSym*state.npsym].reshape((npSym, state.npsym), order="F")
    with open(f"fort.{iu2}", "rb") as f2, open(f"fort.{iu1}", "wb") as f1:
        for _ in range(nop):
            for lBlk in range(1, state.nblsym + 1):
                j2 = state.nlsym if lBlk == state.nblsym else state.npsym
                aSym[:, :j2] = npy.fromfile(
                    f2, dtype=a.dtype, count=npSym*j2
                ).reshape((npSym, j2), order="F")
                blckot(a, 1, i2, 1, 193, f1)

    facio(state, a, scratc, npSym, nop, ix, iu1, iu2, iu3, iu4, ld2)
    lunscr(state, a, npSym, nop, ip, ix, iu2, iu3, iu4)


def reblk(
    state: NECState, b: npy.ndarray, bx: npy.ndarray, nb: int,
    nbx: int, n2c: int
):
    """Reblocks N.G.F. matrix B from rows on fort.14 to columns on fort.16."""
    bMat = b[:nb*state.npbl].reshape((nb, state.npbl), order="F")
    bxMat = bx[:nbx*n2c].reshape((nbx, n2c), order="F")
    with open("fort.16", "wb") as f16:
        nib = 0
        npb = state.npbl
        for ib in range(1, state.nbbl + 1):
            if ib == state.nbbl:
                npb = state.nlbl
            with open("fort.14", "rb") as f14:
                nix = 0
                npx = state.npbx
                for ibx in range(1, state.nbbx + 1):
                    if ibx == state.nbbx:
                        npx = state.nlbx
                    chunk = npy.fromfile(
                        f14, dtype=bx.dtype, count=npx*n2c
                    ).reshape((npx, n2c), order="F")
                    bxMat[:npx, :n2c] = chunk
                    bMat[nix:nix + npx, :npb] = bxMat[:npx, nib:nib + npb]
                    nix += state.npbx
            bMat[:nb, :npb].flatten(order="F").tofile(f16)
            nib += state.npbl


def facio(
    state: NECState, a: npy.ndarray, d: npy.ndarray, nrow: int, nop: int,
    ip: npy.ndarray, iu1: int, iu2: int, iu3: int, iu4: int, ld2: int
):
    """Controls out-of-core I/O passes for block Gauss-Doolittle."""
    it = 2*state.npsym*nrow
    nbm = state.nblsym - 1
    i1, i2, i3, i4 = 1, it, it + 1, 2*it
    elapsed = 0.0

    f1 = open(f"fort.{iu1}", "rb")
    f2 = open(f"fort.{iu2}", "wb")

    for kk in range(nop):
        ka = kk*nrow
        curInUnit = iu1
        curIn = f1
        for ixblk1 in range(1, nbm + 1):
            outUnit = iu3 if (ixblk1%2 == 1) else iu4
            fOut = open(f"fort.{outUnit}", "wb")
            blckin(a, i1, i2, 1, 17, curIn)
            ixbp = ixblk1 + 1
            for ixblk2 in range(ixbp, state.nblsym + 1):
                blckin(a, i3, i4, 1, 18, curIn)
                t1 = second()
                lfactr(
                    state, a, d, nrow, ixblk1, ixblk2,
                    ip[ka:ka + nrow], ld2
                )
                elapsed += second() - t1
                if ixblk2 == ixbp:
                    blckot(a, i1, i2, 1, 19, f2)
                if ixblk1 == nbm and ixblk2 == state.nblsym:
                    blckot(a, i3, i4, 1, 20, f2)
                else:
                    blckot(a, i3, i4, 1, 20, fOut)
            if curIn is not f1:
                curIn.close()
            fOut.close()
            curInUnit = outUnit
            curIn = open(f"fort.{curInUnit}", "rb")

        if curIn is not f1:
            curIn.close()

    f1.close()
    f2.close()
    print(f" CP TIME TAKEN FOR FACTORIZATION = {elapsed:12.5E}")
def lunscr(
    state: NECState, a: npy.ndarray, nrow: int, nop: int,
    ix: npy.ndarray, ip: npy.ndarray, iu2: int, iu3: int, iu4: int
):
    """Unscrambles out-of-core factored blocks and reverses block order."""
    i2 = 2*state.npsym*nrow
    nm1 = nrow - 1
    aMat = a[:nrow*state.npsym].reshape((nrow, state.npsym), order="F")

    with open(f"fort.{iu2}", "rb") as f2, open(f"fort.{iu4}", "wb") as f4:
        for kk in range(nop):
            ka = kk*nrow
            modeBlocks = []
            for ixblk1 in range(1, state.nblsym + 1):
                blckin(a, 1, i2, 1, 121, f2)
                k1 = (ixblk1 - 1)*state.npsym + 2
                if nm1 >= k1:
                    j2 = 0
                    for k in range(k1 - 1, nm1):
                        if j2 < state.npsym:
                            j2 += 1
                        ipk = int(ip[k + ka]) - 1
                        aMat[[k, ipk], :j2] = aMat[[ipk, k], :j2]
                modeBlocks.append(a[:nrow*state.npsym].copy())

            for blk in reversed(modeBlocks):
                a[:nrow*state.npsym] = blk
                blckot(a, 1, i2, 1, 124, f4)

            ix[ka:ka + nrow] = npy.arange(1, nrow + 1, dtype=ix.dtype)
            for i in range(nrow):
                ipi = int(ip[i + ka]) - 1
                ix[i + ka], ix[ipi + ka] = ix[ipi + ka], ix[i + ka]


def lfactr(
    state: NECState, a: npy.ndarray, d: npy.ndarray, nrow: int,
    ix1: int, ix2: int, ip: npy.ndarray, ld2: int
):
    """Performs block Gauss-Doolittle elimination on two in-core blocks."""
    aMat = a[:nrow*2*state.npsym].reshape((nrow, 2*state.npsym), order="F")
    l1 = (ix1 == 1 and ix2 == 2)
    l2 = ((ix2 - 1) == ix1)
    l3 = (ix2 == state.nblsym)

    if l1:
        r1, r2, j1, j2 = 0, 2*state.npsym, 0, -1
    else:
        r1, r2 = state.npsym, 2*state.npsym
        j1 = (ix1 - 1)*state.npsym
        j2 = j1 + state.npsym - 2 if l2 else j1 + state.npsym - 1
    if l3:
        r2 = state.npsym + state.nlsym

    for r in range(r1, r2):
        d[j1:nrow] = aMat[j1:nrow, r]
        if l1 or l2:
            j2 += 1
        if j1 <= j2:
            for ixj, j in enumerate(range(j1, j2 + 1)):
                pj = int(ip[j]) - 1
                ajr = d[pj]
                aMat[j, r] = ajr
                d[pj] = d[j]
                if j + 1 < nrow:
                    d[j + 1:nrow] -= aMat[j + 1:nrow, ixj]*ajr

        j2p1 = j2 + 1
        if not (l1 or l2):
            if nrow >= j2p1 + 1:
                aMat[j2p1:nrow, r] = d[j2p1:nrow]
            continue

        mags = (d[j2p1:nrow]*d[j2p1:nrow].conjugate()).real
        relIdx = int(npy.argmax(mags))
        maxVal = mags[relIdx]
        ties = npy.where(mags >= maxVal)
        if len(ties) > 0:
            relIdx = int(ties[-1])
        dmax = float(mags[relIdx])
        ip[j2p1] = j2p1 + relIdx + 1

        if dmax < 1.0e-10:
            print(f" PIVOT({j2 + 1:3d})={dmax:16.8E}")

        pr = int(ip[j2p1]) - 1
        aMat[j2p1, r] = d[pr]
        d[pr] = d[j2p1]
        if j2p1 + 1 < nrow:
            ajr = 1.0/aMat[j2p1, r]
            aMat[j2p1 + 1:nrow, r] = d[j2p1 + 1:nrow]*ajr


def facgf(
    state: NECState, a, b, c, d, bx, scratc, ip, ix, npSym, n1,
    mpSym, m1, n1c, n2c, ld2, iresrv
):
    """Computes and factors Schur complement D - C*(inv(A)*B) for N.G.F."""
    if n2c == 0:
        return
    if state.icasx >= 3:
        reblk(state, b, c, n1c, state.npbx, n2c)
        ibfl = 16
    else:
        ibfl = 14

    npb = state.npbl
    fIn = open(f"fort.{ibfl}", "rb") if state.icasx > 1 else None
    f14Out = open("fort.14", "wb") if state.icasx > 1 else None
    bxMat = bx[:n1c*state.npbl].reshape((n1c, state.npbl), order="F")

    for ib in range(1, state.nbbl + 1):
        if ib == state.nbbl:
            npb = state.nlbl
        if state.icasx > 1:
            bxMat[:, :npb] = npy.fromfile(
                fIn, dtype=bx.dtype, count=n1c*npb
            ).reshape((n1c, npb), order="F")
        solves(
            state, a, bx, scratc, n1c, npSym, n1, mpSym, m1,
            ip, npb, 13, 13, ld2, iresrv
        )
        if state.icasx > 1:
            bxMat[:, :npb].flatten(order="F").tofile(f14Out)

    if fIn is not None:
        fIn.close()
    if f14Out is not None:
        f14Out.close()
    cMat = c[:n1c*state.npbl].reshape((n1c, state.npbl), order="F")
    dMat = d[:n2c*state.npbl].reshape((n2c, state.npbl), order="F")
    bMat = b[:n1c*state.npbl].reshape((n1c, state.npbl), order="F")

    f15 = open("fort.15", "rb") if state.icasx != 1 else None
    f12 = open("fort.12", "rb") if state.icasx != 1 else None
    f11 = open("fort.11", "wb") if state.icasx != 1 else None

    npc = state.npbl
    for ic in range(1, state.nbbl + 1):
        if ic == state.nbbl:
            npc = state.nlbl
        if state.icasx != 1:
            cMat[:, :npc] = npy.fromfile(
                f15, dtype=c.dtype, count=n1c*npc
            ).reshape((n1c, npc), order="F")
            dMat[:, :npc] = npy.fromfile(
                f12, dtype=d.dtype, count=n2c*npc
            ).reshape((n2c, npc), order="F")

        f14In = open("fort.14", "rb") if state.icasx > 1 else None
        npb = state.npbl
        nic = 0
        for ib in range(1, state.nbbl + 1):
            if ib == state.nbbl:
                npb = state.nlbl
            if state.icasx > 1:
                bMat[:, :npb] = npy.fromfile(
                    f14In, dtype=b.dtype, count=n1c*npb
                ).reshape((n1c, npb), order="F")
            dMat[nic:nic + npb, :npc] -= (
                bMat[:, :npb].T@cMat[:, :npc]
            ).astype(d.dtype)
            nic += state.npbl
        if f14In is not None:
            f14In.close()
        if state.icasx > 1:
            dMat[:, :state.npbl].flatten(order="F").tofile(f11)

    for fp in (f15, f12, f11):
        if fp is not None:
            fp.close()

    if state.icasx <= 1:
        factr(
            d[:n2c*n2c].reshape((n2c, n2c), order="F"),
            scratc, n2c, n2c, ip[n1c:n1c + n2c], ld2
        )
    elif state.icasx != 4:
        with open("fort.11", "rb") as f11R:
            npb = state.npbl
            icOff = 0
            for ib in range(1, state.nbbl + 1):
                if ib == state.nbbl:
                    npb = state.nlbl
                cnt = n2c*npb
                b[icOff:icOff + cnt] = npy.fromfile(
                    f11R, dtype=b.dtype, count=cnt
                )
                icOff += cnt
        factr(
            b[:n2c*n2c].reshape((n2c, n2c), order="F"),
            scratc, n2c, n2c, ip[n1c:n1c + n2c], ld2
        )
        with open("fort.11", "wb") as f11W:
            b[:n2c*n2c].tofile(f11W)
    else:
        nblsys = state.nblsym
        npsys = state.npsym
        nlsys = state.nlsym
        icass = state.icase
        state.nblsym = state.nbbl
        state.npsym = state.npbl
        state.nlsym = state.nlbl
        state.icase = 3
        facio(state, b, scratc, n2c, 1, ix[n1c:], 11, 12, 16, 11, ld2)
        lunscr(state, b, n2c, 1, ip[n1c:], ix[n1c:], 12, 11, 16)
        state.nblsym = nblsys
        state.npsym = npsys
        state.nlsym = nlsys
        state.icase = icass

