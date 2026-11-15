import math
import numpy as npy
from .nec_state import NECState
from .main2 import blckin, blckot


def fbngf(state: NECState, neq: int, neq2: int, iresrv: int, iw):
    """Sets blocking parameters for B, C, and D arrays in N.G.F. solution. Returns (ib11, ic11, id11, ix11)."""
    iresx = iresrv - state.imat
    nbln = neq * neq2
    ndln = neq2 * neq2
    nbcd = 2 * nbln + ndln

    if nbcd <= iresx:
        state.icasx = 1
        ib11 = state.imat + 1
        state.nbbx = state.nbbl = 1
        state.npbx = state.nlbx = neq
        state.npbl = state.nlbl = neq2
    else:
        for unit in (11, 12, 13, 14, 15):
            open(f"fort.{unit}", "wb").close()

        if state.icase >= 3 and nbcd <= iresrv and nbln <= iresx:
            state.icasx = 2
            ib11 = 1
            state.nbbx = state.nbbl = 1
            state.npbx = state.nlbx = neq
            state.npbl = state.nlbl = neq2
        else:
            ir = iresrv if state.icase >= 3 else iresx
            state.icasx = 3
            open("fort.16", "wb").close()
            if ndln > ir:
                state.icasx = 4
            nbcd_row = 2 * neq + neq2
            state.npbl = min(ir // nbcd_row, ir // (2 * neq2))
            if state.icase >= 3:
                state.npbl = min(state.npbl, iresx // neq)
            if state.npbl < 1:
                msg = f" ERROR - INSUFFICIENT STORAGE FOR INTERACTION MATRICIES  IRESRV,IMAT,NEQ,NEQ2 ={iresrv:5d}{state.imat:5d}{neq:5d}{neq2:5d}"
                iw.write(msg + "\n")
                raise MemoryError(msg)

            state.nbbl = (neq2 - 1) // state.npbl
            state.nlbl = neq2 - state.nbbl * state.npbl
            state.nbbl += 1
            nbln = neq * state.npbl
            ir -= nbln
            state.npbx = min(ir // neq2, neq)
            state.nbbx = (neq - 1) // state.npbx
            state.nlbx = neq - state.nbbx * state.npbx
            state.nbbx += 1
            ib11 = 1 if state.icase >= 3 else state.imat + 1

    ic11 = ib11 + nbln
    id11 = ic11 + nbln
    ix11 = state.imat + 1
    iw.write(f"\n\n N.G.F. - NUMBER OF NEW UNKNOWNS IS{neq2:4d}\n")
    if state.icasx != 1:
        iw.write(f" FILE STORAGE FOR NEW MATRIX SECTIONS -  ICASX ={state.icasx:2d}\n")
        iw.write(f" B FILLED BY ROWS-               NO. BLOCKS ={state.nbbx:3d}   ROWS PER BLOCK ={state.npbx:3d}   ROWS IN LAST BLOCK ={state.nlbx:3d}\n")
        iw.write(f" B BY COLUMNS, C AND D BY ROWS -  NO. BLOCKS ={state.nbbl:3d}    R/C PER BLOCK ={state.npbl:3d}    R/C IN LAST BLOCK ={state.nlbl:3d}\n")
    return ib11, ic11, id11, ix11


def fblock(state: NECState, nrow: int, ncol: int, imax: int, irngf: int, ipsym: int, iw):
    """Sets blocking and symmetry parameters for primary interaction matrix (A)."""
    imx1 = imax - irngf
    if nrow * ncol <= imx1:
        state.nbloks = 1
        state.npblk = state.nlast = nrow
        state.imat = nrow * ncol
        state.icase = 1 if nrow == ncol else 2
        if state.icase == 1:
            return
    else:
        for unit in (11, 12, 13, 14):
            open(f"fort.{unit}", "wb").close()

        if nrow == ncol:
            state.icase = 3
            state.npblk = min(imax // (2 * ncol), imx1 // ncol)
            if state.npblk < 1:
                raise MemoryError(f" ERROR - INSUFFICIENT STORAGE FOR MATRIX{nrow:5d}{ncol:5d}")
            state.nbloks = (nrow - 1) // state.npblk
            state.nlast = nrow - state.nbloks * state.npblk
            state.nbloks += 1
            state.nblsym, state.npsym, state.nlsym = state.nbloks, state.npblk, state.nlast
            state.imat = state.npblk * ncol
            iw.write(f"\n\n MATRIX FILE STORAGE-NO. BLOCKS={state.nbloks:5d} COLUMNS PER BLOCK={state.npblk:5d} COLUMNS IN LAST BLOCK={state.nlast:5d}\n")
            return

        state.npblk = min(imax // ncol, nrow)
        if state.npblk < 1:
            raise MemoryError(f" ERROR - INSUFFICIENT STORAGE FOR MATRIX{nrow:5d}{ncol:5d}")
        state.nbloks = (nrow - 1) // state.npblk
        state.nlast = nrow - state.nbloks * state.npblk
        state.nbloks += 1
        iw.write(f"\n\n MATRIX FILE STORAGE-NO. BLOCKS={state.nbloks:5d} COLUMNS PER BLOCK={state.npblk:5d} COLUMNS IN LAST BLOCK={state.nlast:5d}\n")

        if nrow * nrow <= imx1:
            state.icase = 4
            state.nblsym = 1
            state.npsym = state.nlsym = nrow
            state.imat = nrow * nrow
            iw.write(" SUBMATRICIES FIT IN CORE\n")
        else:
            state.icase = 5
            state.npsym = min(imax // (2 * nrow), imx1 // nrow)
            if state.npsym < 1:
                raise MemoryError(f" ERROR - INSUFFICIENT STORAGE FOR MATRIX{nrow:5d}{ncol:5d}")
            state.nblsym = (nrow - 1) // state.npsym
            state.nlsym = nrow - state.nblsym * state.npsym
            state.nblsym += 1
            iw.write(f" SUBMATRIX PARTITIONING-NO. BLOCKS={state.nblsym:5d} COLUMNS PER BLOCK={state.npsym:5d} COLUMNS IN LAST BLOCK={state.nlsym:5d}\n")
            state.imat = state.npsym * nrow

    nop = ncol // nrow
    if nop * nrow != ncol:
        raise ValueError(f" SYMMETRY ERROR - NROW,NCOL={nrow:5d}{ncol:5d}")

    if ipsym <= 0:
        phaz = 6.2831853072 / float(nop)
        for i in range(1, nop):
            for j in range(i, nop):
                val = complex(math.cos(phaz * i * j), math.sin(phaz * i * j))
                state.ssx[i, j] = state.ssx[j, i] = val
    else:
        kk = 1
        state.ssx.fill(0j)
        state.ssx[0,0] = 1.0 + 0j
        if nop not in (2, 4, 8):
            raise ValueError(f"Invalid plane symmetry NOP={nop}")
        ka = 3 if nop == 8 else nop // 2
        for _ in range(ka):
            for i in range(kk):
                for j in range(kk):
                    deter = state.ssx[i, j]
                    state.ssx[i, j + kk] = state.ssx[i + kk, j] = deter
                    state.ssx[i + kk, j + kk] = -deter
            kk *= 2
def gfout(state: NECState, cm, zarray, x, y, z, si, bi, alp, bet, salp, icon1, icon2, itag, ip, iw, igfl: int, ld: int, ld2: int, iresrv: int):
    """Writes Numerical Green's Function (N.G.F.) file."""
    neq = state.n + 2 * state.m
    npeq = state.np + 2 * state.mp
    nop = neq // npeq
    ngf_path = input(f" OPEN UNIT {igfl:2d} FOR N.G.F. OUTPUT FILE: ").strip()

    with open(ngf_path, "wb") as f_ngf:
        npy.array([state.n, state.np, state.m, state.mp], dtype=npy.int32).tofile(f_ngf)
        npy.array([state.wlam, state.fmhz], dtype=npy.float32).tofile(f_ngf)
        npy.array([state.ipsym, state.ksymp, state.iperf, state.nradl], dtype=npy.int32).tofile(f_ngf)
        npy.array([state.epsr, state.sig, state.scrwlt, state.scrwrt], dtype=npy.float32).tofile(f_ngf)
        npy.array([state.nload, state.kcom], dtype=npy.int32).tofile(f_ngf)

        if state.n != 0:
            n = state.n
            for arr in (x, y, z, si, bi, alp, bet, salp):
                npy.asarray(arr[:n], dtype=npy.float64).tofile(f_ngf)
            for arr in (icon1, icon2, itag):
                npy.asarray(arr[:n], dtype=npy.int32).tofile(f_ngf)
            if state.nload > 0:
                npy.asarray(zarray[:n], dtype=npy.complex128).tofile(f_ngf)

        if state.m != 0:
            j = ld - state.m
            for arr in (x, y, z, si, bi, alp, bet, salp):
                npy.asarray(arr[j:ld], dtype=npy.float64).tofile(f_ngf)
            for arr in (icon1, icon2, itag):
                npy.asarray(arr[j:ld], dtype=npy.int32).tofile(f_ngf)

        npy.array(
            [state.icase, state.nbloks, state.npblk, state.nlast, state.nblsym, state.npsym, state.nlsym, state.imat],
            dtype=npy.int32,
        ).tofile(f_ngf)

        if state.iperf == 2:
            state.ar1.flatten(order="F").tofile(f_ngf)
            state.ar2.flatten(order="F").tofile(f_ngf)
            state.ar3.flatten(order="F").tofile(f_ngf)
            npy.array([state.epscf], dtype=npy.complex64).tofile(f_ngf)
            for arr in (state.dxa, state.dya, state.xsa, state.ysa):
                npy.asarray(arr, dtype=npy.float32).tofile(f_ngf)
            for arr in (state.nxa, state.nya):
                npy.asarray(arr, dtype=npy.int32).tofile(f_ngf)

        if nop > 1:
            state.ssx[:nop, :nop].flatten(order="F").tofile(f_ngf)
        npy.asarray(ip[:neq], dtype=npy.int32).tofile(f_ngf)

        com_bytes = bytearray()
        for col in range(5):
            for row in range(19):
                com_bytes.extend(state.com[row][col].ljust(4)[:4].encode("latin1"))
        f_ngf.write(com_bytes)

        if state.icase <= 2:
            cm[: neq * npeq].tofile(f_ngf)
        elif state.icase == 4:
            with open("fort.13", "rb") as f13:
                cnt = npeq * npeq
                for _ in range(nop):
                    chunk = npy.fromfile(f13, dtype=npy.complex64, count=cnt)
                    cm[:cnt] = chunk
                    chunk.tofile(f_ngf)
        else:
            with open("fort.13", "rb") as f13, open("fort.14", "rb") as f14:
                if state.icase != 5:
                    iout = state.npblk * neq * 2
                    for _ in range(state.nbloks):
                        blckin(cm, 1, iout, 1, 201, f13)
                        blckot(cm, 1, iout, 1, 202, f_ngf)
                    for _ in range(state.nbloks):
                        blckin(cm, 1, iout, 1, 203, f14)
                        blckot(cm, 1, iout, 1, 204, f_ngf)
                else:
                    iout = state.npsym * npeq * 2
                    for _ in range(nop):
                        for _ in range(state.nblsym):
                            blckin(cm, 1, iout, 1, 205, f13)
                            blckot(cm, 1, iout, 1, 206, f_ngf)
                        for _ in range(state.nblsym):
                            blckin(cm, 1, iout, 1, 207, f14)
                            blckot(cm, 1, iout, 1, 208, f_ngf)

    msg = f"\n\n\n **** NUMERICAL GREEN'S FUNCTION FILE ON TAPE.{igfl:2d} ****\n        MATRIX STORAGE -{state.imat:7d} COMPLEX NUMBERS\n\n\n"
    iw.write(msg)
    print(msg)

