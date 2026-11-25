import math
import cmath
import numpy as npy
from .cmset import efldVec
from .necState import NECState, TA
from .main2 import cang
from .hintg import _set_ext_wire_flags, unere, hsfld, hintg
from .eksc import eksc, ekscx, rom2, sflds
from .plots3d import recordScalarField

RPD = 57.29577951

def nfpat(
    state: NECState, x, y, z, si, bi, salp,
    t1x, t1y, t1z, t2x, t2y, t2z,
    icon1, icon2, air, aii, bir, bii, cir, cii, cur, iw, ld: int, ld3: int
):
    """Computes near E or H fields over a 3D rectangular or spherical grid."""
    if state.nfeh == 1:
        iw.write(
            "\n\n\n                                   - - - NEAR MAGNETIC"
            " FIELDS - - -\n\n"
            "            -  LOCATION  -                     -  HX  -          "
            "     -  HY  -               -  HZ  -          - PEAK FIELDS -\n"
            "        X          Y          Z          MAGNITUDE   PHASE      "
            "MAGNITUDE   PHASE      MAGNITUDE   PHASE         MAGNITUDE\n"
            "      METERS     METERS     METERS        AMPS/M   DEGREES       "
            "AMPS/M   DEGREES       AMPS/M   DEGREES          AMPS/M\n"
        )
    else:
        iw.write(
            "\n\n\n                                   - - - NEAR ELECTRIC"
            " FIELDS - - -\n\n"
            "            -  LOCATION  -                     -  EX  -          "
            "     -  EY  -               -  EZ  -          - PEAK FIELDS -\n"
            "        X          Y          Z          MAGNITUDE   PHASE      "
            "MAGNITUDE   PHASE      MAGNITUDE   PHASE         MAGNITUDE\n"
            "      METERS     METERS     METERS       VOLTS/M   DEGREES      "
            "VOLTS/M   DEGREES      VOLTS/M   DEGREES         VOLTS/M\n"
        )

    xobAll, yobAll, zobAll = _buildObsGrid(state)
    nPts = len(xobAll)
    if nPts == 0:
        return

    invWlam = 1.0/state.wlam
    if state.nfeh == 1:
        exAll = npy.zeros(nPts, dtype=npy.complex128)
        eyAll = npy.zeros(nPts, dtype=npy.complex128)
        ezAll = npy.zeros(nPts, dtype=npy.complex128)
        for idx in range(nPts):
            exAll[idx], eyAll[idx], ezAll[idx] = nhfld(
                state, float(xobAll[idx])*invWlam,
                float(yobAll[idx])*invWlam, float(zobAll[idx])*invWlam,
                ld, ld3, x, y, z, si, bi, salp, t1y, t1z,
                t1x, t1y, t1z, t2x, t2y, t2z,
                air, aii, bir, bii, cir, cii, cur
            )
    else:
        exAll, eyAll, ezAll = _nefldGrid(
            state, xobAll*invWlam, yobAll*invWlam, zobAll*invWlam,
            ld, ld3, x, y, z, si, bi, salp, t1y, t1z,
            t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2,
            air, aii, bir, bii, cir, cii, cur
        )

    magX = npy.abs(exAll)
    magY = npy.abs(eyAll)
    magZ = npy.abs(ezAll)
    phX = npy.degrees(npy.angle(exAll))
    phY = npy.degrees(npy.angle(eyAll))
    phZ = npy.degrees(npy.angle(ezAll))

    ex2, ey2, ez2 = magX*magX, magY*magY, magZ*magZ
    at = (
        ez2*npy.cos(2.0*npy.angle(ezAll))
        + ey2*npy.cos(2.0*npy.angle(eyAll))
        + ex2*npy.cos(2.0*npy.angle(exAll))
    )
    bt = (
        ez2*npy.sin(2.0*npy.angle(ezAll))
        + ey2*npy.sin(2.0*npy.angle(eyAll))
        + ex2*npy.sin(2.0*npy.angle(exAll))
    )
    etotAll = npy.sqrt(
        0.5*(ez2 + ey2 + ex2) + 0.5*npy.sqrt(at*at + bt*bt)
    )

    fieldName = (
        "Near Magnetic Field" if state.nfeh == 1
        else "Near Electric Field"
    )
    fieldUnit = "A/m" if state.nfeh == 1 else "V/m"

    for idx in range(nPts):
        xob = float(xobAll[idx])
        yob = float(yobAll[idx])
        zob = float(zobAll[idx])
        tmp1, tmp2 = float(magX[idx]), float(phX[idx])
        tmp3, tmp4 = float(magY[idx]), float(phY[idx])
        tmp5, tmp6 = float(magZ[idx]), float(phZ[idx])
        etotal = float(etotAll[idx])

        iw.write(
            f"  {xob:11.4f}{yob:11.4f}{zob:11.4f} "
            f"   {tmp1:11.4E}  {tmp2:7.2f}   {tmp3:11.4E}"
            f"  {tmp4:7.2f}   {tmp5:11.4E}  {tmp6:7.2f}"
            f"      {etotal:11.4E}\n"
        )
        recordScalarField(fieldName, xob, yob, zob, etotal, fieldUnit)

        if state.iplp1 == 2:
            ex, ey, ez = exAll[idx], eyAll[idx], ezAll[idx]
            xxx = (
                xob if state.iplp4 == 1
                else (yob if state.iplp4 == 2 else zob)
            )
            with open("fort.8", "a") as f8:
                if state.iplp2 == 2:
                    if state.iplp3 == 1:
                        f8.write(f" {xxx} {tmp1} {tmp2}\n")
                    elif state.iplp3 == 2:
                        f8.write(f" {xxx} {tmp3} {tmp4}\n")
                    elif state.iplp3 == 3:
                        f8.write(f" {xxx} {tmp5} {tmp6}\n")
                    elif state.iplp3 == 4:
                        f8.write(
                            f" {xxx} {tmp1} {tmp2} {tmp3}"
                            f" {tmp4} {tmp5} {tmp6}\n"
                        )
                    elif state.iplp3 == 5:
                        f8.write(f" {xxx} {etotal}\n")
                elif state.iplp2 == 1:
                    if state.iplp3 == 1:
                        f8.write(f" ({ex.real},{ex.imag})\n")
                    elif state.iplp3 == 2:
                        f8.write(f" ({ey.real},{ey.imag})\n")
                    elif state.iplp3 == 3:
                        f8.write(f" ({ez.real},{ez.imag})\n")
                    elif state.iplp3 == 4:
                        f8.write(
                            f" ({ex.real},{ex.imag})"
                            f" ({ey.real},{ey.imag})"
                            f" ({ez.real},{ez.imag})\n"
                        )

def _buildObsGrid(state: NECState):
    """Builds flattened (xob, yob, zob) arrays for all 3D near-field points."""
    xVals = state.xnr + npy.arange(state.nrx, dtype=npy.float64)*state.dxnr
    yVals = state.ynr + npy.arange(state.nry, dtype=npy.float64)*state.dynr
    zVals = state.znr + npy.arange(state.nrz, dtype=npy.float64)*state.dznr
    # Preserve exact NEC-81 loop nesting: outer Z, middle Y, inner X
    zG, yG, xG = npy.meshgrid(zVals, yVals, xVals, indexing="ij")
    xn = xG.ravel()
    yn = yG.ravel()
    zn = zG.ravel()
    if state.near != 0:
        cth = npy.cos(TA*zn)
        sth = npy.sin(TA*zn)
        cph = npy.cos(TA*yn)
        sph = npy.sin(TA*yn)
        return xn*sth*cph, xn*sth*sph, xn*cth
    return xn, yn, zn


def _nefldGrid(
    state: NECState, xobA: npy.ndarray, yobA: npy.ndarray, zobA: npy.ndarray,
    ld: int, ld3: int, x, y, z, si, bi, salp, cab, sab,
    t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2,
    air, aii, bir, bii, cir, cii, cur
):
    """Vectorized near electric field across all grid observation points."""
    nPts = len(xobA)
    ex = npy.zeros(nPts, dtype=npy.complex128)
    ey = npy.zeros(nPts, dtype=npy.complex128)
    ez = npy.zeros(nPts, dtype=npy.complex128)
    nSeg = state.n

    if nSeg != 0:
        axRad = npy.zeros(nPts, dtype=npy.float64)
        for i in range(nSeg):
            xj = xobA - x[i]
            yj = yobA - y[i]
            zj = zobA - z[i]
            zp = cab[i]*xj + sab[i]*yj + salp[i]*zj
            mHit = (
                (axRad == 0.0)
                & (npy.abs(zp) <= 0.5001*si[i])
                & ((xj*xj + yj*yj + zj*zj - zp*zp) <= 0.9*bi[i]*bi[i])
            )
            if npy.any(mHit):
                axRad[mHit] = bi[i]

        ijOnes = npy.ones(nPts, dtype=npy.int32)
        for i in range(nSeg):
            state.s, state.b = si[i], bi[i]
            state.xj, state.yj, state.zj = x[i], y[i], z[i]
            state.cabj, state.sabj, state.salpj = cab[i], sab[i], salp[i]
            _set_ext_wire_flags(state, i + 1, cab, sab, salp, bi, icon1, icon2)
            exk, eyk, ezk, exs, eys, ezs, exc, eyc, ezc = efldVec(
                state, xobA, yobA, zobA, axRad, ijOnes
            )
            acx = complex(air[i], aii[i])
            bcx = complex(bir[i], bii[i])
            ccx = complex(cir[i], cii[i])
            ex += exk*acx + exs*bcx + exc*ccx
            ey += eyk*acx + eys*bcx + eyc*ccx
            ez += ezk*acx + ezs*bcx + ezc*ccx

    if state.m != 0:
        for pIdx in range(nPts):
            px, py, pz = float(xobA[pIdx]), float(yobA[pIdx]), float(zobA[pIdx])
            jc, jl = nSeg - 3, ld
            for _ in range(state.m):
                jl -= 1
                jc += 3
                state.s = bi[jl]
                state.xj, state.yj, state.zj = x[jl], y[jl], z[jl]
                state.cabj, state.sabj, state.salpj = t1x[jl], t1y[jl], t1z[jl]
                state.b, state.ind1, state.ind2 = t2x[jl], t2y[jl], t2z[jl]
                cx = complex(cur[jc])
                cy = complex(cur[jc + 1])
                cz = complex(cur[jc + 2])
                acx = t1x[jl]*cx + t1y[jl]*cy + t1z[jl]*cz
                bcx = t2x[jl]*cx + t2y[jl]*cy + t2z[jl]*cz
                for ip in range(1, state.ksymp + 1):
                    state.ipgnd = ip
                    unere(state, px, py, pz)
                    ex[pIdx] += acx*state.exk + bcx*state.exs
                    ey[pIdx] += acx*state.eyk + bcx*state.eys
                    ez[pIdx] += acx*state.ezk + bcx*state.ezs

    return ex, ey, ez



def nefld(
    state: NECState, xob, yob, zob, ld, ld3, x, y, z, si, bi, salp,
    cab, sab, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2,
    air, aii, bir, bii, cir, cii, cur
):
    """Computes near electric field (ex, ey, ez) at (xob, yob, zob)."""
    ex = ey = ez = 0j
    axRad = 0.0
    nSeg = state.n

    if nSeg != 0:
        xjA = xob - x[:nSeg]
        yjA = yob - y[:nSeg]
        zjA = zob - z[:nSeg]
        zpA = cab[:nSeg]*xjA + sab[:nSeg]*yjA + salp[:nSeg]*zjA
        mZp = (npy.abs(zpA) <= 0.5001*si[:nSeg])
        if npy.any(mZp):
            rho2A = (
                xjA[mZp]**2 + yjA[mZp]**2 + zjA[mZp]**2 - zpA[mZp]**2
            )
            mHit = (rho2A <= 0.9*bi[:nSeg][mZp]**2)
            if npy.any(mHit):
                axRad = float(bi[:nSeg][mZp][mHit].item(0))

        for i in range(nSeg):
            state.s, state.b = si[i], bi[i]
            state.xj, state.yj, state.zj = x[i], y[i], z[i]
            state.cabj, state.sabj, state.salpj = cab[i], sab[i], salp[i]
            _set_ext_wire_flags(state, i + 1, cab, sab, salp, bi, icon1, icon2)
            efld(state, xob, yob, zob, axRad, 1)
            acx = complex(air[i], aii[i])
            bcx = complex(bir[i], bii[i])
            ccx = complex(cir[i], cii[i])
            ex += state.exk*acx + state.exs*bcx + state.exc*ccx
            ey += state.eyk*acx + state.eys*bcx + state.eyc*ccx
            ez += state.ezk*acx + state.ezs*bcx + state.ezc*ccx

    if state.m != 0:
        jc, jl = nSeg - 3, ld
        for _ in range(state.m):
            jl -= 1
            jc += 3
            state.s = bi[jl]
            state.xj, state.yj, state.zj = x[jl], y[jl], z[jl]
            state.cabj, state.sabj, state.salpj = t1x[jl], t1y[jl], t1z[jl]
            state.b, state.ind1, state.ind2 = t2x[jl], t2y[jl], t2z[jl]
            cx = complex(cur[jc])
            cy = complex(cur[jc + 1])
            cz = complex(cur[jc + 2])
            acx = t1x[jl]*cx + t1y[jl]*cy + t1z[jl]*cz
            bcx = t2x[jl]*cx + t2y[jl]*cy + t2z[jl]*cz
            for ip in range(1, state.ksymp + 1):
                state.ipgnd = ip
                unere(state, xob, yob, zob)
                ex += acx*state.exk + bcx*state.exs
                ey += acx*state.eyk + bcx*state.eys
                ez += acx*state.ezk + bcx*state.ezs

    return ex, ey, ez


def nhfld(
    state: NECState, xob, yob, zob, ld, ld3, x, y, z, si, bi, salp,
    cab, sab, t1x, t1y, t1z, t2x, t2y, t2z,
    air, aii, bir, bii, cir, cii, cur
):
    """Computes near magnetic field (hx, hy, hz) at (xob, yob, zob)."""
    hx = hy = hz = 0j
    axRad = 0.0
    nSeg = state.n

    if nSeg != 0:
        xjA = xob - x[:nSeg]
        yjA = yob - y[:nSeg]
        zjA = zob - z[:nSeg]
        zpA = cab[:nSeg]*xjA + sab[:nSeg]*yjA + salp[:nSeg]*zjA
        mZp = (npy.abs(zpA) <= 0.5001*si[:nSeg])
        if npy.any(mZp):
            rho2A = (
                xjA[mZp]**2 + yjA[mZp]**2 + zjA[mZp]**2 - zpA[mZp]**2
            )
            mHit = (rho2A <= 0.9*bi[:nSeg][mZp]**2)
            if npy.any(mHit):
                axRad = float(bi[:nSeg][mZp][mHit].item(0))

        for i in range(nSeg):
            state.s, state.b = si[i], bi[i]
            state.xj, state.yj, state.zj = x[i], y[i], z[i]
            state.cabj, state.sabj, state.salpj = cab[i], sab[i], salp[i]
            hsfld(state, xob, yob, zob, axRad)
            acx = complex(air[i], aii[i])
            bcx = complex(bir[i], bii[i])
            ccx = complex(cir[i], cii[i])
            hx += state.exk*acx + state.exs*bcx + state.exc*ccx
            hy += state.eyk*acx + state.eys*bcx + state.eyc*ccx
            hz += state.ezk*acx + state.ezs*bcx + state.ezc*ccx

    if state.m != 0:
        jc, jl = nSeg - 3, ld
        for _ in range(state.m):
            jl -= 1
            jc += 3
            state.s = bi[jl]
            state.xj, state.yj, state.zj = x[jl], y[jl], z[jl]
            state.cabj, state.sabj, state.salpj = t1x[jl], t1y[jl], t1z[jl]
            state.b, state.ind1, state.ind2 = t2x[jl], t2y[jl], t2z[jl]
            hintg(state, xob, yob, zob)
            cx = complex(cur[jc])
            cy = complex(cur[jc + 1])
            cz = complex(cur[jc + 2])
            acx = t1x[jl]*cx + t1y[jl]*cy + t1z[jl]*cz
            bcx = t2x[jl]*cx + t2y[jl]*cy + t2z[jl]*cz
            hx += acx*state.exk + bcx*state.exs
            hy += acx*state.eyk + bcx*state.eys
            hz += acx*state.ezk + bcx*state.ezs

    return hx, hy, hz

