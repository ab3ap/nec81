import math
import cmath
import numpy as npy
from .nec_state import NECState, TA
from .main2 import cang
from .hintg import _set_ext_wire_flags
from .eksc import eksc, ekscx, rom2, sflds
from .hintg import unere, hsfld, hintg
from .plots3d import recordScalarField

RPD = 57.29577951


def nfpat(
    state: NECState,
    x, y, z, si, bi, salp,
    t1x, t1y, t1z, t2x, t2y, t2z,
    icon1, icon2, air, aii, bir, bii, cir, cii, cur, iw, ld: int, ld3: int
):
    """Computes near E or H fields over a 3D rectangular or spherical grid."""
    if state.nfeh == 1:
        iw.write(
            "\n\n\n                                   - - - NEAR MAGNETIC FIELDS - - -\n\n"
            "            -  LOCATION  -                     -  HX  -               -  HY  -               -  HZ  -          - PEAK FIELDS -\n"
            "        X          Y          Z          MAGNITUDE   PHASE      MAGNITUDE   PHASE      MAGNITUDE   PHASE         MAGNITUDE\n"
            "      METERS     METERS     METERS        AMPS/M   DEGREES       AMPS/M   DEGREES       AMPS/M   DEGREES          AMPS/M\n"
        )
    else:
        iw.write(
            "\n\n\n                                   - - - NEAR ELECTRIC FIELDS - - -\n\n"
            "            -  LOCATION  -                     -  EX  -               -  EY  -               -  EZ  -          - PEAK FIELDS -\n"
            "        X          Y          Z          MAGNITUDE   PHASE      MAGNITUDE   PHASE      MAGNITUDE   PHASE         MAGNITUDE\n"
            "      METERS     METERS     METERS       VOLTS/M   DEGREES      VOLTS/M   DEGREES      VOLTS/M   DEGREES         VOLTS/M\n"
        )

    znrt = state.znr - state.dznr
    for _ in range(state.nrz):
        znrt += state.dznr
        cth = math.cos(TA * znrt) if state.near != 0 else 0.0
        sth = math.sin(TA * znrt) if state.near != 0 else 0.0
        ynrt = state.ynr - state.dynr
        for _ in range(state.nry):
            ynrt += state.dynr
            cph = math.cos(TA * ynrt) if state.near != 0 else 0.0
            sph = math.sin(TA * ynrt) if state.near != 0 else 0.0
            xnrt = state.xnr - state.dxnr
            for _ in range(state.nrx):
                xnrt += state.dxnr
                if state.near != 0:
                    xob = xnrt * sth * cph
                    yob = xnrt * sth * sph
                    zob = xnrt * cth
                else:
                    xob, yob, zob = xnrt, ynrt, znrt

                tmp1, tmp2, tmp3 = xob / state.wlam, yob / state.wlam, zob / state.wlam
                if state.nfeh == 1:
                    ex, ey, ez = nhfld(
                        state, tmp1, tmp2, tmp3, ld, ld3,
                        x, y, z, si, bi, salp, t1y, t1z, t1x, t1y, t1z, t2x, t2y, t2z,
                        air, aii, bir, bii, cir, cii, cur
                    )
                else:
                    ex, ey, ez = nefld(
                        state, tmp1, tmp2, tmp3, ld, ld3,
                        x, y, z, si, bi, salp, t1y, t1z, t1x, t1y, t1z, t2x, t2y, t2z,
                        icon1, icon2, air, aii, bir, bii, cir, cii, cur
                    )

                tmp1, tmp2 = abs(ex), cang(ex)
                tmp3, tmp4 = abs(ey), cang(ey)
                tmp5, tmp6 = abs(ez), cang(ez)

                ez2, ey2, ex2 = tmp5 * tmp5, tmp3 * tmp3, tmp1 * tmp1
                at = ez2 * math.cos(tmp6 * 2.0 / RPD) + ey2 * math.cos(tmp4 * 2.0 / RPD) + ex2 * math.cos(tmp2 * 2.0 / RPD)
                bt = ez2 * math.sin(tmp6 * 2.0 / RPD) + ey2 * math.sin(tmp4 * 2.0 / RPD) + ex2 * math.sin(tmp2 * 2.0 / RPD)
                etotal = math.sqrt(0.5 * (ez2 + ey2 + ex2) + 0.5 * math.sqrt(at * at + bt * bt))

                iw.write(
                    f"  {xob:11.4f}{yob:11.4f}{zob:11.4f} "
                    f"   {tmp1:11.4E}  {tmp2:7.2f}   {tmp3:11.4E}  {tmp4:7.2f}   {tmp5:11.4E}  {tmp6:7.2f}      {etotal:11.4E}\n"
                )
                field_name = "Near Magnetic Field" if state.nfeh == 1 else "Near Electric Field"
                field_unit = "A/m" if state.nfeh == 1 else "V/m"
                recordScalarField(field_name, xob, yob, zob, etotal, field_unit)


                if state.iplp1 == 2:
                    xxx = xob if state.iplp4 == 1 else (yob if state.iplp4 == 2 else zob)
                    with open("fort.8", "a") as f8:
                        if state.iplp2 == 2:
                            if state.iplp3 == 1:
                                f8.write(f" {xxx} {tmp1} {tmp2}\n")
                            elif state.iplp3 == 2:
                                f8.write(f" {xxx} {tmp3} {tmp4}\n")
                            elif state.iplp3 == 3:
                                f8.write(f" {xxx} {tmp5} {tmp6}\n")
                            elif state.iplp3 == 4:
                                f8.write(f" {xxx} {tmp1} {tmp2} {tmp3} {tmp4} {tmp5} {tmp6}\n")
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
                                f8.write(f" ({ex.real},{ex.imag}) ({ey.real},{ey.imag}) ({ez.real},{ez.imag})\n")


def nefld(state: NECState, xob, yob, zob, ld, ld3, x, y, z, si, bi, salp, cab, sab, t1x, t1y, t1z, t2x, t2y, t2z, icon1, icon2, air, aii, bir, bii, cir, cii, cur):
    """Computes near electric field (ex, ey, ez) at (xob, yob, zob)."""
    ex = ey = ez = 0j
    ax_rad = 0.0

    if state.n != 0:
        for i in range(state.n):
            xj, yj, zj = xob - x[i], yob - y[i], zob - z[i]
            zp = cab[i] * xj + sab[i] * yj + salp[i] * zj
            if abs(zp) <= 0.5001 * si[i]:
                rho2 = xj * xj + yj * yj + zj * zj - zp * zp
                if rho2 <= 0.9 * bi[i] * bi[i]:
                    ax_rad = bi[i]
                    break

        for i in range(state.n):
            state.s, state.b = si[i], bi[i]
            state.xj, state.yj, state.zj = x[i], y[i], z[i]
            state.cabj, state.sabj, state.salpj = cab[i], sab[i], salp[i]
            _set_ext_wire_flags(state, i + 1, cab, sab, salp, bi, icon1, icon2)
            efld(state, xob, yob, zob, ax_rad, 1)
            acx = complex(air[i], aii[i])
            bcx = complex(bir[i], bii[i])
            ccx = complex(cir[i], cii[i])
            ex += state.exk * acx + state.exs * bcx + state.exc * ccx
            ey += state.eyk * acx + state.eys * bcx + state.eyc * ccx
            ez += state.ezk * acx + state.ezs * bcx + state.ezc * ccx

    if state.m != 0:
        jc, jl = state.n - 3, ld
        for _ in range(state.m):
            jl -= 1
            jc += 3
            state.s = bi[jl]
            state.xj, state.yj, state.zj = x[jl], y[jl], z[jl]
            state.cabj, state.sabj, state.salpj = t1x[jl], t1y[jl], t1z[jl]
            state.b, state.ind1, state.ind2 = t2x[jl], t2y[jl], t2z[jl]
            cx, cy, cz = complex(cur[jc]), complex(cur[jc + 1]), complex(cur[jc + 2])
            acx = t1x[jl] * cx + t1y[jl] * cy + t1z[jl] * cz
            bcx = t2x[jl] * cx + t2y[jl] * cy + t2z[jl] * cz
            for ip in range(1, state.ksymp + 1):
                state.ipgnd = ip
                unere(state, xob, yob, zob)
                ex += acx * state.exk + bcx * state.exs
                ey += acx * state.eyk + bcx * state.eys
                ez += acx * state.ezk + bcx * state.ezs

    return ex, ey, ez
def nhfld(state: NECState, xob, yob, zob, ld, ld3, x, y, z, si, bi, salp, cab, sab, t1x, t1y, t1z, t2x, t2y, t2z, air, aii, bir, bii, cir, cii, cur):
    """Computes near magnetic field (hx, hy, hz) at (xob, yob, zob)."""
    hx = hy = hz = 0j
    ax_rad = 0.0

    if state.n != 0:
        for i in range(state.n):
            xj, yj, zj = xob - x[i], yob - y[i], zob - z[i]
            zp = cab[i] * xj + sab[i] * yj + salp[i] * zj
            if abs(zp) <= 0.5001 * si[i]:
                rho2 = xj * xj + yj * yj + zj * zj - zp * zp
                if rho2 <= 0.9 * bi[i] * bi[i]:
                    ax_rad = bi[i]
                    break

        for i in range(state.n):
            state.s, state.b = si[i], bi[i]
            state.xj, state.yj, state.zj = x[i], y[i], z[i]
            state.cabj, state.sabj, state.salpj = cab[i], sab[i], salp[i]
            hsfld(state, xob, yob, zob, ax_rad)
            acx = complex(air[i], aii[i])
            bcx = complex(bir[i], bii[i])
            ccx = complex(cir[i], cii[i])
            hx += state.exk * acx + state.exs * bcx + state.exc * ccx
            hy += state.eyk * acx + state.eys * bcx + state.eyc * ccx
            hz += state.ezk * acx + state.ezs * bcx + state.ezc * ccx

    if state.m != 0:
        jc, jl = state.n - 3, ld
        for _ in range(state.m):
            jl -= 1
            jc += 3
            state.s = bi[jl]
            state.xj, state.yj, state.zj = x[jl], y[jl], z[jl]
            state.cabj, state.sabj, state.salpj = t1x[jl], t1y[jl], t1z[jl]
            state.b, state.ind1, state.ind2 = t2x[jl], t2y[jl], t2z[jl]
            hintg(state, xob, yob, zob)
            cx, cy, cz = complex(cur[jc]), complex(cur[jc + 1]), complex(cur[jc + 2])
            acx = t1x[jl] * cx + t1y[jl] * cy + t1z[jl] * cz
            bcx = t2x[jl] * cx + t2y[jl] * cy + t2z[jl] * cz
            hx += acx * state.exk + bcx * state.exs
            hy += acx * state.eyk + bcx * state.eys
            hz += acx * state.ezk + bcx * state.ezs

    return hx, hy, hz


def efld(state: NECState, xi: float, yi: float, zi: float, ai: float, ij: int):
    """Computes near E fields of a segment with sine, cosine, and constant currents (ground included)."""
    eta, pi, tp = 376.73, 3.141592654, 6.283185308
    xij, yij = xi - state.xj, yi - state.yj
    ijx = ij
    rfl = -1.0

    for ip in range(1, state.ksymp + 1):
        if ip == 2:
            ijx = 1
        rfl = -rfl
        salpr = state.salpj * rfl
        zij = zi - rfl * state.zj
        zp = xij * state.cabj + yij * state.sabj + zij * salpr
        rhox, rhoy, rhoz = xij - state.cabj * zp, yij - state.sabj * zp, zij - salpr * zp
        rh = math.sqrt(rhox * rhox + rhoy * rhoy + rhoz * rhoz + ai * ai)
        if rh <= 1.0e-10:
            rhox = rhoy = rhoz = 0.0
        else:
            rhox, rhoy, rhoz = rhox / rh, rhoy / rh, rhoz / rh

        r = math.sqrt(zp * zp + rh * rh)
        if r >= state.rkh:
            rmag = tp * r
            cth, px = zp / r, rh / r
            txk = complex(math.cos(rmag), -math.sin(rmag))
            py = tp * r * r
            tyk = eta * cth * txk * complex(1.0, -1.0 / rmag) / py
            tzk = eta * px * txk * complex(1.0, rmag - 1.0 / rmag) / (2.0 * py)
            tezk, terk = tyk * cth - tzk * px, tyk * px + tzk * cth
            rmag_s = math.sin(pi * state.s) / pi
            tezc, terc = tezk * rmag_s, terk * rmag_s
            tezk, terk = tezk * state.s, terk * state.s
            txs = tys = tzs = 0j
        else:
            if state.iexk == 1:
                tezs, ters, tezc, terc, tezk, terk = ekscx(state, state.b, state.s, zp, rh, tp, ijx, state.ind1, state.ind2)
            else:
                tezs, ters, tezc, terc, tezk, terk = eksc(state, state.s, zp, rh, tp, ijx)
            txs = tezs * state.cabj + ters * rhox
            tys = tezs * state.sabj + ters * rhoy
            tzs = tezs * salpr + ters * rhoz

        txk = tezk * state.cabj + terk * rhox
        tyk = tezk * state.sabj + terk * rhoy
        tzk = tezk * salpr + terk * rhoz
        txc = tezc * state.cabj + terc * rhox
        tyc = tezc * state.sabj + terc * rhoy
        tzc = tezc * salpr + terc * rhoz

        if ip != 2:
            state.exk, state.eyk, state.ezk = txk, tyk, tzk
            state.exs, state.eys, state.ezs = txs, tys, tzs
            state.exc, state.eyc, state.ezc = txc, tyc, tzc
        else:
            if state.iperf <= 0:
                zratx, rmag = state.zrati, r
                xymag = math.sqrt(xij * xij + yij * yij)
                if state.nradl != 0:
                    xspec, yspec = (xi * state.zj + zi * state.xj) / (zi + state.zj), (yi * state.zj + zi * state.yj) / (zi + state.zj)
                    rhospc = math.sqrt(xspec * xspec + yspec * yspec + state.t2 * state.t2)
                    if rhospc <= state.scrwl:
                        zscrn = state.t1 * rhospc * math.log(rhospc / state.t2)
                        zratx = (zscrn * state.zrati) / (eta * state.zrati + zscrn)
                if xymag <= 1.0e-6:
                    px, py, cth, zrsin = 0.0, 0.0, 1.0, 1.0 + 0j
                else:
                    px, py, cth = -yij / xymag, xij / xymag, zij / rmag
                    zrsin = cmath.sqrt(1.0 - zratx * zratx * (1.0 - cth * cth))
                refs = (cth - zratx * zrsin) / (cth + zratx * zrsin)
                refps = -(zratx * cth - zrsin) / (zratx * cth + zrsin) - refs
                for t_vec in ((txk, tyk, tzk), (txs, tys, tzs), (txc, tyc, tzc)):
                    pass
                epy = px * txk + py * tyk
                txk, tyk, tzk = refs * txk + refps * px * epy, refs * tyk + refps * py * epy, refs * tzk
                epy = px * txs + py * tys
                txs, tys, tzs = refs * txs + refps * px * epy, refs * tys + refps * py * epy, refs * tzs
                epy = px * txc + py * tyc
                txc, tyc, tzc = refs * txc + refps * px * epy, refs * tyc + refps * py * epy, refs * tzc

            state.exk -= txk * state.frati
            state.eyk -= tyk * state.frati
            state.ezk -= tzk * state.frati
            state.exs -= txs * state.frati
            state.eys -= tys * state.frati
            state.ezs -= tzs * state.frati
            state.exc -= txc * state.frati
            state.eyc -= tyc * state.frati
            state.ezc -= tzc * state.frati

    if state.iperf != 2:
        return

    sn = math.sqrt(state.cabj * state.cabj + state.sabj * state.sabj)
    if sn < 1.0e-5:
        state.sn, state.xsn, state.ysn = 0.0, 1.0, 0.0
    else:
        state.sn, state.xsn, state.ysn = sn, state.cabj / sn, state.sabj / sn

    zij = zi + state.zj
    salpr = -state.salpj
    rhox = state.sabj * zij - salpr * yij
    rhoy = salpr * xij - state.cabj * zij
    rhoz = state.cabj * yij - state.sabj * xij
    rh = rhox * rhox + rhoy * rhoy + rhoz * rhoz
    if rh <= 1.0e-10:
        state.xo, state.yo, state.zo = xi - ai * state.ysn, yi + ai * state.xsn, zi
    else:
        rh = ai / math.sqrt(rh)
        if rhoz < 0.0:
            rh = -rh
        state.xo, state.yo, state.zo = xi + rh * rhox, yi + rh * rhoy, zi + rh * rhoz

    r_sq = xij * xij + yij * yij + zij * zij
    if r_sq <= 0.95:
        state.isnor = 1
        dmin = 0.01 * math.sqrt((state.exk * state.exk.conjugate() + state.eyk * state.eyk.conjugate() + state.ezk * state.ezk.conjugate()).real)
        shaf = 0.5 * state.s
        egnd = rom2(state, -shaf, shaf, dmin)
        zp = xij * state.cabj + yij * state.sabj + zij * salpr
        rh_p = r_sq - zp * zp
        dmin_p = 0.0 if rh_p <= 1.0e-10 else math.sqrt(rh_p / (rh_p + ai * ai))
        if dmin_p <= 0.95:
            px = 1.0 - dmin_p
            for idx_base in (0, 3, 6):
                ter = (egnd[idx_base] * state.cabj + egnd[idx_base + 1] * state.sabj + egnd[idx_base + 2] * salpr) * px
                egnd[idx_base] = dmin_p * egnd[idx_base] + ter * state.cabj
                egnd[idx_base + 1] = dmin_p * egnd[idx_base + 1] + ter * state.sabj
                egnd[idx_base + 2] = dmin_p * egnd[idx_base + 2] + ter * salpr
    else:
        state.isnor = 2
        egnd = sflds(state, 0.0)

    state.exk += egnd
    state.eyk += egnd
    state.ezk += egnd
    state.exs += egnd
    state.eys += egnd
    state.ezs += egnd
    state.exc += egnd
    state.eyc += egnd
    state.ezc += egnd

