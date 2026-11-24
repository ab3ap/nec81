import math
import cmath
import numpy as npy
from .nec_state import NECState, TA
from .main2 import atgn2, cang, db10, gwave
from .nsmaExport import recordAdfSample
from .plots3d import recordRadiationSample

PI = 3.141592654
TD = 57.29577951
NORMAX = 1200


def rdpat(state: NECState, cur, gain, air, aii, bir, bii, cir, cii, si, cab, sab, bi, salp, x, y, z, ld: int, ld3: int, ld4: int, iw):
    """Computes far-field radiation pattern, gain, axial ratio, and normalized gain."""
    hpol = ["LINEAR", "RIGHT", "LEFT"]
    igtp = ["    - ", "POWER ", "- DIRE", "CTIVE "]
    igax = [" MAJOR", " MINOR", " VERT.", " HOR. "]
    igntp = [" MAJOR", " AXIS ", " MINOR", " AXIS ", "   VER", "TICAL ", " HORIZ", "ONTAL ", "      ", "TOTAL "]

    if state.ifar >= 2:
        iw.write("\n\n\n                               - - - FAR FIELD GROUND PARAMETERS - - -\n\n\n")
        if state.ifar > 3:
            iw.write(
                f"                                        RADIAL WIRE GROUND SCREEN\n"
                f"                                        {state.nradl:5d} WIRES\n"
                f"                                        WIRE LENGTH={state.scrwlt:8.2f} METERS\n"
                f"                                        WIRE RADIUS={state.scrwrt:10.3E} METERS\n"
            )
        if state.ifar != 4:
            hclif = hpol[0] if state.ifar in (2, 5) else "CIRCLE"
            state.cl = state.clt / state.wlam
            state.ch = state.cht / state.wlam
            state.zrati2 = cmath.sqrt(1.0 / complex(state.epsr2, -state.sig2 * state.wlam * 59.96))
            iw.write(
                f"                                        {hclif:6s} CLIFF\n"
                f"                                        EDGE DISTANCE={state.clt:9.2f} METERS\n"
                f"                                        HEIGHT={state.cht:8.2f} METERS\n"
                f"                                        SECOND MEDIUM -\n"
                f"                                        RELATIVE DIELECTRIC CONST.={state.epsr2:7.3f}\n"
                f"                                        CONDUCTIVITY={state.sig2:10.3E} MHOS\n"
            )

    exrm = exra = 0.0
    if state.ifar == 1:
        iw.write(
            "\n\n\n                            - - - RADIATED FIELDS NEAR GROUND - - -\n\n"
            "        - - - LOCATION - - -          - - E(THETA) - -        - - E(PHI) - -        - - E(RADIAL) - -\n"
            "       RHO      PHI         Z            MAG      PHASE         MAG      PHASE         MAG      PHASE\n"
            "     METERS   DEGREES    METERS        VOLTS/M   DEGREES      VOLTS/M   DEGREES      VOLTS/M   DEGREES\n\n"
        )
    else:
        i_gt = 2 * state.ipd
        itmp1 = 2 * state.iax
        iw.write("\n\n\n                                                - - - RADIATION PATTERNS - - -\n")
        if state.rfld >= 1.0e-20:
            exrm = 1.0 / state.rfld
            exra = state.rfld / state.wlam
            exra = -360.0 * (exra - math.trunc(exra))
            iw.write(
                f"                                                      RANGE={state.rfld:13.6E} METERS\n"
                f"                                                      EXP(-JKR)/R={exrm:12.5E} AT PHASE{exra:7.2f} DEGREES\n\n"
            )
        iw.write(
            f"\n  - - ANGLES - -       {igtp[i_gt]:6s}{igtp[i_gt + 1]:6s}GAINS -       - - - POLARIZATION - - -    "
            f"- - - E(THETA) - - -    - - - E(PHI) - --\n"
            f"  THETA     PHI       {igax[itmp1]:6s}  {igax[itmp1 + 1]:6s}   TOTAL      AXIAL     TILT   SENSE     "
            f"MAGNITUDE    PHASE      MAGNITUDE    PHASE \n"
            f" DEGREES  DEGREES       DB      DB      DB        RATIO     DEG.              VOLTS/M    DEGREES      VOLTS/M    DEGREES\n"
        )

    if state.ixtyp in (0, 5):
        gcop = state.wlam * state.wlam * 2.0 * PI / (376.73 * state.pinr)
        prad = state.pinr - state.ploss - state.pnlr
        gcon = gcop * state.pinr / prad if state.ipd != 0 else gcop
    elif state.ixtyp == 4:
        state.pinr = 394.51 * state.xpr6 * state.xpr6 * state.wlam * state.wlam
        gcop = state.wlam * state.wlam * 2.0 * PI / (376.73 * state.pinr)
        prad = state.pinr - state.ploss - state.pnlr
        gcon = gcop * state.pinr / prad if state.ipd != 0 else gcop
    else:
        gcon = gcop = 4.0 * PI / (1.0 + state.xpr6 * state.xpr6)
    i_nor_cnt = 0
    gmax = -1.0e10
    pint = 0.0
    tmp1_dph = state.dph * TA
    tmp2_dth = 0.5 * state.dth * TA
    phi = state.phis - state.dph

    for kph in range(1, state.nph + 1):
        phi += state.dph
        pha = phi * TA
        thet = state.thets - state.dth
        for kth in range(1, state.nth + 1):
            thet += state.dth
            if state.ksymp == 2 and thet > 90.01 and state.ifar != 1:
                continue
            tha = thet * TA
            if state.ifar == 1:
                eth, eph, erd = gfld(state, state.rfld / state.wlam, pha, thet / state.wlam, cur, x, y, z, si, bi, salp, air, aii, bir, bii, cir, cii, cab, sab, ld, ld3)
            else:
                eth, eph = ffld(state, cur, tha, pha, x, y, z, si, bi, salp, air, aii, bir, bii, cir, cii, cab, sab, ld, ld3)
                erd = 0j

            ethm2 = (eth * eth.conjugate()).real
            ethm, etha = math.sqrt(ethm2), cang(eth)
            ephm2 = (eph * eph.conjugate()).real
            ephm, epha = math.sqrt(ephm2), cang(eph)

            if state.ifar == 1:
                iw.write(f"   {state.rfld:9.2f}  {phi:7.2f}  {thet:9.2f}    {ethm:11.4E}  {etha:7.2f}   {ephm:11.4E}  {epha:7.2f}   {abs(erd):11.4E}  {cang(erd):7.2f}\n")
                if state.iplp1 == 3 and state.iplp3 != 0:
                    with open("fort.8", "a") as f8:
                        val_x = thet if state.iplp2 == 1 else (phi if state.iplp2 == 2 else state.rfld)
                        if state.iplp3 == 1:
                            f8.write(f" {val_x} {ethm} {etha}\n")
                        elif state.iplp3 == 2:
                            f8.write(f" {val_x} {ephm} {epha}\n")
                        elif state.iplp3 == 3:
                            f8.write(f" {val_x} {abs(erd)} {cang(erd)}\n")
                continue

            if ethm2 <= 1.0e-20 and ephm2 <= 1.0e-20:
                tilta = emajr2 = eminr2 = axrat = 0.0
                isens = " "
            else:
                dfaz = epha - etha
                dfaz2 = dfaz - 360.0 if epha >= 0.0 else dfaz + 360.0
                if abs(dfaz) > abs(dfaz2):
                    dfaz = dfaz2
                tstor1 = ethm2 - ephm2
                tstor2 = 2.0 * ephm * ethm * math.cos(dfaz * TA)
                tilta = 0.5 * atgn2(tstor2, tstor1)
                stilta = math.sin(tilta)
                t1_s = tstor1 * stilta * stilta
                t2_s = tstor2 * stilta * math.cos(tilta)
                emajr2 = -t1_s + t2_s + ethm2
                eminr2 = max(0.0, t1_s - t2_s + ephm2)
                axrat = math.sqrt(eminr2 / emajr2)
                tilta *= TD
                isens = hpol[0] if axrat <= 1.0e-5 \
                    else (hpol[2] if dfaz > 0.0 else hpol[1])

            gnmj, gnmn = db10(gcon * emajr2), db10(gcon * eminr2)
            gnv, gnh = db10(gcon * ethm2), db10(gcon * ephm2)
            gtot = db10(gcon * (ethm2 + ephm2))
            recordRadiationSample(thet, phi, gtot, state.fmhz)
            freqMhz = 299.792458 / max(float(state.wlam), 1e-12)
            recordAdfSample(freqMhz, thet, phi, gtot, gnv, gnh, etha)

            if state.inor >= 1 and i_nor_cnt < NORMAX:
                tstor1 = (gnmj, gnmn, gnv, gnh, gtot)[state.inor - 1]
                gain[i_nor_cnt] = tstor1
                i_nor_cnt += 1
                if tstor1 > gmax:
                    gmax = tstor1

            if state.iavp != 0:
                tmp3 = tha if kth == 1 else tha - tmp2_dth
                tmp4 = tha if kth == state.nth else tha + tmp2_dth
                da = abs(tmp1_dph * (math.cos(tmp3) - math.cos(tmp4)))
                if kph == 1 or kph == state.nph:
                    da *= 0.5
                pint += gcop * (ethm2 + ephm2) * da
                if state.iavp == 2:
                    continue

            tmp5, tmp6 = (gnv, gnh) if state.iax == 1 else (gnmj, gnmn)
            ethm *= state.wlam
            ephm *= state.wlam
            if state.rfld >= 1.0e-20:
                ethm, etha = ethm * exrm, etha + exra
                ephm, epha = ephm * exrm, epha + exra

            iw.write(f" {thet:7.2f}{phi:9.2f}   {tmp5:8.2f}{tmp6:8.2f}{gtot:8.2f}{axrat:11.5f}{tilta:9.2f}  {isens:6s}{ethm:15.5E}{etha:9.2f}{ephm:15.5E}{epha:9.2f}\n")
            if state.iplp1 == 3:
                with open("fort.8", "a") as f8:
                    ang_v = thet if state.iplp2 == 1 else phi
                    if state.iplp3 == 1:
                        f8.write(f" {ang_v} {ethm} {etha}\n")
                    elif state.iplp3 == 2:
                        f8.write(f" {ang_v} {ephm} {epha}\n")
                    if state.iplp3 == 0 or state.iplp4 != 0:
                        if state.iplp4 in (1, 2, 3):
                            f8.write(f" {ang_v} {(tmp5, tmp6, gtot)[state.iplp4 - 1]}\n")
                        elif state.iplp4 == 4 and state.iplp2 == 1:
                            f8.write(f" {thet} {phi} {tmp5} {tmp6} {gtot}\n")

    if state.iavp != 0:
        tmp3 = state.thets * TA
        tmp4 = tmp3 + state.dth * TA * float(state.nth - 1)
        solid = abs(state.dph * TA * float(state.nph - 1) * (math.cos(tmp3) - math.cos(tmp4)))
        pint /= solid
        iw.write(f"\n\n   AVERAGE POWER GAIN={pint:12.5E}       SOLID ANGLE USED IN AVERAGING=({solid / PI:7.4f})*PI STERADIANS.\n\n")

    if state.inor != 0:
        if abs(state.gnor) > 1.0e-20:
            gmax = state.gnor
        itmp1 = (state.inor - 1) * 2
        iw.write(
            f"\n\n                                     - - - - NORMALIZED GAIN - - - -\n\n"
            f"                                     {igntp[itmp1]:6s}{igntp[itmp1 + 1]:6s}GAIN\n"
            f"                                      NORMALIZATION FACTOR ={gmax:9.2f} DB\n\n"
            f"    - - ANGLES - -      GAIN           - - ANGLES - -      GAIN           - - ANGLES - -      GAIN       \n"
            f"    THETA     PHI        DB            THETA     PHI        DB            THETA     PHI        DB        \n"
            f"   DEGREES  DEGREES                   DEGREES  DEGREES                   DEGREES  DEGREES                \n"
        )
        itmp2 = min(state.nph * state.nth, NORMAX)
        itmp1 = (itmp2 + 2) // 3
        rem = itmp1 * 3 - itmp2
        itmp3, itmp4 = itmp1, 2 * itmp1 - (1 if rem == 2 else 0)
        for i in range(itmp1):
            j1 = i // state.nth
            t1, p1 = state.thets + (i - j1 * state.nth) * state.dth, state.phis + j1 * state.dph
            s1 = gain[i] - gmax
            if i == itmp1 - 1 and rem != 0:
                if rem == 2:
                    iw.write(f" {t1:9.2f}{p1:9.2f} {s1:9.2f}\n")
                else:
                    j2 = itmp3 // state.nth
                    t2, p2 = state.thets + (itmp3 - j2 * state.nth) * state.dth, state.phis + j2 * state.dph
                    iw.write(f" {t1:9.2f}{p1:9.2f} {s1:9.2f}      {t2:9.2f}{p2:9.2f} {gain[itmp3] - gmax:9.2f}\n")
                break
            j2, j3 = itmp3 // state.nth, itmp4 // state.nth
            t2, p2 = state.thets + (itmp3 - j2 * state.nth) * state.dth, state.phis + j2 * state.dph
            t3, p3 = state.thets + (itmp4 - j3 * state.nth) * state.dth, state.phis + j3 * state.dph
            iw.write(f" {t1:9.2f}{p1:9.2f} {s1:9.2f}      {t2:9.2f}{p2:9.2f} {gain[itmp3] - gmax:9.2f}      {t3:9.2f}{p3:9.2f} {gain[itmp4] - gmax:9.2f}\n")
            itmp3 += 1
            itmp4 += 1
def ffld(state: NECState, cur, thet: float, phi: float, x, y, z, si, bi, salp, air, aii, bir, bii, cir, cii, cab, sab, ld: int, ld3: int):
    """Calculates far-zone radiated electric fields (eth, eph)."""
    tp, eta = 6.283185308, 376.73
    const = complex(0.0, -29.97922085)
    phx, phy = -math.sin(phi), math.cos(phi)
    roz = roz_s = math.cos(thet)
    thx, thy, thz = roz * phy, -roz * phx, -math.sin(thet)
    rox, roy = -thz * phy, thz * phx

    cix = ciy = ciz = 0j
    if state.n != 0:
        for k in range(1, state.ksymp + 1):
            if k != 1:
                if state.iperf == 1:
                    rrv = rrh = complex(-1.0, 0.0)
                else:
                    zrsin = cmath.sqrt(1.0 - state.zrati * state.zrati * thz * thz)
                    rrv = -(roz - state.zrati * zrsin) / (roz + state.zrati * zrsin)
                    rrh = (state.zrati * roz - zrsin) / (state.zrati * roz + zrsin)
                if state.ifar > 1:
                    rrv1, rrh1 = rrv, rrh
                    tthet = math.tan(thet)
                    if state.ifar != 4:
                        zrsin = cmath.sqrt(1.0 - state.zrati2 * state.zrati2 * thz * thz)
                        rrv2 = -(roz - state.zrati2 * zrsin) / (roz + state.zrati2 * zrsin)
                        rrh2 = (state.zrati2 * roz - zrsin) / (state.zrati2 * roz + zrsin)
                        darg = -tp * 2.0 * state.ch * roz
                roz = -roz
                ccx, ccy, ccz = cix, ciy, ciz

            cix = ciy = ciz = 0j
            for i in range(state.n):
                omega = -(rox * cab[i] + roy * sab[i] + roz * salp[i])
                el = PI * si[i]
                sill = omega * el
                top, bot = el + sill, el - sill
                a = (2.0 - omega * omega * el * el / 3.0) * el if abs(omega) < 1.0e-7 else 2.0 * math.sin(sill) / omega
                too = 1.0 - top * top / 6.0 if abs(top) < 1.0e-7 else math.sin(top) / top
                boo = 1.0 - bot * bot / 6.0 if abs(bot) < 1.0e-7 else math.sin(bot) / bot
                b, c = el * (boo - too), el * (boo + too)
                rr = a * air[i] + b * bii[i] + c * cir[i]
                ri = a * aii[i] - b * bir[i] + c * cii[i]
                arg = tp * (x[i] * rox + y[i] * roy + z[i] * roz)

                if k == 2 and state.ifar >= 2:
                    dr = z[i] * tthet
                    d = dr * phy + x[i]
                    if state.ifar != 2:
                        d = math.sqrt(d * d + (y[i] - dr * phx) ** 2)
                    if state.ifar != 3 and (state.scrwl - d) >= 0.0:
                        d += state.t2
                        zscrn = state.t1 * d * math.log(d / state.t2)
                        zscrn = (zscrn * state.zrati) / (eta * state.zrati + zscrn)
                        zrsin = cmath.sqrt(1.0 - zscrn * zscrn * thz * thz)
                        rrv = (roz + zscrn * zrsin) / (-roz + zscrn * zrsin)
                        rrh = (zscrn * roz + zrsin) / (zscrn * roz - zrsin)
                    else:
                        if state.ifar == 5:
                            d = dr * phy + x[i]
                        if state.ifar == 4 or (state.cl - d) > 0.0:
                            rrv, rrh = rrv1, rrh1
                        else:
                            rrv, rrh = rrv2, rrh2
                            arg += darg
                    exa = complex(math.cos(arg), math.sin(arg)) * complex(rr, ri)
                    tix, tiy, tiz = exa * cab[i], exa * sab[i], exa * salp[i]
                    cdp = (tix * phx + tiy * phy) * (rrh - rrv)
                    cix += tix * rrv + cdp * phx
                    ciy += tiy * rrv + cdp * phy
                    ciz -= tiz * rrv
                else:
                    exa = complex(math.cos(arg), math.sin(arg)) * complex(rr, ri)
                    cix += exa * cab[i]
                    ciy += exa * sab[i]
                    ciz += exa * salp[i]

            if k != 1:
                if state.ifar < 2:
                    cdp = (cix * phx + ciy * phy) * (rrh - rrv)
                    cix = ccx + cix * rrv + cdp * phx
                    ciy = ccy + ciy * rrv + cdp * phy
                    ciz = ccz - ciz * rrv
                else:
                    cix += ccx
                    ciy += ccy
                    ciz += ccz

        if state.m <= 0:
            return (cix * thx + ciy * thy + ciz * thz) * const, (cix * phx + ciy * phy) * const

    roz = roz_s
    rfl = -1.0
    ex = ey = ez = 0j
    for ip in range(1, state.ksymp + 1):
        rfl = -rfl
        gx, gy, gz = fflds(state, cur[state.n :], x, y, z, bi, rox, roy, roz * rfl, ld)
        if ip != 2:
            ex, ey, ez = gx, gy, gz
        else:
            if state.iperf == 1:
                gx, gy, gz = -gx, -gy, -gz
            else:
                rrv = cmath.sqrt(1.0 - state.zrati * state.zrati * thz * thz)
                rrh = (state.zrati * roz - rrv) / (state.zrati * roz + rrv)
                rrv = -(roz - state.zrati * rrv) / (roz + state.zrati * rrv)
                eth_t = (gx * phx + gy * phy) * (rrh - rrv)
                gx, gy, gz = gx * rrv + eth_t * phx, gy * rrv + eth_t * phy, gz * rrv
            ex, ey, ez = ex + gx, ey + gy, ez - gz

    ex += cix * const
    ey += ciy * const
    ez += ciz * const
    return ex * thx + ey * thy + ez * thz, ex * phx + ey * phy


def gfld(state: NECState, rho: float, phi: float, rz: float, cur, x, y, z, si, bi, salp, air, aii, bir, bii, cir, cii, cab, sab, ld: int, ld3: int):
    """Computes radiated field including Norton ground wave (eth, epi, erd)."""
    tp = 6.283185308
    r = math.sqrt(rho * rho + rz * rz)
    if state.ksymp == 1 or abs(state.zrati) > 0.5 or r > 1.0e5:
        thet = PI * 0.5 if rz < 1.0e-20 else math.atan(rho / rz)
        eth, epi = ffld(state, cur, thet, phi, x, y, z, si, bi, salp, air, aii, bir, bii, cir, cii, cab, sab, ld, ld3)
        exa = complex(math.cos(-tp * r), math.sin(-tp * r)) / r
        return eth * exa, epi * exa, 0j

    state.u = state.zrati
    state.u2 = state.u * state.u
    phx, phy = -math.sin(phi), math.cos(phi)
    rx, ry = rho * phy, -rho * phx
    cix = ciy = ciz = 0j

    for i in range(state.n):
        dx, dy, dz = cab[i], sab[i], salp[i]
        rix, riy = rx - x[i], ry - y[i]
        rhp = math.sqrt(rix * rix + riy * riy)
        rhx, rhy = (1.0, 0.0) if rhp < 1.0e-6 else (rix / rhp, riy / rhp)
        calp = 1.0 - dz * dz
        if calp < 1.0e-6:
            cph, sph = rhx, rhy
        else:
            calp = math.sqrt(calp)
            cbet, sbet = dx / calp, dy / calp
            cph, sph = rhx * cbet + rhy * sbet, rhy * cbet - rhx * sbet

        el = PI * si[i]
        rfl = -1.0
        for k in (1, 2):
            rfl = -rfl
            riz = rz - z[i] * rfl
            rxyz = math.sqrt(rix * rix + riy * riy + riz * riz)
            rnx, rny, rnz = rix / rxyz, riy / rxyz, riz / rxyz
            omega = -(rnx * dx + rny * dy + rnz * dz * rfl)
            sill = omega * el
            top, bot = el + sill, el - sill
            a = (2.0 - omega * omega * el * el / 3.0) * el if abs(omega) < 1.0e-7 else 2.0 * math.sin(sill) / omega
            too = 1.0 - top * top / 6.0 if abs(top) < 1.0e-7 else math.sin(top) / top
            boo = 1.0 - bot * bot / 6.0 if abs(bot) < 1.0e-7 else math.sin(bot) / bot
            b, c = el * (boo - too), el * (boo + too)
            rr = a * air[i] + b * bii[i] + c * cir[i]
            ri = a * aii[i] - b * bir[i] + c * cii[i]
            arg = tp * (x[i] * rnx + y[i] * rny + z[i] * rnz * rfl)
            exa = complex(math.cos(arg), math.sin(arg)) * complex(rr, ri) / tp
            if k == 1:
                state.xx1, state.r1, state.zmh = exa, rxyz, riz
            else:
                state.xx2, state.r2, state.zph = exa, rxyz, riz

        erv, ezv, erh, ezh, eph = gwave(state)
        erh = erh * cph * calp + erv * dz
        eph = eph * sph * calp
        ezh = ezh * cph * calp + ezv * dz
        cix += erh * rhx - eph * rhy
        ciy += erh * rhy + eph * rhx
        ciz += ezh

    exa = complex(math.cos(-tp * r), math.sin(-tp * r))
    cix, ciy, ciz = cix * exa, ciy * exa, ciz * exa
    rnx, rny, rnz = rx / r, ry / r, rz / r
    thx, thy, thz = rnz * phy, -rnz * phx, -rho / r
    return cix * thx + ciy * thy + ciz * thz, cix * phx + ciy * phy, cix * rnx + ciy * rny + ciz * rnz


def fflds(state: NECState, scur, xs, ys, zs, s, rox: float, roy: float, roz: float, ld: int):
    """Calculates XYZ components of far electric field due to surface patch currents."""
    tpi = 6.283185308
    cons = complex(0.0, 188.365)
    ex = ey = ez = 0j
    i = ld
    for j in range(state.m):
        i -= 1
        arg = tpi * (rox * xs[i] + roy * ys[i] + roz * zs[i])
        ct = complex(math.cos(arg) * s[i], math.sin(arg) * s[i])
        k = 3 * j
        ex += complex(scur[k]) * ct
        ey += complex(scur[k + 1]) * ct
        ez += complex(scur[k + 2]) * ct
    ct = rox * ex + roy * ey + roz * ez
    return cons * (ct * rox - ex), cons * (ct * roy - ey), cons * (ct * roz - ez)

