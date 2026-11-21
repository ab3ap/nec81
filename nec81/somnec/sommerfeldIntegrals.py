import numpy as np
try:
    from .besselHankel import SpecialFunctions
except ImportError:
    from besselHankel import SpecialFunctions

class SommerfeldEvaluator:
    def __init__(self):
        self.spec = SpecialFunctions()

        # COMMON /EVLCOM/
        self.cksm = 0j
        self.ct1 = 0j
        self.ct2 = 0j
        self.ct3 = 0j
        self.ck1 = 0j
        self.ck1sq = 0j
        self.ck2 = 0.0
        self.ck2sq = 0.0
        self.tkmag = 0.0
        self.tsmag = 0.0
        self.ck1r = 0.0
        self.zph = 0.0
        self.rho = 0.0
        self.jh = 0

        # COMMON /CNTOUR/
        self.a = 0j
        self.b = 0j

    def lambdaContour(self, t):
        # COMPUTE INTEGRATION PARAMETER XLAM=LAMBDA FROM PARAMETER T.
        dxlam = self.b - self.a
        xlam = self.a + dxlam*t
        return xlam, dxlam

    @staticmethod
    def testConv(f1r, f2r, f1i, f2i, dmin):
        # TEST FOR CONVERGENCE IN NUMERICAL INTEGRATION
        den = abs(f2r)
        tr = abs(f2i)
        if den < tr:
            den = tr
        if den < dmin:
            den = dmin
        if den < 1.0e-37:
            return 0.0, 0.0
        tr = abs((f1r - f2r)/den)
        ti = abs((f1i - f2i)/den)
        return tr, ti

    def saoa(self, t):
        # SAOA COMPUTES THE INTEGRAND FOR EACH OF THE 6
        # SOMMERFELD INTEGRALS FOR SOURCE AND OBSERVER ABOVE GROUND
        ans = np.zeros(6, dtype=np.complex128)
        xl, dxl = self.lambdaContour(t)

        if self.jh <= 0:
            # BESSEL FUNCTION FORM
            b0, b0p = self.spec.bessel(xl*self.rho)
            b0 = 2.0*b0
            b0p = 2.0*b0p
            cgam1 = np.sqrt(xl*xl - self.ck1sq)
            cgam2 = np.sqrt(xl*xl - self.ck2sq)
            if cgam1.real == 0.0:
                cgam1 = complex(0.0, -abs(cgam1.imag))
            if cgam2.real == 0.0:
                cgam2 = complex(0.0, -abs(cgam2.imag))
        else:
            # HANKEL FUNCTION FORM
            b0, b0p = self.spec.hankel(xl*self.rho)
            com = xl - self.ck1
            cgam1 = np.sqrt(xl + self.ck1)*np.sqrt(com)
            if com.real < 0.0 and com.imag >= 0.0:
                cgam1 = -cgam1
            com = xl - self.ck2
            cgam2 = np.sqrt(xl + self.ck2)*np.sqrt(com)
            if com.real < 0.0 and com.imag >= 0.0:
                cgam2 = -cgam2

        xlr = (xl*np.conj(xl)).real
        if xlr < self.tsmag:
            dgam = cgam2 - cgam1
        else:
            if xl.imag < 0.0:
                sign = 1.0
            else:
                xlrReal = xl.real
                if xlrReal < self.ck2:
                    sign = -1.0
                elif xlrReal > self.ck1r:
                    sign = 1.0
                else:
                    dgam = cgam2 - cgam1
                    sign = 0.0

            if sign != 0.0:
                dgam = 1.0/(xl*xl)
                poly = (self.ct3*dgam + self.ct2)*dgam + self.ct1
                dgam = sign*poly/xl

        denom2 = cgam2*(self.ck1sq*cgam2 + self.ck2sq*cgam1)
        den2 = self.cksm*dgam/denom2
        den1 = 1.0/(cgam1 + cgam2) - self.cksm/cgam2
        com = dxl*xl*np.exp(-cgam2*self.zph)
        ans.__setitem__(5, com*b0*den1/self.ck1)
        com = com*den2

        if self.rho == 0.0:
            val0 = -com*xl*xl*0.5
            ans.__setitem__(0, val0)
            ans.__setitem__(3, val0)
        else:
            b0p = b0p/self.rho
            ans.__setitem__(0, -com*xl*(b0p + b0*xl))
            ans.__setitem__(3, com*xl*b0p)

        ans.__setitem__(1, com*cgam2*cgam2*b0)
        ans.__setitem__(2, -ans.item(3)*cgam2*self.rho)
        ans.__setitem__(4, com*b0)
        return ans

    def rom1(self, n, nx):
        # ROM1 INTEGRATES THE 6 SOMMERFELD INTEGRALS FROM A TO B IN LAMBDA.
        # THE METHOD OF VARIABLE INTERVAL WIDTH ROMBERG INTEGRATION IS USED.
        nm = 131072
        nts = 4
        rx = 1.0e-4

        lstep = 0
        z = 0.0
        ze = 1.0
        s = 1.0
        ep = s/(1.0e4*nm)
        zend = ze - ep

        totalSum = np.zeros(n, dtype=np.complex128)
        t01 = np.zeros(n, dtype=np.complex128)
        t10 = np.zeros(n, dtype=np.complex128)
        t20 = np.zeros(n, dtype=np.complex128)

        ns = nx
        nt = 0
        g1 = self.saoa(z)

        while True:
            dz = s/ns
            if z + dz > ze:
                dz = ze - z
                if dz <= ep:
                    break

            dzot = dz*0.5
            g3 = self.saoa(z + dzot)
            g5 = self.saoa(z + dz)

            while True:
                nogo = 0
                for i in range(n):
                    t00 = (g1.item(i) + g5.item(i))*dzot
                    t01Val = (t00 + dz*g3.item(i))*0.5
                    t10Val = (4.0*t01Val - t00)/3.0
                    t01.__setitem__(i, t01Val)
                    t10.__setitem__(i, t10Val)
                    # TEST CONVERGENCE OF 3 POINT ROMBERG RESULT
                    tr, ti = self.testConv(
                        t01Val.real,
                        t10Val.real,
                        t01Val.imag,
                        t10Val.imag,
                        0.0,
                    )
                    if tr > rx or ti > rx:
                        nogo = 1

                if nogo == 0:
                    for i in range(n):
                        totalSum.__setitem__(i, totalSum.item(i) + t10.item(i))
                    nt = nt + 2
                    break

                g2 = self.saoa(z + dz*0.25)
                g4 = self.saoa(z + dz*0.75)
                nogo = 0
                for i in range(n):
                    g24 = g2.item(i) + g4.item(i)
                    t02 = (t01.item(i) + dzot*g24)*0.5
                    t11 = (4.0*t02 - t01.item(i))/3.0
                    t20Val = (16.0*t11 - t10.item(i))/15.0
                    t20.__setitem__(i, t20Val)
                    # TEST CONVERGENCE OF 5 POINT ROMBERG RESULT
                    tr, ti = self.testConv(
                        t11.real,
                        t20Val.real,
                        t11.imag,
                        t20Val.imag,
                        0.0,
                    )
                    if tr > rx or ti > rx:
                        nogo = 1

                if nogo == 0:
                    for i in range(n):
                        totalSum.__setitem__(i, totalSum.item(i) + t20.item(i))
                    nt = nt + 1
                    break

                nt = 0
                if ns < nm:
                    ns = ns*2
                    dz = s/ns
                    dzot = dz*0.5
                    g5 = g3.copy()
                    g3 = g2.copy()
                else:
                    if lstep != 1:
                        lstep = 1
                        t00, _ = self.lambdaContour(z)
                        print(
                            f' ROM1 -- STEP SIZE LIMITED AT LAMBDA ='
                            f'{t00.real:12.5E}{t00.imag:12.5E}'
                        )
                        print(
                            f'{z:12.5E}{dz:12.5E}'
                            f'{self.a.real:12.5E}{self.a.imag:12.5E}'
                            f'{self.b.real:12.5E}{self.b.imag:12.5E}'
                        )
                        for i in range(n):
                            g1i = g1.item(i)
                            g2i = g2.item(i)
                            g3i = g3.item(i)
                            g4i = g4.item(i)
                            g5i = g5.item(i)
                            print(
                                f'{g1i.real:12.5E}{g1i.imag:12.5E}'
                                f'{g2i.real:12.5E}{g2i.imag:12.5E}'
                                f'{g3i.real:12.5E}{g3i.imag:12.5E}'
                                f'{g4i.real:12.5E}{g4i.imag:12.5E}'
                                f'{g5i.real:12.5E}{g5i.imag:12.5E}'
                            )
                    for i in range(n):
                        totalSum.__setitem__(i, totalSum.item(i) + t20.item(i))
                    nt = nt + 1
                    break

            z = z + dz
            if z > zend:
                break
            g1 = g5.copy()
            if nt >= nts and ns > nx:
                ns = ns//2
                nt = 1

        return totalSum

    def gshank(self, start, dela, nans, seed, ibk, bk, delb):
        # GSHANK INTEGRATES THE 6 SOMMERFELD INTEGRALS FROM START TO
        # INFINITY (UNTIL CONVERGENCE) IN LAMBDA. AT THE BREAK POINT, BK,
        # THE STEP INCREMENT MAY BE CHANGED FROM DELA TO DELB. SHANK'S
        # ALGORITHM TO ACCELERATE CONVERGENCE OF A SLOWLY CONVERGING SERIES
        # IS USED
        crit = 1.0e-4
        maxh = 20

        q1 = np.zeros((6, 20), dtype=np.complex128)
        q2 = np.zeros((6, 20), dtype=np.complex128)
        ans1 = np.zeros(6, dtype=np.complex128)
        ans2 = np.zeros(6, dtype=np.complex128)

        rbk = bk.real
        delStep = dela
        ibx = 1 if ibk == 0 else 0
        for i in range(nans):
            ans2.__setitem__(i, seed.item(i))
        self.b = start

        inx = 0
        converged = False

        while not converged:
            hitBreak = False
            for intIdx in range(1, maxh + 1):
                inx = intIdx
                self.a = self.b
                self.b = self.b + delStep

                if ibx == 0 and self.b.real >= rbk:
                    ibx = 1
                    hitBreak = True
                    break

                s = self.rom1(nans, 2)
                for i in range(nans):
                    ans1.__setitem__(i, ans2.item(i) + s.item(i))

                self.a = self.b
                self.b = self.b + delStep

                if ibx == 0 and self.b.real >= rbk:
                    ibx = 2
                    hitBreak = True
                    break

                s = self.rom1(nans, 2)
                for i in range(nans):
                    ans2.__setitem__(i, ans1.item(i) + s.item(i))

                den = 0.0
                for i in range(nans):
                    as1 = ans1.item(i)
                    as2 = ans2.item(i)
                    if intIdx >= 2:
                        for j in range(2, intIdx + 1):
                            jm = j - 2
                            aa = q2.item((i, jm))
                            q1ijm = q1.item((i, jm))
                            a1 = q1ijm + as1 - 2.0*aa
                            if a1.real == 0.0 and a1.imag == 0.0:
                                a1 = q1ijm
                            else:
                                a2 = aa - q1ijm
                                a1 = q1ijm - a2*a2/a1

                            a2 = aa + as2 - 2.0*as1
                            if a2.real == 0.0 and a2.imag == 0.0:
                                a2 = aa
                            else:
                                diff = as1 - aa
                                a2 = aa - diff*diff/a2

                            q1.__setitem__((i, jm), as1)
                            q2.__setitem__((i, jm), as2)
                            as1 = a1
                            as2 = a2

                    q1.__setitem__((i, intIdx - 1), as1)
                    q2.__setitem__((i, intIdx - 1), as2)
                    amg = abs(as2.real) + abs(as2.imag)
                    if amg > den:
                        den = amg

                denm = 1.0e-3*den*crit
                jmStart = max(1, intIdx - 3)
                stepOk = True
                for j in range(jmStart, intIdx + 1):
                    for i in range(nans):
                        a1 = q2.item((i, j - 1))
                        denVal = (abs(a1.real) + abs(a1.imag))*crit
                        if denVal < denm:
                            denVal = denm
                        a1 = q1.item((i, j - 1)) - a1
                        amg = abs(a1.real) + abs(a1.imag)
                        if amg > denVal:
                            stepOk = False
                            break
                    if not stepOk:
                        break

                if stepOk:
                    converged = True
                    break

            if hitBreak:
                # HIT BREAK POINT. RESET SEED AND START OVER.
                self.b = bk
                delStep = delb
                s = self.rom1(nans, 2)
                if ibx == 2:
                    for i in range(nans):
                        ans2.__setitem__(i, ans1.item(i) + s.item(i))
                else:
                    for i in range(nans):
                        ans2.__setitem__(i, ans2.item(i) + s.item(i))
            elif not converged:
                print(' **** NO CONVERGENCE IN SUBROUTINE GSHANK ****')
                for i in range(nans):
                    q1v = q1.item((i, inx - 1))
                    q2v = q2.item((i, inx - 1))
                    print(
                        f'{q1v.real:12.5E}{q1v.imag:12.5E}'
                        f'{q2v.real:12.5E}{q2v.imag:12.5E}'
                    )
                break

        res = np.zeros(6, dtype=np.complex128)
        for i in range(nans):
            avg = 0.5*(q1.item((i, inx - 1)) + q2.item((i, inx - 1)))
            res.__setitem__(i, avg)
        return res

    def evlua(self):
        # EVLUA CONTROLS THE INTEGRATION CONTOUR IN THE COMPLEX LAMBDA
        # PLANE FOR EVALUATION OF THE SOMMERFELD INTEGRALS.
        ptp = 0.6283185308
        delVal = self.zph
        if self.rho > delVal:
            delVal = self.rho

        if self.zph >= 2.0*self.rho:
            # BESSEL FUNCTION FORM OF SOMMERFELD INTEGRALS
            self.jh = 0
            self.a = 0j
            delVal = 1.0/delVal
            if delVal > self.tkmag:
                self.b = complex(0.1*self.tkmag, -0.1*self.tkmag)
                sumVec = self.rom1(6, 2)
                self.a = self.b
                self.b = complex(delVal, -delVal)
                ansVec = self.rom1(6, 2)
                for i in range(6):
                    sumVec.__setitem__(i, sumVec.item(i) + ansVec.item(i))
            else:
                self.b = complex(delVal, -delVal)
                sumVec = self.rom1(6, 2)

            delta = ptp*delVal
            ansVec = self.gshank(self.b, delta, 6, sumVec, 0, self.b, self.b)
        else:
            # HANKEL FUNCTION FORM OF SOMMERFELD INTEGRALS
            self.jh = 1
            cp1 = complex(0.0, 0.4*self.ck2)
            cp2 = complex(0.6*self.ck2, -0.2*self.ck2)
            cp3 = complex(1.02*self.ck2, -0.2*self.ck2)

            self.a = cp1
            self.b = cp2
            sumVec = self.rom1(6, 2)
            self.a = cp2
            self.b = cp3
            ansVec = self.rom1(6, 2)
            for i in range(6):
                sumVec.__setitem__(i, -(sumVec.item(i) + ansVec.item(i)))

            # PATH FROM IMAGINARY AXIS TO -INFINITY
            slope = 1000.0
            if self.zph > 0.001*self.rho:
                slope = self.rho/self.zph
            delVal = ptp/delVal
            delta = complex(-1.0, slope)*delVal/np.sqrt(1.0 + slope*slope)
            delta2 = -np.conj(delta)
            ansVec = self.gshank(cp1, delta, 6, sumVec, 0, 0j, 0j)

            rmis = self.rho*(self.ck1.real - self.ck2)
            goBelow = False

            if rmis < 2.0*self.ck2 or self.rho < 1.0e-10:
                goBelow = True
            elif self.zph >= 1.0e-10:
                bk = complex(-self.zph, self.rho)*(self.ck1 - cp3)
                rmis = -bk.real/abs(bk.imag)
                if rmis > 4.0*self.rho/self.zph:
                    goBelow = True

            if not goBelow:
                # INTEGRATE UP BETWEEN BRANCH CUTS, THEN TO + INFINITY
                cp1 = self.ck1 - complex(0.1, 0.2)
                cp2 = cp1 + 0.2
                bk = complex(0.0, delVal)
                sumVec = self.gshank(cp1, bk, 6, ansVec, 0, bk, bk)
                self.a = cp1
                self.b = cp2
                ansVec = self.rom1(6, 1)
                for i in range(6):
                    ansVec.__setitem__(i, ansVec.item(i) - sumVec.item(i))
                sumVec = self.gshank(cp3, bk, 6, ansVec, 0, bk, bk)
                ansVec = self.gshank(cp2, delta2, 6, sumVec, 0, bk, bk)
            else:
                # INTEGRATE BELOW BRANCH POINTS, THEN TO + INFINITY
                for i in range(6):
                    sumVec.__setitem__(i, -ansVec.item(i))
                rmis = self.ck1.real*1.01
                if self.ck2 + 1.0 > rmis:
                    rmis = self.ck2 + 1.0
                bk = complex(rmis, 0.99*self.ck1.imag)
                delta = bk - cp3
                delta = delta*delVal/abs(delta)
                ansVec = self.gshank(cp3, delta, 6, sumVec, 1, bk, delta2)

        ansVec.__setitem__(5, ansVec.item(5)*self.ck1)

        # CONJUGATE SINCE NEC USES EXP(+JWT)
        erv = np.conj(self.ck1sq*ansVec.item(2))
        ezv = np.conj(
            self.ck1sq*(ansVec.item(1) + self.ck2sq*ansVec.item(4))
        )
        erh = np.conj(self.ck2sq*(ansVec.item(0) + ansVec.item(5)))
        eph = -np.conj(self.ck2sq*(ansVec.item(3) + ansVec.item(5)))
        return erv, ezv, erh, eph

