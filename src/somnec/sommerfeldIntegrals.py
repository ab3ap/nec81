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
    def testConvVec(f1: np.ndarray, f2: np.ndarray, dmin: float):
        # VECTORIZED CONVERGENCE TEST ACROSS ALL 6 INTEGRALS
        den = np.maximum(np.maximum(np.abs(f2.real), np.abs(f2.imag)), dmin)
        safeDen = np.where(den < 1.0e-37, 1.0, den)
        tr = np.where(den < 1.0e-37, 0.0, np.abs(f1.real - f2.real)/safeDen)
        ti = np.where(den < 1.0e-37, 0.0, np.abs(f1.imag - f2.imag)/safeDen)
        return tr, ti

    def saoa(self, t):
        # SAOA COMPUTES THE INTEGRAND FOR EACH OF THE 6
        # SOMMERFELD INTEGRALS FOR SOURCE AND OBSERVER ABOVE GROUND
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
        a5 = com*b0*den1/self.ck1
        com = com*den2

        if self.rho == 0.0:
            a0 = -com*xl*xl*0.5
            a3 = a0
        else:
            b0p = b0p/self.rho
            a0 = -com*xl*(b0p + b0*xl)
            a3 = com*xl*b0p

        a1 = com*cgam2*cgam2*b0
        a2 = -a3*cgam2*self.rho
        a4 = com*b0
        return np.array([a0, a1, a2, a3, a4, a5], dtype=np.complex128)

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
        ns = nx
        nt = 0
        g1 = self.saoa(z)[:n]

        while True:
            dz = s/ns
            if z + dz > ze:
                dz = ze - z
                if dz <= ep:
                    break

            dzot = dz*0.5
            g3 = self.saoa(z + dzot)[:n]
            g5 = self.saoa(z + dz)[:n]

            while True:
                t00 = (g1 + g5)*dzot
                t01 = (t00 + dz*g3)*0.5
                t10 = (4.0*t01 - t00)/3.0
                tr, ti = self.testConvVec(t01, t10, 0.0)
                if np.all(tr <= rx) and np.all(ti <= rx):
                    totalSum += t10
                    nt += 2
                    break

                g2 = self.saoa(z + dz*0.25)[:n]
                g4 = self.saoa(z + dz*0.75)[:n]
                t02 = (t01 + dzot*(g2 + g4))*0.5
                t11 = (4.0*t02 - t01)/3.0
                t20 = (16.0*t11 - t10)/15.0
                tr, ti = self.testConvVec(t11, t20, 0.0)
                if np.all(tr <= rx) and np.all(ti <= rx):
                    totalSum += t20
                    nt += 1
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
                    totalSum += t20
                    nt += 1
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

        q1 = np.zeros((nans, 20), dtype=np.complex128)
        q2 = np.zeros((nans, 20), dtype=np.complex128)
        ans2 = seed[:nans].copy()

        rbk = bk.real
        delStep = dela
        ibx = 1 if ibk == 0 else 0
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

                ans1 = ans2 + self.rom1(nans, 2)

                self.a = self.b
                self.b = self.b + delStep

                if ibx == 0 and self.b.real >= rbk:
                    ibx = 2
                    hitBreak = True
                    break

                ans2 = ans1 + self.rom1(nans, 2)

                as1 = ans1.copy()
                as2 = ans2.copy()
                if intIdx >= 2:
                    for j in range(2, intIdx + 1):
                        jm = j - 2
                        aa = q2[:, jm]
                        q1jm = q1[:, jm]
                        a1 = q1jm + as1 - 2.0*aa
                        z1 = (a1.real == 0.0) & (a1.imag == 0.0)
                        a2 = aa - q1jm
                        safeA1 = np.where(z1, 1.0 + 0j, a1)
                        as1New = np.where(z1, q1jm, q1jm - a2*a2/safeA1)

                        a2 = aa + as2 - 2.0*as1
                        z2 = (a2.real == 0.0) & (a2.imag == 0.0)
                        diff = as1 - aa
                        safeA2 = np.where(z2, 1.0 + 0j, a2)
                        as2New = np.where(z2, aa, aa - diff*diff/safeA2)

                        q1[:, jm] = as1
                        q2[:, jm] = as2
                        as1 = as1New
                        as2 = as2New

                q1[:, intIdx - 1] = as1
                q2[:, intIdx - 1] = as2
                den = float(np.max(np.abs(as2.real) + np.abs(as2.imag)))
                denm = 1.0e-3*den*crit

                jmStart = max(1, intIdx - 3)
                q1Sub = q1[:, jmStart - 1:intIdx]
                q2Sub = q2[:, jmStart - 1:intIdx]
                denVal = np.maximum(
                    (np.abs(q2Sub.real) + np.abs(q2Sub.imag))*crit, denm
                )
                diffSub = q1Sub - q2Sub
                amg = np.abs(diffSub.real) + np.abs(diffSub.imag)
                if np.all(amg <= denVal):
                    converged = True
                    break

            if hitBreak:
                # HIT BREAK POINT. RESET SEED AND START OVER.
                self.b = bk
                delStep = delb
                s = self.rom1(nans, 2)
                ans2 = (ans1 + s) if ibx == 2 else (ans2 + s)
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
        res[:nans] = 0.5*(q1[:, inx - 1] + q2[:, inx - 1])
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
                sumVec = sumVec + self.rom1(6, 2)
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
            sumVec = -(sumVec + self.rom1(6, 2))

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
                ansVec = self.rom1(6, 1) - sumVec
                sumVec = self.gshank(cp3, bk, 6, ansVec, 0, bk, bk)
                ansVec = self.gshank(cp2, delta2, 6, sumVec, 0, bk, bk)
            else:
                # INTEGRATE BELOW BRANCH POINTS, THEN TO + INFINITY
                sumVec = -ansVec
                rmis = self.ck1.real*1.01
                if self.ck2 + 1.0 > rmis:
                    rmis = self.ck2 + 1.0
                bk = complex(rmis, 0.99*self.ck1.imag)
                delta = bk - cp3
                delta = delta*delVal/abs(delta)
                ansVec = self.gshank(cp3, delta, 6, sumVec, 1, bk, delta2)

        a0, a1, a2, a3, a4, a5 = ansVec
        a5 = a5*self.ck1

        # CONJUGATE SINCE NEC USES EXP(+JWT)
        erv = np.conj(self.ck1sq*a2)
        ezv = np.conj(self.ck1sq*(a1 + self.ck2sq*a4))
        erh = np.conj(self.ck2sq*(a0 + a5))
        eph = -np.conj(self.ck2sq*(a3 + a5))
        return erv, ezv, erh, eph

