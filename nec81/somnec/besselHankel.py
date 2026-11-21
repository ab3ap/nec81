import numpy as np


class SpecialFunctions:
    def __init__(self):
        self._besselInit = False
        self._besselM = np.zeros(101, dtype=np.int32)
        self._besselA1 = np.zeros(25, dtype=np.float64)
        self._besselA2 = np.zeros(25, dtype=np.float64)

        self._hankelInit = False
        self._hankelM = np.zeros(101, dtype=np.int32)
        self._hankelA1 = np.zeros(25, dtype=np.float64)
        self._hankelA2 = np.zeros(25, dtype=np.float64)
        self._hankelA3 = np.zeros(25, dtype=np.float64)
        self._hankelA4 = np.zeros(25, dtype=np.float64)

    def bessel(self, z):
        # BESSEL EVALUATES THE ZERO-ORDER BESSEL FUNCTION AND ITS DERIVATIVE
        # FOR COMPLEX ARGUMENT Z.
        c3 = 0.7978845608
        p10, p20 = 0.0703125, 0.1121520996
        q10, q20 = 0.125, 0.0732421875
        p11, p21 = 0.1171875, 0.1441955566
        q11, q21 = 0.375, 0.1025390625
        pof = 0.7853981635
        fj = 1j

        if not self._besselInit:
            # INITIALIZATION OF CONSTANTS
            for k in range(1, 26):
                self._besselA1.__setitem__(k - 1, -0.25/(k*k))
                self._besselA2.__setitem__(k - 1, 1.0/(k + 1.0))
            for i in range(1, 102):
                test = 1.0
                initVal = 1
                for k in range(1, 25):
                    initVal = k
                    test = -test*i*self._besselA1.item(k - 1)
                    if test < 1.0e-6:
                        break
                self._besselM.__setitem__(i - 1, initVal)
            self._besselInit = True

        zms = (z*np.conj(z)).real
        if zms <= 1.0e-12:
            return 1.0 + 0j, -0.5*z

        ib = 0
        j0x = 0j
        j0px = 0j

        if zms <= 37.21:
            if zms > 36.0:
                ib = 1
            # SERIES EXPANSION
            iz = int(1.0 + zms)
            miz = self._besselM.item(iz - 1)
            j0 = 1.0 + 0j
            j0p = j0
            zk = j0
            zi = z*z
            for k in range(miz):
                zk = zk*self._besselA1.item(k)*zi
                j0 = j0 + zk
                j0p = j0p + self._besselA2.item(k)*zk
            j0p = -0.5*z*j0p
            if ib == 0:
                return j0, j0p
            j0x = j0
            j0px = j0p

        # ASYMPTOTIC EXPANSION
        zi = 1.0/z
        zi2 = zi*zi
        p0z = 1.0 + (p20*zi2 - p10)*zi2
        p1z = 1.0 + (p11 - p21*zi2)*zi2
        q0z = (q20*zi2 - q10)*zi
        q1z = (q11 - q21*zi2)*zi
        zk = np.exp(fj*(z - pof))
        zi2 = 1.0/zk
        cz = 0.5*(zk + zi2)
        sz = fj*0.5*(zi2 - zk)
        zk = c3*np.sqrt(zi)
        j0 = zk*(p0z*cz - q0z*sz)
        j0p = -zk*(p1z*sz + q1z*cz)

        if ib == 0:
            return j0, j0p

        zmsCos = np.cos((np.sqrt(zms) - 6.0)*31.41592654)
        j0 = 0.5*(j0x*(1.0 + zmsCos) + j0*(1.0 - zmsCos))
        j0p = 0.5*(j0px*(1.0 + zmsCos) + j0p*(1.0 - zmsCos))
        return j0, j0p

    def hankel(self, z):
        # HANKEL EVALUATES HANKEL FUNCTION OF THE FIRST KIND, ORDER ZERO,
        # AND ITS DERIVATIVE FOR COMPLEX ARGUMENT Z.
        pi = 3.141592654
        gamma = 0.5772156649
        c1 = -0.0245785095
        c2 = 0.3674669052
        c3 = 0.7978845608
        p10, p20 = 0.0703125, 0.1121520996
        q10, q20 = 0.125, 0.0732421875
        p11, p21 = 0.1171875, 0.1441955566
        q11, q21 = 0.375, 0.1025390625
        p0f = 0.7853981635
        fj = 1j

        if not self._hankelInit:
            # INITIALIZATION OF CONSTANTS
            psi = -gamma
            for k in range(1, 26):
                self._hankelA1.__setitem__(k - 1, -0.25/(k*k))
                self._hankelA2.__setitem__(k - 1, 1.0/(k + 1.0))
                psi = psi + 1.0/k
                self._hankelA3.__setitem__(k - 1, psi + psi)
                numVal = psi + psi + 1.0/(k + 1.0)
                self._hankelA4.__setitem__(k - 1, numVal/(k + 1.0))
            for i in range(1, 102):
                test = 1.0
                initVal = 1
                for k in range(1, 25):
                    initVal = k
                    test = -test*i*self._hankelA1.item(k - 1)
                    if test*self._hankelA3.item(k - 1) < 1.0e-6:
                        break
                self._hankelM.__setitem__(i - 1, initVal)
            self._hankelInit = True

        zms = (z*np.conj(z)).real
        if zms == 0.0:
            raise ValueError('ERROR - HANKEL NOT VALID FOR Z=0.')

        ib = 0
        y0Save = 0j
        y0pSave = 0j

        if zms <= 16.81:
            if zms > 16.0:
                ib = 1
            # SERIES EXPANSION
            iz = int(1.0 + zms)
            miz = self._hankelM.item(iz - 1)
            j0 = 1.0 + 0j
            j0p = j0
            y0 = 0j
            y0p = 0j
            zk = j0
            zi = z*z
            for k in range(miz):
                zk = zk*self._hankelA1.item(k)*zi
                j0 = j0 + zk
                j0p = j0p + self._hankelA2.item(k)*zk
                y0 = y0 + self._hankelA3.item(k)*zk
                y0p = y0p + self._hankelA4.item(k)*zk
            j0p = -0.5*z*j0p
            clogz = np.log(0.5*z)
            y0 = (2.0*j0*clogz - y0)/pi + c2
            term = 2.0/z + 2.0*j0p*clogz + 0.5*y0p*z
            y0p = term/pi + c1*z
            h0 = j0 + fj*y0
            h0p = j0p + fj*y0p
            if ib == 0:
                return h0, h0p
            y0Save = h0
            y0pSave = h0p

        # ASYMPTOTIC EXPANSION
        zi = 1.0/z
        zi2 = zi*zi
        p0z = 1.0 + (p20*zi2 - p10)*zi2
        p1z = 1.0 + (p11 - p21*zi2)*zi2
        q0z = (q20*zi2 - q10)*zi
        q1z = (q11 - q21*zi2)*zi
        zk = np.exp(fj*(z - p0f))*np.sqrt(zi)*c3
        h0 = zk*(p0z + fj*q0z)
        h0p = fj*zk*(p1z + fj*q1z)

        if ib == 0:
            return h0, h0p

        zmsCos = np.cos((np.sqrt(zms) - 4.0)*31.41592654)
        h0 = 0.5*(y0Save*(1.0 + zmsCos) + h0*(1.0 - zmsCos))
        h0p = 0.5*(y0pSave*(1.0 + zmsCos) + h0p*(1.0 - zmsCos))
        return h0, h0p

