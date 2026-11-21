#!/usr/bin/env python

import time
import numpy as np
from somnecMain import SomnecGridGenerator


def runBenchmark():
    cases = (
        ('Average Ground 14.2 MHz', 13.0, 0.005, 14.2),
        ('Poor Ground 7.15 MHz', 4.0, 0.001, 7.15),
        ('Rich Soil 28.5 MHz', 20.0, 0.030, 28.5),
        ('Salt Water 3.8 MHz', 81.0, 5.0, 3.8),
        ('Complex Epscf (SIG<0)', 13.0, -5.0, 14.2),
    )

    print(f'{"Case":<24} {"Time (ms)":>10} {"Max |AR1|":>12} {"Status":>8}')
    print('-'*58)

    for label, epr, cond_Spm, freq_MHz in cases:
        gen = SomnecGridGenerator()
        t0 = time.perf_counter()
        gen.computeGrids(epr, cond_Spm, freq_MHz)
        dt_ms = (time.perf_counter() - t0)*1000.0

        ok1 = np.all(np.isfinite(gen.ar1))
        ok2 = np.all(np.isfinite(gen.ar2))
        ok3 = np.all(np.isfinite(gen.ar3))
        maxVal = float(np.max(np.abs(gen.ar1)))
        status = 'PASS' if (ok1 and ok2 and ok3 and maxVal > 0.0) else 'FAIL'

        print(f'{label:<24} {dt_ms:10.2f} {maxVal:12.4e} {status:>8}')


if __name__ == '__main__':
    runBenchmark()

