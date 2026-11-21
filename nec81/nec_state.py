from dataclasses import dataclass, field
import numpy as npy

# NEC-81 / OH3MCK Parameters
IRESRV = 1_000_000
LD = 3000
LD2 = 2 * LD
LD3 = 3 * LD
LD4 = 4 * LD

LOADMX = 100
NSMAX = 50
NETMX = 100
NORMF = 200
CVEL = 299.8
TA = 1.745329252e-02


@dataclass
class NECState:
    # /DATAD/
    n1: int = 0
    n2: int = 0
    n: int = 0
    np: int = 0
    m1: int = 0
    m2: int = 0
    m: int = 0
    mp: int = 0
    ipsym: int = 0
    wlam: float = 0.0

    # /DATAJ/
    s: float = 0.0
    b: float = 0.0
    xj: float = 0.0
    yj: float = 0.0
    zj: float = 0.0
    cabj: float = 0.0
    sabj: float = 0.0
    salpj: float = 0.0
    exk: complex = 0j
    eyk: complex = 0j
    ezk: complex = 0j
    exs: complex = 0j
    eys: complex = 0j
    ezs: complex = 0j
    exc: complex = 0j
    eyc: complex = 0j
    ezc: complex = 0j
    rkh: float = 1.0
    ind1: int = 0
    ind2: int = 0
    ipgnd: int = 0
    iexk: int = 0

    # /MATPAR/
    icase: int = 0
    nbloks: int = 0
    npblk: int = 0
    nlast: int = 0
    nblsym: int = 0
    npsym: int = 0
    nlsym: int = 0
    imat: int = 0
    icasx: int = 0
    nbbx: int = 0
    npbx: int = 0
    nlbx: int = 0
    nbbl: int = 0
    npbl: int = 0
    nlbl: int = 0

    # /SAVE/
    kcom: int = 0
    com: list = field(default_factory=lambda: [[""] * 5 for _ in range(19)])
    epsr: float = 0.0
    sig: float = 0.0
    scrwlt: float = 0.0
    scrwrt: float = 0.0
    fmhz: float = CVEL

    # /GND/
    zrati: complex = 1.0 + 0j
    zrati2: complex = 0j
    frati: complex = 1.0 + 0j
    cl: float = 0.0
    ch: float = 0.0
    scrwl: float = 0.0
    scrwr: float = 0.0
    nradl: int = 0
    ksymp: int = 1
    ifar: int = -1
    iperf: int = 0
    t1: complex = 0j
    t2: float = 0.0

    # /ZLOAD/
    nload: int = 0
    nlodf: int = 0

    # /YPARM/
    ncoup: int = 0
    icoup: int = 0
    nctag: npy.ndarray = field(default_factory=lambda: npy.zeros(5, dtype=npy.int32))
    ncseg: npy.ndarray = field(default_factory=lambda: npy.zeros(5, dtype=npy.int32))
    y11a: npy.ndarray = field(default_factory=lambda: npy.zeros(5, dtype=npy.complex128))
    y12a: npy.ndarray = field(default_factory=lambda: npy.zeros(20, dtype=npy.complex128))

    # /SEGJ/
    ax: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.float64))
    bx: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.float64))
    cx: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.float64))
    jco: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.int32))
    jsno: int = 0
    iscon: npy.ndarray = field(default_factory=lambda: npy.zeros(50, dtype=npy.int32))
    nscon: int = 0
    ipcon: npy.ndarray = field(default_factory=lambda: npy.zeros(10, dtype=npy.int32))
    npcon: int = 0

    # /VSORC/ (sized to NSMAX=50 per DATA override)
    vqd: npy.ndarray = field(default_factory=lambda: npy.zeros(NSMAX, dtype=npy.complex128))
    vsant: npy.ndarray = field(default_factory=lambda: npy.zeros(NSMAX, dtype=npy.complex128))
    vqds: npy.ndarray = field(default_factory=lambda: npy.zeros(NSMAX, dtype=npy.complex128))
    ivqd: npy.ndarray = field(default_factory=lambda: npy.zeros(NSMAX, dtype=npy.int32))
    isant: npy.ndarray = field(default_factory=lambda: npy.zeros(NSMAX, dtype=npy.int32))
    iqds: npy.ndarray = field(default_factory=lambda: npy.zeros(NSMAX, dtype=npy.int32))
    nvqd: int = 0
    nsant: int = 0
    nqds: int = 0

    # /NETCX/ (sized to NETMX=100 per DATA override)
    zped: complex = 0j
    pin: float = 0.0
    pnls: float = 0.0
    neq: int = 0
    npeq: int = 0
    neq2: int = 0
    nonet: int = 0
    ntsol: int = 0
    nprint: int = 0
    masym: int = 0
    iseg1: npy.ndarray = field(default_factory=lambda: npy.zeros(NETMX, dtype=npy.int32))
    iseg2: npy.ndarray = field(default_factory=lambda: npy.zeros(NETMX, dtype=npy.int32))
    x11r: npy.ndarray = field(default_factory=lambda: npy.zeros(NETMX, dtype=npy.float64))
    x11i: npy.ndarray = field(default_factory=lambda: npy.zeros(NETMX, dtype=npy.float64))
    x12r: npy.ndarray = field(default_factory=lambda: npy.zeros(NETMX, dtype=npy.float64))
    x12i: npy.ndarray = field(default_factory=lambda: npy.zeros(NETMX, dtype=npy.float64))
    x22r: npy.ndarray = field(default_factory=lambda: npy.zeros(NETMX, dtype=npy.float64))
    x22i: npy.ndarray = field(default_factory=lambda: npy.zeros(NETMX, dtype=npy.float64))
    ntyp: npy.ndarray = field(default_factory=lambda: npy.zeros(NETMX, dtype=npy.int32))

    # /NETWKC/
    cmn: npy.ndarray = field(default_factory=lambda: npy.zeros((30, 30), dtype=npy.complex64))
    rhnt: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.complex64))
    ipnt: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.int32))
    nteqa: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.int32))
    ntsca: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.int32))
    vsrc: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.complex128))
    rhnx: npy.ndarray = field(default_factory=lambda: npy.zeros(30, dtype=npy.complex128))
    namprt: int = 0

    # /FPAT/
    nth: int = 0
    nph: int = 0
    ipd: int = 0
    iavp: int = 0
    inor: int = 0
    iax: int = 0
    thets: float = 0.0
    phis: float = 0.0
    dth: float = 0.0
    dph: float = 0.0
    rfld: float = 0.0
    gnor: float = 0.0
    clt: float = 0.0
    cht: float = 0.0
    epsr2: float = 0.0
    sig2: float = 0.0
    ixtyp: int = 0
    xpr6: float = 0.0
    pinr: float = 0.0
    pnlr: float = 0.0
    ploss: float = 0.0
    near: int = -1
    nfeh: int = 0
    nrx: int = 0
    nry: int = 0
    nrz: int = 0
    xnr: float = 0.0
    ynr: float = 0.0
    znr: float = 0.0
    dxnr: float = 0.0
    dynr: float = 0.0
    dznr: float = 0.0

    # /GGRID/
    ar1: npy.ndarray = field(default_factory=lambda: npy.zeros((11, 10, 4), dtype=npy.complex64))
    ar2: npy.ndarray = field(default_factory=lambda: npy.zeros((17, 5, 4), dtype=npy.complex64))
    ar3: npy.ndarray = field(default_factory=lambda: npy.zeros((9, 8, 4), dtype=npy.complex64))
    epscf: complex = 0j
    dxa: npy.ndarray = field(default_factory=lambda: npy.zeros(3, dtype=npy.float32))
    dya: npy.ndarray = field(default_factory=lambda: npy.zeros(3, dtype=npy.float32))
    xsa: npy.ndarray = field(default_factory=lambda: npy.zeros(3, dtype=npy.float32))
    ysa: npy.ndarray = field(default_factory=lambda: npy.zeros(3, dtype=npy.float32))
    nxa: npy.ndarray = field(default_factory=lambda: npy.zeros(3, dtype=npy.int32))
    nya: npy.ndarray = field(default_factory=lambda: npy.zeros(3, dtype=npy.int32))

    # /GWAV/
    u: complex = 0j
    u2: complex = 0j
    xx1: complex = 0j
    xx2: complex = 0j
    r1: float = 0.0
    r2: float = 0.0
    zmh: float = 0.0
    zph: float = 0.0

    # /PLOT/
    iplp1: int = 0
    iplp2: int = 0
    iplp3: int = 0
    iplp4: int = 0

    # /SMAT/
    ssx: npy.ndarray = field(default_factory=lambda: npy.zeros((16, 16), dtype=npy.complex128))

    # /TMI/
    zpk: float = 0.0
    rkb2: float = 0.0
    ijx: int = 0

    # /INCOM/
    xo: float = 0.0
    yo: float = 0.0
    zo: float = 0.0
    sn: float = 0.0
    xsn: float = 0.0
    ysn: float = 0.0
    isnor: int = 0

    # /TMH/
    zpk_h: float = 0.0
    rhks: float = 0.0

    iresrv: int = IRESRV
    cm: npy.ndarray = field(
        default_factory=lambda: npy.zeros(IRESRV, dtype=npy.complex64)
    )
    patch_polygons: list = field(default_factory=list)

    def ensure_matrix_capacity(self, min_elements: int) -> None:
        """Ensure `cm` and `iresrv` can hold at least `min_elements`
        entries."""
        if min_elements > self.iresrv or self.cm.size < min_elements:
            self.iresrv = int(min_elements)
            self.cm = npy.zeros(self.iresrv, dtype=npy.complex64)

