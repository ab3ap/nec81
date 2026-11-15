"""Optional 3D plots for NEC-81 geometry, patterns, and field data."""

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class PlotSession:
    """Hold plotting options and samples produced during one run."""

    enabled: bool = False
    geometryEnabled: bool = True
    patternsEnabled: bool = True
    fieldsEnabled: bool = True
    showSources: bool = True
    showLoads: bool = False
    outputDirectory: Path = Path("plots")
    imageFormat: str = "png"
    showPlots: bool = True
    patternStyle: str = "wireframe"  # "wireframe" or "surface"
    patternSamples: list = field(default_factory=list)
    fieldSamples: list = field(default_factory=list)
    sweepSamples: list = field(default_factory=list)


plotSession = PlotSession()

def configurePlots(
    enabled=False,
    geometryEnabled=True,
    patternsEnabled=True,
    fieldsEnabled=True,
    showSources=True,
    showLoads=False,
    outputDirectory="plots",
    imageFormat="png",
    showPlots=True,
    patternStyle="wireframe",
):
    """Configure plotting without importing Matplotlib until it is needed."""

    global plotSession
    plotSession = PlotSession(
        enabled=enabled,
        geometryEnabled=geometryEnabled,
        patternsEnabled=patternsEnabled,
        fieldsEnabled=fieldsEnabled,
        showSources=showSources,
        showLoads=showLoads,
        outputDirectory=Path(outputDirectory),
        imageFormat=imageFormat.lstrip("."),
        showPlots=showPlots,
        patternStyle=patternStyle,
    )


def importPlotLibraries():
    """Import optional plotting dependencies and explain missing packages."""

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        message = ("3D plotting requires numpy and matplotlib. Install the "
                   "packages listed in requirements.txt.")
        raise RuntimeError(message) from error
    return np, plt


def prepareOutputPath(baseName):
    """Create the plot directory and return a configured image path."""

    plotSession.outputDirectory.mkdir(parents=True, exist_ok=True)
    fileName = f"{baseName}.{plotSession.imageFormat}"
    return plotSession.outputDirectory/fileName


def finishFigure(figure, baseName, plt):
    """Save a figure and retain it when interactive viewing is enabled."""

    outputPath = prepareOutputPath(baseName)
    figure.tight_layout()
    figure.savefig(outputPath, dpi=160, bbox_inches="tight")
    if not plotSession.showPlots:
        plt.close(figure)
    return outputPath


def setEqualAxes(axis, x_m, y_m, z_m, np):
    """Use equal scales on all three Cartesian axes."""

    xRange = float(np.ptp(x_m))
    yRange = float(np.ptp(y_m))
    zRange = float(np.ptp(z_m))
    radius_m = max(xRange, yRange, zRange, 1e-12)*0.5
    xCenter_m = (float(np.max(x_m)) + float(np.min(x_m)))*0.5
    yCenter_m = (float(np.max(y_m)) + float(np.min(y_m)))*0.5
    zCenter_m = (float(np.max(z_m)) + float(np.min(z_m)))*0.5
    axis.set_xlim(xCenter_m - radius_m, xCenter_m + radius_m)
    axis.set_ylim(yCenter_m - radius_m, yCenter_m + radius_m)
    axis.set_zlim(zCenter_m - radius_m, zCenter_m + radius_m)


def solverPatchPolygons(x_m, y_m, z_m, area_m2, t1x, t1y, t1z, t2x, t2y,
                        t2z, patchCount, arraySize):
    """Return one square outline per patch from the solver's patch arrays.

    NEC keeps only each patch's center, area, and two unit tangent vectors,
    so every patch is drawn as the square of equal area. Patch number i
    (1 to patchCount) is stored at index arraySize + 1 - i, as in NEC.
    Placeholder patches parked at z = 10000 are skipped."""

    if (patchCount < 1 or not plotSession.enabled
            or not plotSession.geometryEnabled):
        return []  # Nothing to draw; avoid importing Matplotlib.
    np, _ = importPlotLibraries()
    polygons = []
    for number in range(1, patchCount + 1):
        index = arraySize - number
        if z_m[index] >= 9000.0:
            continue
        center = np.array([x_m[index], y_m[index], z_m[index]], dtype=float)
        tangent1 = np.array([t1x[index], t1y[index], t1z[index]], dtype=float)
        tangent2 = np.array([t2x[index], t2y[index], t2z[index]], dtype=float)
        half_m = 0.5*np.sqrt(abs(float(area_m2[index])))
        polygons.append(np.array([
            center + half_m*(sign1*tangent1 + sign2*tangent2)
            for sign1, sign2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]))
    return polygons

def plotAntennaGeometry(
    xCenter_m,
    yCenter_m,
    zCenter_m,
    length_m,
    directionX,
    directionY,
    directionZ,
    patchPolygons=None,
    currents_A=None,
    sourcePoints=None,
    loadPoints=None,
    sourceSummary="",
):

    """Plot wire segments (optionally colored by RF current |I|) and surface patches."""
    if not plotSession.enabled or not plotSession.geometryEnabled:
        return None
    np, plt = importPlotLibraries()
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    xCenter_m = np.asarray(xCenter_m, dtype=float)
    yCenter_m = np.asarray(yCenter_m, dtype=float)
    zCenter_m = np.asarray(zCenter_m, dtype=float)
    length_m = np.asarray(length_m, dtype=float)
    directionX = np.asarray(directionX, dtype=float)
    directionY = np.asarray(directionY, dtype=float)
    directionZ = np.asarray(directionZ, dtype=float)

    halfLength_m = length_m * 0.5
    xStart_m = xCenter_m - halfLength_m * directionX
    yStart_m = yCenter_m - halfLength_m * directionY
    zStart_m = zCenter_m - halfLength_m * directionZ
    xEnd_m = xCenter_m + halfLength_m * directionX
    yEnd_m = yCenter_m + halfLength_m * directionY
    zEnd_m = zCenter_m + halfLength_m * directionZ

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    patchPolygons = [] if patchPolygons is None else list(patchPolygons)
    if patchPolygons:
        axis.add_collection3d(Poly3DCollection(
            patchPolygons,
            facecolors="tab:orange",
            edgecolors="saddlebrown",
            linewidths=0.6,
            alpha=0.35,
        ))

    nSegs = len(xCenter_m)
    if nSegs > 0:
        segments = [
            [(xStart_m[i], yStart_m[i], zStart_m[i]),
             (xEnd_m[i], yEnd_m[i], zEnd_m[i])]
            for i in range(nSegs)
        ]
        if currents_A is not None and len(currents_A) == nSegs:
            iMag = np.asarray(currents_A, dtype=float)
            iMax = float(np.max(iMag))
            if np.isclose(iMax, 0.0):
                iMax = 1e-6
            norm = colors.Normalize(vmin=0.0, vmax=iMax, clip=True)
            cmap = cm.get_cmap("turbo")
            lineCol = Line3DCollection(
                segments, colors=cmap(norm(iMag)), linewidths=2.2
            )
            axis.add_collection3d(lineCol)
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(iMag)
            figure.colorbar(
                mappable, ax=axis, shrink=0.65, label="RF Current |I| (A)"
            )
            title = f"Antenna Geometry & Current (Peak: {iMax:.4g} A)"
        else:
            lineCol = Line3DCollection(
                segments, colors="tab:blue", linewidths=1.6
            )
            axis.add_collection3d(lineCol)
            title = "Antenna Geometry"
    else:
        title = "Antenna Geometry"

    corners = [np.asarray(polygon, dtype=float).reshape(-1, 3)
               for polygon in patchPolygons]
    allX_m = np.concatenate([xStart_m, xEnd_m] + [c[:, 0] for c in corners])
    allY_m = np.concatenate([yStart_m, yEnd_m] + [c[:, 1] for c in corners])
    allZ_m = np.concatenate([zStart_m, zEnd_m] + [c[:, 2] for c in corners])
    if allX_m.size == 0:
        plt.close(figure)
        return None

    has_markers = False
    if loadPoints and plotSession.showLoads:
        lp = np.asarray(loadPoints, dtype=float)
        axis.scatter(
            lp[:, 0], lp[:, 1], lp[:, 2],
            marker="s", s=26, color="limegreen", edgecolors="black",
            linewidths=0.6, alpha=0.85, depthshade=False, zorder=9, label=f"Load ({len(lp)})",
        )
        has_markers = True
    if sourcePoints and plotSession.showSources:
        sp = np.asarray(sourcePoints, dtype=float)
        axis.scatter(
            sp[:, 0], sp[:, 1], sp[:, 2],
            marker="D", s=60, color="magenta", edgecolors="black",
            linewidths=0.9, depthshade=False, zorder=10, label=f"Source ({len(sp)})",
        )
        has_markers = True
    if has_markers:
        axis.legend(loc="upper right", fontsize=8.5)
    if sourceSummary and plotSession.showSources:
        axis.text2D(
            0.02,
            0.97,
            sourceSummary,
            transform=axis.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="magenta", alpha=0.88),
        )

    setEqualAxes(axis, allX_m, allY_m, allZ_m, np)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(title)
    return finishFigure(figure, "antennaGeometry3d", plt)


def gridPattern(theta_deg, phi_deg, gain_dB, np):
    """Arrange samples on a regular theta/phi grid; None if not a grid."""

    thetaKey = np.round(theta_deg, 6)
    phiKey = np.round(phi_deg, 6)
    thetaGrid = np.unique(thetaKey)
    phiGrid = np.unique(phiKey)
    if thetaGrid.size < 2 or phiGrid.size < 2:
        return None
    thetaIndex = np.searchsorted(thetaGrid, thetaKey)
    phiIndex = np.searchsorted(phiGrid, phiKey)
    gainGrid = np.full((phiGrid.size, thetaGrid.size), np.nan)
    gainGrid[phiIndex, thetaIndex] = gain_dB
    # Repeat the first phi cut at phi + 360 so the wireframe closes.
    span_deg = phiGrid[-1] - phiGrid[0]
    step_deg = float(np.median(np.diff(phiGrid)))
    if span_deg < 360 - 1e-6 and span_deg + step_deg >= 360 - 1e-6:
        phiGrid = np.append(phiGrid, phiGrid[0] + 360)
        gainGrid = np.vstack((gainGrid, gainGrid[:1]))
    return thetaGrid, phiGrid, gainGrid

def plotRadiationWireframe3d(theta_deg, phi_deg, gain_dB,
    dynamicRange_dB=35.0, frequency_MHz=None, suffix=""):
    """Plot gain samples as a normalized, gain-colored 3D wireframe pattern."""
    np, plt = importPlotLibraries()
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    theta_deg = np.asarray(theta_deg, dtype=float)
    phi_deg = np.asarray(phi_deg, dtype=float)
    gain_dB = np.asarray(gain_dB, dtype=float)
    grid = gridPattern(theta_deg, phi_deg, gain_dB, np)
    if grid is None:
        return None
    thetaGrid, phiGrid, gainGrid = grid
    theta_rad, phi_rad = np.meshgrid(np.radians(thetaGrid), np.radians(phiGrid))

    gMax = float(np.nanmax(gainGrid))
    gFloor = max(float(np.nanmin(gainGrid)), gMax - abs(float(dynamicRange_dB)))
    if np.isclose(gMax, gFloor):
        gFloor = gMax - 1.0

    # Clamp radial scale to the same dynamic range so nulls converge cleanly at origin
    relGain = np.clip(gainGrid - gMax, -abs(float(dynamicRange_dB)), 0.0)
    radius = 10 ** (relGain / 20.0)
    radius = np.nan_to_num(radius, nan=0.0)
    xValue = radius * np.sin(theta_rad) * np.cos(phi_rad)
    yValue = radius * np.sin(theta_rad) * np.sin(phi_rad)
    zValue = radius * np.cos(theta_rad)

    segments = []
    segGains = []
    nPhi, nTheta = xValue.shape

    for i in range(nPhi):
        for j in range(nTheta - 1):
            segments.append([(xValue[i, j], yValue[i, j], zValue[i, j]),
                             (xValue[i, j + 1], yValue[i, j + 1], zValue[i, j + 1])])
            segGains.append(0.5 * (gainGrid[i, j] + gainGrid[i, j + 1]))

    for i in range(nPhi - 1):
        for j in range(nTheta):
            segments.append([(xValue[i, j], yValue[i, j], zValue[i, j]),
                             (xValue[i + 1, j], yValue[i + 1, j], zValue[i + 1, j])])
            segGains.append(0.5 * (gainGrid[i, j] + gainGrid[i + 1, j]))

    norm = colors.Normalize(vmin=gFloor, vmax=gMax, clip=True)
    cmap = cm.get_cmap("turbo")
    lineCol = Line3DCollection(segments, colors=cmap(norm(np.asarray(segGains))), linewidths=0.9)

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(lineCol)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.clip(gainGrid, gFloor, gMax))
    figure.colorbar(mappable, ax=axis, shrink=0.65, label=f"Total Gain (dBi, {dynamicRange_dB:.0f} dB range)")

    setEqualAxes(axis, xValue, yValue, zValue, np)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    freqLabel = f" @ {frequency_MHz:.4g} MHz" if frequency_MHz is not None else ""
    axis.set_title(f"3D Radiation Pattern{freqLabel} (Peak Gain: {gMax:.2f} dBi)")
    return finishFigure(figure, f"radiationPatternWireframe3d{suffix}", plt)


def recordRadiationSample(theta_deg, phi_deg, gain_dB, frequency_MHz=None):
    """Record one far-field gain sample tagged with its operating frequency."""
    if not plotSession.enabled or not plotSession.patternsEnabled:
        return
    freqKey = round(float(frequency_MHz), 6) if frequency_MHz is not None else None
    plotSession.patternSamples.append(
        (freqKey, float(theta_deg), float(phi_deg), float(gain_dB))
    )


def plotRadiationPattern3d(theta_deg, phi_deg, gain_dB, frequency_MHz=None, suffix="", dynamicRange_dB=35.0):
    """Plot gain samples as a normalized three-dimensional pattern."""

    if not plotSession.enabled or not plotSession.patternsEnabled:
        return None
    if plotSession.patternStyle == "wireframe":
        path = plotRadiationWireframe3d(
            theta_deg,
            phi_deg,
            gain_dB,
            dynamicRange_dB=dynamicRange_dB,
            frequency_MHz=frequency_MHz,
            suffix=suffix,
        )
        if path is not None:
            return path
        # Samples are not on a theta/phi grid, so fall back to a surface.
    np, plt = importPlotLibraries()
    from matplotlib import cm, colors

    theta_rad = np.radians(np.asarray(theta_deg, dtype=float))
    phi_rad = np.radians(np.asarray(phi_deg, dtype=float))
    gain_dB = np.asarray(gain_dB, dtype=float)
    if gain_dB.size < 3:
        return None

    gMax = float(np.nanmax(gain_dB))
    gFloor = max(float(np.nanmin(gain_dB)), gMax - abs(float(dynamicRange_dB)))
    if np.isclose(gMax, gFloor):
        gFloor = gMax - 1.0

    relGain = np.clip(gain_dB - gMax, -abs(float(dynamicRange_dB)), 0.0)
    radius = 10 ** (relGain / 20.0)
    xValue = radius * np.sin(theta_rad) * np.cos(phi_rad)
    yValue = radius * np.sin(theta_rad) * np.sin(phi_rad)
    zValue = radius * np.cos(theta_rad)

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    norm = colors.Normalize(vmin=gFloor, vmax=gMax, clip=True)
    cmap = cm.get_cmap("turbo")

    drawn_surface = False
    if np.unique(np.round(phi_rad, 6)).size >= 2 and np.unique(np.round(theta_rad, 6)).size >= 2:
        try:
            axis.plot_trisurf(
                xValue,
                yValue,
                zValue,
                cmap=cmap,
                norm=norm,
                linewidth=0.15,
                antialiased=True,
            )
            drawn_surface = True
        except (RuntimeError, ValueError):
            drawn_surface = False

    if not drawn_surface:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        pts = np.column_stack((xValue, yValue, zValue))
        segs = np.stack((pts[:-1], pts[1:]), axis=1)
        segGains = 0.5 * (gain_dB[:-1] + gain_dB[1:])
        lineCol = Line3DCollection(segs, colors=cmap(norm(segGains)), linewidths=2.0)
        axis.add_collection3d(lineCol)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.clip(gain_dB, gFloor, gMax))
    figure.colorbar(mappable, ax=axis, shrink=0.65, label=f"Total Gain (dBi, {dynamicRange_dB:.0f} dB range)")
    setEqualAxes(axis, xValue, yValue, zValue, np)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    freqLabel = f" @ {frequency_MHz:.4g} MHz" if frequency_MHz is not None else ""
    axis.set_title(f"3D Radiation Pattern{freqLabel} (Peak Gain: {gMax:.2f} dBi)")
    return finishFigure(figure, f"radiationPattern3d{suffix}", plt)



def recordScalarField(name, x_m, y_m, z_m, value, unit=""):
    """Record named scalar data for a generic 3D scatter plot."""

    if not plotSession.enabled or not plotSession.fieldsEnabled:
        return
    sample = {
        "name": str(name),
        "x_m": float(x_m),
        "y_m": float(y_m),
        "z_m": float(z_m),
        "value": float(value),
        "unit": str(unit),
    }
    plotSession.fieldSamples.append(sample)


def plotScalarField3d(name, x_m, y_m, z_m, values, unit=""):
    """Plot arbitrary scalar samples in Cartesian three-space."""

    if not plotSession.enabled or not plotSession.fieldsEnabled:
        return None
    np, plt = importPlotLibraries()
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    z_m = np.asarray(z_m, dtype=float)
    values = np.asarray(values, dtype=float)
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    points = axis.scatter(x_m, y_m, z_m, c=values, cmap="plasma")
    label = str(name) if not unit else f"{name} ({unit})"
    figure.colorbar(points, ax=axis, shrink=0.65, label=label)
    setEqualAxes(axis, x_m, y_m, z_m, np)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(str(name))
    safeName = "".join(character if character.isalnum() else "_"
                       for character in str(name)).strip("_")
    return finishFigure(figure, f"{safeName or 'field'}3d", plt)


def recordSweepPoint(frequency_MHz, z_ohms=None, peakGain_dBi=None, z0_ohms=50.0):
    """Record one frequency sweep sample for 2D VSWR, impedance, and gain plots."""
    if not plotSession.enabled:
        return
    freq = float(frequency_MHz)
    r_val = float(z_ohms.real) if z_ohms is not None else float("nan")
    x_val = float(z_ohms.imag) if z_ohms is not None else float("nan")
    vswr = float("nan")
    if z_ohms is not None and z0_ohms > 0.0:
        gamma = abs((z_ohms - z0_ohms) / (z_ohms + z0_ohms))
        vswr = (1.0 + gamma) / max(1.0 - gamma, 1e-6)
        vswr = min(vswr, 99.9)
    gain = float(peakGain_dBi) if peakGain_dBi is not None else float("nan")
    plotSession.sweepSamples.append((freq, r_val, x_val, vswr, gain, float(z0_ohms)))


def plotFrequencySweep():
    """Plot 2D VSWR, feedpoint impedance (R + jX), and peak gain vs. frequency."""
    if not plotSession.enabled or len(plotSession.sweepSamples) < 2:
        return None
    np, plt = importPlotLibraries()

    data = sorted(plotSession.sweepSamples, key=lambda row: row[0])
    freqs = np.array([f for f, _, _, _, _, _ in data], dtype=float)
    r_vals = np.array([r for _, r, _, _, _, _ in data], dtype=float)
    x_vals = np.array([x for _, _, x, _, _, _ in data], dtype=float)
    vswr_vals = np.array([v for _, _, _, v, _, _ in data], dtype=float)
    gain_vals = np.array([g for _, _, _, _, g, _ in data], dtype=float)
    _, _, _, _, _, z0 = data[-1]

    has_z = not np.all(np.isnan(r_vals))
    has_gain = not np.all(np.isnan(gain_vals))
    if not has_z and not has_gain:
        return None

    panels = []
    if has_z:
        panels.extend(["vswr", "z"])
    if has_gain:
        panels.append("gain")

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3.0 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    ax_idx = 0
    if has_z:
        ax_vswr = axes[ax_idx]
        ax_idx += 1
        ax_vswr.plot(freqs, vswr_vals, marker="o", markersize=4, color="tab:red", linewidth=1.6)
        ax_vswr.axhline(2.0, color="gray", linestyle="--", linewidth=0.9, label="2.0:1 VSWR")
        min_i = int(np.nanargmin(vswr_vals))
        ax_vswr.annotate(
            f"Min: {vswr_vals[min_i]:.2f}:1 @ {freqs[min_i]:.4g} MHz",
            xy=(freqs[min_i], vswr_vals[min_i]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="tab:red", alpha=0.85),
        )
        ax_vswr.set_ylabel(f"VSWR ({z0:.0f} Ω)")
        ax_vswr.set_ylim(bottom=1.0, top=max(3.5, min(float(np.nanmax(vswr_vals)) * 1.15, 20.0)))
        ax_vswr.grid(True, linestyle=":", alpha=0.6)
        ax_vswr.legend(loc="upper right", fontsize=8.5)
        ax_vswr.set_title("Frequency Sweep Performance")

        ax_z = axes[ax_idx]
        ax_idx += 1
        ax_z.plot(freqs, r_vals, marker="o", markersize=4, color="tab:blue", linewidth=1.6, label="Resistance R (Ω)")
        ax_z.plot(freqs, x_vals, marker="s", markersize=4, color="tab:orange", linewidth=1.6, label="Reactance X (Ω)")
        ax_z.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
        ax_z.set_ylabel("Impedance (Ω)")
        ax_z.grid(True, linestyle=":", alpha=0.6)
        ax_z.legend(loc="best", fontsize=8.5)

    if has_gain:
        ax_g = axes[ax_idx]
        ax_g.plot(freqs, gain_vals, marker="^", markersize=4, color="tab:green", linewidth=1.6, label="Peak Gain (dBi)")
        max_g = int(np.nanargmax(gain_vals))
        ax_g.annotate(
            f"Peak: {gain_vals[max_g]:.2f} dBi @ {freqs[max_g]:.4g} MHz",
            xy=(freqs[max_g], gain_vals[max_g]),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="tab:green", alpha=0.85),
        )
        ax_g.set_ylabel("Peak Gain (dBi)")
        ax_g.grid(True, linestyle=":", alpha=0.6)
        if not has_z:
            ax_g.set_title("Frequency Sweep Performance")

    axes[-1].set_xlabel("Frequency (MHz)")
    return finishFigure(fig, "frequencySweep", plt)


def _beamwidth_deg(angles_deg, rel_dB, peak_idx):
    """Estimate -3 dB beamwidth around peak_idx on a sorted 1D cut."""
    import numpy as np
    n = len(angles_deg)
    if n < 3:
        return None
    left = peak_idx
    while left > 0 and rel_dB[left] > -3.0:
        left -= 1
    right = peak_idx
    while right < n - 1 and rel_dB[right] > -3.0:
        right += 1
    if rel_dB[left] > -3.0 or rel_dB[right] > -3.0:
        return None
    # Linear interpolation at -3 dB crossings
    ang_l = angles_deg[left]
    if left + 1 <= peak_idx and rel_dB[left + 1] != rel_dB[left]:
        frac = (-3.0 - rel_dB[left]) / (rel_dB[left + 1] - rel_dB[left])
        ang_l = angles_deg[left] + frac * (angles_deg[left + 1] - angles_deg[left])
    ang_r = angles_deg[right]
    if right - 1 >= peak_idx and rel_dB[right - 1] != rel_dB[right]:
        frac = (-3.0 - rel_dB[right]) / (rel_dB[right - 1] - rel_dB[right])
        ang_r = angles_deg[right] + frac * (angles_deg[right - 1] - angles_deg[right])
    bw = abs(ang_r - ang_l)
    return bw if 0.1 < bw < 359.0 else None


def plotPolarCuts2d(series_list, dynamicRange_dB=35.0):
    """Plot 2D polar azimuth & elevation cuts (single frequency or multi-frequency overlay)."""
    if not plotSession.enabled or not plotSession.patternsEnabled or not series_list:
        return None
    np, plt = importPlotLibraries()
    from matplotlib import cm

    # Find reference peak direction & global peak gain from the highest-gain frequency
    best_idx = 0
    g_global_max = -1e9
    for idx, (freq, ths, phs, gs) in enumerate(series_list):
        g_arr = np.asarray(gs, dtype=float)
        if g_arr.size >= 3 and float(np.nanmax(g_arr)) > g_global_max:
            g_global_max = float(np.nanmax(g_arr))
            best_idx = idx

    ref_freq, ref_th, ref_ph, ref_g = series_list[best_idx]
    ref_th_arr = np.asarray(ref_th, dtype=float)
    ref_ph_arr = np.asarray(ref_ph, dtype=float)
    ref_g_arr = np.asarray(ref_g, dtype=float)
    if ref_g_arr.size < 3:
        return None

    peak_i = int(np.nanargmax(ref_g_arr))
    th_peak = float(ref_th_arr.item(peak_i))
    ph_peak = float(ref_ph_arr.item(peak_i)) % 360.0
    ph_opp = (ph_peak + 180.0) % 360.0
    elev_peak = 90.0 - th_peak

    has_az = np.unique(np.round(ref_ph_arr[np.isclose(ref_th_arr, th_peak, atol=1e-3)] % 360.0, 4)).size >= 4
    has_el = np.unique(np.round(ref_th_arr[np.isclose(ref_ph_arr % 360.0, ph_peak, atol=1e-3)], 4)).size >= 4
    if not has_az and not has_el:
        return None

    ncols = int(has_az) + int(has_el)
    fig, axes = plt.subplots(1, ncols, figsize=(5.8 * ncols, 5.4), subplot_kw={"projection": "polar"})
    if ncols == 1:
        axes = [axes]

    dr = abs(float(dynamicRange_dB))
    rings = np.arange(-dr, 1.0, 5.0)
    cmap = cm.get_cmap("turbo")
    n_series = len(series_list)
    colors_list = ["tab:blue"] if n_series == 1 else [cmap(i / max(n_series - 1, 1)) for i in range(n_series)]

    ax_az = axes[0] if has_az else None
    ax_el = axes[1 if has_az else 0] if has_el else None
    single_fb_text = ""
    single_bw_az = ""
    single_bw_el = ""

    for s_idx, (freq, ths, phs, gs) in enumerate(series_list):
        theta_arr = np.asarray(ths, dtype=float)
        phi_arr = np.asarray(phs, dtype=float)
        gain_arr = np.asarray(gs, dtype=float)
        col = colors_list[s_idx]
        lbl = f"{freq:.4g} MHz" if freq is not None else "Pattern"

        if ax_az is not None:
            az_mask = np.isclose(theta_arr, th_peak, atol=1e-3)
            az_phi = phi_arr[az_mask] % 360.0
            az_gain = gain_arr[az_mask]
            if az_phi.size >= 4:
                order = np.argsort(az_phi)
                p_sorted = az_phi[order]
                g_sorted = az_gain[order]
                rel_az = np.clip(g_sorted - g_global_max, -dr, 0.0)

                if n_series == 1:
                    rear_diff = np.abs(((p_sorted - ph_opp + 180.0) % 360.0) - 180.0)
                    rear_mask = rear_diff <= 30.0
                    if np.any(rear_mask):
                        fb_db = g_global_max - float(np.nanmax(g_sorted[rear_mask]))
                        single_fb_text = f" | F/B: {fb_db:.1f} dB"
                    p_shifted = ((p_sorted - ph_peak + 180.0) % 360.0) - 180.0
                    s_ord = np.argsort(p_shifted)
                    pk_local = int(np.argmin(np.abs(p_shifted[s_ord])))
                    bw_az = _beamwidth_deg(p_shifted[s_ord], (g_sorted[s_ord] - g_global_max), pk_local)
                    if bw_az is not None:
                        single_bw_az = f" | BW: {bw_az:.1f}°"

                p_first = float(p_sorted.item(0))
                p_last = float(p_sorted.item(-1))
                if (p_last - p_first) >= 340.0:
                    p_plot = np.append(p_sorted, p_first + 360.0)
                    r_plot = np.append(rel_az, float(rel_az.item(0)))
                else:
                    p_plot, r_plot = p_sorted, rel_az

                ax_az.plot(np.radians(p_plot), r_plot, color=col, linewidth=1.6, label=lbl)

        if ax_el is not None:
            el_mask_fwd = np.isclose(phi_arr % 360.0, ph_peak, atol=1e-3)
            el_mask_rev = np.isclose(phi_arr % 360.0, ph_opp, atol=1e-3)
            if np.any(el_mask_fwd):
                th_f = theta_arr[el_mask_fwd]
                g_f = gain_arr[el_mask_fwd]
                ord_f = np.argsort(th_f)
                th_f, g_f = th_f[ord_f], g_f[ord_f]

                pol_ang = list(90.0 - th_f)
                pol_gain = list(g_f)
                if np.any(el_mask_rev):
                    th_r = theta_arr[el_mask_rev]
                    g_r = gain_arr[el_mask_rev]
                    ord_r = np.argsort(th_r)[::-1]
                    pol_ang.extend(list(90.0 + th_r[ord_r]))
                    pol_gain.extend(list(g_r[ord_r]))

                pol_ang = np.asarray(pol_ang, dtype=float)
                pol_gain = np.asarray(pol_gain, dtype=float)
                rel_el = np.clip(pol_gain - g_global_max, -dr, 0.0)

                if n_series == 1:
                    pk_el = int(np.nanargmax(g_f))
                    bw_el = _beamwidth_deg(90.0 - th_f[::-1], (g_f[::-1] - g_global_max), len(th_f) - 1 - pk_el)
                    if bw_el is not None:
                        single_bw_el = f" | BW: {bw_el:.1f}°"

                ang_first = float(pol_ang.item(0))
                ang_last = float(pol_ang.item(-1))
                if pol_ang.size >= 4 and abs((ang_last - ang_first) % 360.0) <= 15.0:
                    pol_ang = np.append(pol_ang, ang_first)
                    rel_el = np.append(rel_el, float(rel_el.item(0)))

                ax_el.plot(np.radians(pol_ang), rel_el, color=col, linewidth=1.6, label=lbl)

    if ax_az is not None:
        ax_az.set_theta_zero_location("E")
        ax_az.set_theta_direction(1)
        ax_az.set_ylim(-dr, 0.0)
        ax_az.set_rticks(rings)
        ax_az.set_yticklabels([f"{int(r)}" if int(r) % 10 == 0 else "" for r in rings], fontsize=7.5)
        ax_az.grid(True, linestyle=":", alpha=0.7)
        ax_az.set_title(f"Azimuth Cut (Elev = {elev_peak:.1f}°){single_bw_az}", pad=14, fontsize=10)

    if ax_el is not None:
        ax_el.set_theta_zero_location("E")
        ax_el.set_theta_direction(1)
        ax_el.set_ylim(-dr, 0.0)
        ax_el.set_rticks(rings)
        ax_el.set_yticklabels([f"{int(r)}" if int(r) % 10 == 0 else "" for r in rings], fontsize=7.5)
        ax_el.grid(True, linestyle=":", alpha=0.7)
        ax_el.set_title(f"Elevation Cut (Az = {ph_peak:.1f}°){single_bw_el}", pad=14, fontsize=10)
        if n_series > 1:
            ax_el.legend(loc="upper left", bbox_to_anchor=(1.08, 1.02), fontsize=8, title="Frequency")
    elif ax_az is not None and n_series > 1:
        ax_az.legend(loc="upper left", bbox_to_anchor=(1.08, 1.02), fontsize=8, title="Frequency")

    if n_series == 1:
        freqLabel = f" @ {ref_freq:.4g} MHz" if ref_freq is not None else ""
        fig.suptitle(f"2D Polar Cuts{freqLabel} — Peak: {g_global_max:.2f} dBi{single_fb_text}", fontsize=11.5, y=0.98)
    else:
        f_lo, _, _, _ = series_list[0]
        f_hi, _, _, _ = series_list[-1]
        fig.suptitle(
            f"2D Polar Cuts ({f_lo:.4g}–{f_hi:.4g} MHz) — Peak: {g_global_max:.2f} dBi @ {ref_freq:.4g} MHz",
            fontsize=11.5,
            y=0.98,
        )
    fig.subplots_adjust(top=0.84)
    return finishFigure(fig, "polarCuts", plt)


def finalizePlots():
    """Render samples, then keep all interactive figures open together."""

    paths = []
    if plotSession.patternSamples:
        freqs = sorted(
            {freq for freq, _, _, _ in plotSession.patternSamples},
            key=lambda f: -1.0 if f is None else f,
        )
        series_list = []
        best_series = None
        best_gain = -1e9

        for freq in freqs:
            subset = [
                (theta, phi, gain)
                for f, theta, phi, gain in plotSession.patternSamples
                if f == freq
            ]
            thetas = [theta for theta, _, _ in subset]
            phis = [phi for _, phi, _ in subset]
            gains = [gain for _, _, gain in subset]
            series_list.append((freq, thetas, phis, gains))
            if gains and max(gains) > best_gain:
                best_gain = max(gains)
                best_series = (freq, thetas, phis, gains)

        # Render a single 3D pattern at the peak-gain frequency (avoids dozens of 3D windows)
        if best_series is not None:
            b_freq, b_th, b_ph, b_g = best_series
            path = plotRadiationPattern3d(
                b_th,
                b_ph,
                b_g,
                frequency_MHz=b_freq,
                suffix="",
            )
            if path is not None:
                paths.append(path)

        # Render all frequencies overlaid on the 2D polar Azimuth & Elevation cuts
        polar_path = plotPolarCuts2d(series_list)
        if polar_path is not None:
            paths.append(polar_path)

    if plotSession.fieldSamples:
        names = sorted({item["name"] for item in plotSession.fieldSamples})
        for name in names:
            samples = [
                item for item in plotSession.fieldSamples
                if item["name"] == name
            ]
            first_sample = next(iter(samples))
            path = plotScalarField3d(
                name,
                [item["x_m"] for item in samples],
                [item["y_m"] for item in samples],
                [item["z_m"] for item in samples],
                [item["value"] for item in samples],
                first_sample["unit"],
            )
            if path is not None:
                paths.append(path)

    sweep_path = plotFrequencySweep()
    if sweep_path is not None:
        paths.append(sweep_path)

    if plotSession.showPlots:
        _, plt = importPlotLibraries()
        if plt.get_fignums():
            plt.show(block=True)
    return paths

