"""NEC input-deck parsing, editing, validation, and geometry helpers."""

from dataclasses import dataclass
from math import cos, pi, radians, sin, sqrt
from pathlib import Path


@dataclass(frozen=True)
class CardSpec:
    """Describe the fields and purpose of one common NEC card."""

    name: str
    fields: tuple
    description: str


@dataclass
class DeckCard:
    """Represent one editable line in a NEC input deck."""

    code: str
    values: list
    original: str = ""

    def format(self):
        """Return a comma-separated card accepted by the NEC parser."""

        code = self.code.upper().strip()[:2]
        if code in {"CM", "CE"}:
            text = self.values[0] if self.values else ""
            if text and not text.startswith(" "):
                text = " " + text
            return f"{code}{text}"
        if not self.values:
            return code
        return f"{code} " + ",".join(self.values)


CARD_SPECS = {
    "CM": CardSpec(
        "Comment",
        ("Text",),
        "Comment line. Initial comments end with a CE card.",
    ),
    "CE": CardSpec(
        "Comment End",
        ("Text",),
        "Ends the initial comment section.",
    ),
    "GW": CardSpec(
        "Wire",
        (
            "Tag",
            "Segments",
            "X1_m",
            "Y1_m",
            "Z1_m",
            "X2_m",
            "Y2_m",
            "Z2_m",
            "Radius_m",
        ),
        "Creates a straight wire divided into equal segments.",
    ),
    "GA": CardSpec(
        "Arc",
        (
            "Tag",
            "Segments",
            "Radius_m",
            "Start_deg",
            "End_deg",
            "WireRadius_m",
        ),
        "Creates a circular wire arc in the X-Z plane.",
    ),
    "GH": CardSpec(
        "Helix",
        (
            "Tag",
            "Segments",
            "TurnSpacing_m",
            "Length_m",
            "StartRadiusX_m",
            "StartRadiusY_m",
            "EndRadiusX_m",
            "EndRadiusY_m",
            "WireRadius_m",
        ),
        "Creates a helix or tapered helix.",
    ),
    "GM": CardSpec(
        "Move",
        (
            "TagIncrement",
            "Copies",
            "RotateX_deg",
            "RotateY_deg",
            "RotateZ_deg",
            "TranslateX_m",
            "TranslateY_m",
            "TranslateZ_m",
            "StartTag",
        ),
        "Rotates, translates, or copies existing geometry.",
    ),
    "GR": CardSpec(
        "Rotate",
        ("TagIncrement", "Copies"),
        "Creates rotational copies about the Z axis.",
    ),
    "GS": CardSpec(
        "Scale",
        ("Conversion", "Unused", "Scale"),
        "Scales all geometry coordinates and wire radii.",
    ),
    "GX": CardSpec(
        "Reflect",
        ("TagIncrement", "Axes"),
        "Reflects geometry about selected coordinate planes.",
    ),
    "SP": CardSpec(
        "Surface Patch",
        (
            "Tag",
            "Shape",
            "X1_m",
            "Y1_m",
            "Z1_m",
            "X2_orTheta",
            "Y2_orPhi",
            "Z2_m",
        ),
        "Starts one arbitrary, rectangular, triangular, or quad patch.",
    ),
    "SC": CardSpec(
        "Patch Continuation",
        (
            "Unused",
            "Shape",
            "X3_m",
            "Y3_m",
            "Z3_m",
            "X4_m",
            "Y4_m",
            "Z4_m",
        ),
        "Continues the preceding SP surface-patch card.",
    ),
    "SM": CardSpec(
        "Multiple Patches",
        (
            "CountX",
            "CountY",
            "X1_m",
            "Y1_m",
            "Z1_m",
            "X2_m",
            "Y2_m",
            "Z2_m",
        ),
        "Starts a rectangular grid of surface patches.",
    ),
    "GE": CardSpec(
        "Geometry End",
        ("GroundFlag", "PlotFlag"),
        "Ends geometry input and establishes connectivity.",
    ),
    "FR": CardSpec(
        "Frequency",
        (
            "StepType",
            "Count",
            "Unused1",
            "Unused2",
            "Start_MHz",
            "Step_MHz_orRatio",
        ),
        "Selects one frequency or a linear or multiplicative sweep.",
    ),
    "EX": CardSpec(
        "Excitation",
        (
            "Type",
            "Tag",
            "Segment",
            "Flags",
            "Real_V",
            "Imag_V",
            "Param3",
            "Param4",
            "Param5",
            "Param6",
        ),
        "Defines a voltage source, plane wave, or current source.",
    ),
    "GN": CardSpec(
        "Ground",
        (
            "Type",
            "Radials",
            "Unused1",
            "Unused2",
            "RelativePermittivity",
            "Conductivity_Spm",
            "RadialLength_m",
            "RadialRadius_m",
            "SecondPermittivity",
            "SecondConductivity_Spm",
        ),
        "Selects free space, perfect ground, or finite ground.",
    ),
    "LD": CardSpec(
        "Load",
        (
            "Type",
            "Tag",
            "FromSegment",
            "ThroughSegment",
            "Resistance_Ohm",
            "Inductance_H",
            "Capacitance_F",
        ),
        "Applies lumped, distributed, or conductivity loading.",
    ),
    "RP": CardSpec(
        "Radiation Pattern",
        (
            "Mode",
            "ThetaCount",
            "PhiCount",
            "Flags",
            "StartTheta_deg",
            "StartPhi_deg",
            "ThetaStep_deg",
            "PhiStep_deg",
            "Range_m",
            "Normalization_dB",
        ),
        "Requests far-field samples used by the 3D pattern plot.",
    ),
    "NE": CardSpec(
        "Near Electric Field",
        (
            "Coordinates",
            "CountX",
            "CountY",
            "CountZ",
            "StartX_m",
            "StartY_m",
            "StartZ_m",
            "StepX_m",
            "StepY_m",
            "StepZ_m",
        ),
        "Requests electric-field samples on a rectangular or spherical grid.",
    ),
    "NH": CardSpec(
        "Near Magnetic Field",
        (
            "Coordinates",
            "CountX",
            "CountY",
            "CountZ",
            "StartX_m",
            "StartY_m",
            "StartZ_m",
            "StepX_m",
            "StepY_m",
            "StepZ_m",
        ),
        "Requests magnetic-field samples on a rectangular or spherical grid.",
    ),
    "GC": CardSpec(
        "Wire Taper",
        (
            "Unused1",
            "Unused2",
            "SegmentLengthRatio",
            "StartRadius_m",
            "EndRadius_m",
        ),
        "Tapers the preceding GW wire, which must have radius 0.",
    ),
    "GF": CardSpec(
        "Green Function Read",
        ("PrintFlag",),
        "Reads a numerical Green's function file. Must be the first "
        "geometry card.",
    ),
    "GP": CardSpec(
        "Suppress Geometry Print",
        (),
        "NEC-81 extension: suppresses the printed geometry listing.",
    ),
    "GD": CardSpec(
        "Second Medium",
        (
            "Unused1",
            "Unused2",
            "Unused3",
            "Unused4",
            "RelativePermittivity",
            "Conductivity_Spm",
            "BoundaryDistance_m",
            "Depth_m",
        ),
        "Ground parameters of a second medium used only for far fields. "
        "Requires a GN card.",
    ),
    "CP": CardSpec(
        "Coupling",
        ("Tag1", "Segment1", "Tag2", "Segment2"),
        "Selects up to five segments for maximum-coupling calculation.",
    ),
    "NT": CardSpec(
        "Network",
        (
            "Tag1",
            "Segment1",
            "Tag2",
            "Segment2",
            "Y11Real_S",
            "Y11Imag_S",
            "Y12Real_S",
            "Y12Imag_S",
            "Y22Real_S",
            "Y22Imag_S",
        ),
        "Two-port network given by its short-circuit admittance matrix. "
        "Segment1 of -1 cancels earlier networks and lines.",
    ),
    "TL": CardSpec(
        "Transmission Line",
        (
            "Tag1",
            "Segment1",
            "Tag2",
            "Segment2",
            "Z0_Ohm",
            "Length_m",
            "Y1Real_S",
            "Y1Imag_S",
            "Y2Real_S",
            "Y2Imag_S",
        ),
        "Transmission line between two segments; a negative Z0 crosses "
        "the line. Length 0 uses the straight-line distance.",
    ),
    "PL": CardSpec(
        "Plot Flags",
        ("Flag1", "Flag2", "Flag3", "Flag4"),
        "Sets the plot-output flags (documented only in the NEC-81 source).",
    ),
    "WG": CardSpec(
        "Green Function Write",
        (),
        "Writes a numerical Green's function file after the calculation.",
    ),
    "XQ": CardSpec(
        "Execute",
        ("Option",),
        "Executes the current structure and requested calculations.",
    ),
    "EN": CardSpec(
        "End",
        (),
        "Ends the input deck and exits NEC-81 normally.",
    ),
    "EK": CardSpec(
        "Extended Kernel",
        ("Option",),
        "Enables or disables the extended thin-wire kernel.",
    ),
    "KH": CardSpec(
        "Interaction Range",
        ("Unused1", "Unused2", "Unused3", "Unused4", "Range"),
        "Sets the matrix integration approximation range.",
    ),
    "PT": CardSpec(
        "Current Print",
        ("Control", "Tag", "FromSegment", "ThroughSegment"),
        "Controls printed wire-current output.",
    ),
    "PQ": CardSpec(
        "Charge Print",
        ("Control", "Tag", "FromSegment", "ThroughSegment"),
        "Controls printed charge-density output.",
    ),
}


def splitValues(code, tail):
    """Split comma-separated or whitespace-separated card values."""

    if code in {"CM", "CE"}:
        return [tail]
    if not tail.strip():
        return []
    if "," in tail:
        return [value.strip() for value in tail.split(",")]
    return tail.split()


def parseCard(line):
    """Parse one deck line while preserving its original text."""

    original = line.rstrip("\r\n")
    if not original.strip():
        return DeckCard("", [], original)
    code = original[:2].upper()
    tail = original[2:]
    return DeckCard(code, splitValues(code, tail), original)


def parseDeck(text):
    """Parse deck text into an editable card list."""

    return [parseCard(line) for line in text.splitlines()]


def formatDeck(cards):
    """Format a card list as a complete text deck."""

    return "\n".join(card.format() for card in cards) + "\n"


def loadDeck(deckPath):
    """Read and parse a NEC input deck."""

    return parseDeck(Path(deckPath).read_text(encoding="utf-8"))


def saveDeck(deckPath, cards):
    """Write cards to a NEC input deck."""

    Path(deckPath).write_text(formatDeck(cards), encoding="utf-8")


def cardSummary(card):
    """Return a concise card summary for the GUI table."""

    if not card.code:
        return "Blank line"
    spec = CARD_SPECS.get(card.code)
    if card.code in {"CM", "CE"}:
        return card.values[0].strip() if card.values else ""
    if spec is None:
        return ", ".join(card.values)
    shown = []
    for fieldName, value in zip(spec.fields, card.values):
        if value:
            shown.append(f"{fieldName}={value}")
        if len(shown) == 3:
            break
    return ", ".join(shown) or spec.name


def validateDeck(cards):
    """Return human-readable deck validation messages."""

    messages = []
    codes = [card.code for card in cards if card.code]
    if "CE" not in codes:
        messages.append("Error: the initial comment section needs a CE card.")
    if "GE" not in codes:
        messages.append("Error: geometry needs a GE end card.")
    if "EN" not in codes:
        messages.append("Error: the deck needs an EN end card.")
    if "GW" not in codes and not any(
        code in codes for code in ("GA", "GH", "SP", "SM")
    ):
        messages.append("Warning: no wire or surface geometry was found.")
    for number, card in enumerate(cards, start=1):
        if not card.code:
            continue
        if len(card.code) != 2 or not card.code.isalpha():
            messages.append(f"Line {number}: invalid card code {card.code!r}.")
        spec = CARD_SPECS.get(card.code)
        if spec and len(card.values) > len(spec.fields):
            messages.append(
                f"Line {number}: {card.code} has more values than expected."
            )
    messages.extend(patchCardMessages(cards))
    if not messages:
        messages.append("No structural deck errors were found.")
    return messages


def patchCardMessages(cards):
    """Check that SP and SM cards are followed by their SC card."""

    messages = []
    previous = ""
    for number, card in enumerate(cards, start=1):
        code = card.code
        if code in {"SP", "SM"} and card.values:
            # SP shape 0 is an arbitrary patch and needs no SC card.
            needsContinuation = code == "SM" or (
                len(card.values) > 1 and card.values[1].strip() not in
                {"", "0"})
            following = cards[number] if number < len(cards) else None
            if needsContinuation and (
                following is None or following.code != "SC"
            ):
                messages.append(
                    f"Error: line {number}: {code} must be followed by "
                    "an SC card."
                )
        elif code == "SC" and previous not in {"SP", "SM", "SC"}:
            messages.append(
                f"Error: line {number}: SC card has no preceding SP or SM."
            )
        if code:
            previous = code
    return messages


def parseFloat(value, cardNumber, fieldName):
    """Parse one numeric field and provide a useful error."""

    try:
        return float(value)
    except ValueError as error:
        message = (
            f"Card {cardNumber}: {fieldName} must be numeric, got {value!r}."
        )
        raise ValueError(message) from error


def floatValues(card, cardNumber, fieldNames):
    """Return a card's numeric fields; missing trailing fields are 0."""

    values = []
    for index, fieldName in enumerate(fieldNames):
        text = card.values[index] if index < len(card.values) else ""
        values.append(parseFloat(text, cardNumber, fieldName)
                      if text.strip() else 0.0)
    return values


def rotationMatrix(rotateX_deg, rotateY_deg, rotateZ_deg):
    """Return the GM rotation: about X, then Y, then Z (as in NEC)."""

    sps, cps = sin(radians(rotateX_deg)), cos(radians(rotateX_deg))
    sth, cth = sin(radians(rotateY_deg)), cos(radians(rotateY_deg))
    sph, cph = sin(radians(rotateZ_deg)), cos(radians(rotateZ_deg))
    return (
        (cph*cth, cph*sth*sps - sph*cps, cph*sth*cps + sph*sps),
        (sph*cth, sph*sth*sps + cph*cps, sph*sth*cps - cph*sps),
        (-sth, cth*sps, cth*cps),
    )


def transformPoint(point, matrix, shift=(0.0, 0.0, 0.0)):
    """Rotate a point by a 3x3 matrix, then translate it."""

    return tuple(
        sum(matrix[row][column]*point[column] for column in range(3))
        + shift[row]
        for row in range(3)
    )


class PreviewModel:
    """Wires and patches after applying geometry cards in deck order."""

    def __init__(self):
        self.wires = []     # [tag, start, end]
        self.patches = []   # tuple of corner points

    def addCopies(self, matrix, shift, copies, tagStep, firstTag=0,
                  lastTag=0):
        """Move structure in place (copies == 0) or add moved copies.

        Each new copy is made from the previous one, as in NEC's GM."""

        wiresIn = self.wires
        if firstTag:
            wiresIn = [wire for wire in self.wires
                       if firstTag <= wire[0] <= lastTag]
        if copies == 0:
            for wire in wiresIn:
                wire[1] = transformPoint(wire[1], matrix, shift)
                wire[2] = transformPoint(wire[2], matrix, shift)
            self.patches = [
                tuple(transformPoint(corner, matrix, shift)
                      for corner in patch)
                for patch in self.patches
            ]
            return
        sourceWires, sourcePatches = wiresIn, list(self.patches)
        for _ in range(copies):
            newWires = [
                [wire[0] + tagStep if wire[0] else 0,
                 transformPoint(wire[1], matrix, shift),
                 transformPoint(wire[2], matrix, shift)]
                for wire in sourceWires
            ]
            newPatches = [
                tuple(transformPoint(corner, matrix, shift)
                      for corner in patch)
                for patch in sourcePatches
            ]
            self.wires.extend(newWires)
            self.patches.extend(newPatches)
            sourceWires, sourcePatches = newWires, newPatches

    def scale(self, factor):
        """Scale every point built so far."""

        def scaled(point):
            return tuple(value*factor for value in point)

        for wire in self.wires:
            wire[1], wire[2] = scaled(wire[1]), scaled(wire[2])
        self.patches = [tuple(scaled(corner) for corner in patch)
                        for patch in self.patches]

    def reflect(self, axes, tagStep):
        """GX: reflect about Z, then Y, then X planes, doubling structure."""

        tag = tagStep
        for axis in (2, 1, 0):
            if not axes[axis]:
                continue

            def mirrored(point, axis=axis):
                values = list(point)
                values[axis] = -values[axis]
                return tuple(values)

            added = [
                [wire[0] + tag if wire[0] else 0,
                 mirrored(wire[1]), mirrored(wire[2])]
                for wire in self.wires
            ]
            if self.wires:
                tag *= 2
            self.wires.extend(added)
            self.patches.extend(
                [tuple(mirrored(corner) for corner in patch)
                 for patch in self.patches]
            )

    def rotateCopies(self, total, tagStep):
        """GR: rotate about Z to make `total` copies including the original."""

        if total < 2:
            return
        angle = 360.0/total
        matrix = rotationMatrix(0.0, 0.0, angle)
        sourceWires, sourcePatches = list(self.wires), list(self.patches)
        for _ in range(total - 1):
            newWires = [
                [wire[0] + tagStep if wire[0] else 0,
                 transformPoint(wire[1], matrix),
                 transformPoint(wire[2], matrix)]
                for wire in sourceWires
            ]
            newPatches = [
                tuple(transformPoint(corner, matrix) for corner in patch)
                for patch in sourcePatches
            ]
            self.wires.extend(newWires)
            self.patches.extend(newPatches)
            sourceWires, sourcePatches = newWires, newPatches


def wirePoints(points, tag):
    """Turn a polyline's points into consecutive wire segments."""

    return [[tag, points[index], points[index + 1]]
            for index in range(len(points) - 1)]


def arcPoints(radius_m, start_deg, end_deg, segments):
    """Return GA arc points: X-Z plane, angle from +X toward +Z."""

    delta = radians(end_deg - start_deg)/segments
    angle = radians(start_deg)
    return [
        (radius_m*cos(angle + index*delta), 0.0,
         radius_m*sin(angle + index*delta))
        for index in range(segments + 1)
    ]


def helixPoints(cardNumber, values, segments):
    """Return GH helix points, following the NEC helix routine."""

    spacing, length, a1, b1, a2, b2 = values[:6]
    if spacing == 0 or length == 0:
        raise ValueError(
            f"Card {cardNumber}: helix spacing and length must be nonzero."
        )
    zinc = abs(length/segments)
    points = []
    for index in range(segments + 1):
        z_m = index*zinc
        if a2 == a1:
            radiusY = a1 if b1 == 0 else b1
            x_m = a1*cos(2*pi*z_m/spacing)
            y_m = radiusY*sin(2*pi*z_m/spacing)
        else:
            radiusB2 = a2 if b2 == 0 else b2
            x_m = (a1 + (a2 - a1)*z_m/abs(length))*cos(2*pi*z_m/spacing)
            y_m = (b1 + (radiusB2 - b1)*z_m/abs(length)
                   )*sin(2*pi*z_m/spacing)
        if length <= 0:
            x_m, y_m = y_m, x_m
        points.append((x_m, y_m, z_m))
    return points


def cross(left, right):
    """Return the cross product of two 3-vectors."""

    return (left[1]*right[2] - left[2]*right[1],
            left[2]*right[0] - left[0]*right[2],
            left[0]*right[1] - left[1]*right[0])


def add(left, right):
    """Return the sum of two 3-vectors."""

    return tuple(a + b for a, b in zip(left, right))


def sub(left, right):
    """Return the difference of two 3-vectors."""

    return tuple(a - b for a, b in zip(left, right))


def arbitraryPatchCorners(center, elevation_deg, azimuth_deg, area_m2):
    """Return a square of the given area for an SP shape-0 patch."""

    cosElevation = cos(radians(elevation_deg))
    normal = (cosElevation*cos(radians(azimuth_deg)),
              cosElevation*sin(radians(azimuth_deg)),
              sin(radians(elevation_deg)))
    horizontal = sqrt(normal[0]**2 + normal[1]**2)
    if horizontal >= 1e-6:
        t1 = (-normal[1]/horizontal, normal[0]/horizontal, 0.0)
    else:
        t1 = (1.0, 0.0, 0.0)
    t2 = cross(normal, t1)
    half = 0.5*sqrt(abs(area_m2))
    return tuple(
        tuple(center[axis] + half*(sign1*t1[axis] + sign2*t2[axis])
              for axis in range(3))
        for sign1, sign2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    )


def previewModel(cards):
    """Build the wire and patch preview by applying cards up to GE."""

    model = PreviewModel()
    index = 0
    chain = None    # (third, fourth) corners of the last rectangle/quad
    while index < len(cards):
        card = cards[index]
        cardNumber = index + 1
        index += 1
        code = card.code
        if code == "GE":
            break
        if code == "SC":
            chain = continuePatch(model, card, cardNumber, chain)
            continue
        if code:
            chain = None
        if code == "GW" and len(card.values) >= 8:
            tag, segments, x1, y1, z1, x2, y2, z2 = floatValues(
                card, cardNumber,
                ("Tag", "Segments", "X1", "Y1", "Z1", "X2", "Y2", "Z2"))
            if int(segments) < 1:
                continue
            count = int(segments)
            points = [
                (x1 + (x2 - x1)*step/count, y1 + (y2 - y1)*step/count,
                 z1 + (z2 - z1)*step/count)
                for step in range(count + 1)
            ]
            model.wires.extend(wirePoints(points, int(tag)))
        elif code == "GA":
            tag, segments, radius_m, start, end = floatValues(
                card, cardNumber,
                ("Tag", "Segments", "Radius", "Start", "End"))
            if int(segments) >= 1:
                model.wires.extend(wirePoints(
                    arcPoints(radius_m, start, end, int(segments)),
                    int(tag)))
        elif code == "GH":
            values = floatValues(
                card, cardNumber,
                ("Tag", "Segments", "Spacing", "Length", "RadiusX1",
                 "RadiusY1", "RadiusX2", "RadiusY2"))
            if int(values[1]) >= 1:
                model.wires.extend(wirePoints(
                    helixPoints(cardNumber, values[2:], int(values[1])),
                    int(values[0])))
        elif code == "GM":
            (tagStep, copies, rotX, rotY, rotZ, moveX, moveY, moveZ,
             startTag) = floatValues(
                card, cardNumber,
                ("TagIncrement", "Copies", "RotateX", "RotateY", "RotateZ",
                 "TranslateX", "TranslateY", "TranslateZ", "StartTag"))
            startTag += 0.5e-3
            firstTag = int(startTag)
            lastTag = int((startTag - firstTag)*1e3) or firstTag
            if firstTag == 0:
                lastTag = 0
            model.addCopies(rotationMatrix(rotX, rotY, rotZ),
                            (moveX, moveY, moveZ), int(copies),
                            int(tagStep), firstTag, lastTag)
        elif code == "GR":
            tagStep, total = floatValues(
                card, cardNumber, ("TagIncrement", "Copies"))
            model.rotateCopies(int(total), int(tagStep))
        elif code == "GS":
            conversion, _, factor = floatValues(
                card, cardNumber, ("Conversion", "Unused", "Scale"))
            if conversion == 1:
                factor = 0.3048
            elif conversion > 1:
                factor = 0.0254
            model.scale(factor)
        elif code == "GX":
            tagStep, axisCode = floatValues(
                card, cardNumber, ("TagIncrement", "Axes"))
            axisCode = int(axisCode)
            if axisCode:
                axes = (axisCode//100 % 10 != 0, axisCode//10 % 10 != 0,
                        axisCode % 10 != 0)
                model.reflect(axes, int(tagStep))
        elif code == "SP":
            index, chain = addSinglePatch(model, cards, index, card,
                                          cardNumber)
        elif code == "SM":
            index = addMultiplePatches(model, cards, index, card,
                                       cardNumber)
    return model


def continuationCard(cards, index, cardNumber):
    """Return the SC card that must follow an SP or SM card."""

    if index >= len(cards) or cards[index].code != "SC":
        raise ValueError(f"Card {cardNumber}: SC card required here.")
    return cards[index], index + 1


def addSinglePatch(model, cards, index, card, cardNumber):
    """Add one SP patch (plus its SC card) to the preview model."""

    _, shape, x1, y1, z1, x2, y2, z2 = floatValues(
        card, cardNumber,
        ("Tag", "Shape", "X1", "Y1", "Z1", "X2", "Y2", "Z2"))
    shape = int(shape)
    first, second = (x1, y1, z1), (x2, y2, z2)
    if shape == 0:
        model.patches.append(
            arbitraryPatchCorners(first, x2, y2, z2))
        return index, None
    continuation, index = continuationCard(cards, index, cardNumber + 1)
    values = floatValues(
        continuation, cardNumber + 1,
        ("Unused", "Shape", "X3", "Y3", "Z3", "X4", "Y4", "Z4"))
    third, fourth = tuple(values[2:5]), tuple(values[5:8])
    chain = None
    if shape == 1:      # Rectangle: fourth corner follows from three.
        fourth = sub(add(first, third), second)
        model.patches.append((first, second, third, fourth))
        chain = (third, fourth)
    elif shape == 2:    # Triangle.
        model.patches.append((first, second, third))
    elif shape == 3:    # Quadrilateral.
        model.patches.append((first, second, third, fourth))
        chain = (third, fourth)
    else:
        raise ValueError(f"Card {cardNumber}: unknown patch shape {shape}.")
    return index, chain


def continuePatch(model, card, cardNumber, chain):
    """Add a patch that shares an edge with the previous rectangle or quad.

    The SC card's shape field is 1 for a rectangle or 3 for a quadrilateral.
    The new patch starts from the previous patch's fourth and third corners."""

    if chain is None:
        raise ValueError(
            f"Card {cardNumber}: SC card has no preceding SP rectangle or "
            "quadrilateral.")
    values = floatValues(
        card, cardNumber,
        ("Unused", "Shape", "X3", "Y3", "Z3", "X4", "Y4", "Z4"))
    shape = int(values[1])
    if shape not in {1, 3}:
        raise ValueError(
            f"Card {cardNumber}: a continuation SC card must have shape 1 "
            "(rectangle) or 3 (quadrilateral).")
    previousThird, previousFourth = chain
    first, second, third = previousFourth, previousThird, tuple(values[2:5])
    fourth = (tuple(values[5:8]) if shape == 3
              else sub(add(first, third), second))
    model.patches.append((first, second, third, fourth))
    return (third, fourth)


def addMultiplePatches(model, cards, index, card, cardNumber):
    """Add an SM grid of rectangular patches (plus its SC card)."""

    countX, countY, x1, y1, z1, x2, y2, z2 = floatValues(
        card, cardNumber,
        ("CountX", "CountY", "X1", "Y1", "Z1", "X2", "Y2", "Z2"))
    countX, countY = int(countX), int(countY)
    if countX < 1 or countY < 1:
        raise ValueError(f"Card {cardNumber}: SM counts must be positive.")
    continuation, index = continuationCard(cards, index, cardNumber + 1)
    values = floatValues(
        continuation, cardNumber + 1,
        ("Unused", "Shape", "X3", "Y3", "Z3"))
    first, second, third = (x1, y1, z1), (x2, y2, z2), tuple(values[2:5])
    side1 = tuple(value/countX for value in sub(second, first))
    side2 = tuple(value/countY for value in sub(third, second))
    for column in range(countY):
        for row in range(countX):
            corner = tuple(
                first[axis] + row*side1[axis] + column*side2[axis]
                for axis in range(3))
            model.patches.append((
                corner, add(corner, side1),
                add(add(corner, side1), side2), add(corner, side2)))
    return index


def wireSegments(cards):
    """Return previewable (start, end) wire segments from GW/GA/GH cards.

    Geometry transforms (GM, GR, GS, GX) are applied in deck order."""

    return [(wire[1], wire[2]) for wire in previewModel(cards).wires]


def patchOutlines(cards):
    """Return patch outlines as tuples of 3 or 4 (x, y, z) corner points."""

    return previewModel(cards).patches


def geometryBounds(segments, patches=()):
    """Return center and radius values for equal 3D preview axes."""

    points = [point for segment in segments for point in segment]
    points.extend(point for patch in patches for point in patch)
    if not points:
        return (0.0, 0.0, 0.0), 1.0
    axes = list(zip(*points))
    minimums = [min(axis) for axis in axes]
    maximums = [max(axis) for axis in axes]
    centers = tuple(
        (minimum + maximum)*0.5
        for minimum, maximum in zip(minimums, maximums)
    )
    spans = [maximum - minimum for minimum, maximum in zip(
        minimums, maximums
    )]
    radius_m = max(max(spans)*0.55, 1e-6)
    return centers, radius_m


def segmentLength(segment):
    """Return the Cartesian length of one preview segment."""

    start, end = segment
    return sqrt(sum((right - left)**2 for left, right in zip(start, end)))
