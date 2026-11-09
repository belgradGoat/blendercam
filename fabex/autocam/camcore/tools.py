"""Tool library: the tools a plan may use.

The default catalogue is built from Fabex's metric cutter presets (`end_cyl_*.py`,
`ball_*.py`: 1-20 mm) plus a metric drill series and 90 degree spot drills. The presets
carry only a type and a diameter, so flute length, reach (stick-out below the holder) and
holder diameter are filled in with typical values ("assumed") - a real turret map should
replace them. Tools from a turret map carry their fixed T numbers; catalogue tools get
numbers when a plan uses them.

Lengths are metres (API units); `tip_angle` is degrees.
"""

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .errors import CamcoreError, check_keys

TOOL_TYPES = ("END", "BALL", "BULL", "DRILL", "SPOT", "TAP")

# Fabex cutter type used to compute paths with each tool
FABEX_TYPES = {
    "END": "END",
    "BALL": "BALLNOSE",
    "BULL": "BULLNOSE",
    "DRILL": "END",
    "SPOT": "VCARVE",
    "TAP": "END",
}

# Typical flute lengths of short carbide end mills (DIN 6527 K), mm -> mm
_LOC = ((1, 3), (1.5, 4.5), (2, 6), (2.5, 8), (3, 8), (4, 11), (5, 13), (6, 13), (8, 19),
        (10, 22), (12, 26), (16, 32), (20, 38))  # fmt: skip

_DRILLS_MM = sorted(
    {x / 2 for x in range(2, 25)}  # 1.0 .. 12.0 by 0.5
    | {3.3, 3.4, 4.2, 6.6, 6.8, 10.2}  # tap drills and clearance holes for M3-M12
)
_SPOTS_MM = (6.0, 10.0)


def _interp_loc(d_mm: float) -> float:
    xs = [a for a, _ in _LOC]
    if d_mm <= xs[0]:
        return _LOC[0][1] * d_mm / xs[0]
    if d_mm >= xs[-1]:
        return _LOC[-1][1] * d_mm / xs[-1]
    for (x0, y0), (x1, y1) in zip(_LOC, _LOC[1:]):
        if x0 <= d_mm <= x1:
            return y0 + (y1 - y0) * (d_mm - x0) / (x1 - x0)
    return 3 * d_mm


def _mm_label(d: float) -> str:
    return f"{d * 1000:g}"


@dataclass
class Tool:
    id: str
    type: str
    diameter: float
    number: int | None = None  # T number (turret pocket); None = assigned by a plan
    corner_radius: float = 0.0  # BALL: diameter / 2; BULL: the corner
    flutes: int = 3
    flute_length: float = 0.0  # 0 = typical for the type and diameter
    reach: float = 0.0  # stick-out below the holder; 0 = typical
    holder_diameter: float = 0.0  # 0 = typical
    tip_angle: float = 0.0  # DRILL / SPOT point angle (deg); 0 = typical
    pitch: float = 0.0  # TAP thread pitch
    material: str = "carbide"
    description: str = ""
    source: str = ""  # where it came from (a preset file, "user", ...)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Tool":
        check_keys(cls, data, "Tool")
        missing = [k for k in ("id", "type", "diameter") if k not in data]
        if missing:
            raise CamcoreError("INVALID_VALUE", f"Tool needs {missing}", {"tool": data})
        t = cls(**data)
        t.type = str(t.type).upper()
        return complete(t)

    def validate(self) -> None:
        where = f"tool {self.id!r}"
        if self.type not in TOOL_TYPES:
            raise CamcoreError(
                "INVALID_VALUE", f"{where}: type must be one of {TOOL_TYPES}", {"tool": self.id}
            )
        if not self.diameter > 0:
            raise CamcoreError("INVALID_VALUE", f"{where}: diameter must be positive")
        if self.number is not None and int(self.number) < 1:
            raise CamcoreError("INVALID_VALUE", f"{where}: number must be at least 1")
        if self.type == "BULL" and not 0 < self.corner_radius < self.diameter / 2:
            raise CamcoreError("INVALID_VALUE", f"{where}: BULL needs 0 < corner_radius < d/2")
        if self.type == "TAP" and not self.pitch > 0:
            raise CamcoreError("INVALID_VALUE", f"{where}: TAP needs a pitch")
        if int(self.flutes) < 1:
            raise CamcoreError("INVALID_VALUE", f"{where}: flutes must be at least 1")
        for name in ("flute_length", "reach", "holder_diameter"):
            if getattr(self, name) <= 0:
                raise CamcoreError("INVALID_VALUE", f"{where}: {name} must be positive")
        if self.reach < self.flute_length:
            raise CamcoreError("INVALID_VALUE", f"{where}: reach is shorter than the flutes")


def complete(t: Tool) -> Tool:
    """Fill in the typical values a tool left at 0, and the ball radius."""
    d_mm = t.diameter * 1000
    t = replace(t)
    if t.type == "BALL":
        t.corner_radius = t.diameter / 2
    elif t.type in ("END", "DRILL", "SPOT", "TAP"):
        t.corner_radius = 0.0
    if not t.flute_length:
        loc = {
            "DRILL": 5 * d_mm,
            "SPOT": d_mm,
            "TAP": 3 * d_mm,
        }.get(t.type, _interp_loc(d_mm))
        t.flute_length = round(loc / 1000, 6)
    if not t.reach:
        t.reach = round(t.flute_length + (0.5 * d_mm + 3) / 1000, 6)
    if not t.holder_diameter:
        t.holder_diameter = round((2.5 * d_mm + 12) / 1000, 6)
    if not t.tip_angle:
        t.tip_angle = {"DRILL": 140.0, "SPOT": 90.0}.get(t.type, 0.0)
    t.validate()
    return t


_TYPE_RE = re.compile(r"d\.cutter_type\s*=\s*[\"'](\w+)[\"']")
_DIAMETER_RE = re.compile(r"d\.cutter_diameter\s*=\s*([0-9.eE+-]+)")


def parse_preset(text: str) -> tuple[str, float] | None:
    """(Fabex cutter type, diameter in m) of a cutter preset script, or None."""
    ty, dia = _TYPE_RE.search(text), _DIAMETER_RE.search(text)
    if not ty or not dia:
        return None
    return ty.group(1), float(dia.group(1))


def default_library(preset_dir: str | Path | None = None) -> list[Tool]:
    """The default catalogue: Fabex's metric end / ball mill presets found in
    `preset_dir`, then drills and spot drills."""
    tools = []
    if preset_dir is not None:
        for path in sorted(Path(preset_dir).glob("*.py")):
            if not re.match(r"(end_cyl|ball)_[0-9.]+mm\.py$", path.name):
                continue
            parsed = parse_preset(path.read_text(encoding="utf-8", errors="replace"))
            if parsed is None or parsed[0] not in ("END", "BALLNOSE"):
                continue
            fabex_type, d = parsed
            kind = "END" if fabex_type == "END" else "BALL"
            label = _mm_label(d)
            tools.append(
                Tool(
                    id=("EM" if kind == "END" else "BM") + label,
                    type=kind,
                    diameter=d,
                    flutes=(3 if d >= 0.003 else 2) if kind == "END" else 2,
                    description=f"{label} mm carbide {'end' if kind == 'END' else 'ball'} mill",
                    source=f"presets/cam_cutters/{path.name}",
                )
            )
        tools.sort(key=lambda t: (t.type, t.diameter))
    for d_mm in _DRILLS_MM:
        tools.append(
            Tool(
                id=f"DR{d_mm:g}",
                type="DRILL",
                diameter=d_mm / 1000,
                flutes=2,
                description=f"{d_mm:g} mm carbide drill 5xD",
                source="default",
            )
        )
    for d_mm in _SPOTS_MM:
        tools.append(
            Tool(
                id=f"SP{d_mm:g}",
                type="SPOT",
                diameter=d_mm / 1000,
                flutes=2,
                description=f"{d_mm:g} mm 90 deg spot drill",
                source="default",
            )
        )
    return [complete(t) for t in tools]


def library_from_list(items: list) -> list[Tool]:
    """A user's library (e.g. the turret map) from JSON-able dicts."""
    if not isinstance(items, list) or not items:
        raise CamcoreError("INVALID_VALUE", "A tool library is a non-empty list of tools")
    tools = [Tool.from_dict(dict(item, source=item.get("source") or "user")) for item in items]
    validate_library(tools)
    return tools


def validate_library(tools: list[Tool]) -> None:
    ids, numbers = {}, {}
    for t in tools:
        if t.id in ids:
            raise CamcoreError("INVALID_VALUE", f"Two tools are called {t.id!r}")
        ids[t.id] = t
        if t.number is not None:
            if t.number in numbers:
                raise CamcoreError(
                    "TOOL_CONFLICT",
                    f"Tools {numbers[t.number]!r} and {t.id!r} both have number T{t.number}",
                    {"number": t.number, "tools": [numbers[t.number], t.id]},
                )
            numbers[t.number] = t.id


def to_fabex_props(tool: Tool) -> dict:
    """Operation properties that make Fabex compute paths with this tool."""
    props = {
        "cutter_type": FABEX_TYPES[tool.type],
        "cutter_diameter": tool.diameter,
        "cutter_flutes": int(tool.flutes),
        "cutter_description": tool.description or tool.id,
    }
    if tool.number is not None:
        props["cutter_id"] = int(tool.number)
    if tool.type == "BULL":
        props["bull_corner_radius"] = tool.corner_radius
    if tool.type == "SPOT":
        props["cutter_tip_angle"] = tool.tip_angle
    return props
