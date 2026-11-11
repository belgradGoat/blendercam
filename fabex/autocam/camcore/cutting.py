"""Cutting data: spindle speed, feeds and engagement for a tool in a material.

Aluminium with carbide tools and flood coolant, on a machining centre (a FANUC Robodrill
by default). The numbers are conservative starting values from common tool makers' tables,
not a substitute for the tool maker's own data:

- mills: cutting speed Vc (m/min) gives the rpm, capped by the spindle; feed = rpm x flutes
  x fz, with fz (mm per tooth) proportional to the diameter for the kind of cut. When the
  spindle limits the rpm the chip load stays the same, so the feed drops with it.
- depth (ap) and width (ae) of cut per kind of cut, as fractions of the diameter, capped by
  the flute length;
- a spindle power cap: the removal rate x the specific cutting force must stay below 80% of
  the spindle's power, or the feed is reduced;
- drills: feed per revolution and a peck (Q) of one diameter; spot drills; rigid taps
  (feed = rpm x pitch).

Units: metres, m/min and rpm (API units), like the tool library.
"""

from dataclasses import asdict, dataclass, field
from math import pi

from .errors import CamcoreError
from .machine import MachineProfile
from .tools import Tool

MATERIALS = {
    "6061": {
        "name": "Aluminium 6061-T6",
        "vc_mill": 400.0,
        "vc_drill": 120.0,
        "vc_spot": 100.0,
        "vc_tap": 20.0,
        "kc": 750.0,  # specific cutting force, N/mm2
        "fz_factor": 1.0,
    },
    "6082": {
        "name": "Aluminium 6082-T6 (EN AW-6082)",
        "vc_mill": 400.0,
        "vc_drill": 120.0,
        "vc_spot": 100.0,
        "vc_tap": 20.0,
        "kc": 750.0,
        "fz_factor": 1.0,
    },
    "7075": {
        "name": "Aluminium 7075-T6",
        "vc_mill": 350.0,
        "vc_drill": 100.0,
        "vc_spot": 90.0,
        "vc_tap": 15.0,
        "kc": 900.0,
        "fz_factor": 0.9,
    },
    "2017": {
        "name": "Aluminium 2017A (dural)",
        "vc_mill": 350.0,
        "vc_drill": 100.0,
        "vc_spot": 90.0,
        "vc_tap": 15.0,
        "kc": 850.0,
        "fz_factor": 0.9,
    },
}
ALIASES = {"aluminium": "6061", "aluminum": "6061", "al": "6061", "6061-t6": "6061"}

MILLS = ("END", "BALL", "BULL")
# kind of cut -> (fz per mm of diameter (mm), ae / D, ap / D) for mills
ENGAGEMENTS = {
    "slot": (0.005, 1.0, 0.5),  # full width
    "pocket": (0.006, 0.4, 0.5),  # concentric rings: the first ring of each layer is a slot
    "rough": (0.008, 0.4, 1.0),  # side milling
    "finish": (0.006, 0.05, 1.5),  # walls: a light width over the flute length
    "floor": (0.006, 0.6, 0.05),  # floors: one light layer
    "face": (0.007, 0.7, 0.1),  # tops
}
HOLE_KINDS = ("drill", "spot", "tap")
FZ_MIN, FZ_MAX = 0.005, 0.10  # mm per tooth
POWER_USE = 0.8  # of the spindle's power
EFFICIENCY = 0.85
PLUNGE_FRACTION = 0.25  # straight plunges of centre-cutting mills: a quarter of the feed
TAP_RPM_MAX = 2000.0


def material(name: str) -> tuple[str, dict]:
    key = str(name).strip().lower()
    key = ALIASES.get(key, key.upper() if key.upper() in MATERIALS else key)
    if key not in MATERIALS:
        raise CamcoreError(
            "INVALID_VALUE",
            f"No cutting data for material {name!r}",
            {"materials": sorted(MATERIALS) + sorted(ALIASES)},
        )
    return key, MATERIALS[key]


@dataclass
class CuttingData:
    tool: str
    material: str
    engagement: str
    rpm: float
    feed: float  # m/min: cutting feed (drills: the drilling feed)
    plunge_feed: float  # m/min: straight down
    vc: float  # m/min actually reached (after the spindle cap)
    fz: float = 0.0  # m per tooth (mills)
    feed_per_rev: float = 0.0  # m (drills, spot drills, taps)
    ap: float = 0.0  # m: depth of cut (stepdown)
    ae: float = 0.0  # m: width of cut (stepover)
    ramp_angle: float = 0.0  # degrees: helix / ramp entry
    peck: float = 0.0  # m (drills)
    power: float = 0.0  # kW at ap x ae
    removal_rate: float = 0.0  # cm3/min at ap x ae
    rpm_limited: bool = False  # the spindle capped Vc
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("rpm", "vc", "power", "removal_rate", "ramp_angle"):
            d[k] = round(d[k], 3)
        for k in ("feed", "plunge_feed"):
            d[k] = round(d[k], 5)
        for k in ("fz", "feed_per_rev", "ap", "ae", "peck"):
            d[k] = round(d[k], 7)
        return d


def _rpm(vc: float, d_mm: float, machine: MachineProfile) -> tuple[float, bool]:
    ideal = vc * 1000 / (pi * d_mm)
    rpm = min(ideal, machine.spindle_max)
    return max(rpm, machine.spindle_min), ideal > machine.spindle_max


def _mill(tool, mat, engagement, machine, ap, ae, notes) -> CuttingData:
    d_mm = tool.diameter * 1000
    k, ae_f, ap_f = ENGAGEMENTS[engagement]
    fz = min(max(k * d_mm, FZ_MIN), FZ_MAX) * mat["fz_factor"]
    if d_mm < 3 and engagement in ("slot", "pocket"):
        ap_f *= 0.6  # small tools: shallower slots
    rpm, limited = _rpm(mat["vc_mill"], d_mm, machine)
    feed = rpm * tool.flutes * fz  # mm/min
    flute_cap = (0.95 if engagement == "finish" else 0.9) * tool.flute_length * 1000
    ap_mm = min(ap * 1000 if ap else ap_f * d_mm, flute_cap)
    ae_mm = min(ae * 1000 if ae else ae_f * d_mm, d_mm)
    if ap and ap * 1000 > flute_cap + 1e-9:
        notes.append(f"ap capped at {flute_cap:.3g} mm by the flute length")
    feed_max = machine.feed_max * 1000
    if feed > feed_max:
        feed = feed_max
        notes.append("feed capped by the machine")
    power = ap_mm * ae_mm * feed * mat["kc"] / (60e6 * EFFICIENCY)
    cap = POWER_USE * machine.spindle_power
    if power > cap:
        feed *= cap / power
        notes.append(f"feed reduced to keep the spindle under {cap:.3g} kW")
        power = cap
    if limited:
        notes.append(
            f"spindle limit: Vc {pi * d_mm * rpm / 1000:.0f} of {mat['vc_mill']:.0f} m/min"
        )
    ramp = 3.0 if d_mm >= 6 else 2.0 if d_mm >= 3 else 1.0
    return CuttingData(
        tool=tool.id,
        material="",
        engagement=engagement,
        rpm=rpm,
        feed=feed / 1000,
        plunge_feed=PLUNGE_FRACTION * feed / 1000,
        vc=pi * d_mm * rpm / 1000,
        fz=fz / 1000,
        ap=ap_mm / 1000,
        ae=ae_mm / 1000,
        ramp_angle=ramp,
        power=power,
        removal_rate=ap_mm * ae_mm * feed / 1000,
        rpm_limited=limited,
        notes=notes,
    )


def _hole(tool, mat, engagement, machine, notes) -> CuttingData:
    d_mm = tool.diameter * 1000
    if engagement == "tap":
        if tool.type != "TAP":
            raise CamcoreError("INVALID_VALUE", f"{tool.id} is not a tap")
        rpm = min(mat["vc_tap"] * 1000 / (pi * d_mm), TAP_RPM_MAX, machine.spindle_max)
        rpm = max(rpm, machine.spindle_min)
        pitch_mm = tool.pitch * 1000
        feed = rpm * pitch_mm
        return CuttingData(
            tool=tool.id,
            material="",
            engagement="tap",
            rpm=rpm,
            feed=feed / 1000,
            plunge_feed=feed / 1000,
            vc=pi * d_mm * rpm / 1000,
            feed_per_rev=pitch_mm / 1000,
            notes=notes + ["rigid tapping: feed = rpm x pitch"],
        )
    if engagement == "spot":
        vc, f_rev = mat["vc_spot"], min(max(0.008 * d_mm, 0.02), 0.1)
    else:
        vc, f_rev = mat["vc_drill"], min(max(0.025 * d_mm, 0.02), 0.3)
    f_rev *= mat["fz_factor"]
    rpm, limited = _rpm(vc, d_mm, machine)
    feed = min(rpm * f_rev, machine.feed_max * 1000)
    if limited:
        notes.append(f"spindle limit: Vc {pi * d_mm * rpm / 1000:.0f} of {vc:.0f} m/min")
    return CuttingData(
        tool=tool.id,
        material="",
        engagement=engagement,
        rpm=rpm,
        feed=feed / 1000,
        plunge_feed=feed / 1000,
        vc=pi * d_mm * rpm / 1000,
        feed_per_rev=f_rev / 1000,
        peck=max(d_mm, 1.0) / 1000 if engagement == "drill" else 0.0,
        rpm_limited=limited,
        notes=notes,
    )


def cutting_data(
    tool: Tool,
    material_name: str,
    engagement: str,
    machine: MachineProfile,
    ap: float | None = None,
    ae: float | None = None,
) -> CuttingData:
    """Cutting data for `tool` in `material_name` for a kind of cut: slot, pocket, rough,
    finish, floor, face (mills) or drill, spot, tap. ap / ae (m) override the default depth
    and width of cut."""
    key, mat = material(material_name)
    notes = []
    if engagement in ENGAGEMENTS:
        if tool.type not in MILLS:
            raise CamcoreError(
                "INVALID_VALUE",
                f"{tool.id} ({tool.type}) can't {engagement}: use drill / spot / tap",
                {"engagements": list(HOLE_KINDS)},
            )
        cd = _mill(tool, mat, engagement, machine, ap, ae, notes)
    elif engagement in HOLE_KINDS:
        expected = {"drill": ("DRILL",), "spot": ("SPOT", "DRILL"), "tap": ("TAP",)}[engagement]
        if tool.type not in expected:
            raise CamcoreError(
                "INVALID_VALUE", f"{tool.id} ({tool.type}) is not a tool for {engagement}"
            )
        cd = _hole(tool, mat, engagement, machine, notes)
    else:
        raise CamcoreError(
            "INVALID_VALUE",
            f"engagement must be one of {list(ENGAGEMENTS) + list(HOLE_KINDS)}",
            {"engagement": engagement},
        )
    cd.material = key
    return cd
