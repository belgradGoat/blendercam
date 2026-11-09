"""The machine a program is written for: limits, Fanuc dialect and shop conventions.

Defaults describe a FANUC Robodrill a-D21MiB5 (31i-B5 control, BT30, 21-tool turret,
10,000 rpm spindle). Every block the writer emits that a machine option or a shop
convention could change (safe start, AICC, coolant M-codes, homing, program end) is an
editable string here, so a profile can match a real machine without touching the writer.
Values marked "assumed" are typical, not read from a machine - check them.
"""

import re
from dataclasses import asdict, dataclass, field, replace

from .errors import CamcoreError, check_keys

COOLANTS = ("FLOOD", "MIST", "AIR", "THROUGH", "OFF")


def _coolant_codes() -> dict:
    # THROUGH / AIR M-codes differ between machines and options: empty = not available
    return {"FLOOD": "M08", "MIST": "M07", "AIR": "", "THROUGH": "", "OFF": "M09"}


@dataclass
class MachineProfile:
    name: str = "FANUC Robodrill a-D21MiB5"
    control: str = "FANUC 31i-B5"
    travel: list = field(default_factory=lambda: [0.5, 0.4, 0.33])  # X, Y, Z (m)
    spindle_min: float = 100.0  # rpm (assumed)
    spindle_max: float = 10000.0  # rpm (24,000 is an option)
    spindle_power: float = 3.7  # kW continuous (assumed): caps the removal rate when planning
    feed_min: float = 0.0  # m/min
    feed_max: float = 30.0  # m/min, cutting feed
    rapid: float = 54.0  # m/min, for time estimates
    magazine: int = 21  # tool pockets
    tool_change_time: float = 1.5  # s, for time estimates (assumed)
    program_memory: int = 1024  # kB of part program storage (assumed); bigger needs DNC
    program_number: int = 1000  # default O-number
    work_offset: str = "G54"
    units_block: str = "G21"
    safe_start: str = "G0 G17 G40 G49 G80 G90 G94"
    home_z: list = field(default_factory=lambda: ["G91 G28 Z0.", "G90"])
    end: list = field(default_factory=lambda: ["G91 G28 Z0.", "G28 Y0.", "G90", "M30"])
    aicc: bool = True  # AI contour control fitted (G05.1 Q1 alarms without the option)
    aicc_on: str = "G05.1 Q1"
    aicc_off: str = "G05.1 Q0"
    coolant: dict = field(default_factory=_coolant_codes)
    h_offset: int = 0  # tool length offset number H = T + h_offset
    spindle_dwell: float = 0.0  # s to wait after M03
    decimals: int = 3  # mm decimals in coordinates
    max_comment: int = 60  # characters

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None = None, base: "MachineProfile | None" = None):
        """A profile from `base` (default: the default profile) with `data` laid over it."""
        data = dict(data or {})
        check_keys(cls, data, "Machine profile")
        p = replace(base or cls(), **data)
        p.coolant = {**_coolant_codes(), **(p.coolant or {})}
        p.validate()
        return p

    def validate(self) -> None:
        problems = []
        try:
            travel = [float(t) for t in self.travel]
        except (TypeError, ValueError):
            travel = []
        if len(travel) != 3 or min(travel) <= 0:
            problems.append("travel must be three positive lengths (m)")
        if not 0 <= self.spindle_min < self.spindle_max:
            problems.append("need 0 <= spindle_min < spindle_max")
        if not 0 <= self.feed_min < self.feed_max:
            problems.append("need 0 <= feed_min < feed_max")
        if self.rapid <= 0 or self.spindle_power <= 0:
            problems.append("rapid and spindle_power must be positive")
        if int(self.magazine) < 1:
            problems.append("magazine must be at least 1")
        if not 1 <= int(self.program_number) <= 9999:
            problems.append("program_number must be 1..9999")
        if not re.fullmatch(r"G5[4-9]|G54\.1 ?P\d{1,3}", str(self.work_offset)):
            problems.append("work_offset must be G54..G59 or G54.1 Pn")
        if int(self.decimals) not in (3, 4):
            problems.append("decimals must be 3 or 4")
        unknown = set(self.coolant) - set(COOLANTS)
        if unknown:
            problems.append(f"coolant keys must be among {COOLANTS}, not {sorted(unknown)}")
        for name in ("home_z", "end"):
            if not isinstance(getattr(self, name), list):
                problems.append(f"{name} must be a list of blocks")
        if problems:
            raise CamcoreError("INVALID_VALUE", "; ".join(problems), {"problems": problems})


# Robodrill models (travel X x Y x Z). Check them against the machine's specification.
PRESETS = {
    "robodrill_d21mib5": {"name": "FANUC Robodrill a-D21MiB5", "travel": [0.5, 0.4, 0.33]},
    "robodrill_d21lib5": {"name": "FANUC Robodrill a-D21LiB5", "travel": [0.7, 0.4, 0.33]},
    "robodrill_d14sib5": {
        "name": "FANUC Robodrill a-D14SiB5",
        "travel": [0.3, 0.3, 0.33],
        "magazine": 14,
    },
}


def preset(name: str) -> MachineProfile:
    if name not in PRESETS:
        raise CamcoreError(
            "INVALID_VALUE", f"No machine preset {name!r}", {"presets": sorted(PRESETS)}
        )
    return MachineProfile.from_dict(PRESETS[name])
