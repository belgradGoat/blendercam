"""What the Fanuc writer needs to write a program: machine-unit (mm, mm/min) data, built
from Fabex operations by the Blender glue (fanuc_export.py) or by tests."""

from dataclasses import dataclass, field

import numpy as np

from .machine import MachineProfile

CYCLES = ("G81", "G82", "G73", "G83", "G84")


@dataclass
class ToolSpec:
    number: int  # T (and H = T + machine.h_offset)
    type: str  # camcore tool type: END BALL BULL DRILL SPOT TAP
    diameter: float  # mm
    corner_radius: float = 0.0  # mm
    flutes: int = 0
    pitch: float = 0.0  # mm, taps
    description: str = ""

    def key(self) -> tuple:
        """What makes two tools different: a T number may only ever mean one of these."""
        return (
            self.type,
            round(self.diameter, 4),
            round(self.corner_radius, 4),
            int(self.flutes),
            round(self.pitch, 4),
        )


@dataclass
class DrillCycle:
    cycle: str  # G81 drill, G82 + dwell, G73 chip-breaking peck, G83 full-retract peck, G84 tap
    holes: list  # [(x, y, z bottom)] mm, in drilling order
    r_plane: float  # mm (absolute): where feeding starts
    feed: float  # mm/min (G84: rpm x pitch)
    peck: float = 0.0  # Q, mm (G73 / G83)
    dwell: float = 0.0  # s at the bottom (G82)
    retract: str = "G98"  # G98: back to the initial level between holes; G99: to R


@dataclass
class OpBlock:
    name: str
    tool: ToolSpec
    rpm: float
    clockwise: bool = True
    feed: float = 0.0  # mm/min
    plunge_feed: float = 0.0  # mm/min
    clearance: float = 0.0  # mm: the free height, also the canned cycles' initial level
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # mm
    kinds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    drill: DrillCycle | None = None  # written as a canned cycle instead of points
    coolant: str = "FLOOD"
    aicc: bool = True  # use AI contour control (if the machine has it)
    header: list = field(default_factory=list)  # raw blocks before the operation
    trailer: list = field(default_factory=list)  # raw blocks after it
    strategy: str = ""


@dataclass
class ProgramSpec:
    number: int
    title: str
    machine: MachineProfile
    blocks: list
    work_offset: str = ""  # "" = the machine's
    comments: list = field(default_factory=list)
    simplify: float = 0.001  # mm chord tolerance for feed moves (0 = keep every point)
