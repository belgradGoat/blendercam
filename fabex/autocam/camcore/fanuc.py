"""Write a FANUC (Robodrill, 31i) milling program from a ProgramSpec.

Program shape:

    %
    O1000 (TITLE)
    (T1 D6. END 3FL - ...)          tool list
    G21                             units in their own block
    G0 G17 G40 G49 G80 G90 G94      safe start (machine profile)
    (OP NAME)
    G91 G28 Z0.                     per tool change: Z home, tool, approach,
    G90                             length offset, coolant, AICC
    T1 M06
    G0 G90 G54 X.. Y.. S10000 M03
    G43 H1 Z..
    M08
    G05.1 Q1
    ...                             moves: G0 never diagonal (up, across, down)
    G05.1 Q0
    M09
    M05
    G91 G28 Z0.                     end (machine profile)
    G28 Y0.
    G90
    M30
    %

Every coordinate and feed carries a decimal point (X10. is 10 mm whatever parameter 3401
says), F is written when it changes, drilling is written as canned cycles (G81 / G82 /
G73 / G83, G84 rigid tapping with M29) ended by G80, and AICC is switched off before
cycles, tool changes and the end.
"""

import math
import unicodedata

import numpy as np

from .errors import CamcoreError
from .machine import COOLANTS
from .moves import FEED, PLUNGE, RAPID, simplify, split_rapid
from .program import CYCLES, ProgramSpec

TYPE_WORDS = {"END": "END", "BALL": "BALL", "BULL": "BULL", "DRILL": "DRILL", "SPOT": "SPOT"}


def num(x: float, decimals: int = 3) -> str:
    """A Fanuc number with a decimal point: 10 -> '10.', 0.5 -> '0.5', -0.0 -> '0.'."""
    s = f"{float(x):.{decimals}f}".rstrip("0")
    return "0." if s in ("-0.", "0.") else s


_ASCII = str.maketrans(
    {"\u2013": "-", "\u2014": "-", "\u00d8": "D", "\u00f8": "D", "\u2300": "D", "\u00b0": " DEG",
     "\u00d7": "X", "\u00b5": "U", "\u00bd": "1/2", "\u00bc": "1/4", "\u215b": "1/8"}
)  # fmt: skip


def comment(text: str, limit: int = 60) -> str:
    """Upper-case ASCII without parentheses, so any control accepts it."""
    s = str(text).translate(_ASCII)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.upper().replace("(", "[").replace(")", "]")
    s = " ".join(s.split())
    return f"({s[:limit]})"


def _tool_comment(tool, limit: int) -> str:
    words = [f"T{tool.number}", f"D{num(tool.diameter)}", tool.type]
    if tool.type == "BULL":
        words.append(f"R{num(tool.corner_radius)}")
    if tool.flutes:
        words.append(f"{tool.flutes}FL")
    if tool.type == "TAP" and tool.pitch:
        words.append(f"P{num(tool.pitch)}")
    if tool.description:
        words += ["-", tool.description]
    return comment(" ".join(words), limit)


def _tapping(block) -> bool:
    return block.drill is not None and block.drill.cycle == "G84"


class _Writer:
    def __init__(self, spec: ProgramSpec):
        self.spec = spec
        self.m = spec.machine
        self.dec = int(self.m.decimals)
        self.lines: list[str] = []
        self.pos = [None, None, None]  # rounded current position (mm), None = unknown
        self.modal = None  # "G0" / "G1" / None
        self.f = None
        self.tool = None
        self.rpm = None  # spindle running at this speed, None = stopped
        self.cw = True
        self.coolant = "OFF"
        self.aicc = False
        self.warnings: list[str] = []
        self.lo = np.full(3, np.inf)
        self.hi = np.full(3, -np.inf)
        self.ops: list[dict] = []
        self.op = None
        self.feed_max = self.m.feed_max * 1000  # mm/min
        self.rapid = self.m.rapid * 1000

    # ---- output helpers
    def emit(self, line: str) -> None:
        self.lines.append(line)

    def _r(self, v: float) -> float:
        return round(float(v), self.dec)

    def _track(self, p) -> None:
        a = np.asarray(p, dtype=float)
        self.lo = np.minimum(self.lo, a)
        self.hi = np.maximum(self.hi, a)

    def _time(self, length: float, rate: float) -> None:
        if self.op is not None and rate > 0:
            self.op["minutes"] += length / rate
            self.op["length_mm"] += length

    def _move(self, code: str, target, feed: float | None = None) -> None:
        target = [self._r(v) for v in target]
        words = []
        for axis, old, new in zip("XYZ", self.pos, target):
            if old is None or new != old:
                words.append(f"{axis}{num(new, self.dec)}")
        if not words:
            return
        if feed is not None:
            feed = self._feed(feed)
        known = [p for p in self.pos if p is not None]
        length = math.dist(self.pos, target) if len(known) == 3 else 0.0  # unknown start: no time
        line = []
        if self.modal != code:
            line.append(code)
            self.modal = code
        line += words
        if feed is not None and feed != self.f:
            line.append(f"F{num(feed, 1)}")
            self.f = feed
        self.emit(" ".join(line))
        self._time(length, self.rapid if code == "G0" else (self.f or 0))
        self.pos = target
        self._track(target)
        if self.op is not None:
            self.op["moves"] += 1

    def _feed(self, feed: float) -> float:
        if feed <= 0:
            raise CamcoreError("INVALID_VALUE", f"Operation {self.op['name']!r} has no feed rate")
        if feed > self.feed_max:
            self._warn(f"F{num(feed, 1)} above the machine's {num(self.feed_max, 1)}: clamped")
            feed = self.feed_max
        return round(feed, 1)

    def _warn(self, text: str) -> None:
        full = f"{self.op['name']}: {text}" if self.op else text
        if full not in self.warnings:
            self.warnings.append(full)

    # ---- machine state
    def aicc_off(self) -> None:
        if self.aicc:
            self.emit(self.m.aicc_off)
            self.aicc = False

    def coolant_off(self) -> None:
        if self.coolant != "OFF":
            self.emit(self.m.coolant["OFF"])
            self.coolant = "OFF"

    def spindle_off(self) -> None:
        if self.rpm is not None:
            self.emit("M05")
            self.rpm = None

    def _rpm(self, block) -> int:
        rpm = float(block.rpm)
        if rpm <= 0:
            raise CamcoreError("INVALID_VALUE", f"Operation {block.name!r} has no spindle speed")
        if rpm > self.m.spindle_max:
            self._warn(f"S{rpm:.0f} above the machine's {self.m.spindle_max:.0f}: clamped")
            rpm = self.m.spindle_max
        if rpm < self.m.spindle_min:
            self._warn(f"S{rpm:.0f} below the machine's {self.m.spindle_min:.0f}: raised")
            rpm = self.m.spindle_min
        return int(round(rpm))

    def change_tool(self, block, first_xy) -> None:
        self.aicc_off()
        self.coolant_off()
        self.spindle_off()
        for line in self.m.home_z:
            self.emit(line)
        t = block.tool
        self.emit(f"T{t.number} M06")
        self.tool = t
        self.pos = [None, None, None]
        self.modal = None
        self.f = None
        if self.op is not None:
            self.op["minutes"] += self.m.tool_change_time / 60
        rpm = self._rpm(block)
        x, y = (self._r(v) for v in first_xy)
        wo = self.spec.work_offset or self.m.work_offset
        start = f"G0 G90 {wo} X{num(x, self.dec)} Y{num(y, self.dec)}"
        if _tapping(block):  # rigid tapping (M29) takes over a stopped spindle
            self.emit(start)
        else:
            self.emit(f"{start} S{rpm} {'M03' if block.clockwise else 'M04'}")
            self.rpm, self.cw = rpm, block.clockwise
        self.modal = "G0"
        z = self._r(block.clearance)
        self.emit(f"G43 H{t.number + int(self.m.h_offset)} Z{num(z, self.dec)}")
        self.pos = [x, y, z]
        self._track(self.pos)
        self._dwell()

    def _dwell(self) -> None:
        if self.m.spindle_dwell > 0:
            self.emit(f"G04 X{num(self.m.spindle_dwell, 3)}")
            if self.op is not None:
                self.op["minutes"] += self.m.spindle_dwell / 60

    def spindle(self, block) -> None:
        if _tapping(block):
            return
        rpm = self._rpm(block)
        if rpm != self.rpm or block.clockwise != self.cw:
            self.emit(f"S{rpm} {'M03' if block.clockwise else 'M04'}")
            self.rpm, self.cw = rpm, block.clockwise
            self._dwell()

    def set_coolant(self, block) -> None:
        want = str(block.coolant).upper()
        if want not in COOLANTS:
            raise CamcoreError(
                "INVALID_VALUE", f"Coolant {block.coolant!r}: use one of {COOLANTS}", {}
            )
        if want == self.coolant:
            return
        if want == "OFF":
            self.coolant_off()
            return
        code = self.m.coolant.get(want, "")
        if not code:
            raise CamcoreError(
                "MACHINE_LIMIT",
                f"The machine profile has no M-code for {want} coolant",
                {"coolant": self.m.coolant},
            )
        self.coolant_off()
        self.emit(code)
        self.coolant = want

    # ---- blocks
    def block(self, b) -> None:
        self.op = {
            "name": b.name,
            "strategy": b.strategy,
            "tool": b.tool.number,
            "moves": 0,
            "points_in": 0,
            "points_written": 0,
            "holes": 0,
            "length_mm": 0.0,
            "minutes": 0.0,
        }
        self.ops.append(self.op)
        if b.tool.number > self.m.magazine or b.tool.number < 1:
            raise CamcoreError(
                "MACHINE_LIMIT",
                f"T{b.tool.number} ({b.name}) is outside the {self.m.magazine}-tool magazine",
                {"tool": b.tool.number, "magazine": self.m.magazine},
            )
        if b.drill is not None:
            if not b.drill.holes:
                self._warn("no holes to drill")
                return
            first_xy = b.drill.holes[0][:2]
            pts = kinds = None
        else:
            pts = np.asarray(b.points, dtype=float).reshape(-1, 3)
            kinds = np.asarray(b.kinds)
            if len(pts) != len(kinds):
                raise CamcoreError("INVALID_VALUE", f"{b.name}: points and kinds differ in length")
            self.op["points_in"] = int(len(pts))
            if not len(pts):
                self._warn("empty path")
                return
            if self.spec.simplify > 0:
                keep = simplify(pts, kinds, self.spec.simplify)
                pts, kinds = pts[keep], kinds[keep]
            self.op["points_written"] = int(len(pts))
            first_xy = pts[0][:2]

        self.emit(
            comment(f"{b.name}" + (f" - {b.strategy}" if b.strategy else ""), self.m.max_comment)
        )
        if self.tool is None or self.tool.number != b.tool.number:
            self.change_tool(b, first_xy)
        else:
            self.spindle(b)
        self.set_coolant(b)
        for line in b.header:
            self.emit(line)
        self.modal = None  # each operation states its first motion code
        if b.drill is not None:
            self.aicc_off()
            self.cycle(b)
        else:
            if self.m.aicc and b.aicc:
                if not self.aicc:
                    self.emit(self.m.aicc_on)
                    self.aicc = True
            else:
                self.aicc_off()
            self.moves(b, pts, kinds)
        for line in b.trailer:
            self.emit(line)

    def moves(self, b, pts, kinds) -> None:
        for p, k in zip(pts, kinds):
            if k == RAPID:
                cur = [self.pos[i] if self.pos[i] is not None else p[i] for i in range(3)]
                for q in split_rapid(cur, [self._r(v) for v in p]):
                    self._move("G0", q)
            elif k == PLUNGE:
                self._move("G1", p, b.plunge_feed)
            elif k == FEED:
                self._move("G1", p, b.feed)
            else:
                raise CamcoreError("INVALID_VALUE", f"{b.name}: unknown move kind {k}")

    def cycle(self, b) -> None:
        c = b.drill
        if c.cycle not in CYCLES:
            raise CamcoreError("INVALID_VALUE", f"{b.name}: cycle must be one of {CYCLES}")
        if c.retract not in ("G98", "G99"):
            raise CamcoreError("INVALID_VALUE", f"{b.name}: retract must be G98 or G99")
        if c.cycle in ("G73", "G83") and not c.peck > 0:
            raise CamcoreError("INVALID_VALUE", f"{b.name}: {c.cycle} needs a peck depth (Q)")
        initial = self._r(b.clearance)
        r = self._r(c.r_plane)
        if r > initial:
            raise CamcoreError(
                "INVALID_VALUE",
                f"{b.name}: R plane {num(r)} is above the initial level {num(initial)}",
            )
        feed = c.feed
        tap_rpm = self._rpm(b)
        if c.cycle == "G84":
            if b.tool.pitch > 0:
                feed = tap_rpm * b.tool.pitch
            if not b.clockwise:
                self._warn("G84 with a left-hand spindle: use a left-hand tap")
        feed = self._feed(feed)
        # go to the initial level (G98 returns here between holes)
        if self.pos[2] is None or self.pos[2] != initial:
            self._move("G0", [self.pos[0], self.pos[1], initial])
        dec = self.dec
        last_z = None
        total_depth = 0.0
        for i, (x, y, z) in enumerate(c.holes):
            x, y, z = self._r(x), self._r(y), self._r(z)
            if z >= r:
                raise CamcoreError(
                    "INVALID_VALUE", f"{b.name}: hole bottom Z{num(z)} is not below R{num(r)}"
                )
            xy = []
            if self.pos[0] != x or i == 0:
                xy.append(f"X{num(x, dec)}")
            if self.pos[1] != y or i == 0:
                xy.append(f"Y{num(y, dec)}")
            if i == 0:
                words = [c.retract, c.cycle] + xy + [f"Z{num(z, dec)}", f"R{num(r, dec)}"]
                if c.cycle in ("G73", "G83"):
                    words.append(f"Q{num(c.peck, dec)}")
                if c.cycle == "G82":
                    words.append(f"P{int(round(c.dwell * 1000))}")
                words.append(f"F{num(feed, 1)}")
                if c.cycle == "G84":
                    self.spindle_off()
                    self.emit(f"M29 S{tap_rpm}")
                self.emit(" ".join(words))
            else:
                words = xy + ([f"Z{num(z, dec)}"] if z != last_z else [])
                if words:
                    self.emit(" ".join(words))
            # time: across at rapid, down to R at rapid, feed to depth (pecks retract), back up
            across = math.dist(self.pos[:2], (x, y)) if i else 0.0
            depth = r - z
            top = initial if c.retract == "G98" else r
            rapid_len = across + 2 * (top - r)
            feed_len = depth
            if c.cycle == "G83":
                n = math.ceil(depth / c.peck)
                rapid_len += 2 * sum(min(c.peck * k, depth) for k in range(1, n))
            if c.cycle == "G84":
                feed_len *= 2  # reverses out at the same feed
            self._time(rapid_len, self.rapid)
            self._time(feed_len, feed)
            if c.cycle == "G82":
                self.op["minutes"] += c.dwell / 60
            total_depth += depth
            self.pos = [x, y, top]
            self._track((x, y, z))
            self._track((x, y, top))
            last_z = z
            self.op["holes"] += 1
        self.emit("G80")
        self.modal = None
        self.f = feed
        if c.cycle == "G84":
            self.rpm = None  # rigid tapping leaves the spindle stopped

    def end(self) -> None:
        self.aicc_off()
        self.coolant_off()
        self.spindle_off()
        for line in self.m.end:
            self.emit(line)


def write(spec: ProgramSpec) -> tuple[str, dict]:
    """(program text, report): the report has the size, tools, envelope, per-operation
    moves / time and warnings."""
    spec.machine.validate()
    m = spec.machine
    if not 1 <= int(spec.number) <= 9999:
        raise CamcoreError("INVALID_VALUE", "Program number must be 1..9999")
    if not spec.blocks:
        raise CamcoreError("INVALID_VALUE", "A program needs at least one operation")
    tools: dict[int, object] = {}
    for b in spec.blocks:
        t = b.tool
        if t.number in tools and tools[t.number].key() != t.key():
            raise CamcoreError(
                "TOOL_CONFLICT",
                f"T{t.number} is used for two different tools "
                f"({tools[t.number].key()} and {t.key()})",
                {"number": t.number, "tools": [list(tools[t.number].key()), list(t.key())]},
            )
        tools.setdefault(t.number, t)
    if len(tools) > m.magazine:
        raise CamcoreError(
            "MACHINE_LIMIT",
            f"{len(tools)} tools, the magazine holds {m.magazine}",
            {"tools": sorted(tools), "magazine": m.magazine},
        )

    w = _Writer(spec)
    w.emit("%")
    w.emit(f"O{int(spec.number):04d} {comment(spec.title or 'FABEX', 30)}")
    for t in tools.values():
        w.emit(_tool_comment(t, m.max_comment))
    for text in spec.comments:
        w.emit(comment(text, m.max_comment))
    w.emit(m.units_block)
    w.emit(m.safe_start)
    for b in spec.blocks:
        w.block(b)
    w.op = None
    w.end()
    w.emit("%")
    text = "\n".join(w.lines) + "\n"

    size = len(text.encode("ascii", "replace"))
    if size > m.program_memory * 1024:
        w.warnings.append(
            f"Program is {size / 1024:.0f} kB, more than the {m.program_memory} kB of program "
            "memory: run it from a data server / memory card (DNC) or split it"
        )
    envelope = None
    if np.all(np.isfinite(w.lo)):
        envelope = {"min": [round(v, 4) for v in w.lo], "max": [round(v, 4) for v in w.hi]}
        span = w.hi - w.lo
        travel = np.asarray(m.travel, dtype=float) * 1000
        over = [a for a, s, t in zip("XYZ", span, travel) if s > t]
        if over:
            w.warnings.append(f"The program spans more than the machine's travel in {over}")
    for op in w.ops:
        op["minutes"] = round(op["minutes"], 3)
        op["length_mm"] = round(op["length_mm"], 3)
    report = {
        "program_number": int(spec.number),
        "title": spec.title,
        "bytes": size,
        "lines": len(w.lines),
        "minutes": round(sum(op["minutes"] for op in w.ops), 3),
        "tools": [
            {
                "number": t.number,
                "type": t.type,
                "diameter_mm": t.diameter,
                "operations": [b.name for b in spec.blocks if b.tool.number == t.number],
            }
            for t in tools.values()
        ],
        "operations": w.ops,
        "envelope_mm": envelope,
        "warnings": w.warnings,
    }
    return text, report
