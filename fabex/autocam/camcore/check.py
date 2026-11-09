"""Check a FANUC milling program as text, the way the control will read it.

It reads the G-code (not the spec it came from), so it also checks programs edited by hand
or written by other posts. Canned cycles are expanded for the envelope. Errors are things
that would alarm, crash or cut wrong; warnings need a look.

Rules (errors unless marked):
- FRAMING: `%` first and last, an O-number, M30 (or M02) before the final `%`
- UNITS: G21 / G20 before any motion
- DECIMAL_POINT: X Y Z R Q I J K F always written with a decimal point
- TOOL: cutting (G1/G2/G3, cycles) only after T.. M06
- LENGTH_OFFSET: G43 H = T (+ the profile's offset) before cutting; no G0 Z between M06
  and G43
- SPINDLE: spindle on (M03/M04 with S > 0) before cutting; S within the machine's range;
  M05 before M06 (warning)
- FEED: F within the machine's maximum
- CYCLE: no canned cycle still active at M06 / M30 (G80 first)
- AICC: AICC off before M06 / M30; on during a canned cycle (warning)
- INCREMENTAL: no motion in G91 except G28 / G30 homing
- RAPID: no G0 that moves across while going down (going up: warning)
- ENVELOPE: the program fits the machine's travel
- SIZE: the program fits the program memory (warning)
- PLUNGE: an END / BALL / BULL tool feeding straight down more than its diameter below
  Z 0 (the stock top) in one move (warning)
"""

import re

from .machine import MachineProfile

_COMMENT = re.compile(r"\([^)]*\)")
_WORD = re.compile(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")
_TOOL_COMMENT = re.compile(r"\(T(\d+) D([0-9.]+) (\w+)")
_NEEDS_POINT = set("XYZRQIJKF")
_CYCLES = {"73", "74", "76", "81", "82", "83", "84", "85", "86", "87", "88", "89"}
_MAX_PER_RULE = 20


class _Checker:
    def __init__(self, machine: MachineProfile):
        self.m = machine
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.counts: dict[str, int] = {}

    def add(self, level: str, rule: str, n: int, text: str, message: str) -> None:
        key = f"{level}:{rule}"
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] > _MAX_PER_RULE:
            return
        entry = {"rule": rule, "line": n, "text": text[:80], "message": message}
        (self.errors if level == "error" else self.warnings).append(entry)

    def error(self, rule, n, text, message):
        self.add("error", rule, n, text, message)

    def warn(self, rule, n, text, message):
        self.add("warning", rule, n, text, message)


def check(text: str, machine: MachineProfile | None = None) -> dict:
    """Check a program. Returns ok (no errors), errors, warnings (each: rule, line, text,
    message), tools, envelope_mm, bytes and lines."""
    m = machine or MachineProfile()
    c = _Checker(m)
    lines = text.splitlines()
    tool_types = {int(a): (float(b), t) for a, b, t in _TOOL_COMMENT.findall(text)}
    nonempty = [(i + 1, ln.strip()) for i, ln in enumerate(lines) if ln.strip()]
    if not nonempty:
        c.error("FRAMING", 0, "", "Empty program")
        return _result(c, text, lines, set(), None)

    # framing
    if nonempty[0][1] != "%":
        c.error("FRAMING", nonempty[0][0], nonempty[0][1], "The first line must be %")
    if len(nonempty) < 2 or not re.match(r"O\d{1,4}\b", nonempty[1][1]):
        n, t = nonempty[min(1, len(nonempty) - 1)]
        c.error("FRAMING", n, t, "The program must start with an O-number after %")
    if nonempty[-1][1] != "%":
        c.error("FRAMING", nonempty[-1][0], nonempty[-1][1], "The last line must be %")

    units = None
    absolute = True
    motion = None  # 0, 1, 2, 3
    cycle = None  # active canned cycle code ("81", ...)
    cycle_r = cycle_z = None
    initial = None
    retract98 = True
    tool = None
    pending_t = None
    h = None
    spindle = False
    rigid = False  # M29: rigid tapping drives the spindle itself
    s = 0.0
    aicc = False
    ended = False
    pos = [None, None, None]
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    tools_used = set()
    since_m6 = False  # between M06 and G43
    feed_max = m.feed_max * 1000

    def track(p):
        for k in range(3):
            if p[k] is not None:
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])

    for n, raw in nonempty[1:-1] if len(nonempty) > 2 else []:
        line = _COMMENT.sub("", raw.upper()).strip()
        if not line or line == "%":
            continue
        if ended:
            c.error("FRAMING", n, raw, "Blocks after M30")
            continue
        words = _WORD.findall(line)
        if not words:
            continue
        g = [w[1] for w in words if w[0] == "G"]
        gnum = {x.lstrip("0") or "0" for x in g}
        mcodes = [int(float(w[1])) for w in words if w[0] == "M"]
        vals = {}
        for letter, value in words:
            if letter == "Q" and "5.1" in gnum:
                pass  # G05.1 Q1 / Q0: a mode, not a length
            elif letter in _NEEDS_POINT and "." not in value:
                c.error(
                    "DECIMAL_POINT",
                    n,
                    raw,
                    f"{letter}{value} has no decimal point (read as {value} um "
                    "unless parameter 3401#0 is set)",
                )
            if letter not in ("G", "M"):
                vals[letter] = float(value)

        if "21" in gnum or "20" in gnum:
            units = "21" if "21" in gnum else "20"
        if "90" in gnum:
            absolute = True
        if "91" in gnum:
            absolute = False
        if "98" in gnum:
            retract98 = True
        if "99" in gnum:
            retract98 = False
        if "49" in gnum:
            h = None
        if "5.1" in gnum:
            q = vals.get("Q")
            aicc = bool(q) if q is not None else aicc
        if "S" in vals:
            s = vals["S"]
            if s > m.spindle_max:
                c.error("SPINDLE", n, raw, f"S{s:g} above the machine's {m.spindle_max:g} rpm")
        if "F" in vals and vals["F"] > feed_max:
            c.error("FEED", n, raw, f"F{vals['F']:g} above the machine's {feed_max:g} mm/min")
        if "T" in vals:
            pending_t = int(vals["T"])
        homing = bool(gnum & {"28", "30", "53"})

        # M codes that end things
        if 6 in mcodes or 30 in mcodes or 2 in mcodes:
            what = "M06" if 6 in mcodes else "the program end"
            if cycle is not None:
                c.error("CYCLE", n, raw, f"Canned cycle G{cycle} still active at {what}: G80")
            if aicc:
                c.error("AICC", n, raw, f"AICC still on at {what}: {m.aicc_off}")
        if 6 in mcodes:
            if spindle:
                c.warn("SPINDLE", n, raw, "Spindle still on at M06 (M05 first)")
            if pending_t is None:
                c.error("TOOL", n, raw, "M06 without a T number")
            else:
                tool = pending_t
                tools_used.add(tool)
            h = None
            since_m6 = True
            pos = [None, None, None]
        if 3 in mcodes or 4 in mcodes:
            spindle = True
            if s <= 0:
                c.error("SPINDLE", n, raw, "Spindle started without an S speed")
        if 5 in mcodes:
            spindle = False
        if 29 in mcodes:
            rigid = True
            if s <= 0:
                c.error("SPINDLE", n, raw, "M29 without an S speed")
        if 30 in mcodes or 2 in mcodes:
            ended = True

        if "43" in gnum:
            if "H" not in vals:
                c.error("LENGTH_OFFSET", n, raw, "G43 without H")
            else:
                h = int(vals["H"])
                if tool is not None and h != tool + int(m.h_offset):
                    c.error("LENGTH_OFFSET", n, raw, f"G43 H{h} does not match T{tool}")
            since_m6 = False

        # motion
        if "80" in gnum:
            cycle = None
            rigid = False
        new_cycle = gnum & _CYCLES
        for code in ("0", "1", "2", "3"):
            if code in gnum:
                motion = int(code)
                cycle = None
        if new_cycle:
            cycle = sorted(new_cycle)[0]
            initial = pos[2]
            cycle_r = cycle_z = None

        has_xyz = any(a in vals for a in "XYZ")
        if homing or not has_xyz and not new_cycle:
            continue
        if not absolute:
            c.error("INCREMENTAL", n, raw, "Motion in G91 (incremental): G90 first")
            continue
        if units is None:
            c.error("UNITS", n, raw, "Motion before G21 / G20")
            units = "?"

        target = [vals.get(a, pos[k]) for k, a in enumerate("XYZ")]

        if cycle is not None:
            if "R" in vals:
                cycle_r = vals["R"]
            if "Z" in vals:
                cycle_z = vals["Z"]
            _require_cut(c, n, raw, tool, h, spindle or (rigid and cycle == "84"), m)
            if cycle == "84" and not rigid:
                c.warn("CYCLE", n, raw, "G84 without M29: floating-holder tapping")
            if aicc:
                c.warn("AICC", n, raw, "Canned cycle with AICC on")
            if cycle_z is None or cycle_r is None:
                c.error("CYCLE", n, raw, "Canned cycle without Z or R")
                continue
            x, y = target[0], target[1]
            track([x, y, cycle_z])
            track([x, y, cycle_r])
            if initial is not None:
                track([x, y, initial])
            pos = [x, y, initial if retract98 and initial is not None else cycle_r]
            continue

        if motion == 0:
            if since_m6 and "Z" in vals:
                c.error("LENGTH_OFFSET", n, raw, "Rapid in Z after M06 before G43")
            if None not in pos:
                across = target[0] != pos[0] or target[1] != pos[1]
                if across and target[2] < pos[2]:
                    c.error("RAPID", n, raw, "G0 moves across while going down")
                elif across and target[2] > pos[2]:
                    c.warn("RAPID", n, raw, "G0 moves across while going up")
        elif motion in (1, 2, 3):
            _require_cut(c, n, raw, tool, h, spindle, m)
            # straight down below Z 0 (the stock top) by more than the tool's diameter
            if (
                None not in pos
                and target[0] == pos[0]
                and target[1] == pos[1]
                and tool in tool_types
                and tool_types[tool][1] in ("END", "BALL", "BULL")
                and min(pos[2], 0.0) - target[2] > tool_types[tool][0]
            ):
                c.warn(
                    "PLUNGE",
                    n,
                    raw,
                    f"T{tool} feeds {min(pos[2], 0.0) - target[2]:.3f} mm straight down below "
                    "Z 0 (more than its diameter): ramp or helix in if this is in material",
                )
        else:
            c.error("FRAMING", n, raw, "Coordinates without a motion mode")
        pos = target
        track(pos)

    if not ended:
        c.error("FRAMING", nonempty[-1][0], nonempty[-1][1], "No M30 / M02 at the end")
    envelope = None
    if all(v != float("inf") for v in lo):
        envelope = {"min": lo, "max": hi}
        travel = [t * 1000 for t in m.travel]
        over = [a for a, a_lo, a_hi, t in zip("XYZ", lo, hi, travel) if a_hi - a_lo > t]
        if over:
            c.error("ENVELOPE", 0, "", f"The program spans more than the travel in {over}")
    return _result(c, text, lines, tools_used, envelope)


def _require_cut(c, n, raw, tool, h, spindle, m) -> None:
    if tool is None:
        c.error("TOOL", n, raw, "Cutting before any tool change (T.. M06)")
    elif h is None:
        c.error("LENGTH_OFFSET", n, raw, f"Cutting with T{tool} without G43 H")
    if not spindle:
        c.error("SPINDLE", n, raw, "Cutting with the spindle stopped")


def _result(c, text, lines, tools, envelope) -> dict:
    size = len(text.encode("ascii", "replace"))
    if size > c.m.program_memory * 1024:
        c.warn(
            "SIZE",
            0,
            "",
            f"{size / 1024:.0f} kB is more than the {c.m.program_memory} kB of program memory",
        )
    suppressed = {k: v - _MAX_PER_RULE for k, v in c.counts.items() if v > _MAX_PER_RULE}
    return {
        "ok": not c.errors,
        "errors": c.errors,
        "warnings": c.warnings,
        "suppressed": suppressed,
        "tools": sorted(tools),
        "envelope_mm": envelope,
        "bytes": size,
        "lines": len(lines),
    }
