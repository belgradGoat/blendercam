# FANUC Machining Centres

Fabex can write programs for FANUC-controlled vertical machining centres such as the
**FANUC Robodrill** (0i / 31i controls). Choose **FANUC (Robodrill)** as the post processor
in the Machine panel, or call `set_machine()` from a script or agent.

The FANUC post is not a variant of the ISO post. It writes what a machining centre with a
tool changer needs:

- **Tool changes:** `T# M06` with `G43 H#` (H = T unless the profile says otherwise), spindle
  and coolant off before each change.
- **Decimal points everywhere.** `X10.` means 10 mm whatever parameter 3401 says. Fabex's
  other posts write `X10`, which a FANUC control without decimal-point programming reads as
  10 µm.
- **Rapids never move across while going down.** A FANUC G0 is not a straight line by
  default, so every rapid goes up first, then across, then down.
- **Canned drilling cycles.** DRILL operations become `G81`, `G82`, `G73`, `G83` or rigid
  tapping `G84` (with `M29`), ended by `G80`.
- **AI contour control** (`G05.1 Q1`) around milling moves, switched off before cycles, tool
  changes and the end of the program.
- **A safe start and end:** `G21` in its own block, then
  `G0 G17 G40 G49 G80 G90 G94`. The end is `M9 M5 G91 G28 Z0. G28 Y0. G90 M30`.
- **No synthetic return to X0 Y0** at the start of every operation, no `G64`, and no
  splitting by line count.
- **F only on feed blocks.** Feed moves are thinned to 1 µm chord tolerance, so straight
  runs collapse to their ends.

Moves are classified the same way as in Fabex's other posts: rapid, feed or plunge (from
the operation's plunge angle).

```
%
O1001 (BRACKET OP1)
(T1 D4. END 3FL)
(T2 D5. DRILL 2FL)
G21
G0 G17 G40 G49 G80 G90 G94
(POCKET - POCKET)
G91 G28 Z0.
G90
T1 M06
G0 G90 G54 X95. Y59. S10000 M03
G43 H1 Z10.
M08
G05.1 Q1
G1 Z-1.5 F300.
X105. F1200.
...
(DRILL 5 - DRILL)
G05.1 Q0
M09
M05
G91 G28 Z0.
G90
T2 M06
G0 G90 G54 X80. Y60. S6000 M03
G43 H2 Z10.
M08
G98 G73 X80. Y60. Z-10. R2. Q5. F400.
X120.
G80
...
M30
%
```

## The machine profile

Everything a machine option or a shop habit can change is part of the **machine
profile**, stored in the .blend file:
- the safe-start block;
- the AICC on and off blocks;
- the coolant M-codes;
- homing and the program end;
- the work offset;
- the tool length offset numbering.

Until you change it, the profile describes a **Robodrill α-D21MiB5**:

| Setting | Default |
|---|---|
| travel | 500 × 400 × 330 mm |
| spindle | 10,000 rpm |
| cutting feed | up to 30 m/min |
| magazine | 21 tools |
| work offset | G54 |
| AICC | on |

Some values are typical rather than read from a machine: spindle power, program memory
(1 MB) and tool change time. Check them against yours.

```python
fx.machine_profile()                                   # the profile, and Fabex's machine
fx.set_machine(preset="robodrill_d21mib5")             # or robodrill_d21lib5 / _d14sib5
fx.set_machine(profile={"spindle_max": 24000, "coolant": {"THROUGH": "M88"}})
fx.set_machine(profile={"aicc": False, "work_offset": "G55", "program_memory": 2048})
```

Setting the profile also updates Fabex's own machine settings to match:
- post processor FANUC;
- millimetres;
- the travel as the working area;
- the feed and spindle limits.

Fabex refuses operations faster than its machine's maximum feed. That maximum defaults to
2 m/min.

## Operation settings

With the FANUC post selected, each operation shows a **Machining Centre** section. It holds:

- **Tool Number (T):** give each different cutter its own number. One number used for two
  different cutters stops the export with `TOOL_CONFLICT`.
- **Coolant:** flood (M08), mist (M07), air or through-spindle (M-codes from the profile),
  or off.
- **AI Contour Control:** on by default.
- **Drill Cycle** for DRILL operations:
  - **Auto** picks G81 up to 1 × D deep, G73 (chip breaking, good in aluminium) up to
    4 × D, and G83 (full retract) deeper.
  - Or choose G81, G82 (dwell), G73, G83 or G84 (rigid tapping, feed = speed × pitch).
  - Peck depth, dwell, R clearance above the depth start, and tap pitch go with it. With
    the peck left at 0, each peck is one drill diameter, or the stepdown if layers are on
    and it is smaller.

Holes are taken from the DRILL operation's computed path, so any DRILL source works. The
centres and depths are the ones Fabex computed. For B-rep parts, `extract_edges(kind='holes',
project='bottom')` puts them at the exact hole centres and bottoms. Holes are drilled at
the operation's plunge feed, as Fabex's other posts do.

## Programs from several operations

Fabex's chains write several operations into one program, with tool changes between them,
in the chain's order:

```python
fx.create_chain("OP1", ["Drill 5", "Drill 4", "Pocket", "Cutout"])
report = fx.export_program(chain="OP1", directory="/path/to/programs", program_number=1001)
report = fx.export_program(["Pocket", "Drill 5"], directory="...", title="Bracket OP1")
```

`export_program()` writes `<title>.nc` and returns a report:

- the file, its size in bytes and lines, and the estimated minutes;
- the tools, with the operations that use each one;
- per operation: moves, points in and written, holes, length and time;
- the envelope in mm, and any warnings, such as clamped speeds or a program bigger than the
  memory;
- `check`: the program read back as G-code and checked (below).

The Export buttons and the chain operators in the UI write the same program when the post
processor is FANUC.

## Checking a program

`check_gcode(path=...)` (or `text=...`) reads a program the way the control will and
reports errors and warnings by rule:

| Rule | Error when |
|---|---|
| FRAMING | `%` first and last, an O-number, `M30` / `M02` before the end, nothing after it |
| UNITS | motion before `G21` / `G20` |
| DECIMAL_POINT | X Y Z R Q I J K F without a decimal point |
| TOOL | cutting before `T.. M06` |
| LENGTH_OFFSET | cutting without `G43 H` = T; a `G0 Z` between `M06` and `G43` |
| SPINDLE | cutting with the spindle stopped (rigid tapping with `M29` is fine); S above the machine's maximum |
| FEED | F above the machine's maximum |
| CYCLE | a canned cycle still active at `M06` / `M30` |
| AICC | AICC still on at `M06` / `M30` |
| INCREMENTAL | moves in `G91` other than `G28` / `G30` homing |
| RAPID | a `G0` that moves across while going down |
| ENVELOPE | the program spans more than the machine's travel |

**Warnings:**
- **RAPID:** a `G0` that moves across while going up.
- **SPINDLE:** no `M05` before `M06`.
- **AICC:** AICC on during a cycle.
- **SIZE:** the program exceeds program memory; use a data server or memory card.
- **PLUNGE:** an end or ball mill feeds straight down below Z 0 by more than its diameter.
  Ramp or helix in instead.

The check also works on programs from other software or edited by hand.

## Tool library

`tool_library()` lists the tools automatic planning may use. Unless you set your own, this
is a catalogue built from Fabex's metric cutter presets:
- end and ball mills, 1–20 mm;
- carbide drills, 1–12 mm, including tap-drill and clearance sizes for M3–M12;
- 6 and 10 mm 90° spot drills.

The presets only give a type and a diameter. Flute length, reach and holder diameter are
typical values.

`set_tool_library([...])` replaces the catalogue with your own tools, such as the turret
map with fixed T numbers:

```python
fx.set_tool_library([
    {"id": "EM10", "type": "END", "diameter": 0.010, "number": 1, "flute_length": 0.022},
    {"id": "DR6.8", "type": "DRILL", "diameter": 0.0068, "number": 2},
    {"id": "M8", "type": "TAP", "diameter": 0.008, "pitch": 0.00125, "number": 3},
])
```

Lengths are metres, like the rest of the API.

## Before the first cut

Run each new program as an air cut, with the work offset raised, before cutting metal.
Check these against your machine and adjust the profile:
- the safe start;
- the coolant M-codes;
- the program end;
- whether the AICC option is fitted (`G05.1 Q1` alarms without it);
- program memory.
