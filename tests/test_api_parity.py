"""Check that the scripting API produces exactly the same G-code as the UI operator.

For every case in test_data/ (a .blend plus reference _*.gcode / _*.tap files) this
works in temporary copies - so no tracked files are touched - and:

1. generates G-code with the Calculate Path operator (gcode_generator.py)
2. generates G-code with fabex.api.compute_operation (api_generator.py)
3. compares both against each other and against the reference files

Stale generated outputs are deleted from the copies first, so an operation that fails
to produce G-code shows up as MISSING instead of silently comparing an old file.

Headless Blender gets a throwaway copy of the user config (BLENDER_USER_CONFIG), so
add-on code that saves preferences can't modify the real ones.

Run:  python3 test_api_parity.py [case ...]
Blender: $BLENDER, else /Applications/Blender.app/Contents/MacOS/Blender, else `blender`.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_DATA = HERE / "test_data"


def blender_binary() -> str:
    candidates = [os.environ.get("BLENDER"), "/Applications/Blender.app/Contents/MacOS/Blender"]
    for c in candidates:
        if c and Path(c).exists():
            return c
    found = shutil.which("blender")
    if not found:
        sys.exit("Blender not found - set $BLENDER")
    return found


def user_config_dir(blender: str) -> Path | None:
    out = subprocess.run(
        [
            blender,
            "-b",
            "--factory-startup",
            "--python-expr",
            "import bpy; print('CFG=' + bpy.utils.user_resource('CONFIG'))",
        ],
        capture_output=True,
        text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("CFG="):
            return Path(line[4:])
    return None


def reference_files(case: Path) -> list[str]:
    return sorted(
        p.name for p in case.iterdir() if p.name.startswith("_") and p.suffix in (".gcode", ".tap")
    )


def body(path: Path) -> str | None:
    if not path.exists():
        return None
    return "".join(path.read_text().splitlines(keepends=True)[1:])  # first line has a date


def run_variant(blender, case: Path, script: Path, workdir: Path, env) -> tuple[Path, str]:
    copy = workdir / case.name
    shutil.copytree(case, copy)
    for ref in reference_files(copy):
        stale = copy / ref[1:]
        if stale.exists():
            stale.unlink()
    blend = next(copy.glob("*.blend"))
    proc = subprocess.run(
        [blender, "-noaudio", "-b", str(blend), "-P", str(script)],
        cwd=copy,
        env=env,
        capture_output=True,
        text=True,
    )
    log = proc.stdout + proc.stderr
    (copy / f"{script.stem}.log").write_text(log)
    missing = [r[1:] for r in reference_files(copy) if not (copy / r[1:]).exists()]
    if missing:
        # keep the evidence for intermittent failures
        important = [
            line
            for line in log.splitlines()
            if any(k in line for k in ("Error", "Traceback", "FAIL", "Segmentation", "Abort"))
        ]
        print(f"  {case.name} via {script.name}: exit {proc.returncode}, missing {missing}")
        for line in (important or log.splitlines())[-12:]:
            print(f"    | {line}")
    return copy, log


def main(selected):
    blender = blender_binary()
    cases = sorted(p for p in TEST_DATA.iterdir() if p.is_dir() and list(p.glob("*.blend")))
    if selected:
        cases = [c for c in cases if c.name in selected]

    with tempfile.TemporaryDirectory(prefix="fabex_parity_") as tmp:
        tmp = Path(tmp)
        env = dict(os.environ)
        cfg = user_config_dir(blender)
        if cfg and cfg.exists():
            shutil.copytree(cfg, tmp / "config")
            env["BLENDER_USER_CONFIG"] = str(tmp / "config")

        rows, failures = [], 0
        for case in cases:
            refs = reference_files(case)
            if not refs:
                continue
            op_dir, _ = run_variant(blender, case, HERE / "gcode_generator.py", tmp / "op", env)
            api_dir, api_log = run_variant(
                blender, case, HERE / "api_generator.py", tmp / "api", env
            )
            api_results = [
                json.loads(line[len("API_RESULT ") :])
                for line in api_log.splitlines()
                if line.startswith("API_RESULT ")
            ]
            for ref in refs:
                expected = body(case / ref)
                mac = case / (ref + ".mac")
                if sys.platform == "darwin" and mac.exists():
                    expected = body(mac)
                got_op = body(op_dir / ref[1:])
                got_api = body(api_dir / ref[1:])

                def verdict(got):
                    if got is None:
                        return "MISSING"
                    return "ok" if got == expected else "DIFF"

                parity = "same" if got_op == got_api else "DIFFERENT"
                rows.append((case.name, ref[1:], verdict(got_op), verdict(got_api), parity))
                if parity != "same" or verdict(got_api) != verdict(got_op):
                    failures += 1
            for r in api_results:
                if not r.get("ok"):
                    print(f"  API error in {case.name}: {r.get('error')}")
                elif not r["result"]["ok"]:
                    print(f"  {case.name}/{r['result']['op']}: {r['result']['message']}")

        width = max(len(f"{c}/{f}") for c, f, *_ in rows) if rows else 10
        print(f"\n{'case/file':<{width}}  operator  api      operator-vs-api")
        for c, f, v_op, v_api, parity in rows:
            print(f"{c + '/' + f:<{width}}  {v_op:<8}  {v_api:<7}  {parity}")
        print(f"\n{len(rows)} files, {failures} parity failures")
        return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
