"""Import every module of mistr/ and exp/, one subprocess each (so one module's native
libraries cannot clash with another's). Exit 1 if any import fails."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
mods = [".".join(p.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
        for pkg in ("mistr", "exp") for p in sorted((ROOT / pkg).rglob("*.py"))]
bad = 0
for m in mods:
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", f"import {m}"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode:
        bad += 1
        print(f"FAIL {m}: {(r.stderr.strip().splitlines() or ['?'])[-1][:200]}")
print(f"{len(mods)} modules, {bad} failed")
sys.exit(1 if bad else 0)
