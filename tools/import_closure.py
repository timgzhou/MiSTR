"""Static import graph of the repo: which modules the entry points reach, and which are dead.

    python tools/import_closure.py [--entries tools/entry_points.txt]
        prints every module of mistr/ and exp/ that no entry point reaches (exit 1 if any)
    python tools/import_closure.py --why <module>
        prints one import chain from an entry point to <module>

Imports inside functions count; so do string literals naming a module ("exp.mistr.train"),
which is how launchers and subprocess calls refer to them.
"""
import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("mistr", "exp")


def path_of(mod: str) -> Path | None:
    p = ROOT / (mod.replace(".", "/") + ".py")
    if p.exists():
        return p
    p = ROOT / mod.replace(".", "/") / "__init__.py"
    return p if p.exists() else None


def imports_of(mod: str, path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    pkg = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                base = pkg.split(".")[: len(pkg.split(".")) - (n.level - 1)]
                m = ".".join(base + ([n.module] if n.module else []))
            else:
                m = n.module or ""
            out.add(m)
            out |= {f"{m}.{a.name}" for a in n.names}
        elif (isinstance(n, ast.Constant) and isinstance(n.value, str)
              and n.value.startswith(tuple(p + "." for p in PACKAGES)) and " " not in n.value):
            out.add(n.value)
    parts = mod.split(".")
    out |= {".".join(parts[:i]) for i in range(1, len(parts))}
    return {m for m in out if path_of(m)}


def closure(entries):
    parent = {e: None for e in entries if path_of(e)}
    todo = list(parent)
    while todo:
        m = todo.pop()
        for d in imports_of(m, path_of(m)):
            if d not in parent:
                parent[d] = m
                todo.append(d)
    return parent


def all_modules():
    for pkg in PACKAGES:
        for p in sorted((ROOT / pkg).rglob("*.py")):
            m = ".".join(p.relative_to(ROOT).with_suffix("").parts)
            yield m.removesuffix(".__init__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", default=str(ROOT / "tools" / "entry_points.txt"))
    ap.add_argument("--why")
    a = ap.parse_args()
    entries = [l.split("#")[0].strip() for l in open(a.entries)]
    entries = [e for e in entries if e]
    missing = [e for e in entries if not path_of(e)]
    if missing:
        sys.exit(f"entry points that do not exist: {missing}")
    parent = closure(entries)
    if a.why:
        m, chain = a.why, []
        while m:
            chain.append(m)
            m = parent.get(m)
        print(" <- ".join(chain) if a.why in parent else f"{a.why}: not reached")
        return
    dead = [m for m in all_modules() if m not in parent]
    print("\n".join(dead) or "no dead modules")
    sys.exit(1 if dead else 0)


if __name__ == "__main__":
    main()
