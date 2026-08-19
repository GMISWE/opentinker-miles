"""Corpus survey: run the front-end + ledger pipeline over a directory of
shipped configs; emit the per-config ledger and the config->program collapse
ratio. "Covered" claims still require certification (DESIGN §5) — this
report is the static half.

Usage: python -m lift.coverage <configs_dir> [--recursive]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lift.frontends.nemo_rl import LiftError, elaborate_sft, load_config
from lift.passes import COLLAPSE_PIPELINE, DEFAULT_PIPELINE, default_registry
from lift.pm import PassManager

ALGO_KEYS = ("sft", "grpo", "dpo", "rm", "distillation")


@dataclass
class Row:
    file: str
    algo: str
    status: str  # "lifted" | "unsupported-class" | "blocked" | "error"
    guarantee: str = ""
    program_hash: str = ""
    approx_hash: str = ""  # after the declared E1 collapse pipeline
    counts: dict[str, int] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)
    inexpressible: list[str] = field(default_factory=list)
    note: str = ""


def _algo_of(path: Path) -> str:
    try:
        cfg = load_config(str(path))  # resolves `defaults:` include chains
    except (yaml.YAMLError, LiftError, OSError) as exc:
        return f"load-error:{exc.__class__.__name__}"
    for k in ALGO_KEYS:
        if k in cfg:
            return k
    return "unknown"


def survey(config_dir: str, recursive: bool = False) -> list[Row]:
    root = Path(config_dir)
    files = sorted(root.rglob("*.yaml") if recursive else root.glob("*.yaml"))
    rows: list[Row] = []
    for f in files:
        rel = str(f.relative_to(root))
        algo = _algo_of(f)
        if algo != "sft":
            rows.append(Row(rel, algo, "unsupported-class"))
            continue
        try:
            r = elaborate_sft(str(f))
        except LiftError as exc:
            rows.append(Row(rel, algo, "error", note=str(exc)))
            continue
        out, rpt = PassManager(DEFAULT_PIPELINE, default_registry()).run(
            r.program, source=rel
        )
        aout, _arpt = PassManager(COLLAPSE_PIPELINE, default_registry()).run(
            r.program, source=rel
        )
        counts = Counter(e.klass for e in r.ledger)
        rows.append(
            Row(
                rel,
                algo,
                "blocked"
                if not r.emittable
                else ("blocked" if rpt.stopped else "lifted"),
                guarantee=rpt.guarantee.name,
                program_hash=out.hash[:12],
                approx_hash=aout.hash[:12],
                counts=dict(sorted(counts.items())),
                unmapped=[e.key for e in r.unmapped],
                inexpressible=[e.key for e in r.inexpressible],
                note="pipeline stopped" if rpt.stopped else "",
            )
        )
    return rows


def collapse(rows: list[Row], attr: str = "program_hash") -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for r in rows:
        if r.status == "lifted":
            groups.setdefault(getattr(r, attr), []).append(r.file)
    return groups


def render(rows: list[Row]) -> str:
    lines = [
        "| config | algo | status | E | P/X/T/I/U | blockers |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r.counts
        pxtiu = "/".join(str(c.get(k, 0)) for k in "PXTIU") if c else ""
        blockers = "; ".join(r.unmapped + r.inexpressible) or r.note
        lines.append(
            f"| {r.file} | {r.algo} | {r.status} | {r.guarantee} | {pxtiu} | {blockers} |"
        )
    lifted = [r for r in rows if r.status == "lifted"]
    lines.append("")
    for label, attr in (
        ("strict (E0)", "program_hash"),
        ("approx (E1 sched)", "approx_hash"),
    ):
        groups = collapse(rows, attr)
        lines.append(
            f"{label}: lifted {len(lifted)}/{len(rows)} -> {len(groups)} programs"
        )
        for h, fs in sorted(groups.items()):
            if len(fs) > 1:
                lines.append(f"  collapse {h}: {', '.join(fs)}")
    return "\n".join(lines)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = survey(args[0], recursive="--recursive" in sys.argv)
    print(render(rows))
    if "--json" in sys.argv:
        print(json.dumps([r.__dict__ for r in rows], indent=1))


if __name__ == "__main__":
    main()
