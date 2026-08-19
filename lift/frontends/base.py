"""Shared front-end result types: the knob ledger and the lowering profile.

Ledger classes (specs/015-lift/DESIGN.md §1, derived not declared):
  P  program-level: consumed into the TrainIR term
  X  execution-level: denotation-invariant; consumed into the LoweringProfile
     (subkinds: exec | observability | inert — inert = a disabled block)
  T  semantic-transform: expressible via a named transform, recorded
  I  inexpressible on the tinker surface today (feeds the surface backlog)
  U  unknown to this front-end version (loud; blocks emission)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lift import ir


@dataclass(frozen=True)
class LedgerEntry:
    key: str  # dotted config path
    klass: str  # "P" | "X" | "T" | "I" | "U"
    value: object = None
    note: str = ""


@dataclass
class LiftResult:
    source: str
    program: ir.Program
    profile: dict[str, object]  # execution attrs, consumed below the waist
    ledger: list[LedgerEntry] = field(default_factory=list)
    # digest -> full text for artifacts the term references only by digest
    # (e.g. jinja chat templates): program identity stays hash-based, while
    # adapters/certifiers can execute the artifact.
    assets: dict[str, str] = field(default_factory=dict)

    @property
    def unmapped(self) -> list[LedgerEntry]:
        return [e for e in self.ledger if e.klass == "U"]

    @property
    def inexpressible(self) -> list[LedgerEntry]:
        return [e for e in self.ledger if e.klass == "I"]

    @property
    def emittable(self) -> bool:
        """U blocks emission; I blocks it too unless the obligation is waived
        explicitly at emission time. Parsing is never coverage (DESIGN §5)."""
        return not self.unmapped and not self.inexpressible
