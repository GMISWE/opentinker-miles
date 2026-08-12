"""Gate suite: specification-first checking layer for the TinkerCloud API.

Design + SOTA positioning: specs/014-gate-suite/design.md (monorepo).
A gate is one composition: invariant x trace x arm -> verdict JSON
(schema: gates/verdicts/schema.json). Invariants encode the paper's
S3 contract; traces make its admission clause executable; drivers
speak only the public SDK/HTTP surface; comparators discharge the
claimed equivalence level (E0/E1/E2).
"""

GATES_VERSION = "0.1.0"
