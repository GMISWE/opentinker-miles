"""Oracles: external references a comparator can discharge E1/E2 against.

P1 placeholder — migrates with the E1/E2 gates:
  fp32 HF forward + HF+PEFT adapter oracle:
      specs/009-e1-8b-calibration/probes/e1_8b_oracle.py,
      scripts/gates/g9_adapter_interchange.py (export mode)
  rerun-envelope estimator:
      specs/013-a1-8b-spine/probes/ (e2_* / a5_* rerun population)
"""
