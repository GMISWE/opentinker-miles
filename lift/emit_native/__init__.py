"""Native emitters: TrainIR (+ LoweringProfile + assets) -> a framework's
native config. The reverse direction of the lift (specs/015-lift/BIDIR.md).

Each emitter reads the same per-framework semantic table right-to-left:
P-values come from the term, X-values from the profile (or the framework's
default profile), and refusals mark the direction's I-ledger (e.g. weighted
losses that NLLLoss cannot express).
"""
