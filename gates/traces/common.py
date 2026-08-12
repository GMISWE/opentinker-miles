"""Shared trace ingredients.

synthetic_lm_datums reproduces the generator of scripts/gates/g1_seam_parity.py
and specs/008+013 g1b probes token-for-token, so gate results stay comparable
with the recorded results JSONs.
"""


def synthetic_token_rows(n: int, seq: int, vocab: int = 50000) -> list[list[int]]:
    return [
        [(1000 + i * 7919 + j * 104729) % vocab for j in range(seq)]
        for i in range(n)
    ]


def synthetic_lm_datums(n: int, seq: int):
    import torch
    from tinker import types

    out = []
    for toks in synthetic_token_rows(n, seq):
        inp, tgt = toks[:-1], toks[1:]
        out.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(inp),
                loss_fn_inputs={
                    "weights": types.TensorData.from_torch(
                        torch.ones(len(inp), dtype=torch.float32)
                    ),
                    "target_tokens": types.TensorData.from_torch(
                        torch.tensor(tgt, dtype=torch.long)
                    ),
                },
            )
        )
    return out
