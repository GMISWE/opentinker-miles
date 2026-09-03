# Server-side fallback dataset (Miles)

`gsm8k_rl.jsonl` — five GSM8K problems (`{"prompt", "response"}` per line).

Miles' `RolloutManager` needs a prompt dataset path at boot even though Tinker
clients send every training sample themselves; `MilesArgumentBuilder` points
`args.prompt_data` at `/data/datasets/gsm8k_rl.jsonl` on the pod, which is a copy
of this file. It is never read for training data. NeMo RL needs nothing of the kind.
