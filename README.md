# Jet

Jet is a small decision model built on **Qwen3-0.6B**. Give it a state and named,
typed questions; it returns choices, scores, and probabilities without generating
free-form text. Answers always follow the requested type, but decisions can still
be wrong.

[Model weights](https://huggingface.co/michaljach/jet) ·
[Hugging Face demo](https://huggingface.co/spaces/michaljach/jet) ·
[Training history](TRAINING_HISTORY.md)

| Type | Criteria | Answer |
|---|---|---|
| `choice` | 2–255 named options | Selected key, probability per key, confidence |
| `score` | 2–10 ordered levels | Fractional score, selected level, probabilities |
| `noul` | None | Probability that the answer is yes |

## Run locally

```sh
git clone https://github.com/jach-labs/jet
cd jet
uv sync                  # Apple Silicon / Metal
# Linux with NVIDIA: uv sync --extra cuda
JET_API_KEY=secret uv run jet-serve --base-model michaljach/jet
```

```sh
curl http://localhost:8000/v1/decide \
  -H 'Authorization: Bearer secret' \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice this month.","questions":{"topic":{"type":"choice","instructions":"What is the primary issue?","criteria":{"billing":"billing or payment problem","bug":"the product is broken"}},"escalate":{"type":"noul","instructions":"Does this require human support?"}}}'
```

The model requires Jet's prompt format and label-token readout. It is not a chat
model. The MLX server shares the state prefix across questions, applies the saved
calibration temperatures, and returns typed answers. Serving truncates long states
in the middle; the Decision Index adapter instead requires complete inputs.

## How it works

Each answer option maps to a single token. Jet reads the next-token logits only
for those labels and applies softmax with a temperature fitted on held-out data.
Standard serving uses one forward pass per question, without sampling.
The fused bf16 weights include the trained LoRA updates.

The benchmark package also provides `decision_index_ensemble:TwoOrderJetEngine`.
It averages original and reversed option-order probabilities using exact option
keys. This approximately doubles forward-pass work and is separate from the
standard serving API.

## Training and evaluation

The current checkpoint was trained locally on an RTX 4080 Super with MLX CUDA.
The final training stage used 15,997 examples, rank-16 LoRA, learning rate `1e-5`,
gradient checkpointing, and one epoch. Selection chose step 3,500 of 3,570.
Public training partitions and deterministic generators cover relevance,
entailment, stance, sarcasm, routing, arithmetic, Boolean rules, code behavior,
commonsense completion, and tool-response preference. Calibration uses a separate
held-out split.

| Held-out set | Rows | Accuracy |
|---|---:|---:|
| General typed decisions | 3,683 | 66.66% |
| Intent, entailment, commonsense and tool-response preference | 600 | 73.50% |
| Transfer tasks | 720 | 67.50% |
| Domain-held-out tool routing | 240 | 85.00% |

These are single-order adapter measurements. They are not a Decision Index
leaderboard score. Partial public benchmark reconstructions do not cover the
complete current suite. Cross-dataset text overlap inherited during training
also prevents a certified decontamination claim. Performance varies by task;
see the [evaluation report](docs/training/jet-v5/summary.md) for full methodology,
limitations, and the separate two-order results.

To reproduce training, source acquisition, audits, and evaluation, use the
[recorded protocol](docs/training/jet-v5/protocol.md) and scripts under `scripts/`.
Dataset sources, revisions, exclusion checks, configuration, and checkpoint hashes
are recorded under `docs/training/`.

```sh
# CUDA launcher prepares the MLX CUDA environment.
./train_cuda.sh --help
uv run jet-calibrate --adapter /path/to/adapter --data /path/to/calibration.jsonl
uv run jet-fuse --adapter /path/to/adapter --out models/jet
uv run jet-eval --base-model models/jet --data data/test.jsonl
```

Optional `jet-distill` data generation uses Anthropic credentials or the
`--backend claude-code` mode. It is not required for serving or reproducing the
public-source training pipeline; hosted generation can incur charges.

## Deployment

The Hugging Face model package includes fused bf16 weights, tokenizer,
calibration, golden reference vectors, and a q8 ONNX export. The Gradio Space
uses a pinned model revision and exposes `/decide`; its availability depends on
Hugging Face's free hosting quota. See [deployment instructions](deploy/huggingface/README.md).

The ONNX model supports CPU and browser runtimes through ONNX Runtime.
`jet-golden` generates reference cases, and `export_web.sh` exports and checks
the browser model. Export validation is recorded with the model package;
quantization can change probabilities.

## Project layout

- `src/format.py`, `src/inference.py`: prompts and typed answers
- `src/model.py`, `src/torch_model.py`, `src/onnx_model.py`: inference backends
- `src/train.py`, `src/evaluate.py`, `src/fuse.py`: training, calibration, evaluation and fusion
- `src/data/`, `scripts/`: data builders and reproducible experiments
- `src/decision_index_engine.py`, `src/decision_index_ensemble.py`: benchmark adapters
- `deploy/huggingface/`: hosted demo and API deployment
