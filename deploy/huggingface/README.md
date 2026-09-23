# Hugging Face deployments

## Current status — 2026-09-23

The model repository and its weights/tokenizer/calibration were also transferred to
https://huggingface.co/michaljach/jet; the pinned revision is unchanged.

Following Hugging Face support’s advice, the existing Space was transferred from
`jach-labs/jet` to `michaljach/jet`. The CPU Docker API is uploaded at
https://huggingface.co/spaces/michaljach/jet with CPU Basic requested and
`JET_CORS_ORIGINS=https://jach.me,https://michaljach-jet.hf.space`.

Hosting remains blocked: runtime is PAUSED with
`Quota exceeded for flavor cpu-basic (requested=1): current=0, limit=0`.
The signed-in browser’s Resume Space action also reports that the CPU Basic
quota is exhausted. A fresh personal Docker Space creation was rejected with
HTTP 402. No paid hardware or subscription was selected. The website continues
to use its working browser GPU option; its server default has not been changed
to the paused endpoint. Support must enable the personal account’s free CPU quota.

## Gradio + ZeroGPU

The `zerogpu/` folder is the server-side deployment matching Kev's architecture.
It uses `src/torch_model.py`, Gradio 6.28.0 and `@spaces.GPU`. Model weights stay
on the server. The queued Gradio API is `/decide` (see its README for a client
example); the separate Docker variant below exposes `/v1/decide`.

```sh
python deploy/huggingface/deploy.py zerogpu --repo michaljach/jet
```

This explicitly requests only `zero-a10g`, never a paid GPU. On 2026-09-23,
Hugging Face rejected this request with HTTP 402. The account's email is verified
and it was created in April 2024, but the signed-in Create Space UI still disables
Gradio. Its static Space settings offer no community GPU grant application.
The original support request was sent; support advised transferring to the older personal account. See the current status above.

Validation: all 36 published reference cases matched in predicted answer using
PyTorch on Apple Silicon, with maximum probability difference 0.01993 from the
bf16 reference. The local Gradio client successfully called `/decide`; malformed
questions were rejected before inference. Cloud CUDA/ZeroGPU execution cannot
be verified until Hugging Face enables hosting.

## CPU Docker deployment

The static Space is a status page only. Browser inference has been removed.
The Docker API uses CPU Basic, which has no
hourly hardware charge, but Hugging Face rejected creation on this account with
HTTP 402 (a PRO plan is required for personal Docker Spaces; Team/Enterprise for
organization Spaces). ZeroGPU creation was also rejected. No plan was purchased.

The server uses `michaljach/jet` at revision
`8a97cfea2df622bb03f5dc9b02567e21abd2551c`, its q8 ONNX weights and calibration.
The smaller runtime does not require MLX, PyTorch, or a GPU. Quantization changes
probabilities slightly; the model card reports up to 0.034 on its golden cases.
The server limits state context to 2048 tokens, truncating the middle, and allow 8
questions with a maximum of 4096 prompt tokens per question.

## Deploy

Authenticate using `hf auth login` (never put tokens in source files), then:

```sh
python deploy/huggingface/deploy.py static --repo michaljach/jet-status
# Only once the account is eligible for CPU Basic Spaces:
python deploy/huggingface/deploy.py api --repo michaljach/jet-api
```

The upload script uses an explicit file list and only requests `cpu-basic` for
the API. It never upgrades hardware, buys storage, or changes billing settings.
The static Space also includes the complete API deployment under `api/`.

To run the API locally without Docker, install `api/requirements.txt` in a
separate virtual environment, then from the repository root:

```sh
PYTHONPATH=src uvicorn app:app --app-dir deploy/huggingface/api --port 8000
curl http://localhost:8000/v1/decide -H 'Content-Type: application/json' \
  -d '{"state":"I love it","questions":{"positive":{"type":"noul","instructions":"Is the sentiment positive?"}}}'
```

Or download the `api/` folder from the published Space and run:

```sh
docker build -t jet-api api
docker run --rm -p 8000:7860 jet-api
```

The live API will have `/health`, `/docs` and `/v1/decide`. Optional settings:
`JET_API_KEY` (secret bearer token), `JET_CORS_ORIGINS` (comma-separated origins).
CPU work is serialized and additional requests get 503 with `Retry-After: 5`.
