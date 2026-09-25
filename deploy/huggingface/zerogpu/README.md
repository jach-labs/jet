---
title: Jet
emoji: ✈️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
short_description: Typed decisions with server-side ZeroGPU inference
startup_duration_timeout: 1h
models:
  - michaljach/jet
---

# Jet on ZeroGPU

Jet's Gradio demo and API run entirely on Hugging Face's servers. The browser
does not download model weights. Inference uses PyTorch and `@spaces.GPU`, with
the model placed on CUDA during startup as required by ZeroGPU.

The model and calibration are pinned to `michaljach/jet` revision
`25ccbd9e09c75643b3c2214e2b2522bec39171a7`, the last Qwen3-0.6B release (V5).
Prompts and typed answers share the same code as the original Jet server. Limits: 8 questions, 4096 state tokens
(middle truncation), 6144 total tokens per prompt, 100 KB per request.

## API

```python
import json
from gradio_client import Client

client = Client("michaljach/jet")
result = client.predict(
    "I love this product.",
    json.dumps({"positive": {"type": "noul", "instructions": "Is the sentiment positive?"}}),
    api_name="/decide",
)
print(result["answers"])
```

This is the queued Gradio `/decide` API, as used by Kev's Space. It is not the
standalone FastAPI `/v1/decide` route. ZeroGPU is subject to Hugging Face quotas
and hosting eligibility. Deploy only with `zero-a10g` hardware; the deployment
script will never upgrade to a paid GPU or change account billing.

For local testing only, set `JET_DEVICE=cpu` (or `mps` on Apple Silicon).
Leave it unset on ZeroGPU so CUDA initialization works correctly.
