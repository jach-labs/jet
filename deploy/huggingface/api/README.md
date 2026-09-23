---
title: Jet API
emoji: ✈️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
models:
  - michaljach/jet
---

CPU inference for Jet using its quantized ONNX export. POST `/v1/decide` uses
the same request format as Jet; interactive API documentation is at `/docs`.
Set the optional `JET_API_KEY` secret to require bearer authentication.
Set `JET_CORS_ORIGINS` to a comma-separated list of allowed browser origins.

Limits: 8 questions, 2048 state tokens (middle truncated), 4096 total prompt
tokens per question. One request at a time; busy requests receive HTTP 503.
Quantization can slightly change probabilities from the original bf16 model.

CPU Basic has no hourly hardware charge, but Hugging Face may require a paid
account plan to create this Docker Space. This deployment never selects paid hardware.
