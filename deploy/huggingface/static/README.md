---
title: Jet
emoji: ✈️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
models:
  - michaljach/jet
short_description: Server inference deployment pending. API source available.
---

# Jet server deployment

Browser inference has been removed. This static Space is a deployment status
page and does not download or initialize any model weights.

The `zerogpu/` directory contains the tested server-side Gradio + ZeroGPU app
and `/decide` API. The `api/` directory contains the CPU Docker deployment for
`POST /v1/decide`; see `DEPLOYMENT.md`. Hosting is currently blocked by Hugging
Face account-plan requirements. No paid resources have been enabled.

Source: https://github.com/michaljach/jet
