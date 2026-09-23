To: website@huggingface.co
Subject: Free ZeroGPU hosting eligibility for michaljach — HTTP 402 despite verified account

Hello Hugging Face team,

I would like to host a public Gradio + ZeroGPU demo for my model, jach-labs/jet,
under my personal account michaljach. I only want free hosting.

The ZeroGPU documentation says a verified personal account older than 30 days
can host up to two ZeroGPU Spaces for free:
https://huggingface.co/docs/hub/spaces-zerogpu

My account was created in April 2024, my primary email is verified, and I have
no personal Spaces. However, creating michaljach/jet with SDK gradio and hardware
zero-a10g returns HTTP 402 with this message:

“You must be subscribed to PRO to host Spaces with ZeroGPU. If you recently
created your account, please wait 30 days or request a community grant.”

Request ID: Root=1-6ab417d8-5113f80268a257b0279d2b16

The signed-in Create Space page also disables Gradio and Docker and requires
PRO. The existing static Space's settings do not show a community GPU grant
application option.

Could you check my free ZeroGPU eligibility and enable it, or advise how to
request a free community ZeroGPU grant?

The server-side Gradio app is implemented and tested. It uses PyTorch with
@spaces.GPU, loads the model onto CUDA at startup, and does not run inference
in the browser. Jet is a 0.6B Qwen3-based decision model with typed answers.

Model: https://huggingface.co/jach-labs/jet
Deployment source: https://huggingface.co/spaces/jach-labs/jet/tree/main/zerogpu

Thank you,
Michal Jach
