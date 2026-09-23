"""Server-side Jet demo and Gradio API on Hugging Face ZeroGPU."""
import spaces  # Must precede torch imports for ZeroGPU initialization.

import json
import os
import gradio as gr

from format import Question
from torch_model import TorchJet

jet = TorchJet(device=os.getenv("JET_DEVICE", "cuda"))

EXAMPLES = [
    ["hey can you set an alarm for 6:30 tomorrow morning", json.dumps({
        "intent": {"type": "choice", "instructions": "What does the user want?",
                   "criteria": {"alarm": "set an alarm", "weather": "ask about the weather",
                                "music": "play music", "calendar": "check the calendar"}},
    }, indent=2)],
    ["The battery lasts all day, the screen is beautiful, and shipping was fast. I love it.", json.dumps({
        "stars": {"type": "score", "instructions": "How many stars would this reviewer give?",
                  "criteria": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]},
        "positive": {"type": "noul", "instructions": "Is the sentiment positive?"},
    }, indent=2)],
]


@spaces.GPU(duration=30)
def infer(prepared):
    return jet.decide_prepared(prepared)


def decide(state: str, questions: str) -> dict:
    """Answer typed questions using Jet on the server.

    state: plain text or a JSON object/list.
    questions: JSON object of named choice, score or noul questions.
    """
    if len(state.encode()) + len(questions.encode()) > 100_000:
        raise gr.Error("Request exceeds 100 KB")
    try:
        parsed = json.loads(questions)
        if not isinstance(parsed, dict) or not 1 <= len(parsed) <= 8:
            raise ValueError("Provide 1–8 named questions")
        if any(not isinstance(q, dict) for q in parsed.values()):
            raise ValueError("Each question must be an object")
        typed = {name: Question.from_dict(q) for name, q in parsed.items()}
        state_value = state
        if state.lstrip().startswith(("{", "[")):
            try:
                state_value = json.loads(state)
            except json.JSONDecodeError:
                pass  # Arbitrary text beginning with a bracket is still valid state.
        prepared = jet.prepare(state_value, typed)
    except (ValueError, KeyError, TypeError) as exc:
        raise gr.Error(f"Invalid request: {exc}") from exc
    return infer(prepared)


with gr.Blocks(title="Jet decision model") as demo:
    gr.Markdown("""# Jet
Give Jet context and typed questions. Get choices, scores and probabilities.
The model runs on Hugging Face's servers. Your browser sends inputs and displays results.
""")
    with gr.Row():
        with gr.Column():
            state = gr.Textbox(label="State", lines=6, value=EXAMPLES[0][0])
            questions = gr.Code(label="Questions (JSON)", language="json", value=EXAMPLES[0][1])
            run = gr.Button("Decide", variant="primary")
        output = gr.JSON(label="Decision")
    gr.Examples(EXAMPLES, inputs=[state, questions], label="Examples")
    gr.Markdown("ZeroGPU requests are queued and subject to Hugging Face usage quotas. "
                "Use **View API** for the `/decide` endpoint. "
                "[Model](https://huggingface.co/michaljach/jet) · "
                "[Source](https://github.com/michaljach/jet)")
    run.click(decide, inputs=[state, questions], outputs=output,
              api_name="decide", concurrency_limit=1)

demo.queue(max_size=16, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
