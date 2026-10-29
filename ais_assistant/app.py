"""Browser chat screen (Gradio), served on http://127.0.0.1:7860 only.

    python -m ais_assistant.app

One Orchestrator (models, index) is shared; each browser tab gets its own
Conversation through gr.State, so two visitors never see each other's history.
"""

import logging
import os

# Gradio sends usage analytics and a version check by default; nothing may leave the machine.
# Must be set before gradio is imported.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr  # noqa: E402

from .orchestrator import Conversation, Orchestrator

EXAMPLES = [
    "What industries does AIS work with?",
    "Where are AIS's offices in India?",
    "How can I apply for a job at AIS?",
]


def build_ui(orchestrator: Orchestrator) -> gr.Blocks:
    with gr.Blocks(title="AIS Website Assistant", analytics_enabled=False) as demo:
        gr.Markdown("## AIS Website Assistant\nAnswers come only from the public "
                    "[aisinfo.com](https://aisinfo.com/) website, with the source page linked.")
        chatbot = gr.Chatbot(height=480)
        conversation = gr.State(None)
        with gr.Row():
            box = gr.Textbox(placeholder="Ask about AIS...", show_label=False, scale=8)
            send = gr.Button("Send", variant="primary", scale=1)
        with gr.Accordion("Agent decisions for the last answer", open=False):
            trace_box = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=box)

        def respond(message, history, conv):
            conv = conv or Conversation(orchestrator)
            if not message.strip():
                return "", history, conv, gr.update()
            reply = conv.ask(message)
            history = (history or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply.as_markdown()},
            ]
            trace = "\n".join(f"- **{t['agent']}** ({t['ms']} ms): {t['decision']}" for t in reply.trace)
            return "", history, conv, trace

        outputs = [box, chatbot, conversation, trace_box]
        box.submit(respond, [box, chatbot, conversation], outputs)
        send.click(respond, [box, chatbot, conversation], outputs)
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # 127.0.0.1 and no share link: nothing is exposed outside this machine
    orchestrator = Orchestrator()
    orchestrator.warm_up()
    build_ui(orchestrator).launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
