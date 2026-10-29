# AIS Website Assistant

A local, open-source, multi-agent chatbot that answers visitor questions using
**only** the content of https://aisinfo.com/. It shows the source page for every
answer, remembers earlier turns for follow-up questions, and says "I don't know"
(with the contact page) when the site doesn't cover the question.

Everything runs on your machine: Ollama (`qwen2.5:3b-instruct`), `BAAI/bge-small-en-v1.5`
embeddings, Chroma, LangGraph. No hosted APIs. No GPU needed (developed on a MacBook Air M1, 8 GB RAM).

## Quick start

Requires Python 3.10+ and [Ollama](https://ollama.com) running (`ollama serve`, or the desktop app).

```bash
python3 -m venv .venv && source .venv/bin/activate   # 1. isolated environment
pip install -r requirements.txt                       # 2. dependencies
ollama pull qwen2.5:3b-instruct                       # 3. local LLM (~2 GB, once)
python -m ais_assistant.index                         # 4. build the search index from the saved site (~1 min)
python -m ais_assistant.app                           # 5. chat at http://127.0.0.1:7860
```

Prefer the terminal? `python -m ais_assistant.chat --trace` is a command-line chat that also prints every agent's decision.

