"""Command-line chat.

    python -m ais_assistant.chat            # plain chat
    python -m ais_assistant.chat --trace    # also print each agent's decision
"""

import argparse
import logging

from .orchestrator import Conversation, Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the AIS website assistant")
    parser.add_argument("--trace", action="store_true", help="show what each agent decided")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    print("Loading models...")
    orchestrator = Orchestrator()
    orchestrator.warm_up()
    conversation = Conversation(orchestrator)
    print("AIS website assistant. Ask me about AIS (type 'quit' to exit).\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            continue
        reply = conversation.ask(message)
        print(f"\nAssistant: {reply.text}")
        for title, url in reply.sources:
            print(f"  Source: {title} - {url}")
        if args.trace:
            for step in reply.trace:
                print(f"  [{step['agent']}] {step['decision']} ({step['ms']} ms)")
        print()


if __name__ == "__main__":
    main()
