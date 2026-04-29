from __future__ import annotations

import argparse

from weather_agent.cmd.bot import cmd_bot
from weather_agent.cmd.worker import cmd_worker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weather_agent",
        description="Telegram Weather AI Agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bot_parser = subparsers.add_parser("bot", help="Start the Telegram bot")
    bot_parser.set_defaults(func=cmd_bot)

    worker_parser = subparsers.add_parser("worker", help="Start the rule evaluation worker")
    worker_parser.set_defaults(func=cmd_worker)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
