"""``python -m skill_factory.ui`` launches the dashboard."""

from __future__ import annotations

import argparse

from skill_factory.ui.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill Factory dashboard")
    parser.add_argument("--runs", default="runs", help="Runs directory to browse")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(runs_dir=args.runs, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
