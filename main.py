
"""
main.py

Unified CLI entry point for all three tasks.
Usage examples are provided in README.md.
"""
from __future__ import annotations

import argparse
import importlib
import sys

def main():
    ap = argparse.ArgumentParser(description="HF Lab 3 — run tasks (a), (b), (c).")
    sub = ap.add_subparsers(dest="task", required=True)

    sub.add_parser("a", help="Task (a): evaluate HF classifiers on your FastText dataset.")
    sub.add_parser("b", help="Task (b): Ukrainian model demos (zero-shot, summarization, translation).")
    sub.add_parser("c", help="Task (c): Diffusers pipelines for text2img/img2img/inpaint.")

    # Parse only the first-level subcommand; keep the rest for the task's own argparse
    args, rest = ap.parse_known_args()

    if args.task == "a":
        mod = importlib.import_module("tasks.task_a")
    elif args.task == "b":
        mod = importlib.import_module("tasks.task_b")
    else:
        mod = importlib.import_module("tasks.task_c")

    # Ensure the task's argparse sees ONLY its own flags (without the subcommand token)
    _backup = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + rest
        mod.main()
    finally:
        sys.argv = _backup

if __name__ == "__main__":
    main()
