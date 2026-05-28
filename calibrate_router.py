"""
Calibration script for SemanticRouter MATCH_THRESHOLD.

Usage:
    python calibrate_router.py --primitives path/to/primitives.yaml

This script lets you type commands and see the similarity scores
against all primitives, so you can decide where to set the threshold.
"""

import argparse
from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer, util

# ── ANSI colors ────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

MATCH_THRESHOLD     = 0.55   # change this to test different values
AMBIGUOUS_THRESHOLD = 0.40   # kept for reference


def color_score(score: float) -> str:
    s = f"{score:.3f}"
    if score >= MATCH_THRESHOLD:
        return f"{GREEN}{BOLD}{s}{RESET}"
    elif score >= AMBIGUOUS_THRESHOLD:
        return f"{YELLOW}{s}{RESET}"
    else:
        return f"{RED}{s}{RESET}"


def load_primitives(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Calibrate SemanticRouter threshold")
    parser.add_argument(
        "--primitives",
        default="primitives.yaml",
        help="Path to primitives.yaml (default: primitives.yaml)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help=f"Match threshold to test (default: {MATCH_THRESHOLD})",
    )
    args = parser.parse_args()

    threshold = args.threshold

    primitives = load_primitives(args.primitives)
    print(f"\n{BOLD}Loading embedding model...{RESET}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"{BOLD}Pre-computing primitive embeddings...{RESET}")
    prim_embeddings = {
        name: embedder.encode(data["description"], convert_to_tensor=True)
        for name, data in primitives.items()
    }

    print(f"\n{BOLD}Primitives loaded:{RESET}")
    for name, data in primitives.items():
        print(f"  {BOLD}{name}{RESET}: {data['description']}")

    print(f"\n{BOLD}Threshold: {GREEN}{threshold}{RESET}")
    print(f"  {GREEN}■{RESET} >= {threshold}  → COVERED  (existing_system)")
    print(f"  {RED}■{RESET} <  {threshold}  → NOT COVERED (code_generation)")
    print("\nType a command and press Enter. Type 'quit' to exit.\n")
    print("─" * 60)

    while True:
        try:
            command = input(f"\n{BOLD}Command:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if command.lower() in ("quit", "exit", "q"):
            break
        if not command:
            continue

        chunk_emb = embedder.encode(command, convert_to_tensor=True)
        scores = {
            name: util.cos_sim(chunk_emb, emb).item()
            for name, emb in prim_embeddings.items()
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Scores for: \"{command}\"")
        print(f"  {'Primitive':<20} {'Score':>8}  Description")
        print(f"  {'─'*20}  {'─'*8}  {'─'*35}")
        for name, score in sorted_scores:
            desc = primitives[name]["description"]
            covered_marker = "✓" if score >= threshold else " "
            print(f"  {covered_marker} {name:<18} {color_score(score)}  {desc}")

        best_name, best_score = sorted_scores[0]
        if best_score >= threshold:
            print(f"\n  → {GREEN}{BOLD}COVERED{RESET} by '{best_name}' ({best_score:.3f})")
        else:
            print(f"\n  → {RED}{BOLD}NOT COVERED{RESET} — would go to code_generation (best: {best_score:.3f})")


if __name__ == "__main__":
    main()