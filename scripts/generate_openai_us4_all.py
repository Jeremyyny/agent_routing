"""Generate OpenAI response JSONL for all three US4 subagent prompt files."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_MAX_TOKENS = {
    "extractor": 1000,
    "reasoner": 1200,
    "verifier": 1000,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/sft_data/deepseek_us4_500")
    parser.add_argument("--output-dir", default="outputs/sft_data/openai_us4_500")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="Per-kind limit; 0 means all rows.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extractor-max-tokens", type=int, default=DEFAULT_MAX_TOKENS["extractor"])
    parser.add_argument("--reasoner-max-tokens", type=int, default=DEFAULT_MAX_TOKENS["reasoner"])
    parser.add_argument("--verifier-max-tokens", type=int, default=DEFAULT_MAX_TOKENS["verifier"])
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("extractor", args.extractor_max_tokens),
        ("reasoner", args.reasoner_max_tokens),
        ("verifier", args.verifier_max_tokens),
    ]

    for idx, (kind, max_tokens) in enumerate(jobs, start=1):
        input_file = input_dir / f"{kind}_deepseek_prompts.jsonl"
        output_file = output_dir / f"{kind}_openai_responses.jsonl"
        if not input_file.exists():
            raise FileNotFoundError(f"Missing input file: {input_file}")

        print("=" * 80, flush=True)
        print(f"[{idx}/{len(jobs)}] {kind}: {input_file} -> {output_file}", flush=True)
        print(f"model={args.model} max_tokens={max_tokens} limit={args.limit} resume={args.resume}", flush=True)

        cmd = [
            sys.executable,
            "scripts/generate_openai_jsonl.py",
            "--input-file", str(input_file),
            "--output-file", str(output_file),
            "--model", args.model,
            "--temperature", str(args.temperature),
            "--max-tokens", str(max_tokens),
            "--limit", str(args.limit),
        ]
        if args.resume:
            cmd.append("--resume")

        subprocess.run(cmd, check=True)

    print("=" * 80, flush=True)
    print("All OpenAI response JSONL files generated.", flush=True)


if __name__ == "__main__":
    main()
