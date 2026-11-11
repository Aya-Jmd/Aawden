"""Fresh-process inference with no intervention hooks installed."""
import argparse
from dataclasses import asdict
from .artifacts import atomic_json, read_json
from .interventions import load_toy_checkpoint
from .models import ToyCircuit, HuggingFaceAdapter
from .prompts import GenerationRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["toy", "hf"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    adapter = ToyCircuit(load_toy_checkpoint(args.checkpoint)) if args.kind == "toy" else HuggingFaceAdapter(read_json(args.checkpoint))
    requests = [GenerationRequest(**r) for r in read_json(args.requests)]
    atomic_json(args.output, [asdict(result) for result in adapter.generate(requests)])


if __name__ == "__main__":
    main()

