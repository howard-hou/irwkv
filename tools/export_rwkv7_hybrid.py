# SPDX-License-Identifier: Apache-2.0
"""Write a weight-free bux-compatible hybrid config for --load-format dummy."""

import argparse
import json
from pathlib import Path

from irwkv.transformers_utils.rwkv7_hybrid import Rwkv7HybridConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=3328)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()
    config = Rwkv7HybridConfig(
        hidden_size=args.width,
        num_hidden_layers=args.layers,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_model_len,
        dtype="bfloat16",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "config.json"
    if target.exists():
        parser.error(f"{target} already exists")
    config.save_pretrained(args.output)
    print(
        json.dumps(
            {
                "model": str(args.output),
                "width": args.width,
                "layers": config.num_hidden_layers,
                **config.parameter_count(),
            }
        )
    )


if __name__ == "__main__":
    main()
