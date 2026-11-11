# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a guarded experimental copy; leave production Marlin untouched."""

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCE_SHA256 = (
    "859b4dc5d242c0a7b1799680eca44c13a5fbfc319738d29c429b357d7b6d7d54"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError("Changed Marlin source requires reviewing the barrier patch")
    needle = "      thread_block_reduce();"
    assert source.count(needle) == 1
    generated = source.replace(
        needle,
        "      // B loads must finish before other warps reuse sh_b as sh_red.\n"
        "      __syncthreads();\n" + needle,
    )
    args.output.write_text(generated)
    args.manifest.write_text(
        json.dumps(
            {
                "source": str(args.source),
                "source_sha256": digest,
                "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
                "change": "One convergent CTA barrier before thread_block_reduce",
                "production_changed": False,
            },
            indent=2,
        )
    )
