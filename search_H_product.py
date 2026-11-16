#!/usr/bin/env python3
"""Search decompositions of products of identical parametrised H-box states.

An ``m``-qubit H-box has computational-basis amplitudes ``[1, ..., 1, a]``:
only the all-one string has amplitude ``a``.  Copies are contiguous in the
qubit order.  Omitting ``--a`` searches one decomposition valid symbolically
for every ``a``. The default automatically uses a small product subset of the
repo's dictionary for intact boxes, or the full dictionary for refinements.
All outputs are reconstructed exactly before they are written.
"""

from __future__ import annotations

import argparse

from h_product_search import HProductSearchError, run_h_search
from stab_search import add_search_arguments


def _complex_parameter(text: str) -> complex:
    """Parse a real or complex command-line value, accepting ``i`` or ``j``."""
    try:
        return complex(text.strip().lower().replace("i", "j"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid complex value: {text!r}") from error


def argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for H-box product searches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m",
        type=int,
        default=2,
        help="number of qubits in each H-box (default: 2)",
    )
    parser.add_argument(
        "--a",
        type=_complex_parameter,
        default=None,
        help="fixed H-box parameter, e.g. 0.5 or 1+2i; omit for general a",
    )
    parser = add_search_arguments(parser, partition_example="2+2+2")
    parser.add_argument(
        "--reduced-pool", action="store_true",
        help="alias for --pool reduced; supports any m and fixed or symbolic a",
    )
    parser.add_argument("--pool", choices=("auto", "reduced", "full"), default="auto",
                        help="dictionary selection (default: auto)")
    parser.add_argument("--report", help="write settings and trial results, including failures")
    parser.add_argument("--start-temperature", type=float, default=2e-2,
                        help="initial LNS temperature")
    parser.add_argument("--end-temperature", type=float, default=2e-4,
                        help="final LNS temperature")
    parser.add_argument("--working-pool-limit", type=int, default=50_000,
                        help="screened full-dictionary pool size")
    parser.add_argument("--pool-memory-mib", type=float, default=512,
                        help="estimated pool memory budget; raise explicitly for larger searches")
    parser.set_defaults(
        trials=8, iterations=5000, oversample=2, max_parts=4,
        destroy_sizes=(1, 2, 3, 4, 6), repair_shortlist=8,
    )
    return parser


def main() -> int:
    """Parse CLI arguments and delegate the H-box search to the helper."""
    parser = argument_parser()
    args = parser.parse_args()
    try:
        return run_h_search(args)
    except HProductSearchError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
