#!/usr/bin/env python3
"""Search symmetric stabiliser decompositions of general product phase states."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from stab_core import orbit_weights
from stab_search import add_search_arguments, run_target_search


def phase_product_state(n: int, w: complex) -> np.ndarray:
    """Return computational-basis amplitudes of ``(|0> + w|1>)**n``."""
    return np.asarray([w ** index.bit_count() for index in range(1 << n)])


def phase_product_orbits(ns: Sequence[int], w: complex) -> np.ndarray:
    """Return one phase-product amplitude per block-weight orbit."""
    return w ** orbit_weights(ns).sum(axis=1)


def general_phase_product_span(ns: Sequence[int]) -> np.ndarray:
    """Orbit-basis columns spanning (|0>+w|1>)^n for every complex w."""
    weights = orbit_weights(ns).sum(axis=1)
    n = sum(ns)
    return np.equal.outer(weights, np.arange(n + 1)).astype(np.complex128)


def argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for phase-product searches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        type=float,
        default=None,
        help=(
            "search one w=exp(i*phase); omit to search the complete "
            "generic-w family"
        ),
    )
    return add_search_arguments(parser, partition_example="3+3+5")


def main() -> None:
    """Parse command-line arguments and run the requested search."""
    args = argument_parser().parse_args()
    if args.phase is None:
        builder = general_phase_product_span
        target_spec = {
            "type": "phase_product",
            "family": True,
            "symbol": "w",
            "normalization": "unnormalised",
            "amplitude": "w**weight",
        }
    else:
        w = np.exp(1j * args.phase)
        builder = lambda ns: phase_product_orbits(ns, w)
        target_spec = {
            "type": "phase_product",
            "family": False,
            "phase": args.phase,
            "w": f"exp(I * ({args.phase!r}))",
            "normalization": "unnormalised",
            "amplitude": "w**weight",
        }
    run_target_search(args, builder, target_spec)


if __name__ == "__main__":
    main()
