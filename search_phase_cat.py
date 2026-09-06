#!/usr/bin/env python3
"""Search symmetric stabiliser decompositions of general phase cat states."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from stab_core import orbit_weights
from stab_search import add_search_arguments, run_target_search


def _parity_bit(parity: str) -> int:
    """Translate an ``even``/``odd`` label to its parity bit."""
    if parity == "even":
        return 0
    if parity == "odd":
        return 1
    raise ValueError("parity must be 'even' or 'odd'")


def phase_cat_state(n: int, w: complex, parity: str = "even") -> np.ndarray:
    """Return the unnormalised computational-basis phase cat target."""
    bit = _parity_bit(parity)
    return np.asarray(
        [w ** index.bit_count() if (index.bit_count() & 1) == bit else 0.0
         for index in range(1 << n)],
        dtype=np.complex128,
    )


def phase_cat_orbits(
    ns: Sequence[int],
    w: complex,
    parity: str = "even",
) -> np.ndarray:
    """Return one unnormalised phase-cat amplitude per block-weight orbit."""
    weights = orbit_weights(ns).sum(axis=1)
    bit = _parity_bit(parity)
    return np.where((weights & 1) == bit, w ** weights, 0.0).astype(np.complex128)


def general_phase_cat_span(
    ns: Sequence[int],
    parity: str = "even",
) -> np.ndarray:
    """Orbit columns spanning the even/odd phase-cat family for every complex w."""
    weights = orbit_weights(ns).sum(axis=1)
    bit = _parity_bit(parity)
    layers = np.arange(bit, sum(ns) + 1, 2)
    return np.equal.outer(weights, layers).astype(np.complex128)


def argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for phase-cat searches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parity",
        choices=("even", "odd"),
        default="even",
        help="cat parity to search (default: even)",
    )
    parser.add_argument(
        "--phase",
        type=float,
        default=None,
        help="search one w=exp(i*phase); omit to search the complete generic-w family",
    )
    return add_search_arguments(parser, partition_example="3+3+5")


def main() -> None:
    """Parse command-line arguments and run the requested search."""
    args = argument_parser().parse_args()
    parity = args.parity
    if args.phase is None:
        builder = lambda ns: general_phase_cat_span(ns, parity)
    else:
        w = np.exp(1j * args.phase)
        builder = lambda ns: phase_cat_orbits(ns, w, parity)
    run_target_search(args, builder)


if __name__ == "__main__":
    main()
