#!/usr/bin/env python3
"""Search symmetric stabiliser decompositions of general product phase states."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from stab_core import orbit_weights
from stab_search import add_search_arguments, run_target_search


def phase_product_state(n: int, w: complex) -> np.ndarray:
    return np.asarray([w ** index.bit_count() for index in range(1 << n)])


def phase_product_orbits(ns: Sequence[int], w: complex) -> np.ndarray:
    return w ** orbit_weights(ns).sum(axis=1)


def general_phase_product_span(ns: Sequence[int]) -> np.ndarray:
    """Orbit-basis columns spanning (|0>+w|1>)^n for every complex w."""
    weights = orbit_weights(ns).sum(axis=1)
    n = sum(ns)
    return np.equal.outer(weights, np.arange(n + 1)).astype(np.complex128)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=float, default=None,
        help='search one w=exp(i*phase); omit to search the complete generic-w family')
    return add_search_arguments(parser, partition_example='3+3+5')


def main() -> None:
    args = argument_parser().parse_args()
    if args.phase is None:
        builder = general_phase_product_span
    else:
        w = np.exp(1j * args.phase)
        builder = lambda ns: phase_product_orbits(ns, w)
    run_target_search(args, builder)


if __name__=='__main__': 
    main()
