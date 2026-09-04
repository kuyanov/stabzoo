#!/usr/bin/env python3
"""Search for partition-symmetric stabiliser decompositions of Dicke states D^n_k.

Examples
--------
python search_Dicke.py 10 11 --k 4
python search_Dicke.py 10 11 --k 4 --partition 5+5
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from stab_core import orbit_weights
from stab_search import add_search_arguments, run_target_search


def dicke_orbits(ns: Sequence[int], k: int) -> np.ndarray:
    """One unnormalised D^n_k amplitude per block-weight orbit."""
    if not 0 <= k <= sum(ns):
        raise ValueError(f"k must lie between 0 and n={sum(ns)}")
    weights = orbit_weights(ns).sum(axis=1)
    return (weights == k).astype(np.complex128)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, required=True, help="Dicke excitation number")
    return add_search_arguments(parser, partition_example="5+5")


def main() -> None:
    args = argument_parser().parse_args()
    if not 0 <= args.k <= args.n:
        raise SystemExit(f"--k must lie between 0 and n={args.n}")
    run_target_search(args, lambda ns: dicke_orbits(ns, args.k))


if __name__ == "__main__":
    main()
