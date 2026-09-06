#!/usr/bin/env python3
"""Search for symmetric stabiliser decompositions of even-parity T cat states."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from stab_core import orbit_weights
from stab_search import add_search_arguments, run_target_search


def T_cat_state(n: int) -> np.ndarray:
    """Return the unnormalised even-parity projection of ``|T>^n``.

    Equivalently this is proportional to

        |T>^n + Z^tensor(n) |T>^n.

    Computational-basis amplitudes of odd Hamming weight vanish.
    """
    omega = (1.0 + 1.0j) / np.sqrt(2.0)
    return np.asarray([
        omega ** index.bit_count() if index.bit_count() % 2 == 0 else 0.0
        for index in range(1 << n)
    ])


def T_cat_state_orbits(ns: Sequence[int]) -> np.ndarray:
    """Return one unnormalised even-T-cat amplitude per block-weight orbit."""
    omega = (1.0 + 1.0j) / np.sqrt(2.0)
    weights = orbit_weights(ns).sum(axis=1)
    return np.where((weights & 1) == 0, omega ** weights, 0.0 + 0.0j)


def argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for even T-cat searches."""
    parser = argparse.ArgumentParser(description=__doc__)
    return add_search_arguments(parser, partition_example="5+5")


def main() -> None:
    """Parse command-line arguments and run the requested search."""
    run_target_search(
        argument_parser().parse_args(),
        T_cat_state_orbits,
        {
            "type": "T_cat",
            "parity": 0,
            "omega": "(1 + I) / sqrt(2)",
            "normalization": "unnormalised",
            "amplitude": "omega**weight if weight % 2 == 0 else 0",
        },
    )


if __name__ == "__main__":
    main()
