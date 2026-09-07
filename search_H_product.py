#!/usr/bin/env python3
"""Search decompositions of products of identical parametrised H-box states.

An ``m``-qubit H-box has computational-basis amplitudes ``[1, ..., 1, a]``:
only the all-one string has amplitude ``a``.  Copies are contiguous in the
qubit order.  Omitting ``--a`` searches one decomposition valid symbolically
for every ``a``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from stab_core import orbit_weights
from stab_search import add_search_arguments, run_target_search


def _complex_parameter(text: str) -> complex:
    """Parse a real or complex command-line value, accepting ``i`` or ``j``."""
    try:
        return complex(text.strip().lower().replace("i", "j"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid complex value: {text!r}") from error


def _box_block_groups(ns: Sequence[int], m: int) -> tuple[tuple[int, ...], ...]:
    """Group partition blocks belonging to each contiguous ``m``-qubit box."""
    partition = tuple(map(int, ns))
    if m <= 0 or any(size <= 0 for size in partition) or sum(partition) % m:
        raise ValueError("partition sizes and m must be positive, and m must divide n")

    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    position = 0
    next_boundary = m
    for block, size in enumerate(partition):
        if position + size > next_boundary:
            raise ValueError(
                f"partition {'+'.join(map(str, partition))} crosses an H-box boundary"
            )
        current.append(block)
        position += size
        if position == next_boundary:
            groups.append(tuple(current))
            current = []
            next_boundary += m
    if current:
        raise ValueError("partition does not refine the contiguous H-boxes")
    return tuple(groups)


def h_box_full_counts(ns: Sequence[int], m: int) -> np.ndarray:
    """Count all-one H-boxes for every block-weight orbit of ``ns``."""
    partition = tuple(map(int, ns))
    groups = _box_block_groups(partition, m)
    weights = orbit_weights(partition)
    sizes = np.asarray(partition)
    counts = np.zeros(len(weights), dtype=np.int16)
    for group in groups:
        columns = np.asarray(group, dtype=np.intp)
        counts += np.all(weights[:, columns] == sizes[columns], axis=1)
    return counts


def h_box_product_orbits(ns: Sequence[int], m: int, a: complex) -> np.ndarray:
    """Return orbit amplitudes of a product of fixed-parameter H-boxes."""
    return np.asarray(a ** h_box_full_counts(ns, m), dtype=np.complex128)


def h_box_product_state(m: int, copies: int, a: complex) -> np.ndarray:
    """Return the full computational-basis vector of an H-box product."""
    if m <= 0 or copies <= 0:
        raise ValueError("m and copies must be positive")
    n = m * copies
    indices = np.arange(1 << n, dtype=np.uint64)
    mask = (1 << m) - 1
    full_boxes = np.zeros(len(indices), dtype=np.int16)
    for copy in range(copies):
        full_boxes += ((indices >> (copy * m)) & mask) == mask
    return np.asarray(a**full_boxes, dtype=np.complex128)


def general_h_box_product_span(ns: Sequence[int], m: int) -> np.ndarray:
    """Return orbit columns spanning the H-box product for every parameter ``a``."""
    counts = h_box_full_counts(ns, m)
    copies = sum(ns) // m
    return np.equal.outer(counts, np.arange(copies + 1)).astype(np.complex128)


def _complex_expression(value: complex) -> str:
    """Format a complex number as a SymPy-compatible expression."""
    return f"({value.real:.17g}) + ({value.imag:.17g})*I"


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
    # Fixed H-box targets have cancellation barriers that purely greedy repair
    # does not cross. Small destruction moves and a modest stochastic shortlist
    # recover the rank-5 H_2(0)^3 decomposition without materially changing the
    # cost of a repair step.
    parser.set_defaults(
        destroy_sizes=(2, 3, 4, 6, 8, 12),
        repair_shortlist=8,
    )
    return parser


def main() -> None:
    """Parse command-line arguments and run the requested H-box search."""
    args = argument_parser().parse_args()
    if args.m <= 0 or args.n % args.m:
        raise SystemExit("--m must be positive and divide n")
    copies = args.n // args.m

    # The target is invariant under permutations inside each H-box.  More
    # refined explicitly supplied partitions are allowed, but no block may
    # cross a box boundary.
    if args.partition is None:
        args.partition = (args.m,) * copies
    try:
        _box_block_groups(args.partition, args.m)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    common_target = {
        "type": "H_product",
        "m": args.m,
        "copies": copies,
        "normalization": "unnormalised",
        "box_order": "copies are contiguous m-qubit blocks",
        "amplitude": "a**full_boxes",
    }
    if args.a is None:
        builder = lambda ns: general_h_box_product_span(ns, args.m)
        target_spec = {
            **common_target,
            "family": True,
            "symbol": "a",
        }
    else:
        builder = lambda ns: h_box_product_orbits(ns, args.m, args.a)
        target_spec = {
            **common_target,
            "family": False,
            "a": _complex_expression(args.a),
        }
    run_target_search(args, builder, target_spec)


if __name__ == "__main__":
    main()
