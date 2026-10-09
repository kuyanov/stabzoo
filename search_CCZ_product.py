#!/usr/bin/env python3
"""Search exact decompositions of CCZ products (n qubits, n/3 copies).

The target is H_3(-1)^{tensor n/3}, with amplitudes +/-1. Keeping every
triple symmetric provably requires 2**(n/3) terms; see CCZ_product/README.md.
For smaller requested ranks, the default partition splits enough triples
into 1+2 to escape that obstruction. Searches still use stab_search's LNS.
Only exact certificates are written. --report also records unsuccessful runs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np

from search_H_product import _box_block_groups, h_box_product_orbits
from stab_core import count_symmetric_stabilisers, orbit_multiplicities, orbit_weights
from stab_export import save_decomposition_json
from stab_search import (
    _normalise_target,
    _prune_support,
    add_search_arguments,
    format_partition,
    omp,
    sparse_search,
)


def symmetric_rank_lower_bound(partition: tuple[int, ...]) -> int:
    """Matrix-flattening lower bound for this dictionary, not unrestricted rank."""
    groups = _box_block_groups(partition, 3)
    return 2 ** sum(len(group) == 1 for group in groups)


def default_partition(copies: int, rank: int) -> tuple[int, ...]:
    """Refine as few triples as necessary to remove the flattening obstruction."""
    if copies <= 0 or rank <= 0:
        raise ValueError("copies and rank must be positive")
    intact = copies
    while 2**intact > rank:
        intact -= 1
    return (3,) * intact + (1, 2) * (copies - intact)


def tensor_indices(partition: tuple[int, ...]) -> list[int]:
    """Locate the terms of (|+++>_unnormalised - 2|111>)^copies.

    Count preceding affine/quadratic batches without enumerating their atoms.
    The active blocks have independent core bits and mode 1; pinned blocks
    have mode 0 and core bit 1. All phase coefficients are zero.
    """
    groups = _box_block_groups(partition, 3)
    blocks = len(partition)

    def rref_count(pivots: tuple[int, ...]) -> int:
        dimension = len(pivots)
        return 1 << (
            blocks * dimension - sum(pivots) - dimension * (dimension + 1) // 2
        )

    starts = [0]
    for dimension in range(blocks):
        subspaces = sum(rref_count(p) for p in combinations(range(blocks), dimension))
        starts.append(
            starts[-1] + subspaces * 2 ** (blocks - dimension) * 3**blocks
            * 4**dimension * 2 ** (dimension * (dimension - 1) // 2)
        )

    result = []
    for pinned in product((False, True), repeat=len(groups)):
        modes = tuple(
            0 if pin else 1 for pin, group in zip(pinned, groups) for _ in group
        )
        pivots = tuple(i for i, mode in enumerate(modes) if mode)
        dimension = len(pivots)
        preceding = sum(
            rref_count(p) for p in combinations(range(blocks), dimension) if p < pivots
        )
        cosets = 2 ** (blocks - dimension)
        mode_code = sum(value * 3 ** (blocks - 1 - i) for i, value in enumerate(modes))
        phase_count = 4**dimension * 2 ** (dimension * (dimension - 1) // 2)
        result.append(
            starts[dimension]
            + ((preceding * cosets + cosets - 1) * 3**blocks + mode_code) * phase_count
        )
    return result


def tensor_start(partition: tuple[int, ...], rank: int, seed: int) -> list[int]:
    """Prune a shuffled exact tensor decomposition to make an alternative start."""
    groups = _box_block_groups(partition, 3)
    weights = orbit_weights(partition)
    sizes = np.asarray(partition)
    columns = []
    for pinned in product((False, True), repeat=len(groups)):
        support = np.ones(len(weights), dtype=bool)
        for pin, group in zip(pinned, groups):
            if pin:
                support &= np.all(weights[:, group] == sizes[list(group)], axis=1)
        columns.append(support)
    pool = np.column_stack(columns) * np.sqrt(orbit_multiplicities(partition))[:, None]
    pool /= np.linalg.norm(pool, axis=0)
    _, target, _ = _normalise_target(partition, h_box_product_orbits(partition, 3, -1))
    order = np.random.default_rng(seed).permutation(pool.shape[1])
    selected = _prune_support(pool, target, order, rank)
    indices = tensor_indices(partition)
    return [indices[i] for i in selected]


def _trial(task: tuple[argparse.Namespace, int]) -> dict:
    args, trial = task
    seed = args.seed + trial
    target = h_box_product_orbits(args.partition, 3, -1)
    started = time.perf_counter()
    method = "omp" if args.omp_only or trial % 2 else "tensor"
    if args.omp_only:
        indices, residual = omp(
            args.partition, target, args.rank, seed=seed, oversample=args.oversample
        )
    else:
        initial = tensor_start(args.partition, args.rank, seed) if method == "tensor" else None
        indices, residual = sparse_search(
            args.partition, target, args.rank, initial_indices=initial, seed=seed,
            iterations=args.iterations, epoch_length=args.epoch_length,
            destroy_sizes=args.destroy_sizes, repair_shortlist=args.repair_shortlist,
            start_temperature=args.start_temperature, end_temperature=args.end_temperature,
            working_pool_limit=args.working_pool_limit, exact_tol=args.tolerance,
            omp_oversample=args.oversample, verbose=args.verbose and args.workers == 1,
        )
    return dict(trial=trial, seed=seed, initialization=method, indices=indices,
                residual=residual, elapsed=time.perf_counter() - started)


def argument_parser() -> argparse.ArgumentParser:
    parser = add_search_arguments(
        argparse.ArgumentParser(description=__doc__, conflict_handler="resolve"),
        partition_example="3+3+1+2",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel independent trials for the selected partition")
    parser.add_argument("--max-parts", type=int, default=4,
                        help="maximum allowed partition blocks (default: 4)")
    parser.set_defaults(trials=8, iterations=2000, max_parts=4, oversample=2,
                        destroy_sizes=(1, 2, 3, 4), repair_shortlist=16)
    parser.add_argument("--start-temperature", type=float, default=2e-2)
    parser.add_argument("--end-temperature", type=float, default=2e-4)
    parser.add_argument("--working-pool-limit", type=int, default=50_000)
    parser.add_argument("--report", help="write settings and trial results, including failures")
    return parser


def main() -> int:
    parser = argument_parser()
    args = parser.parse_args()
    if args.n <= 0 or args.n % 3:
        parser.error("n is the number of qubits and must be a positive multiple of 3")
    if args.rank <= 0 or args.workers <= 0 or args.trials <= 0 or args.max_parts <= 0:
        parser.error("rank, --workers, --trials and --max-parts must be positive")
    if (args.iterations < 0 or args.oversample < 0 or args.epoch_length <= 0
            or args.repair_shortlist <= 0 or args.working_pool_limit <= 0
            or not args.destroy_sizes or min(args.destroy_sizes) <= 0):
        parser.error("invalid search budget or repair settings")
    for name in ("tolerance", "start_temperature", "end_temperature"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.report and Path(args.report).resolve() == Path(args.output).resolve():
        parser.error("--report and --output must be different paths")

    copies = args.n // 3
    baseline = 2**copies
    if args.partition is None:
        args.partition = default_partition(copies, args.rank)
    try:
        _box_block_groups(args.partition, 3)
    except ValueError as error:
        parser.error(str(error))
    if sum(args.partition) != args.n:
        parser.error("--partition must sum to n")
    lower_bound = symmetric_rank_lower_bound(args.partition)
    if args.rank < lower_bound:
        parser.error(
            f"this partition requires at least {lower_bound} terms by matrix flattening; "
            "refine the triples with --partition, or omit it for automatic refinement"
        )
    if len(args.partition) > args.max_parts:
        parser.error(
            f"partition {format_partition(args.partition)} has {len(args.partition)} blocks; "
            "the dictionary grows very quickly. Increase --max-parts explicitly to proceed"
        )

    count = count_symmetric_stabilisers(args.partition)
    print(f"CCZ copies={copies}, qubits={args.n}, tensor upper bound={baseline}, "
          f"requested rank={args.rank}")
    print(f"partition={format_partition(args.partition)}, emitted atoms={count}, "
          f"dictionary lower bound={lower_bound}", flush=True)
    results = []
    best = None
    if args.rank >= baseline:
        best = dict(indices=tensor_indices(args.partition), residual=0.0)
        print("Using the exact tensor construction.")
    else:
        tasks = [(args, trial) for trial in range(1 if args.omp_only else args.trials)]
        pool = None
        try:
            if args.workers == 1:
                outcomes = map(_trial, tasks)
            else:
                pool = mp.get_context("spawn").Pool(min(args.workers, len(tasks)))
                outcomes = pool.imap_unordered(_trial, tasks)
            for result in outcomes:
                results.append(result)
                print(f"trial={result['trial']} start={result['initialization']} "
                      f"residual={result['residual']:.9g} time={result['elapsed']:.3f}s",
                      flush=True)
                if best is None or result["residual"] < best["residual"]:
                    best = result
                if best["residual"] <= args.tolerance:
                    break
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()

    success = best is not None and best["residual"] <= args.tolerance
    if success:
        target_spec = dict(type="H_product", m=3, copies=copies, family=False, a="-1",
                           normalization="unnormalised",
                           box_order="copies are contiguous m-qubit blocks",
                           amplitude="a**full_boxes")
        # Re-solve over Q(I) before allowing a floating-point hit to become a certificate.
        from stab_export import _exact_coefficients, _qlab_orbit_codes, qlab_terms_from_indices

        terms = qlab_terms_from_indices(args.partition, best["indices"])
        codes = _qlab_orbit_codes(args.partition, terms)
        try:
            _exact_coefficients(args.partition, codes, target_spec)
        except ValueError:
            success = False
            print("Numerical candidate failed exact coefficient reconstruction.")
        if success:
            save_decomposition_json(
                args.output, partition=args.partition, indices=best["indices"],
                target=h_box_product_orbits(args.partition, 3, -1),
                target_spec=target_spec, residual=best["residual"],
            )
            print(f"Exact upper bound: {len(best['indices'])}; wrote {args.output}")
    if not success:
        print(f"No exact decomposition found at rank {args.rank}; "
              f"the certified tensor upper bound remains {baseline}. No certificate written.")
    if args.report:
        report = dict(settings=vars(args), copies=copies, tensor_upper_bound=baseline,
                      dictionary_lower_bound=lower_bound, emitted_atoms=count,
                      success=success, best=best, trials=sorted(results, key=lambda r: r["trial"]))
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
