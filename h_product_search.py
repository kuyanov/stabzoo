"""H-box states, product dictionaries, and search with the existing OMP/LNS.

Fixed parameters use a state target; symbolic parameters use the entire
coefficient subspace. Reduced pools are subsets of the original dictionary.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

import numpy as np

from stab_core import count_symmetric_stabilisers, orbit_multiplicities, orbit_weights
from stab_export import (
    _exact_coefficients, _qlab_orbit_codes, qlab_terms_from_indices,
    save_decomposition_json,
)
from stab_search import (
    _accurate_residual, _complete_support, _large_neighbourhood_search,
    _normalise_target, _prune_support, omp, sparse_search,
)


class HProductSearchError(ValueError):
    """An invalid H-product search configuration."""


def box_block_groups(ns: Sequence[int], m: int) -> tuple[tuple[int, ...], ...]:
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
    groups = box_block_groups(partition, m)
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
    """Format a fixed parameter as an exact Gaussian-rational expression."""
    real = Fraction(str(value.real))
    imaginary = Fraction(str(value.imag))
    return f"({real}) + ({imaginary})*I"


def prepare_h_search(args) -> dict:
    """Validate options, select a dictionary, and describe the exact target."""
    if args.n <= 0 or args.m <= 0 or args.n % args.m:
        raise HProductSearchError("n and --m must be positive, and --m must divide n")
    if args.a is not None and not np.isfinite(args.a):
        raise HProductSearchError("--a must be finite")
    if (args.rank <= 0 or args.workers <= 0 or args.trials <= 0 or args.max_parts <= 0
            or args.iterations < 0 or args.epoch_length <= 0 or args.oversample < 0
            or args.repair_shortlist <= 0 or args.working_pool_limit <= 0
            or not args.destroy_sizes or min(args.destroy_sizes) <= 0):
        raise HProductSearchError("invalid rank, search budget, or repair settings")
    for name in ("tolerance", "start_temperature", "end_temperature", "pool_memory_mib"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise HProductSearchError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.report and Path(args.report).resolve() == Path(args.output).resolve():
        raise HProductSearchError("--report and --output must be different paths")
    if args.reduced_pool:
        if args.pool == "full":
            raise HProductSearchError("--reduced-pool and --pool full select different dictionaries")
        args.pool = "reduced"
    copies = args.n // args.m

    # The target is invariant under permutations inside each H-box.  More
    # refined explicitly supplied partitions are allowed, but no block may
    # cross a box boundary.
    if args.partition is None:
        args.partition = (args.m,) * copies
    try:
        box_block_groups(args.partition, args.m)
    except ValueError as error:
        raise HProductSearchError(str(error))
    if sum(args.partition) != args.n:
        raise HProductSearchError("--partition must sum to n")
    intact = args.partition == (args.m,) * copies
    if args.pool == "reduced" and not intact:
        raise HProductSearchError("--pool reduced requires one intact partition block per H-box")
    args.backend = args.pool
    if args.backend == "auto":
        args.backend = "reduced" if intact else "full"
        if args.m == 1 and copies <= 4:
            args.backend = "full"

    dimension = int(np.prod([size + 1 for size in args.partition], dtype=object))
    target_columns = copies + 1 if args.a is None else 1
    if 32 * dimension * target_columns > args.pool_memory_mib * 1024**2:
        raise HProductSearchError("estimated target memory exceeds --pool-memory-mib")
    if stabilizer_indices(args, args.a) is None:
        if args.backend == "reduced":
            atoms = (2 if args.m == 1 else 3)**copies
            estimated = 4 * dimension * atoms * 8
        else:
            if len(args.partition) > args.max_parts:
                raise HProductSearchError("full dictionary exceeds --max-parts; increase it explicitly")
            count = count_symmetric_stabilisers(args.partition)
            estimated = 16 * count + 16 * dimension * min(count, args.working_pool_limit)
        if estimated > args.pool_memory_mib * 1024**2:
            raise HProductSearchError("estimated pool memory exceeds --pool-memory-mib; increase it explicitly")

    common_target = {
        "type": "H_product",
        "m": args.m,
        "copies": copies,
        "normalization": "unnormalised",
        "box_order": "copies are contiguous m-qubit blocks",
        "amplitude": "a**full_boxes",
    }
    if args.a is None:
        target_spec = {
            **common_target,
            "family": True,
            "symbol": "a",
        }
    else:
        target_spec = {
            **common_target,
            "family": False,
            "a": _complex_expression(args.a),
        }
    return target_spec


def _product_indices(blocks: int, modes: np.ndarray) -> np.ndarray:
    """Locate independent zero-phase product atoms in the public emitted pool."""
    def rref_count(pivots):
        dimension = len(pivots)
        return 1 << (blocks * dimension - sum(pivots)
                     - dimension * (dimension + 1) // 2)

    starts = [0]
    for dimension in range(blocks):
        subspaces = sum(rref_count(p) for p in combinations(range(blocks), dimension))
        starts.append(starts[-1] + subspaces * 2**(blocks - dimension)
                      * 3**blocks * 4**dimension
                      * 2**(dimension * (dimension - 1) // 2))
    result = []
    for mode in modes:
        pivots = tuple(i for i, value in enumerate(mode) if value)
        dimension = len(pivots)
        preceding = sum(rref_count(p) for p in combinations(range(blocks), dimension)
                        if p < pivots)
        cosets = 2**(blocks - dimension)
        mode_code = sum(int(value) * 3**(blocks - 1 - i)
                        for i, value in enumerate(mode))
        phase_count = 4**dimension * 2**(dimension * (dimension - 1) // 2)
        result.append(starts[dimension]
                      + ((preceding * cosets + cosets - 1) * 3**blocks
                         + mode_code) * phase_count)
    dtype = object if result and max(result) > np.iinfo(np.int64).max else np.int64
    return np.asarray(result, dtype=dtype)


@lru_cache(maxsize=2)
def reduced_pool(copies: int, m: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return product atoms weighted in the intact H-box orbit basis.

    Modes are pin-all-ones, plus, and the symmetric complete-graph state.
    The last state has phase (-1)**binom(weight, 2), and is distinct from
    H_m(-1) when m > 2. At m=1 it duplicates plus and is omitted.
    """
    if copies <= 0 or m <= 0:
        raise ValueError("copies and m must be positive")
    partition = (m,) * copies
    weights = orbit_weights(partition)
    roots = np.sqrt(orbit_multiplicities(partition))
    modes = np.asarray(list(product(range(2 if m == 1 else 3), repeat=copies)),
                       dtype=np.int8)
    columns = []
    for mode in modes:
        column = np.ones(len(weights))
        for block, value in enumerate(mode):
            if value == 0:
                column *= weights[:, block] == m
            elif value == 2:
                quadratic = (weights[:, block] * (weights[:, block] - 1) // 2) & 1
                column *= 1 - 2 * quadratic
        columns.append(column)
    pool = np.asarray(columns).T * roots[:, None]
    pool = np.asfortranarray(pool / np.linalg.norm(pool, axis=0), dtype=np.complex64)
    emitted = _product_indices(copies, modes)
    pool.flags.writeable = False
    emitted.flags.writeable = False
    return pool, emitted


def reduced_start(copies: int, rng: np.random.Generator, m: int = 2) -> np.ndarray:
    """Use paired rank-three factors only for m=2; otherwise use plus/pin."""
    permutation = rng.permutation(copies)
    initial = []
    radix = 2 if m == 1 else 3
    if m == 2:
        choices = ((1, 1), (2, 2), (0, 0))
        for chosen in product(range(3), repeat=copies // 2):
            for tail in ((0, 1) if copies % 2 else (1,)):
                mode = np.ones(copies, dtype=np.int8)
                for pair, choice in enumerate(chosen):
                    mode[permutation[2 * pair:2 * pair + 2]] = choices[choice]
                if copies % 2:
                    mode[permutation[-1]] = tail
                initial.append(sum(int(value) * radix**(copies - 1 - block)
                                   for block, value in enumerate(mode)))
    else:
        for chosen in product((1, 0), repeat=copies):
            mode = np.zeros(copies, dtype=np.int8)
            mode[permutation] = chosen
            initial.append(sum(int(value) * radix**(copies - 1 - block)
                               for block, value in enumerate(mode)))
    return np.asarray(initial, dtype=np.intp)


def target_orbits(args, parameter):
    if parameter is None:
        return general_h_box_product_span(args.partition, args.m)
    return h_box_product_orbits(args.partition, args.m, parameter)


def stabilizer_indices(args, parameter):
    """Recognize single dictionary atoms already used by the tensor starts."""
    if parameter is None:
        return None
    blocks = len(args.partition)
    if parameter == 1:
        return _product_indices(blocks, np.ones((1, blocks), dtype=np.int8)).tolist()
    if args.m == 1 and parameter == 0:
        return [0]
    if args.m == 1 and parameter in (1j, -1, -1j):
        linear = {1j: 1, -1: 2, -1j: 3}[parameter]
        index = int(_product_indices(blocks, np.ones((1, blocks), dtype=np.int8))[0])
        index += sum(linear * 4**i for i in range(blocks)) * 2**(blocks * (blocks - 1) // 2)
        return [index]
    if (args.m == 2 and parameter == -1
            and args.partition == (2,) * (args.n // 2)):
        return _product_indices(blocks, np.full((1, blocks), 2, dtype=np.int8)).tolist()
    return None


def reduced_trial(args, trial: int) -> dict:
    """Apply unchanged pruning and conditional-OMP/LNS repair to the subset."""
    started = time.perf_counter()
    m = args.m
    copies = args.n // m
    partition = (m,) * copies
    pool, emitted = reduced_pool(copies, m)
    multiplicities = orbit_multiplicities(partition)
    target_values = target_orbits(args, args.a)
    _, target, norm = _normalise_target(partition, target_values)
    seed = args.seed + trial
    rng = np.random.default_rng(seed)
    if args.omp_only:
        selected, _ = _complete_support(
            pool, pool.conj().T, target.astype(np.complex64),
            np.empty(0, dtype=np.intp), args.rank + args.oversample, rng, 1,
        )
        initialization = "reduced_omp"
    else:
        selected = reduced_start(copies, rng, m)
        initialization = "reduced_tensor"
    selected = _prune_support(pool, target, selected, args.rank)
    residual = _accurate_residual(pool, multiplicities, target, selected)
    if not args.omp_only and residual > args.tolerance / norm:
        selected, residual = _large_neighbourhood_search(
            pool, target, selected, rng, iterations=args.iterations,
            epoch_length=args.epoch_length, destroy_sizes=args.destroy_sizes,
            repair_shortlist=args.repair_shortlist,
            start_temperature=args.start_temperature,
            end_temperature=args.end_temperature,
            relative_tolerance=args.tolerance / norm,
            multiplicities=multiplicities,
            verbose=args.verbose and args.workers == 1,
        )
    return dict(trial=trial, seed=seed, initialization=initialization,
                indices=emitted[selected].tolist(), residual=float(residual * norm),
                elapsed=time.perf_counter() - started, unique_atoms=pool.shape[1])


def _full_tensor_start(args, target, seed):
    groups = box_block_groups(args.partition, args.m)
    weights = orbit_weights(args.partition)
    sizes = np.asarray(args.partition)
    columns, modes = [], []
    for pinned in product((False, True), repeat=len(groups)):
        support = np.ones(len(weights), dtype=bool)
        mode = np.ones(len(args.partition), dtype=np.int8)
        for pin, group in zip(pinned, groups):
            if pin:
                support &= np.all(weights[:, group] == sizes[list(group)], axis=1)
                mode[list(group)] = 0
        columns.append(support)
        modes.append(mode)
    pool = np.column_stack(columns) * np.sqrt(orbit_multiplicities(args.partition))[:, None]
    pool /= np.linalg.norm(pool, axis=0)
    _, unit, _ = _normalise_target(args.partition, target)
    order = np.random.default_rng(seed).permutation(pool.shape[1])
    chosen = _prune_support(pool, unit, order, args.rank)
    emitted = _product_indices(len(args.partition), np.asarray(modes))
    return emitted[chosen].tolist()


def _trial(task) -> dict:
    args, trial = task
    direct = stabilizer_indices(args, args.a)
    if direct is not None:
        return dict(trial=trial, seed=args.seed + trial, initialization="single_atom",
                    indices=direct, residual=0.0, elapsed=0.0)
    if args.backend == "reduced":
        return reduced_trial(args, trial)
    started = time.perf_counter()
    seed = args.seed + trial
    target = target_orbits(args, args.a)
    if args.omp_only:
        indices, residual = omp(args.partition, target, args.rank, seed=seed,
                                oversample=args.oversample)
        initialization = "omp"
    else:
        initialization = "tensor" if trial % 2 == 0 else "omp"
        initial = _full_tensor_start(args, target, seed) if trial % 2 == 0 else None
        if initial is not None and args.rank >= 2**(args.n // args.m):
            # This support is already an exact elementary tensor construction.
            return dict(trial=trial, seed=seed, initialization=initialization,
                        indices=initial, residual=0.0,
                        elapsed=time.perf_counter() - started)
        indices, residual = sparse_search(
            args.partition, target, args.rank, initial_indices=initial, seed=seed,
            iterations=args.iterations, epoch_length=args.epoch_length,
            destroy_sizes=args.destroy_sizes, repair_shortlist=args.repair_shortlist,
            start_temperature=args.start_temperature, end_temperature=args.end_temperature,
            working_pool_limit=args.working_pool_limit, exact_tol=args.tolerance,
            omp_oversample=args.oversample, verbose=args.verbose and args.workers == 1,
        )
    return dict(trial=trial, seed=seed, initialization=initialization,
                indices=indices, residual=residual, elapsed=time.perf_counter() - started)


def run_h_search(args) -> int:
    """Validate and run a fixed or symbolic H-box search with exact export."""
    target_spec = prepare_h_search(args)
    copies = args.n // args.m
    count = count_symmetric_stabilisers(args.partition)
    label = "a" if args.a is None else target_spec["a"]
    print(f"H_{args.m}({label}) copies={copies}, qubits={args.n}, requested rank={args.rank}")
    print(f"partition={'+'.join(map(str, args.partition))}, emitted atoms={count}, "
          f"pool={args.backend}", flush=True)
    tasks = [(args, trial) for trial in range(1 if args.omp_only else args.trials)]
    results, best, pool = [], None, None
    try:
        if args.workers == 1:
            outcomes = map(_trial, tasks)
        else:
            pool = mp.get_context("spawn").Pool(min(args.workers, len(tasks)))
            outcomes = pool.imap_unordered(_trial, tasks)
        for result in outcomes:
            results.append(result)
            print(f"trial={result['trial']} start={result['initialization']} "
                  f"residual={result['residual']:.9g} time={result['elapsed']:.3f}s", flush=True)
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
        terms = qlab_terms_from_indices(args.partition, best["indices"])
        try:
            if _exact_coefficients(args.partition, _qlab_orbit_codes(args.partition, terms),
                                   target_spec) is None:
                raise ValueError("the target does not permit exact reconstruction")
        except ValueError as error:
            success = False
            print(f"Numerical candidate failed exact reconstruction: {error}")
        if success:
            save_decomposition_json(
                args.output, partition=args.partition, indices=best["indices"],
                target=target_orbits(args, args.a), target_spec=target_spec,
                residual=best["residual"],
            )
            print(f"Exact upper bound: {len(best['indices'])}; wrote {args.output}")
    if not success:
        print(f"No exact decomposition found at rank {args.rank}. No certificate written.")
    if args.report:
        report = dict(settings={**vars(args), "a": target_spec.get("a")},
                      target=target_spec, emitted_atoms=count, success=success, best=best,
                      trials=sorted(results, key=lambda result: result["trial"]))
        if args.backend == "reduced":
            report["search_pool_atoms"] = (2 if args.m == 1 else 3)**copies
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if success else 1

