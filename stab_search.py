"""Sparse approximation over partition-symmetric stabiliser dictionaries.

Searches use the weighted block-orbit basis, whose dimension is
``prod(n_i + 1)``. No computational-basis vector of length ``2**sum(n_i)`` is
constructed.

The search starts from OMP and then uses one large-neighbourhood loop: remove
several atoms and rebuild the missing subspace by conditional OMP.  If a modest
dictionary is still not exact, a deterministic two-atom polish handles narrow
local barriers.  The same pipeline is used at every target rank.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, cast

import numpy as np

from stab_core import (
    _PHASES,
    core_conjugate_phase_table,
    count_symmetric_stabilisers,
    gen_partitions,
    gen_symmetric_stabiliser_batch_data,
    gen_symmetric_stabilisers_batched,
    orbit_multiplicities,
    target_in_orbit_basis,
)

_INDEPENDENCE_TOL = 2e-6  # Search geometry is complex64.
_PINV_RCOND = 1e-8
_SCORE_TOL = 0.0
_PAIR_SHORTLIST = 64


class Progress(Protocol):
    """Minimal interface implemented by progress-bar objects."""

    def update(self, n: int = 1) -> object:
        """Advance the progress count by ``n``."""
        ...

    def close(self) -> object:
        """Release resources associated with the progress display."""
        ...


class _NullProgress:
    """No-op progress bar used when progress reporting is disabled."""

    def update(self, n: int = 1) -> None:
        """Discard a progress increment."""
        del n

    def close(self) -> None:
        """Close the no-op progress bar."""
        return


def _progress(total: int, enabled: bool) -> Progress:
    """Create a tqdm progress bar, or a no-op replacement if unavailable."""
    if not enabled:
        return _NullProgress()
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
    except ImportError:
        return _NullProgress()
    return cast(Progress, tqdm(total=total))


def _normalise_target(
    ns: tuple[int, ...],
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return orbit data and a normalized weighted target.

    A one-dimensional target is treated as one state, as before.  A matrix
    target is treated as a *target subspace*: its columns are converted to
    weighted orbit coordinates and orthonormalized.  The search residual is
    then the Frobenius norm of the part of that subspace missed by the chosen
    stabilizer span.
    """
    target = np.asarray(target)
    multiplicities = orbit_multiplicities(ns)
    roots = np.sqrt(multiplicities)

    if target.ndim == 1:
        orbit_target = target_in_orbit_basis(ns, target)
        weighted = orbit_target * roots
        norm = float(np.linalg.norm(weighted))
        unit = weighted if norm == 0.0 else weighted / norm
        return orbit_target, unit.astype(np.complex128, copy=False), norm

    if target.ndim != 2:
        raise ValueError("target must be a vector or a matrix of target columns")
    orbit_target = np.column_stack(
        [
            target_in_orbit_basis(ns, target[:, column])
            for column in range(target.shape[1])
        ]
    )
    weighted = orbit_target * roots[:, None]
    if weighted.shape[1] == 0:
        return orbit_target, weighted.astype(np.complex128), 0.0
    u, singular, _ = np.linalg.svd(weighted, full_matrices=False)
    if not len(singular):
        return orbit_target, weighted[:, :0].astype(np.complex128), 0.0
    threshold = 1e-11 * max(1.0, float(singular[0]))
    dimension = int(np.count_nonzero(singular > threshold))
    basis = u[:, :dimension].astype(np.complex128, copy=False)
    return orbit_target, basis, (1.0 if dimension else 0.0)


def _target_dimension(target: np.ndarray) -> int:
    """Return one for a state target, or the number of subspace columns."""
    return 1 if target.ndim == 1 else target.shape[1]


def _overlap_gains(overlap: np.ndarray) -> np.ndarray:
    """Return squared correlation norms for all candidate atoms."""
    if overlap.ndim == 1:
        return np.abs(overlap) ** 2
    return np.sum(np.abs(overlap) ** 2, axis=1)


@lru_cache(maxsize=1)
def normalised_orbit_pool(ns: tuple[int, ...]) -> np.ndarray:
    """Materialise all normalized emitted atoms in weighted orbit coordinates.

    Columns follow the public emitted-index order and may contain duplicate
    rays when a block has size one or two.  The cached result is read-only.
    """
    multiplicities = orbit_multiplicities(ns)
    roots = np.sqrt(multiplicities).astype(np.float32)
    count = count_symmetric_stabilisers(ns)
    pool = np.empty((len(multiplicities), count), dtype=np.complex64)

    left = 0
    for encoded in gen_symmetric_stabilisers_batched(ns):
        support_sizes = (encoded != 0) @ multiplicities
        piece = (
            _PHASES[encoded]
            * roots[None, :]
            / np.sqrt(support_sizes)[:, None]
        ).astype(np.complex64, copy=False)
        right = left + len(piece)
        pool[:, left:right] = piece.T
        left = right

    if left != count:
        raise RuntimeError(f"generator emitted {left} atoms; expected {count}")
    pool.flags.writeable = False
    return pool


@dataclass(frozen=True)
class SearchPool:
    """Unique rays, plus maps preserving public emitted-pool indices."""

    vectors: np.ndarray
    representatives: np.ndarray
    full_to_unique: np.ndarray


@dataclass(frozen=True)
class _WorkingPool:
    """Target-screened atoms used when the complete pool is too large."""

    vectors: np.ndarray
    emitted_indices: np.ndarray


@lru_cache(maxsize=1)
def normalised_search_pool(ns: tuple[int, ...]) -> SearchPool:
    """Return a deduplicated, deterministic dictionary for ``ns``.

    The accompanying index maps translate between emitted indices and unique
    dictionary columns, preserving the public indexing convention.
    """
    full = normalised_orbit_pool(ns)
    rows, representatives, inverse = np.unique(
        full.T,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    vectors = np.asfortranarray(rows.T, dtype=np.complex64)
    representatives = representatives.astype(np.intp, copy=False)
    inverse = inverse.astype(np.intp, copy=False)
    vectors.flags.writeable = False
    representatives.flags.writeable = False
    inverse.flags.writeable = False
    return SearchPool(vectors, representatives, inverse)


@lru_cache(maxsize=8)
def _batch_count(ns: tuple[int, ...]) -> int:
    """Return the number of structural batches emitted for ``ns``."""
    return sum(1 for _ in gen_symmetric_stabiliser_batch_data(ns))


def _screened_pool(
    ns: tuple[int, ...],
    target: np.ndarray,
    required_indices: Sequence[int],
    limit: int,
) -> _WorkingPool:
    """Build a target-screened dictionary of approximately ``limit`` atoms.

    Every structural batch contributes its strongest phase choices, and all
    ``required_indices`` are retained so that an OMP start remains available.
    """
    batches = _batch_count(ns)
    quota = max(1, limit // batches)
    required = np.unique(np.asarray(required_indices, dtype=np.intp))
    capacity = min(
        count_symmetric_stabilisers(ns),
        quota * batches + len(required),
    )
    multiplicities = orbit_multiplicities(ns)
    roots = np.sqrt(multiplicities).astype(np.float32)
    vectors = np.empty((len(multiplicities), capacity), dtype=np.complex64)
    emitted = np.empty(capacity, dtype=np.intp)

    left = 0
    output = 0
    search_target = target.astype(np.complex64)
    for encoded in gen_symmetric_stabilisers_batched(ns):
        support_sizes = (encoded != 0) @ multiplicities
        piece = (
            _PHASES[encoded]
            * roots[None, :]
            / np.sqrt(support_sizes)[:, None]
        ).astype(np.complex64, copy=False)
        count = len(piece)
        keep_count = min(quota, count)
        correlations = piece @ search_target.conj()
        if correlations.ndim == 2:
            correlations = np.linalg.norm(correlations, axis=1)
        else:
            correlations = np.abs(correlations)
        local = np.argpartition(correlations, -keep_count)[-keep_count:]

        lo = np.searchsorted(required, left)
        hi = np.searchsorted(required, left + count)
        if hi > lo:
            local = np.unique(
                np.concatenate((local, required[lo:hi] - left))
            )
        right = output + len(local)
        vectors[:, output:right] = piece[local].T
        emitted[output:right] = left + local
        output = right
        left += count

    vectors = np.asfortranarray(vectors[:, :output])
    emitted = emitted[:output].copy()
    vectors.flags.writeable = False
    emitted.flags.writeable = False
    return _WorkingPool(vectors, emitted)


@dataclass(frozen=True)
class _Geometry:
    """Target and dictionary data residualized against a selected span."""

    basis: np.ndarray
    residual: np.ndarray
    squared_norm: np.ndarray
    overlap: np.ndarray
    score: float


def _geometry(
    pool: np.ndarray,
    pool_adjoint: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
) -> _Geometry:
    """Residualize the target and every dictionary atom against ``selected``.

    The returned score is the squared target projection captured by the
    selected span.  For a matrix target, scores are summed over its columns.
    """
    if len(selected):
        basis, triangular = np.linalg.qr(pool[:, selected], mode="reduced")
        basis = basis[:, np.abs(np.diag(triangular)) > 1e-6]
    else:
        basis = np.empty((pool.shape[0], 0), dtype=pool.dtype)

    residual = target - basis @ (basis.conj().T @ target)
    cross = pool_adjoint @ basis
    squared_norm = 1.0 - np.sum(np.abs(cross) ** 2, axis=1).real
    overlap = pool_adjoint @ residual
    score = _target_dimension(target) - float(np.vdot(residual, residual).real)
    return _Geometry(basis, residual, squared_norm, overlap, score)


def _complete_support(
    pool: np.ndarray,
    pool_adjoint: np.ndarray,
    target: np.ndarray,
    retained: np.ndarray,
    size: int,
    rng: np.random.Generator,
    shortlist: int,
) -> tuple[np.ndarray, float]:
    """Greedily extend ``retained`` to ``size`` independent atoms."""
    selected = list(map(int, retained))
    geometry = _geometry(pool, pool_adjoint, target, retained)
    basis = geometry.basis
    residual = geometry.residual
    squared_norm = geometry.squared_norm
    overlap = geometry.overlap
    score = geometry.score

    excluded = np.zeros(pool.shape[1], dtype=bool)
    excluded[retained] = True
    while len(selected) < size:
        values = np.full(pool.shape[1], -np.inf, dtype=np.float32)
        valid = (~excluded) & (squared_norm > _INDEPENDENCE_TOL)
        values[valid] = _overlap_gains(overlap[valid]) / squared_norm[valid]
        available = int(np.count_nonzero(valid))
        if available == 0:
            break

        width = min(max(1, shortlist), available)
        if width == 1:
            chosen = int(np.argmax(values))
        else:
            top = np.argpartition(values, -width)[-width:]
            top_values = values[top]
            scale = max(1e-8, 0.25 * float(np.max(top_values)))
            probabilities = np.exp((top_values - np.max(top_values)) / scale)
            probabilities /= probabilities.sum()
            chosen = int(rng.choice(top, p=probabilities))

        direction = pool[:, chosen] - basis @ (
            basis.conj().T @ pool[:, chosen]
        )
        direction_norm = np.sqrt(float(np.vdot(direction, direction).real))
        if direction_norm <= _INDEPENDENCE_TOL:
            excluded[chosen] = True
            continue
        direction /= direction_norm

        coefficient = direction.conj() @ residual
        projection = pool_adjoint @ direction
        if residual.ndim == 1:
            residual -= direction * coefficient
            overlap -= projection * coefficient
            gain = float(abs(coefficient) ** 2)
        else:
            residual -= direction[:, None] * coefficient[None, :]
            overlap -= projection[:, None] * coefficient[None, :]
            gain = float(np.sum(np.abs(coefficient) ** 2))
        squared_norm -= np.abs(projection) ** 2
        basis = np.column_stack((basis, direction))
        selected.append(chosen)
        excluded[chosen] = True
        score += gain

    return np.asarray(selected, dtype=np.intp), score


def _complete_pair(
    pool: np.ndarray,
    pool_adjoint: np.ndarray,
    target: np.ndarray,
    retained: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Add the best pair from a shortlist to a retained support.

    The gain is evaluated jointly, so this can cross local barriers where
    neither new atom is useful on its own.  At most ``_PAIR_SHORTLIST`` atoms
    are considered, keeping the quadratic pair calculation small.
    """
    geometry = _geometry(pool, pool_adjoint, target, retained)
    valid = geometry.squared_norm > _INDEPENDENCE_TOL
    valid[retained] = False
    available = int(np.count_nonzero(valid))
    if available < 2:
        return retained.copy(), geometry.score

    gains = np.full(pool.shape[1], -np.inf, dtype=np.float32)
    gains[valid] = (
        _overlap_gains(geometry.overlap[valid])
        / geometry.squared_norm[valid]
    )
    width = min(_PAIR_SHORTLIST, available)
    candidates = np.argpartition(gains, -width)[-width:]
    directions = pool[:, candidates] - geometry.basis @ (
        geometry.basis.conj().T @ pool[:, candidates]
    )
    gram = directions.conj().T @ directions
    diagonal = np.real(np.diag(gram))
    determinant = (
        diagonal[:, None] * diagonal[None, :] - np.abs(gram) ** 2
    )

    correlations = geometry.overlap[candidates]
    if correlations.ndim == 1:
        correlations = correlations[:, None]
    correlation_norm = np.sum(np.abs(correlations) ** 2, axis=1)
    correlation_cross = correlations.conj() @ correlations.T
    pair_gain = (
        diagonal[None, :] * correlation_norm[:, None]
        + diagonal[:, None] * correlation_norm[None, :]
        - 2.0 * np.real(gram * correlation_cross)
    )
    independent = np.triu(determinant > _INDEPENDENCE_TOL, k=1)
    pair_gain = np.divide(
        pair_gain,
        determinant,
        out=np.full_like(pair_gain, -np.inf),
        where=independent,
    )
    first, second = np.unravel_index(
        int(np.argmax(pair_gain)), pair_gain.shape
    )
    gain = float(pair_gain[first, second])
    if not np.isfinite(gain):
        return retained.copy(), geometry.score
    selected = np.append(retained, candidates[[first, second]])
    return selected.astype(np.intp, copy=False), geometry.score + gain


def _deletion_losses(
    pool: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Return the exact projection-score loss from deleting each selected atom."""
    atoms = pool[:, selected]
    gram = atoms.conj().T @ atoms
    inverse = np.linalg.pinv(gram, rcond=_PINV_RCOND, hermitian=True)
    coefficients = inverse @ (atoms.conj().T @ target)
    diagonal = np.maximum(np.real(np.diag(inverse)), 1e-12)
    if coefficients.ndim == 1:
        numerator = np.abs(coefficients) ** 2
    else:
        numerator = np.sum(np.abs(coefficients) ** 2, axis=1)
    return numerator / diagonal


def _prune_support(
    pool: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
    size: int,
) -> np.ndarray:
    """Delete minimum-loss atoms until ``selected`` has the requested size."""
    selected = selected.copy()
    while len(selected) > size:
        loss = _deletion_losses(pool, target, selected)
        selected = np.delete(selected, int(np.argmin(loss)))
    return selected


def _accurate_residual(
    pool: np.ndarray,
    multiplicities: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
) -> float:
    """Recompute the residual in complex128 from exact encoded atom phases."""
    approximate = pool[:, selected]
    support = approximate != 0
    phase = np.zeros(approximate.shape, dtype=np.complex128)
    phase[np.real(approximate) > 0] = 1.0
    phase[np.real(approximate) < 0] = -1.0
    phase[np.imag(approximate) > 0] = 1.0j
    phase[np.imag(approximate) < 0] = -1.0j
    support_sizes = support.T @ multiplicities
    atoms = (
        phase
        * np.sqrt(multiplicities)[:, None]
        / np.sqrt(support_sizes)[None, :]
    )
    coefficients = np.linalg.lstsq(atoms, target, rcond=None)[0]
    return float(np.linalg.norm(target - atoms @ coefficients))


def _validate_problem(
    ns: Sequence[int],
    target: np.ndarray,
    rank: int,
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Validate inputs and return the partition, unit target, and target norm."""
    ns_tuple = tuple(map(int, ns))
    if not ns_tuple or any(value <= 0 for value in ns_tuple):
        raise ValueError("ns must contain positive block sizes")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    _, unit_target, norm = _normalise_target(ns_tuple, target)
    return ns_tuple, unit_target, norm


@dataclass(frozen=True)
class _OmpGroup:
    """All structural batches having the same core dimension."""

    start: int
    phases: np.ndarray
    conjugates: np.ndarray
    bins: np.ndarray
    columns: np.ndarray
    orbit_y: np.ndarray
    internal_phase: np.ndarray
    structure_offsets: np.ndarray
    inverse_norms: np.ndarray
    route_scale: np.ndarray

    @property
    def structure_count(self) -> int:
        """Number of affine-support/block-mode structures in the group."""
        return len(self.structure_offsets) - 1

    @property
    def phase_count(self) -> int:
        """Number of core quadratic phases available per structure."""
        return len(self.phases)

    @property
    def emitted_count(self) -> int:
        """Total number of emitted atoms in the group."""
        return self.structure_count * self.phase_count


@lru_cache(maxsize=4)
def _omp_groups(ns: tuple[int, ...]) -> tuple[_OmpGroup, ...]:
    """Pack thousands of tiny generator batches into at most ``k+1`` groups."""
    multiplicities = orbit_multiplicities(ns)
    structures: dict[int, list[tuple[np.ndarray, ...]]] = {}
    phase_tables: dict[int, np.ndarray] = {}

    for phases, columns, orbit_y, internal_phase in (
        gen_symmetric_stabiliser_batch_data(ns)
    ):
        dimension = phases.shape[1].bit_length() - 1
        inverse_norm = np.asarray(
            [1.0 / np.sqrt(multiplicities[columns].sum())],
            dtype=np.float32,
        )
        structures.setdefault(dimension, []).append(
            (columns, orbit_y, internal_phase, inverse_norm)
        )
        phase_tables[dimension] = phases

    groups = []
    start = 0
    for dimension in sorted(structures):
        entries = structures[dimension]
        sizes = np.fromiter(
            (len(entry[0]) for entry in entries),
            dtype=np.intp,
            count=len(entries),
        )
        offsets = np.empty(len(entries) + 1, dtype=np.intp)
        offsets[0] = 0
        np.cumsum(sizes, out=offsets[1:])

        columns = np.concatenate([entry[0] for entry in entries])
        orbit_y = np.concatenate([entry[1] for entry in entries])
        internal = np.concatenate([entry[2] for entry in entries])
        inverse_norms = np.concatenate([entry[3] for entry in entries])
        structure_ids = np.repeat(np.arange(len(entries)), sizes)
        bins = structure_ids * (1 << dimension) + orbit_y
        route_scale = (
            (1 - internal).astype(np.float32)
            * np.repeat(inverse_norms, sizes)
        )
        phases = phase_tables[dimension]
        group = _OmpGroup(
            start,
            phases,
            core_conjugate_phase_table(dimension),
            bins,
            columns,
            orbit_y,
            internal,
            offsets,
            inverse_norms,
            route_scale,
        )
        groups.append(group)
        start += group.emitted_count

    return tuple(groups)


def _group_correlations(
    group: _OmpGroup,
    weighted_vector: np.ndarray,
) -> np.ndarray:
    """Correlate one orbit vector with every atom in a packed group."""
    terms = group.route_scale * weighted_vector[group.columns]
    bin_count = group.structure_count * group.phases.shape[1]
    aggregate = np.bincount(
        group.bins,
        weights=np.real(terms),
        minlength=bin_count,
    ) + 1j * np.bincount(
        group.bins,
        weights=np.imag(terms),
        minlength=bin_count,
    )
    aggregate = aggregate.astype(np.complex64, copy=False).reshape(
        group.structure_count, group.phases.shape[1]
    )
    return (aggregate @ group.conjugates.T).reshape(-1)


def _group_gains(
    group: _OmpGroup,
    weighted_residual: np.ndarray,
) -> np.ndarray:
    """Squared correlations, summed over target-subspace columns."""
    if weighted_residual.ndim == 1:
        correlation = _group_correlations(group, weighted_residual)
        return np.abs(correlation) ** 2

    gains = np.zeros(group.emitted_count, dtype=np.float32)
    for column in range(weighted_residual.shape[1]):
        correlation = _group_correlations(
            group, weighted_residual[:, column]
        )
        gains += np.abs(correlation) ** 2
    return gains


def _group_atom(
    group: _OmpGroup,
    emitted_index: int,
    roots: np.ndarray,
) -> np.ndarray:
    """Materialise one normalized atom from its emitted pool index."""
    local = emitted_index - group.start
    structure, phase = divmod(local, group.phase_count)
    lo = group.structure_offsets[structure]
    hi = group.structure_offsets[structure + 1]
    columns = group.columns[lo:hi]
    encoded = 1 + (
        (
            group.phases[phase, group.orbit_y[lo:hi]]
            + group.internal_phase[lo:hi]
        )
        & 3
    )
    atom = np.zeros(len(roots), dtype=np.complex64)
    atom[columns] = (
        _PHASES[encoded]
        * roots[columns]
        * group.inverse_norms[structure]
    )
    return atom


def _selected_group(
    groups: tuple[_OmpGroup, ...],
    emitted_index: int,
) -> _OmpGroup:
    """Locate the packed group containing an emitted dictionary index."""
    for group in groups:
        if emitted_index < group.start + group.emitted_count:
            return group
    raise IndexError("emitted stabiliser index outside grouped pool")


def _factored_omp(
    ns: tuple[int, ...],
    target: np.ndarray,
    rank: int,
    oversample: int,
    seed: int,
    verbose: bool,
) -> tuple[list[int], float]:
    """Run OMP over the complete dictionary without materialising it.

    Structures of equal core dimension are packed together, allowing all
    correlations to be evaluated with a handful of dense transforms.
    """
    roots = np.sqrt(orbit_multiplicities(ns)).astype(np.float32)
    groups = _omp_groups(ns)
    emitted_count = sum(group.emitted_count for group in groups)
    target_size = min(rank + oversample, emitted_count)
    squared_norm = np.ones(emitted_count, dtype=np.float32)
    excluded = np.zeros(emitted_count, dtype=bool)
    selected_ids: list[int] = []
    selected_vectors: list[np.ndarray] = []
    basis = np.empty((len(roots), 0), dtype=np.complex64)
    residual = target.astype(np.complex64)
    rng = np.random.default_rng(seed)
    progress = _progress(2 * emitted_count * target_size, verbose)

    try:
        while len(selected_ids) < target_size:
            weighted_residual = (
                roots * residual
                if residual.ndim == 1
                else roots[:, None] * residual
            )
            maximum = -np.inf
            choices: list[np.ndarray] = []
            for group in groups:
                lo = group.start
                hi = lo + group.emitted_count
                gains = _group_gains(group, weighted_residual)
                valid = (~excluded[lo:hi]) & (
                    squared_norm[lo:hi] > _INDEPENDENCE_TOL
                )
                gains[valid] /= squared_norm[lo:hi][valid]
                gains[~valid] = -np.inf
                local_maximum = float(np.max(gains))
                previous_scale = abs(maximum) if np.isfinite(maximum) else 0.0
                tolerance = 1e-6 * max(
                    1.0, previous_scale, abs(local_maximum)
                )
                local = np.flatnonzero(
                    gains >= local_maximum - tolerance
                ) + lo
                if not np.isfinite(maximum) or (
                    local_maximum > maximum + tolerance
                ):
                    maximum = local_maximum
                    choices = [local]
                elif abs(local_maximum - maximum) <= tolerance:
                    choices.append(local)
                progress.update(group.emitted_count)

            if not choices or maximum <= 1e-12:
                break
            candidates = np.concatenate(choices)
            emitted_index = int(candidates[rng.integers(len(candidates))])
            group = _selected_group(groups, emitted_index)
            atom = _group_atom(group, emitted_index, roots)
            direction = atom - basis @ (basis.conj().T @ atom)
            if len(selected_ids):
                direction -= basis @ (basis.conj().T @ direction)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= _INDEPENDENCE_TOL:
                excluded[emitted_index] = True
                continue
            direction /= direction_norm

            coefficient = direction.conj() @ residual
            if residual.ndim == 1:
                residual -= direction * coefficient
            else:
                residual -= direction[:, None] * coefficient[None, :]
            basis = np.column_stack((basis, direction))
            selected_ids.append(emitted_index)
            selected_vectors.append(atom)
            excluded[emitted_index] = True

            weighted_direction = roots * direction
            for packed in groups:
                lo = packed.start
                hi = lo + packed.emitted_count
                projection = _group_correlations(
                    packed, weighted_direction
                )
                squared_norm[lo:hi] -= np.abs(projection) ** 2
                np.maximum(squared_norm[lo:hi], 0.0, out=squared_norm[lo:hi])
                progress.update(packed.emitted_count)
    finally:
        progress.close()

    if not selected_vectors:
        return [], float(np.linalg.norm(target))
    atoms = np.column_stack(selected_vectors).astype(np.complex128)
    selected = np.arange(len(selected_ids), dtype=np.intp)
    selected = _prune_support(atoms, target, selected, rank)
    atoms = atoms[:, selected]
    coefficients = np.linalg.lstsq(atoms, target, rcond=None)[0]
    residual_norm = float(np.linalg.norm(target - atoms @ coefficients))
    return [selected_ids[index] for index in selected], residual_norm


def omp(
    ns: Sequence[int],
    target: np.ndarray,
    rank: int,
    verbose: bool = False,
    *,
    seed: int = 0,
    oversample: int = 0,
    materialise_limit: int = 200_000,
) -> tuple[list[int], float]:
    """Approximate ``target`` with ``rank`` symmetric stabiliser atoms.

    OMP operates in the weighted block-orbit basis and returns emitted-pool
    indices together with the residual norm in the original target scaling.
    ``oversample`` permits extra atoms before minimum-loss pruning.

    ``materialise_limit`` is retained for API compatibility; the current OMP
    implementation always uses the packed dictionary.
    """
    ns_tuple, unit_target, norm = _validate_problem(ns, target, rank)
    if oversample < 0:
        raise ValueError("oversample must be nonnegative")
    if materialise_limit <= 0:
        raise ValueError("materialise_limit must be positive")
    if norm == 0.0 or rank == 0:
        return [], norm

    count = count_symmetric_stabilisers(ns_tuple)
    if rank > count:
        raise ValueError("rank exceeds the emitted stabiliser pool size")
    indices, relative_residual = _factored_omp(
        ns_tuple, unit_target, rank, oversample, seed, verbose
    )
    return indices, norm * relative_residual


@dataclass
class _SearchState:
    """Current and best supports maintained by simulated annealing."""

    current: np.ndarray
    current_score: float
    best: np.ndarray
    best_score: float


def _removed_positions(
    pool: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
    count: int,
    iteration: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose support positions using alternating random and guided deletion."""
    if iteration % 3 == 0:
        return rng.choice(len(selected), count, replace=False)
    loss = _deletion_losses(pool, target, selected)
    priority = np.log(loss + 1e-9) + 1.5 * rng.gumbel(size=len(loss))
    return np.argpartition(priority, count - 1)[:count]


def _large_neighbourhood_search(
    pool: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    rng: np.random.Generator,
    *,
    iterations: int,
    epoch_length: int,
    destroy_sizes: Sequence[int],
    repair_shortlist: int,
    start_temperature: float,
    end_temperature: float,
    relative_tolerance: float,
    multiplicities: np.ndarray,
    verbose: bool,
) -> tuple[np.ndarray, float]:
    """Improve a support by repeatedly destroying and conditionally repairing it.

    Complete repaired supports are accepted with a geometric temperature
    schedule.  The best support is verified in complex128 before returning.
    """
    pool_adjoint = pool.conj().T
    search_target = target.astype(np.complex64)
    initial_score = _geometry(
        pool, pool_adjoint, search_target, initial
    ).score
    state = _SearchState(initial.copy(), initial_score, initial.copy(), initial_score)
    sizes = tuple(
        sorted({min(int(value), len(initial)) for value in destroy_sizes})
    )
    sizes = tuple(value for value in sizes if value > 0)
    if not sizes or iterations <= 0:
        return state.best, _accurate_residual(
            pool, multiplicities, target, state.best
        )

    for iteration in range(iterations):
        within_epoch = iteration % epoch_length
        if within_epoch == 0 and iteration:
            state.current = state.best.copy()
            state.current_score = state.best_score
        fraction = within_epoch / max(1, epoch_length - 1)
        temperature = start_temperature * (
            end_temperature / start_temperature
        ) ** fraction

        destroy = sizes[iteration % len(sizes)]
        positions = _removed_positions(
            pool,
            search_target,
            state.current,
            destroy,
            iteration,
            rng,
        )
        keep = np.ones(len(state.current), dtype=bool)
        keep[positions] = False
        trial, trial_score = _complete_support(
            pool,
            pool_adjoint,
            search_target,
            state.current[keep],
            len(initial),
            rng,
            repair_shortlist,
        )
        if len(trial) != len(initial):
            continue

        improvement = trial_score - state.current_score
        if improvement >= 0.0 or rng.random() < np.exp(
            improvement / max(temperature, 1e-12)
        ):
            state.current = trial
            state.current_score = trial_score

        if trial_score <= state.best_score + _SCORE_TOL:
            continue
        state.best = trial.copy()
        state.best_score = trial_score
        if verbose:
            print(
                f"iteration={iteration} destroy={destroy} "
                f"defect={max(0.0, _target_dimension(target) - trial_score):.9g}"
            )
        if _target_dimension(target) - trial_score > 1e-5:
            continue
        residual = _accurate_residual(
            pool, multiplicities, target, state.best
        )
        if residual <= relative_tolerance:
            return state.best, residual

    residual = _accurate_residual(pool, multiplicities, target, state.best)
    return state.best, residual


def _pair_polish(
    pool: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
    *,
    passes: int,
    relative_tolerance: float,
    multiplicities: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Sweep over all two-atom deletions and insert each best completing pair.

    This deterministic local search is restricted to modest dictionaries by
    :func:`sparse_search`, because a pass requires ``rank choose 2`` repairs.
    """
    pool_adjoint = pool.conj().T
    current = selected.copy()
    current_score = _geometry(pool, pool_adjoint, target, current).score
    rank = len(current)

    for _ in range(max(0, passes)):
        improved = False
        for first in range(rank):
            for second in range(first + 1, rank):
                keep = np.ones(rank, dtype=bool)
                keep[[first, second]] = False
                trial, score = _complete_pair(
                    pool,
                    pool_adjoint,
                    target,
                    current[keep],
                )
                if len(trial) != rank or score <= current_score + 1e-7:
                    continue
                current = trial
                current_score = score
                improved = True
                if _target_dimension(target) - score <= 1e-5:
                    residual = _accurate_residual(
                        pool, multiplicities, target, current
                    )
                    if residual <= relative_tolerance:
                        return current, residual
        if not improved:
            break

    return current, _accurate_residual(pool, multiplicities, target, current)


def _prepare_search_pool(
    ns: tuple[int, ...],
    target: np.ndarray,
    emitted: Sequence[int],
    emitted_count: int,
    materialise_limit: int,
    working_pool_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the LNS dictionary and translate its initial emitted indices.

    Small dictionaries are materialised and deduplicated completely.  Large
    dictionaries are screened against the target, while forcibly retaining
    every atom in the initial support.

    Returns
    -------
    pool
        Normalized dictionary columns in weighted orbit coordinates.
    selected
        Initial support as column indices into ``pool``.
    output_indices
        Map from dictionary columns back to public emitted indices.
    """
    emitted_array = np.asarray(emitted, dtype=np.intp)
    if np.any((emitted_array < 0) | (emitted_array >= emitted_count)):
        raise ValueError("initial index outside the emitted stabiliser pool")

    if emitted_count <= materialise_limit:
        complete = normalised_search_pool(ns)
        selected = complete.full_to_unique[emitted_array]
        if len(np.unique(selected)) != len(selected):
            raise ValueError("initial_indices contain repeated stabiliser rays")
        return complete.vectors, selected, complete.representatives

    working = _screened_pool(
        ns, target, emitted_array, working_pool_limit
    )
    locations = {
        int(index): position
        for position, index in enumerate(working.emitted_indices)
    }
    selected = np.fromiter(
        (locations[int(index)] for index in emitted_array),
        dtype=np.intp,
        count=len(emitted_array),
    )
    return working.vectors, selected, working.emitted_indices


def sparse_search(
    ns: Sequence[int],
    target: np.ndarray,
    rank: int,
    *,
    initial_indices: Sequence[int] | None = None,
    seed: int = 0,
    iterations: int = 1000,
    epoch_length: int = 100,
    destroy_sizes: Sequence[int] = (4, 6, 8, 12),
    repair_shortlist: int = 1,
    start_temperature: float = 2e-3,
    end_temperature: float = 2e-4,
    exact_tol: float = 1e-10,
    materialise_limit: int = 200_000,
    working_pool_limit: int = 50_000,
    omp_oversample: int = 0,
    pair_passes: int = 4,
    pair_pool_limit: int = 5000,
    verbose: bool = False,
) -> tuple[list[int], float]:
    """Search for a rank-``rank`` symmetric stabiliser approximation.

    The pipeline consists of packed OMP, one simulated-annealing
    large-neighbourhood search, and (for small dictionaries) deterministic
    two-atom polishing.  Every LNS iteration removes one of ``destroy_sizes``
    atoms and refills the holes by conditional OMP.  Temperatures apply to
    complete repaired supports rather than individual insertions.

    Returns emitted-pool indices and the residual norm in the original target
    scaling.  A residual at most ``exact_tol`` is returned as exactly zero.
    """
    ns_tuple, unit_target, norm = _validate_problem(ns, target, rank)
    if norm == 0.0 or rank == 0:
        return [], norm
    if iterations < 0 or epoch_length <= 0:
        raise ValueError("iterations must be nonnegative and epoch_length positive")
    if start_temperature <= 0.0 or end_temperature <= 0.0:
        raise ValueError("temperatures must be positive")
    if repair_shortlist <= 0:
        raise ValueError("repair_shortlist must be positive")
    if materialise_limit <= 0 or working_pool_limit <= 0 or pair_pool_limit <= 0:
        raise ValueError("pool limits must be positive")
    if pair_passes < 0:
        raise ValueError("pair_passes must be nonnegative")

    emitted_count = count_symmetric_stabilisers(ns_tuple)
    if rank > emitted_count:
        raise ValueError("rank exceeds the emitted stabiliser pool size")
    if initial_indices is None:
        emitted, initial_residual = _factored_omp(
            ns_tuple,
            unit_target,
            rank,
            omp_oversample,
            seed,
            verbose,
        )
    else:
        emitted = list(map(int, initial_indices))
        initial_residual = float("inf")
    if initial_indices is None and initial_residual <= exact_tol / norm:
        return emitted, 0.0
    if initial_indices is None and len(emitted) < rank:
        return emitted, float(norm * initial_residual)
    if len(emitted) != rank:
        raise ValueError("initial_indices must contain exactly rank entries")
    pool, selected, output_indices = _prepare_search_pool(
        ns_tuple,
        unit_target,
        emitted,
        emitted_count,
        materialise_limit,
        working_pool_limit,
    )
    selected, relative_residual = _large_neighbourhood_search(
        pool,
        unit_target,
        selected,
        np.random.default_rng(seed),
        iterations=iterations,
        epoch_length=epoch_length,
        destroy_sizes=destroy_sizes,
        repair_shortlist=repair_shortlist,
        start_temperature=start_temperature,
        end_temperature=end_temperature,
        relative_tolerance=exact_tol / norm,
        multiplicities=orbit_multiplicities(ns_tuple),
        verbose=verbose,
    )
    if (
        relative_residual > exact_tol / norm
        and pair_passes
        and pool.shape[1] <= pair_pool_limit
    ):
        selected, relative_residual = _pair_polish(
            pool,
            unit_target,
            selected,
            passes=pair_passes,
            relative_tolerance=exact_tol / norm,
            multiplicities=orbit_multiplicities(ns_tuple),
        )

    residual = norm * relative_residual
    if residual <= exact_tol:
        residual = 0.0
    return output_indices[selected].tolist(), float(residual)


# ---------------------------------------------------------------------------
# Generic command-line orchestration for target-specific search scripts
# ---------------------------------------------------------------------------

TargetBuilder = Callable[[tuple[int, ...]], np.ndarray]


@dataclass(frozen=True)
class PartitionSearchResult:
    """Best result obtained for one block partition."""

    partition: tuple[int, ...]
    indices: list[int]
    residual: float
    elapsed: float


def parse_partition(text: str) -> tuple[int, ...]:
    """Parse ``1+5+5`` or ``1,5,5`` into a partition tuple."""
    return tuple(int(part) for part in text.replace(",", "+").split("+") if part)


def format_partition(ns: Sequence[int]) -> str:
    """Format a partition tuple for command-line output."""
    return "+".join(map(str, ns))


def add_search_arguments(
    parser: argparse.ArgumentParser,
    *,
    partition_example: str = "1+5+5",
) -> argparse.ArgumentParser:
    """Add the common sparse-search CLI to a target-specific parser.

    A new target script therefore only has to define a function mapping a
    partition ``ns`` either to one amplitude per block-weight orbit (one
    target state) or to a matrix whose columns span a target family, then call
    :func:`run_target_search`.
    """
    parser.add_argument("n", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("--max-parts", type=int, default=3)
    parser.add_argument(
        "--partition",
        type=parse_partition,
        help=f"search one partition, for example --partition {partition_example}",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--oversample", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--epoch-length", type=int, default=100)
    parser.add_argument(
        "--destroy-sizes",
        type=lambda value: tuple(map(int, value.split(","))),
        default=(4, 6, 8, 12),
    )
    parser.add_argument("--repair-shortlist", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="partition-search processes; use more than one for hard scans",
    )
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--omp-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output",
        default="out.json",
        help="QlAb decomposition JSON path (default: out.json)",
    )
    return parser


def _search_trials(
    ns: tuple[int, ...],
    target: np.ndarray,
    args: argparse.Namespace,
    *,
    emit_progress: bool,
) -> tuple[list[int], float]:
    """Run the common trial schedule for one already-built target."""
    best_indices: list[int] = []
    best_residual = float("inf")
    trial_count = 1 if args.omp_only else max(1, args.trials)

    for trial in range(trial_count):
        seed = args.seed + trial
        if args.omp_only:
            indices, residual = omp(
                ns,
                target,
                args.rank,
                seed=seed,
                oversample=args.oversample,
            )
        else:
            indices, residual = sparse_search(
                ns,
                target,
                args.rank,
                seed=seed,
                iterations=args.iterations,
                epoch_length=args.epoch_length,
                destroy_sizes=args.destroy_sizes,
                repair_shortlist=args.repair_shortlist,
                exact_tol=args.tolerance,
                omp_oversample=args.oversample,
                verbose=args.verbose and emit_progress,
            )
        if residual < best_residual:
            best_indices = indices
            best_residual = residual
        if emit_progress:
            print(f"  trial={trial} final residual={residual:.6g}")
        if best_residual < args.tolerance:
            break

    return best_indices, best_residual


def _partition_worker(
    task: tuple[tuple[int, ...], np.ndarray, argparse.Namespace],
) -> PartitionSearchResult:
    """Multiprocessing worker; targets are built in the parent process."""
    ns, target, args = task
    emit_progress = args.workers == 1
    started = time.perf_counter()
    indices, residual = _search_trials(
        ns, target, args, emit_progress=emit_progress
    )
    return PartitionSearchResult(
        ns, indices, residual, time.perf_counter() - started
    )


def report_partition_result(
    result: PartitionSearchResult,
    tolerance: float,
) -> bool:
    """Print one partition result and return whether it is exact."""
    success = result.residual < tolerance
    status = "SUCCESS" if success else "best approximation"
    print(
        f"{status}: partition={format_partition(result.partition)} "
        f"rank={len(result.indices)} residual={result.residual:.6g} "
        f"time={result.elapsed:.3f}s\nindices={result.indices}"
    )
    return success


def eligible_partitions(
    n: int,
    rank: int,
    max_parts: int,
) -> list[tuple[int, ...]]:
    """Partitions whose emitted symmetric dictionary can contain ``rank`` rays."""
    return [
        ns
        for ns in gen_partitions(n, max_parts)
        if count_symmetric_stabilisers(ns) >= rank
    ]


def run_target_search(
    args: argparse.Namespace,
    target_builder: TargetBuilder,
    target_spec: dict[str, object],
) -> PartitionSearchResult | None:
    """Run the standard partition/trial pipeline for an orbit-basis target.

    ``target_builder(ns)`` may return either one amplitude for every
    block-weight orbit of ``ns``, or an orbit-by-k matrix whose columns span a
    family of targets.  In the latter case the search minimizes the Frobenius
    residual of the whole target subspace.  Target scripts can therefore
    remain tiny and contain no search-engine policy.  The best decomposition
    found is written to ``args.output`` in the common QlAb JSON format.
    """
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    if args.partition is not None:
        if sum(args.partition) != args.n:
            raise SystemExit("--partition must sum to n")
        if count_symmetric_stabilisers(args.partition) < args.rank:
            raise SystemExit(
                f"partition {format_partition(args.partition)} emits fewer "
                f"than rank {args.rank} stabilisers"
            )
        partitions = [args.partition]
    else:
        partitions = eligible_partitions(args.n, args.rank, args.max_parts)
        if not partitions:
            raise SystemExit(
                "no partition up to --max-parts has enough emitted stabilisers "
                f"for rank {args.rank}"
            )

    # Build the (small) orbit target in the parent.  Workers then only receive
    # plain arrays, so target functions never need multiprocessing/pickling
    # boilerplate.
    tasks = [
        (ns, np.asarray(target_builder(ns)), args)
        for ns in partitions
    ]

    best: PartitionSearchResult | None = None
    if args.workers == 1:
        for task in tasks:
            result = _partition_worker(task)
            if best is None or result.residual < best.residual:
                best = result
            if report_partition_result(result, args.tolerance):
                break
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(processes=min(args.workers, len(tasks)))
        terminated = False
        try:
            for result in pool.imap_unordered(_partition_worker, tasks):
                if best is None or result.residual < best.residual:
                    best = result
                if report_partition_result(result, args.tolerance):
                    pool.terminate()
                    terminated = True
                    break
        finally:
            if not terminated:
                pool.close()
            pool.join()

    if best is not None and best.indices:
        from stab_export import save_decomposition_json

        target = np.asarray(target_builder(best.partition))
        save_decomposition_json(
            args.output,
            partition=best.partition,
            indices=best.indices,
            target=target,
            target_spec=target_spec,
            residual=best.residual,
        )
        print(f"wrote QlAb decomposition to {args.output}")
    return best
