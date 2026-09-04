from collections.abc import Iterator, Sequence

from functools import cache, lru_cache
from itertools import combinations, product
from math import comb

import numpy as np


def gen_rrefs(n: int, r: int) -> Iterator[np.ndarray]:
    """
    Generate all rank-r, r-by-n matrices in reduced row echelon form.
    """
    for pivot_cols in combinations(range(n), r):
        cnt_free = n * r - sum(pivot_cols) - r * (r + 1) // 2
        for free_vars in product(range(2), repeat=cnt_free):
            mat = np.zeros((r, n), dtype=np.int8)
            for i in range(r):
                mat[i][pivot_cols[i]] = 1
            loc = 0
            for i in range(r):
                for j in range(pivot_cols[i] + 1, n):
                    if j in pivot_cols:
                        continue
                    mat[i][j] = free_vars[loc]
                    loc += 1
            yield mat


def canonical_cs(BT: np.ndarray) -> Iterator[np.ndarray]:
    d, k = BT.shape
    pivots = [int(np.flatnonzero(BT[r])[0]) for r in range(d)]
    free = [a for a in range(k) if a not in pivots]
    for values in product(range(2), repeat=len(free)):
        c = np.zeros(k, dtype=np.int8)
        c[free] = values
        yield c


def count_symmetric_stabilisers(ns: Sequence[int]) -> int:
    """
    Count the states emitted by the canonical generator.

    This is also the number of distinct stabiliser rays when every block has
    size at least three.  Blocks of size one or two have additional
    active/inactive
    degeneracies.
    """
    k = len(ns)
    cnt = 0
    for d in range(k + 1):
        cnt += 6**k * 2 ** (d * (d + 1) // 2) * len(list(gen_rrefs(k, d)))
    return cnt


def orbit_weights(ns: Sequence[int]) -> np.ndarray:
    """
    Return the block-weight tuples in the order used by the orbit routines.

    The result has shape (prod_a (n_a + 1), len(ns)).
    """
    ns = tuple(map(int, ns))
    shape = tuple(n + 1 for n in ns)
    return np.indices(shape, dtype=np.int16).reshape(len(ns), -1).T


def orbit_multiplicities(ns: Sequence[int]) -> np.ndarray:
    """Number of computational-basis strings in each block-weight orbit."""
    ns = tuple(map(int, ns))
    weights = orbit_weights(ns)
    result = np.ones(len(weights), dtype=np.int64)
    for a, n in enumerate(ns):
        result *= np.fromiter(
            (comb(n, int(w)) for w in weights[:, a]),
            dtype=np.int64,
            count=len(weights),
        )
    return result


_PHASES = np.asarray([0, 1, 1j, -1, -1j], dtype=np.complex64)
_CONJ_PHASES = _PHASES.conj()


@cache
def core_phase_table(d: int) -> np.ndarray:
    """
    All phases l.y + 2 y^T Q y mod 4 on F_2^d.

    Rows enumerate (l,Q), columns are little-endian integer encodings of y.
    The shape is (4^d 2^binom(d,2), 2^d).
    """
    y_int = np.arange(1 << d, dtype=np.int64)
    y = ((y_int[:, None] >> np.arange(d)) & 1).astype(np.int8)

    l_code = np.arange(4**d, dtype=np.int64)
    l = ((l_code[:, None] >> (2 * np.arange(d))) & 3).astype(np.int8)
    linear = (l @ y.T) & 3

    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    nq = len(pairs)
    q_code = np.arange(1 << nq, dtype=np.int64)
    q_bits = ((q_code[:, None] >> np.arange(nq)) & 1).astype(np.int8)

    if nq:
        pair_values = np.stack([y[:, i] * y[:, j] for i, j in pairs])
        quadratic = (2 * (q_bits @ pair_values)) & 3
    else:
        quadratic = np.zeros((1, 1 << d), dtype=np.int8)

    table = ((linear[:, None, :] + quadratic[None, :, :]) & 3).reshape(-1, 1 << d)
    table = table.astype(np.int8, copy=False)
    table.flags.writeable = False
    return table


@cache
def core_conjugate_phase_table(d: int) -> np.ndarray:
    """Complex conjugates of all core phase functions, cached by d."""
    table = _CONJ_PHASES[1 + core_phase_table(d)]
    table.flags.writeable = False
    return table


@lru_cache(maxsize=8)
def orbit_mode_tables(ns: tuple[int, ...]):
    """Precompute support/phase data for the three modes of every block."""
    k = len(ns)
    ns_array = np.asarray(ns, dtype=np.int16)
    weights = orbit_weights(ns)
    h = ((weights * (weights - 1) // 2) & 1).astype(np.int8)
    powers = 1 << np.arange(k)

    result = []
    for mode_tuple in product(range(3), repeat=k):
        modes = np.asarray(mode_tuple, dtype=np.int8)
        active = modes != 0
        q = np.maximum(modes - 1, 0)

        # An inactive block has weight only 0 or n_a. Its core bit says
        # which of these two constant strings occurs. An active block's
        # core bit is its weight parity.
        support = np.all(
            active[None, :] | ((weights == 0) | (weights == ns_array)),
            axis=1,
        )
        z = np.where(active[None, :], weights & 1, weights == ns_array)
        z_int = (z @ powers).astype(np.intp)
        internal_phase = (2 * ((h @ q) & 1)).astype(np.int8)
        result.append((support, z_int, internal_phase))
    return tuple(result)


def gen_symmetric_stabiliser_batch_data(ns: tuple[int, ...]):
    """
    Yield the factored data underlying each stabiliser batch.

    Each item is ``(core_phases, columns, orbit_y, internal_phase)``. For a
    row r and a valid orbit j, its phase exponent is

        core_phases[r, orbit_y[j]] + internal_phase[j]  (mod 4).

    Keeping this factorisation is substantially faster for operations such as
    OMP that only need correlations with the dictionary.
    """
    k = len(ns)
    powers = 1 << np.arange(k)
    mode_tables = orbit_mode_tables(ns)
    rrefs = tuple(tuple(gen_rrefs(k, d)) for d in range(k + 1))

    for d in range(k + 1):
        core_phases = core_phase_table(d)
        y_int = np.arange(1 << d, dtype=np.intp)
        y = ((y_int[:, None] >> np.arange(d)) & 1).astype(np.int8)
        for BT in rrefs[d]:
            direction_z = (y @ BT) & 1
            for c in canonical_cs(BT):
                z_to_y = np.full(1 << k, -1, dtype=np.int16)
                affine_z = direction_z ^ c
                z_to_y[(affine_z @ powers).astype(np.intp)] = y_int
                for support, z_int, internal_phase in mode_tables:
                    orbit_y = z_to_y[z_int]
                    valid = support & (orbit_y >= 0)
                    columns = np.flatnonzero(valid)
                    yield (
                        core_phases,
                        columns,
                        orbit_y[valid].astype(np.intp, copy=False),
                        internal_phase[valid],
                    )


def gen_symmetric_stabilisers_batched(ns: Sequence[int]) -> Iterator[np.ndarray]:
    """
    Yield the symmetric stabiliser pool in vectorised batches.

    An amplitude is encoded as 0, or 1+p for i^p. Columns are block-weight
    orbits in ``orbit_weights(ns)`` order. Each batch contains every core
    quadratic phase for one fixed affine support and block-mode choice.

    When every n_a >= 3, every stabiliser ray occurs exactly once. For blocks
    of size one or two, different active/inactive parameter choices can encode
    the same ray; those repetitions are intentionally retained.
    """
    ns = tuple(map(int, ns))
    n_orbits = int(np.prod(np.asarray(ns, dtype=np.int64) + 1))
    for data in gen_symmetric_stabiliser_batch_data(ns):
        core_phases, columns, orbit_y, internal_phase = data
        n_phases = len(core_phases)
        batch = np.zeros((n_phases, n_orbits), dtype=np.int8)
        batch[:, columns] = 1 + ((core_phases[:, orbit_y] + internal_phase) & 3)
        yield batch


@lru_cache(maxsize=2)
def basis_to_orbit(ns: tuple[int, ...]):
    """Map each computational-basis index to its block-weight orbit index."""
    n = sum(ns)
    x = np.arange(1 << n, dtype=np.int64)
    weights = []
    offset = 0
    for block_size in ns:
        mask = (1 << block_size) - 1
        weights.append(np.bitwise_count((x >> offset) & mask))
        offset += block_size
    orbit_shape = tuple(block_size + 1 for block_size in ns)
    return np.ravel_multi_index(tuple(weights), orbit_shape)


def target_in_orbit_basis(ns: tuple[int, ...], target: np.ndarray) -> np.ndarray:
    """
    Return one target amplitude per block-weight orbit.

    An orbit-sized target is returned directly.  A computational-basis target
    is averaged within each orbit.  Averaging is exact for symmetric targets;
    for a general target it gives its orthogonal projection onto the symmetric
    subspace, which has the same inner products with every dictionary atom.
    """
    target = np.asarray(target)
    if target.ndim != 1:
        raise ValueError("target must be a one-dimensional vector")

    n_orbits = int(np.prod(np.asarray(ns, dtype=np.int64) + 1))
    if len(target) == n_orbits:
        return target.astype(np.complex128, copy=False)

    n_amplitudes = 1 << sum(ns)
    if len(target) != n_amplitudes:
        raise ValueError(
            f"target has length {len(target)}; expected {n_orbits} orbit "
            f"amplitudes or {n_amplitudes} computational-basis amplitudes"
        )

    basis2orbit = basis_to_orbit(ns)
    multiplicities = orbit_multiplicities(ns)
    real = np.bincount(
        basis2orbit,
        weights=np.asarray(np.real(target), dtype=np.float64),
        minlength=n_orbits,
    )
    imag = np.bincount(
        basis2orbit,
        weights=np.asarray(np.imag(target), dtype=np.float64),
        minlength=n_orbits,
    )
    return (real + 1j * imag) / multiplicities


def decode_orbits(encoded_orbits: np.ndarray, ns: Sequence[int]) -> np.ndarray:
    """Expand encoded orbit amplitudes into the computational basis."""
    ns = tuple(map(int, ns))
    phases = np.asarray(encoded_orbits)[..., basis_to_orbit(ns)]
    return _PHASES[phases]


def integer_partitions_fixed_length(
    n: int,
    m: int,
    lo: int = 1,
) -> Iterator[tuple[int, ...]]:
    """Nondecreasing m-part partitions of n."""
    if m == 1:
        if n >= lo:
            yield (n,)
        return
    for a in range(lo, n + 1):
        rem = n - a
        if rem < a * (m - 1):
            break
        for tail in integer_partitions_fixed_length(rem, m - 1, a):
            yield (a,) + tail


def gen_partitions(n: int, max_parts: int) -> list[tuple[int, ...]]:
    """All partitions of n, with number of parts ranging from 1 to max_parts."""
    parts = []
    for m in range(1, max_parts + 1):
        parts.extend(integer_partitions_fixed_length(n, m))
    return parts
