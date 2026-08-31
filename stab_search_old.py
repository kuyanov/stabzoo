#!/usr/bin/env python3
"""Core routines for partition-symmetric stabilizer decomposition search.

The module deliberately separates four concerns:

  * build_symmetric_pool(partition): compressed partition-symmetric stabilizer pool;
  * greedy_initial_decomposition(...): simultaneous-OMP warm start for an arbitrary target span;
  * anneal_decomposition(...): heat-bath / coordinate annealing;
  * descriptor_to_Q_l_A_b(...): recover an explicit computational-basis stabilizer description.

Two phase-enumeration modes are supported.

``phase_mode='complete'`` (default)
    Enumerates phase *functions on the support* that are invariant under the
    Young subgroup.  The displayed (Q,l) gauge need not itself be block
    constant.  This fixes the hidden-gauge omission of the older
    symmetric_stabilizer_anneal.py.  For example, the S_6-invariant state
        |0^6> - i |1^6>
    is present in build_symmetric_pool((6,)).

``phase_mode='manifest'``
    Reproduces the older, very fast canonical pool: Q is constant on block
    pairs and l is constant on blocks.  It is useful for large exploratory
    scans, but it is not a complete state-level symmetry classification.

All pool vectors are normalized and stored in the normalized block-Dicke basis
indexed by block weights.  Its dimension is prod_a (n_a+1).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, Sequence
import math
import time

import numpy as np


# ---------------------------------------------------------------------------
# GF(2) bit-vector helpers
# ---------------------------------------------------------------------------

def canonical_basis(vectors: Iterable[int], n: int) -> tuple[int, ...]:
    rows = [int(v) for v in vectors if int(v)]
    r = 0
    for p in range(n):
        idx = next((i for i in range(r, len(rows))
                   if (rows[i] >> p) & 1), None)
        if idx is None:
            continue
        rows[r], rows[idx] = rows[idx], rows[r]
        for i in range(len(rows)):
            if i != r and ((rows[i] >> p) & 1):
                rows[i] ^= rows[r]
        r += 1
        if r == len(rows):
            break
    return tuple(rows[:r])


def swap_bits(mask: int, i: int, j: int) -> int:
    if ((mask >> i) ^ (mask >> j)) & 1:
        mask ^= (1 << i) | (1 << j)
    return int(mask)


def partition_generators(partition: Sequence[int]) -> list[tuple[int, int]]:
    out = []
    off = 0
    for m in partition:
        out.extend((i, i + 1) for i in range(off, off + m - 1))
        off += m
    return out


def orbit_of_mask(mask: int, generators: Sequence[tuple[int, int]]) -> set[int]:
    seen = {int(mask)}
    stack = [int(mask)]
    while stack:
        x = stack.pop()
        for i, j in generators:
            y = swap_bits(x, i, j)
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return seen


def syndrome(basis: Sequence[int], x: int) -> int:
    out = 0
    for i, row in enumerate(basis):
        out |= (((row & x).bit_count() & 1) << i)
    return out


def _gf2_rref(rows: Sequence[int], ncols: int):
    """RREF of bit-mask rows; also tracks row operations.

    Returns (rrows, transform, pivots). transform[i] is a bit mask of original
    equations whose xor produced rrows[i].
    """
    a = [int(x) for x in rows]
    m = len(a)
    tr = [1 << i for i in range(m)]
    pivots = []
    r = 0
    for c in range(ncols):
        p = next((i for i in range(r, m) if (a[i] >> c) & 1), None)
        if p is None:
            continue
        a[r], a[p] = a[p], a[r]
        tr[r], tr[p] = tr[p], tr[r]
        for i in range(m):
            if i != r and ((a[i] >> c) & 1):
                a[i] ^= a[r]
                tr[i] ^= tr[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return a, tr, pivots


def _gf2_nullspace(rows: Sequence[int], ncols: int) -> tuple[int, ...]:
    rr, _, piv = _gf2_rref(rows, ncols)
    rank = len(piv)
    pset = set(piv)
    out = []
    for f in range(ncols):
        if f in pset:
            continue
        x = 1 << f
        for i in range(rank):
            p = piv[i]
            if (rr[i] >> f) & 1:
                x |= 1 << p
        out.append(x)
    return tuple(out)


class GF2LinearSystem:
    """One fixed A over F2, many right-hand sides."""

    def __init__(self, rows: Sequence[int], ncols: int):
        self.rows = tuple(int(x) for x in rows)
        self.ncols = int(ncols)
        self.rrows, self.transform, self.pivots = _gf2_rref(
            self.rows, self.ncols)
        self.rank = len(self.pivots)
        self.free = tuple(c for c in range(ncols) if c not in set(self.pivots))
        nb = []
        for f in self.free:
            x = 1 << f
            for i, p in enumerate(self.pivots):
                if (self.rrows[i] >> f) & 1:
                    x |= 1 << p
            nb.append(x)
        self.nullspace = tuple(nb)

    def solve(self, rhs_bits: int) -> int | None:
        """Particular solution with all free variables zero, or None."""
        rhs_r = [((self.transform[i] & rhs_bits).bit_count() & 1)
                 for i in range(len(self.rrows))]
        for i in range(self.rank, len(self.rrows)):
            if self.rrows[i] == 0 and rhs_r[i]:
                return None
        x = 0
        for i, p in enumerate(self.pivots):
            if rhs_r[i]:
                x |= 1 << p
        return x


# ---------------------------------------------------------------------------
# Invariant affine supports
# ---------------------------------------------------------------------------

def invariant_row_spaces(partition: Sequence[int]) -> list[tuple[int, ...]]:
    n = sum(partition)
    gens = partition_generators(partition)
    cyclic: set[tuple[int, ...]] = set()
    for ws in product(*[range(m + 1) for m in partition]):
        if not any(ws):
            continue
        mask = 0
        off = 0
        for m, w in zip(partition, ws):
            mask |= ((1 << w) - 1) << off
            off += m
        cyclic.add(canonical_basis(orbit_of_mask(mask, gens), n))
    generators = list(cyclic)
    spaces: set[tuple[int, ...]] = {()}
    queue: list[tuple[int, ...]] = [()]
    while queue:
        B = queue.pop()
        for C in generators:
            D = canonical_basis(B + C, n)
            if D not in spaces:
                spaces.add(D)
                queue.append(D)
    return sorted(spaces, key=lambda B: (len(B), B))


@dataclass(frozen=True)
class AffineSupport:
    rows: tuple[int, ...]
    rhs: int
    representative: int

    @property
    def codimension(self) -> int:
        return len(self.rows)


def invariant_affine_supports(partition: Sequence[int]) -> list[AffineSupport]:
    n = sum(partition)
    gens = partition_generators(partition)
    out = []
    for B in invariant_row_spaces(partition):
        r = len(B)
        reps: list[int | None] = [None] * (1 << r)
        for x in range(1 << n):
            s = syndrome(B, x)
            if reps[s] is None:
                reps[s] = x
        for b, x0_ in enumerate(reps):
            assert x0_ is not None
            x0 = int(x0_)
            if all(syndrome(B, swap_bits(x0, i, j)) == b for i, j in gens):
                out.append(AffineSupport(tuple(B), b, x0))
    return out


# ---------------------------------------------------------------------------
# Block-weight compression
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitData:
    weights: tuple[tuple[int, ...], ...]
    representatives: np.ndarray
    sizes: np.ndarray


def orbit_data(partition: Sequence[int]) -> OrbitData:
    weights = tuple(product(*[range(m + 1) for m in partition]))
    reps = []
    sizes = []
    for ws in weights:
        mask = 0
        off = 0
        sz = 1
        for m, w in zip(partition, ws):
            mask |= ((1 << w) - 1) << off
            sz *= math.comb(m, w)
            off += m
        reps.append(mask)
        sizes.append(sz)
    return OrbitData(weights, np.asarray(reps, dtype=np.int64), np.asarray(sizes, dtype=float))


@dataclass(frozen=True)
class PhaseDescriptor:
    # General ambient gauge. qmask indexes all pairs i<j; linear is length n.
    qmask: int
    linear: tuple[int, ...]


@dataclass(frozen=True)
class CandidateDescriptor:
    support: AffineSupport
    phase: PhaseDescriptor
    support_size: int


@dataclass
class CandidatePool:
    partition: tuple[int, ...]
    vectors: np.ndarray
    descriptors: list[CandidateDescriptor]
    orbit_data: OrbitData
    pair_slots: tuple[tuple[int, int], ...]
    phase_mode: str

    @property
    def n(self) -> int:
        return sum(self.partition)

    @property
    def size(self) -> int:
        return self.vectors.shape[1]


# ---------------------------------------------------------------------------
# Phase enumeration: fast manifest mode
# ---------------------------------------------------------------------------

def _manifest_phase_table(partition: Sequence[int], od: OrbitData):
    k = len(partition)
    qslots = [(a, a) for a, m in enumerate(partition) if m >= 2]
    qslots += list(combinations(range(k), 2))
    nq = 1 << len(qslots)
    nl = 4 ** k
    P = np.zeros((nq * nl, len(od.weights)), dtype=np.uint8)
    desc = []
    row = 0
    blocks = []
    off = 0
    for m in partition:
        blocks.append(list(range(off, off + m)))
        off += m
    all_pairs = tuple(combinations(range(sum(partition)), 2))
    pair_index = {p: i for i, p in enumerate(all_pairs)}
    for qmask in range(nq):
        for lcode in range(nl):
            z = lcode
            block_l = []
            for _ in range(k):
                block_l.append(z & 3)
                z >>= 2
            linear = [0] * sum(partition)
            for a, B in enumerate(blocks):
                for i in B:
                    linear[i] = block_l[a]
            ambient_q = 0
            for s, (a, b) in enumerate(qslots):
                if not ((qmask >> s) & 1):
                    continue
                if a == b:
                    pairs = combinations(blocks[a], 2)
                else:
                    pairs = ((i, j) for i in blocks[a] for j in blocks[b])
                for i, j in pairs:
                    if i > j:
                        i, j = j, i
                    ambient_q |= 1 << pair_index[(i, j)]
            for j, ws in enumerate(od.weights):
                e = sum(block_l[a] * ws[a] for a in range(k))
                q = 0
                for s, (a, b) in enumerate(qslots):
                    if (qmask >> s) & 1:
                        q += ws[a]*(ws[a]-1)//2 if a == b else ws[a]*ws[b]
                P[row, j] = (e + 2*q) & 3
            desc.append(PhaseDescriptor(ambient_q, tuple(linear)))
            row += 1
    return P, tuple(desc)


def _build_manifest_pool(partition: tuple[int, ...], od: OrbitData,
                         supports: Sequence[AffineSupport]) -> tuple[np.ndarray, list[CandidateDescriptor]]:
    P, pdesc = _manifest_phase_table(partition, od)
    phases = np.asarray([1, 1j, -1, -1j], complex)
    vecs, descs = [], []
    for sup in supports:
        inside = np.asarray([syndrome(sup.rows, int(rep)) == sup.rhs
                             for rep in od.representatives], bool)
        cols = np.flatnonzero(inside)
        support_size = int(round(float(od.sizes[inside].sum())))
        A = P[:, cols]
        relative = (A - A[:, [0]]) & 3
        _, first = np.unique(relative, axis=0, return_index=True)
        first.sort()
        scale = np.sqrt(od.sizes[inside] / support_size)
        for idx in first:
            e = (P[idx, cols] - P[idx, cols[0]]) & 3
            v = np.zeros(len(od.weights), complex)
            v[cols] = scale * phases[e]
            vecs.append(v)
            descs.append(CandidateDescriptor(sup, pdesc[idx], support_size))
    C = np.column_stack(vecs) if vecs else np.zeros(
        (len(od.weights), 0), complex)
    return C, descs


# ---------------------------------------------------------------------------
# Phase enumeration: complete state-level symmetry mode
# ---------------------------------------------------------------------------

def _support_direction_basis(sup: AffineSupport, n: int) -> tuple[int, ...]:
    return _gf2_nullspace(sup.rows, n)


def _quadratic_pair_bit(mask: int, pairs: Sequence[tuple[int, int]]) -> int:
    out = 0
    for k, (i, j) in enumerate(pairs):
        if ((mask >> i) & 1) and ((mask >> j) & 1):
            out |= 1 << k
    return out


def _phase_eval_descriptor(desc: PhaseDescriptor, x: int, pairs) -> int:
    e = 0
    for i, a in enumerate(desc.linear):
        if (x >> i) & 1:
            e += int(a)
    e += 2 * ((desc.qmask & _quadratic_pair_bit(x, pairs)).bit_count() & 1)
    return e & 3


def _independent_image_generators(output_masks: Sequence[int], preimages: Sequence[int]):
    """Basis of a binary image, retaining one preimage for every basis vector."""
    piv: dict[int, tuple[int, int]] = {}
    for out, pre in zip(output_masks, preimages):
        y, p = int(out), int(pre)
        while y:
            b = y.bit_length() - 1
            if b in piv:
                y ^= piv[b][0]
                p ^= piv[b][1]
            else:
                # clean lower pivots from the new vector, and new pivot from old vectors
                for bb in sorted(list(piv)):
                    if (y >> bb) & 1:
                        y ^= piv[bb][0]
                        p ^= piv[bb][1]
                for bb in list(piv):
                    oy, op = piv[bb]
                    if (oy >> b) & 1:
                        piv[bb] = (oy ^ y, op ^ p)
                piv[b] = (y, p)
                break
    return tuple(piv[b] for b in sorted(piv))


def _complete_phases_on_support(
    partition: tuple[int, ...], od: OrbitData, sup: AffineSupport,
    pairs: tuple[tuple[int, int], ...],
):
    """Yield (relative_phase_vector, PhaseDescriptor) for one invariant support.

    Derivation. Write l = u + 2v with u,v in F_2^n and Q in F_2^{n choose 2}.
    For a symmetry generator g and x in the support, f(gx)=f(x) mod 4.
    Modulo 2 this is linear in u.  Once u is fixed, division by two gives a
    second linear system over F_2 in (v,Q).  Checking x0, x0+b_i and
    x0+b_i+b_j is sufficient because f(gx)-f(x) is quadratic on the support.
    """
    n = sum(partition)
    nq = len(pairs)
    zdim = n + nq
    gens = partition_generators(partition)
    dirs = _support_direction_basis(sup, n)

    # A quadratic polynomial on the affine support is determined by these points.
    pts = {sup.representative}
    for a in dirs:
        pts.add(sup.representative ^ a)
    for i in range(len(dirs)):
        for j in range(i+1, len(dirs)):
            pts.add(sup.representative ^ dirs[i] ^ dirs[j])

    stage1_rows = []
    stage2_rows = []
    posmasks = []
    negmasks = []
    for x in pts:
        qx = _quadratic_pair_bit(x, pairs)
        for i, j in gens:
            y = swap_bits(x, i, j)
            if y == x:
                continue
            c = x ^ y
            stage1_rows.append(c)
            qy = _quadratic_pair_bit(y, pairs)
            stage2_rows.append(c | ((qx ^ qy) << n))
            posmasks.append(x & ~y)
            negmasks.append(y & ~x)

    stage1_rows = tuple(set(stage1_rows))
    ubasis = _gf2_nullspace(stage1_rows, n)
    sys2 = GF2LinearSystem(stage2_rows, zdim)

    inside = np.asarray([syndrome(sup.rows, int(rep)) == sup.rhs
                         for rep in od.representatives], bool)
    cols = np.flatnonzero(inside)
    reps = [int(od.representatives[c]) for c in cols]
    if len(reps) <= 1:
        yield np.zeros(len(reps), dtype=np.uint8), PhaseDescriptor(0, tuple([0]*n))
        return
    r0 = reps[0]
    qr0 = _quadratic_pair_bit(r0, pairs)
    eval_masks = []
    for r in reps[1:]:
        c = r ^ r0
        eval_masks.append(c | ((_quadratic_pair_bit(r, pairs) ^ qr0) << n))

    # Enumerating u itself can be wasteful on one-orbit supports; handled above.
    seen: set[bytes] = set()
    for code in range(1 << len(ubasis)):
        u = 0
        for k, b in enumerate(ubasis):
            if (code >> k) & 1:
                u ^= b

        rhs = 0
        good = True
        for t, (pm, nm) in enumerate(zip(posmasks, negmasks)):
            delta = (u & pm).bit_count() - (u & nm).bit_count()
            if delta & 1:
                good = False
                break
            if (delta // 2) & 1:
                rhs |= 1 << t
        if not good:
            continue
        z0 = sys2.solve(rhs)
        if z0 is None:
            continue

        out_masks = []
        for nb in sys2.nullspace:
            om = 0
            for j, em in enumerate(eval_masks):
                if (em & nb).bit_count() & 1:
                    om |= 1 << j
            out_masks.append(om)
        image = _independent_image_generators(out_masks, sys2.nullspace)

        # Fixed relative phases from u and the particular z0.
        base = np.zeros(len(reps), dtype=np.uint8)
        for jj, r in enumerate(reps[1:], start=1):
            # Integer, not xor, difference: this carries the low Z4 bit correctly.
            delta = 0
            for ii in range(n):
                if (u >> ii) & 1:
                    delta += ((r >> ii) & 1) - ((r0 >> ii) & 1)
            hi = (eval_masks[jj-1] & z0).bit_count() & 1
            base[jj] = (delta + 2*hi) & 3

        for ic in range(1 << len(image)):
            z = z0
            e = base.copy()
            for k, (om, pre) in enumerate(image):
                if (ic >> k) & 1:
                    z ^= pre
                    # Nullspace changes only the high bit.
                    mm = om
                    while mm:
                        b = (mm & -mm).bit_length()-1
                        e[b+1] ^= 2
                        mm &= mm-1
            key = bytes(e.tolist())
            if key in seen:
                continue
            seen.add(key)
            vmask = z & ((1 << n)-1)
            qmask = z >> n
            linear = tuple((((u >> i) & 1) + 2*((vmask >> i) & 1))
                           & 3 for i in range(n))
            yield e, PhaseDescriptor(qmask, linear)


def _build_complete_pool(partition: tuple[int, ...], od: OrbitData,
                         supports: Sequence[AffineSupport], pairs):
    phases = np.asarray([1, 1j, -1, -1j], complex)
    vecs, descs = [], []
    for sup in supports:
        inside = np.asarray([syndrome(sup.rows, int(rep)) == sup.rhs
                             for rep in od.representatives], bool)
        cols = np.flatnonzero(inside)
        support_size = int(round(float(od.sizes[inside].sum())))
        scale = np.sqrt(od.sizes[inside] / support_size)
        for e, pdesc in _complete_phases_on_support(partition, od, sup, pairs):
            v = np.zeros(len(od.weights), complex)
            v[cols] = scale * phases[e]
            vecs.append(v)
            descs.append(CandidateDescriptor(sup, pdesc, support_size))
    C = np.column_stack(vecs) if vecs else np.zeros(
        (len(od.weights), 0), complex)
    return C, descs


def _deduplicate_columns(C: np.ndarray, descs: list[CandidateDescriptor]):
    seen = {}
    keep = []
    for j in range(C.shape[1]):
        # Every vector is globally phase-fixed by the first support orbit.
        key = np.round(np.real(C[:, j]), 13).tobytes() + \
            np.round(np.imag(C[:, j]), 13).tobytes()
        if key not in seen:
            seen[key] = j
            keep.append(j)
    return C[:, keep], [descs[j] for j in keep]


def build_symmetric_pool(partition: Sequence[int], *, phase_mode: str = "complete",
                         verbose: bool = False) -> CandidatePool:
    """Build a pool of partition-symmetric stabilizer states.

    phase_mode='complete' is state-level complete for the standard affine-
    quadratic stabilizer amplitude normal form; phase_mode='manifest' is the
    older fast block-constant-gauge subset.
    """
    t0 = time.perf_counter()
    partition = tuple(int(x) for x in partition)
    if any(x <= 0 for x in partition):
        raise ValueError("partition parts must be positive")
    n = sum(partition)
    od = orbit_data(partition)
    supports = invariant_affine_supports(partition)
    pairs = tuple(combinations(range(n), 2))
    if phase_mode == "manifest":
        C, descs = _build_manifest_pool(partition, od, supports)
    elif phase_mode == "complete":
        C, descs = _build_complete_pool(partition, od, supports, pairs)
    else:
        raise ValueError("phase_mode must be 'complete' or 'manifest'")
    C, descs = _deduplicate_columns(C, descs)
    if verbose:
        print(f"pool partition={partition} mode={phase_mode} supports={len(supports)} "
              f"states={C.shape[1]} orbit_dim={C.shape[0]} time={time.perf_counter()-t0:.3f}s")
    return CandidatePool(partition, C, descs, od, pairs, phase_mode)


# ---------------------------------------------------------------------------
# Target spans
# ---------------------------------------------------------------------------

def orthonormalize_span(T: np.ndarray, tol: float = 1e-11) -> np.ndarray:
    T = np.asarray(T, complex)
    if T.ndim == 1:
        T = T[:, None]
    U, s, _ = np.linalg.svd(T, full_matrices=False)
    r = int(np.sum(s > tol))
    return U[:, :r]


def general_w_target_span(pool: CandidatePool, kind: str) -> np.ndarray:
    n = pool.n
    k = kind.lower().replace("-", "_")
    if k in {"cat", "even", "evencat"}:
        k = "even_cat"
    if k in {"odd", "oddcat"}:
        k = "odd_cat"
    if k == "product":
        layers = tuple(range(n+1))
    elif k == "even_cat":
        layers = tuple(range(0, n+1, 2))
    elif k == "odd_cat":
        layers = tuple(range(1, n+1, 2))
    else:
        raise ValueError("kind must be product, even_cat, or odd_cat")
    T = np.zeros((len(pool.orbit_data.weights), len(layers)), complex)
    for j, w in enumerate(layers):
        den = math.comb(n, w)
        for a, ws in enumerate(pool.orbit_data.weights):
            if sum(ws) == w:
                T[a, j] = math.sqrt(pool.orbit_data.sizes[a]/den)
    return T


def T_target(pool: CandidatePool, kind: str) -> np.ndarray:
    omega = np.exp(1j*np.pi/4)
    n = pool.n
    v = np.zeros(len(pool.orbit_data.weights), complex)
    for a, ws in enumerate(pool.orbit_data.weights):
        w = sum(ws)
        if kind == "product":
            amp = omega**w / (2**(n/2))
        elif kind in {"cat", "even-cat", "even_cat"}:
            amp = 0 if (w & 1) else omega**w / (2**((n-1)/2))
        else:
            raise ValueError("kind must be product or cat")
        v[a] = math.sqrt(pool.orbit_data.sizes[a]) * amp
    v /= np.linalg.norm(v)
    return v[:, None]


# ---------------------------------------------------------------------------
# Generic span score and greedy warm start
# ---------------------------------------------------------------------------

def projection_score(V: np.ndarray, T: np.ndarray) -> float:
    T = orthonormalize_span(T)
    if V.shape[1] == 0:
        return 0.0
    Q, R = np.linalg.qr(V, mode="reduced")
    if R.size and np.min(np.abs(np.diag(R))) < 1e-11:
        U, s, _ = np.linalg.svd(V, full_matrices=False)
        Q = U[:, :int(np.sum(s > 1e-11))]
    return float(np.linalg.norm(Q.conj().T @ T, "fro")**2)


def greedy_initial_decomposition(pool_or_C, target_span: np.ndarray, rank: int,
                                 *, seed: int = 0):
    """Simultaneous OMP: choose R atoms maximizing target-span projection gain."""
    C = pool_or_C.vectors if isinstance(
        pool_or_C, CandidatePool) else np.asarray(pool_or_C, complex)
    T = orthonormalize_span(target_span)
    if rank > C.shape[1]:
        raise ValueError("rank exceeds pool size")
    rng = np.random.default_rng(seed)
    selected = []
    Q = np.zeros((C.shape[0], 0), complex)
    for _ in range(rank):
        U = C - Q @ (Q.conj().T @ C) if Q.shape[1] else C.copy()
        den = np.sum(np.abs(U)**2, axis=0)
        corr = U.conj().T @ T
        gain = np.sum(np.abs(corr)**2, axis=1) / np.maximum(den, 1e-30)
        gain[den < 1e-12] = -np.inf
        if selected:
            gain[selected] = -np.inf
        mx = float(np.max(gain))
        choices = np.flatnonzero(gain >= mx-1e-13)
        j = int(rng.choice(choices))
        selected.append(j)
        Q = np.linalg.qr(C[:, selected], mode="reduced")[0]
    sel = np.asarray(selected, int)
    return sel, projection_score(C[:, sel], T)


@dataclass
class SearchResult:
    partition: tuple[int, ...]
    rank: int
    indices: np.ndarray
    score: float
    target_dim: int
    pool_size: int
    method: str
    seed: int
    sweeps: int
    elapsed: float

    @property
    def defect(self): return self.target_dim - self.score
    @property
    def exact(self): return self.defect < 1e-9


def _all_replacement_scores_gram(G, H, others):
    if len(others) == 0:
        den = np.real(np.diag(G)).copy()
        return np.sum(np.abs(H)**2, axis=1)/np.maximum(den, 1e-30)
    Goo = G[np.ix_(others, others)]
    Ho = H[others, :]
    try:
        if np.linalg.cond(Goo) > 1e10:
            raise np.linalg.LinAlgError
        Y = np.linalg.solve(Goo, Ho)
        X = np.linalg.solve(Goo, G[others, :])
    except np.linalg.LinAlgError:
        K = np.linalg.pinv(Goo, rcond=1e-10, hermitian=True)
        Y = K@Ho
        X = K@G[others, :]
    base = float(np.real(np.sum(np.conj(Ho)*Y)))
    den = 1.0-np.real(np.sum(np.conj(G[others, :])*X, axis=0))
    corr = H-G[:, others]@Y
    out = np.full(G.shape[0], -np.inf, float)
    valid = den > 1e-8
    out[valid] = base+np.sum(np.abs(corr[valid])**2, axis=1)/den[valid]
    return out


def _replacement_scores_direct(C, T, others, candidates):
    if len(others):
        Q = np.linalg.qr(C[:, others], mode='reduced')[0]
        base = float(np.linalg.norm(Q.conj().T@T, 'fro')**2)
        U = C[:, candidates]-Q@(Q.conj().T@C[:, candidates])
    else:
        base = 0.0
        U = C[:, candidates]
    den = np.sum(np.abs(U)**2, axis=0)
    corr = U.conj().T@T
    gain = np.sum(np.abs(corr)**2, axis=1)/np.maximum(den, 1e-30)
    out = base+gain
    out[den < 1e-10] = -np.inf
    return out


def anneal_decomposition(
    pool: CandidatePool, target_span: np.ndarray, rank: int, *,
    initial_indices: Sequence[int] | None = None,
    seed: int = 0, sweeps: int = 300, reheats: int = 5,
    start_temperature: float = 0.08, end_temperature: float = 1e-6,
    exact_tol: float = 1e-10, max_gram_pool: int = 5000,
    batch_size: int = 3000, final_full_polish: bool = True,
    verbose: bool = False,
) -> SearchResult:
    """Heat-bath coordinate annealing for an arbitrary target span.

    For small pools it precomputes C^*C and scores every replacement exactly,
    matching the efficient old implementation.  For large pools it uses exact
    residualized scores on random candidate batches and an optional full-pool
    deterministic polish; this still performs genuine annealing rather than
    silently falling back to greedy.
    """
    t0 = time.perf_counter()
    C = pool.vectors
    T = orthonormalize_span(target_span)
    d = T.shape[1]
    N = C.shape[1]
    rng = np.random.default_rng(seed)
    if initial_indices is None:
        sel, score = greedy_initial_decomposition(pool, T, rank, seed=seed)
    else:
        sel = np.asarray(initial_indices, int).copy()
        score = projection_score(C[:, sel], T)
    best_sel = sel.copy()
    best_score = score
    if d-best_score < exact_tol:
        return SearchResult(pool.partition, rank, best_sel, best_score, d, N, 'initial', seed, 0, time.perf_counter()-t0)

    use_gram = N <= max_gram_pool
    if use_gram:
        G = C.conj().T@C
        H = C.conj().T@T
    total = 0
    for rh in range(max(1, reheats)):
        if rh:
            sel = best_sel.copy()
            for _ in range(max(1, rank//4)):
                p = int(rng.integers(rank))
                j = int(rng.integers(N))
                while j in sel:
                    j = int(rng.integers(N))
                sel[p] = j
        for sw in range(max(1, sweeps)):
            frac = sw/max(1, sweeps-1)
            temp = start_temperature*(end_temperature/start_temperature)**frac
            for pos in rng.permutation(rank):
                others = np.delete(sel, pos)
                if use_gram:
                    scores = _all_replacement_scores_gram(G, H, others)
                    scores[others] = -np.inf
                    cand = np.flatnonzero(np.isfinite(scores))
                    vals = scores[cand]
                else:
                    b = min(batch_size, N)
                    cand = rng.choice(
                        N, size=b, replace=False) if b < N else np.arange(N)
                    cand = np.unique(np.concatenate([cand, [sel[pos]]]))
                    scores = _replacement_scores_direct(C, T, others, cand)
                    bad = np.isin(cand, others)
                    scores[bad] = -np.inf
                    vals = scores
                finite = np.isfinite(vals)
                if not np.any(finite):
                    continue
                cc = cand[finite]
                vv = vals[finite]
                mx = float(np.max(vv))
                logits = np.maximum((vv-mx)/max(temp, 1e-15), -60)
                p = np.exp(logits)
                p /= p.sum()
                j = int(rng.choice(cc, p=p))
                sel[pos] = j
                score = float(vv[np.where(cc == j)[0][0]])
                if score > best_score+1e-13:
                    best_score = score
                    best_sel = sel.copy()
                    if verbose:
                        print(
                            f"  reheat={rh} sweep={sw} defect={d-best_score:.9g}")
                    if d-best_score < exact_tol:
                        return SearchResult(pool.partition, rank, best_sel, best_score, d, N, 'anneal', seed, total+sw+1, time.perf_counter()-t0)
            total += 1

    # Deterministic coordinate descent.  Large-pool mode can scan all atoms.
    sel = best_sel.copy()
    for _ in range(10):
        changed = False
        for pos in range(rank):
            others = np.delete(sel, pos)
            if use_gram:
                scores = _all_replacement_scores_gram(G, H, others)
                scores[others] = -np.inf
                j = int(np.argmax(scores))
                ns = float(scores[j])
            else:
                cand = np.arange(N) if final_full_polish else rng.choice(
                    N, size=min(batch_size, N), replace=False)
                scores = _replacement_scores_direct(C, T, others, cand)
                scores[np.isin(cand, others)] = -np.inf
                jj = int(np.argmax(scores))
                j = int(cand[jj])
                ns = float(scores[jj])
            if ns > best_score+1e-12:
                sel[pos] = j
                best_sel = sel.copy()
                best_score = ns
                changed = True
                if d-best_score < exact_tol:
                    return SearchResult(pool.partition, rank, best_sel, best_score, d, N, 'anneal+polish', seed, total, time.perf_counter()-t0)
        if not changed:
            break
    return SearchResult(pool.partition, rank, best_sel, best_score, d, N, 'anneal+polish', seed, total, time.perf_counter()-t0)


# ---------------------------------------------------------------------------
# Export / exact computational-basis descriptors
# ---------------------------------------------------------------------------

def descriptor_to_Q_l_A_b(pool: CandidatePool, desc: CandidateDescriptor):
    n = pool.n
    Q = np.zeros((n, n), np.uint8)
    for k, (i, j) in enumerate(pool.pair_slots):
        if (desc.phase.qmask >> k) & 1:
            Q[i, j] = 1
    l = np.asarray(desc.phase.linear, dtype=np.uint8)
    A = np.asarray([[(row >> i) & 1 for i in range(n)]
                   for row in desc.support.rows], dtype=np.uint8)
    b = np.asarray([(desc.support.rhs >> i) & 1 for i in range(
        len(desc.support.rows))], dtype=np.uint8)
    return Q, l, A, b


def write_selected_descriptors(path: str | Path, pool: CandidatePool, result: SearchResult,
                               *, header: Sequence[str] = ()):
    lines = list(header)+[
        f"partition = {'+'.join(map(str, pool.partition))}",
        f"phase_mode = {pool.phase_mode}",
        f"rank = {result.rank}", f"score = {result.score:.16g}",
        f"defect = {result.defect:.12g}", f"pool_size = {pool.size}", ""]
    for r, idx in enumerate(result.indices):
        d = pool.descriptors[int(idx)]
        Q, l, A, b = descriptor_to_Q_l_A_b(pool, d)
        lines += [f"Stabiliser {r}:  # pool id {int(idx)}", "Q =", str(Q),
                  f"l = {l}", "A =", str(A), f"b = {b}",
                  f"support_size = {d.support_size}", ""]
    Path(path).write_text("\n".join(lines))


def contains_state(pool: CandidatePool, v: np.ndarray, tol: float = 1e-10):
    v = np.asarray(v, complex)
    v = v/np.linalg.norm(v)
    nz = np.flatnonzero(np.abs(v) > tol)
    if len(nz):
        v = v/(v[nz[0]]/abs(v[nz[0]]))
    errs = np.linalg.norm(pool.vectors-v[:, None], axis=0)
    j = int(np.argmin(errs))
    return float(errs[j]), j


def initial_decomposition(pool: CandidatePool, target_span: np.ndarray, rank: int, *,
                          seed: int = 0, oversample: int = 0,
                          exact_tol: float = 1e-10):
    """Stronger warm start: simultaneous OMP, optionally overshoot and prune.

    ``oversample`` asks OMP for rank+oversample atoms first.  If that span is
    already exact, the routine greedily removes atoms while preserving as much
    target projection as possible until ``rank`` atoms remain.  This is often a
    much better annealing seed than direct rank-R OMP.
    """
    T = orthonormalize_span(target_span)
    R0 = min(pool.size, rank + max(0, int(oversample)))
    sel, score = greedy_initial_decomposition(pool, T, R0, seed=seed)
    sel = list(map(int, sel))
    while len(sel) > rank:
        best_score = -1.0
        best_pos = None
        # R0 is normally small (tens), so exact leave-one-out scoring is cheap.
        for pos in range(len(sel)):
            ss = np.delete(np.asarray(sel, int), pos)
            sc = projection_score(pool.vectors[:, ss], T)
            if sc > best_score:
                best_score, best_pos = sc, pos
        assert best_pos is not None
        sel.pop(int(best_pos))
        score = best_score
    return np.asarray(sel, int), float(score)
