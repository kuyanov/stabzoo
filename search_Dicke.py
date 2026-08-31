#!/usr/bin/env python3
"""Search a stabilizer decomposition of a Dicke state |D_k^n>.

Only n, k and the requested stabilizer rank R are required.  By default the
script scans inequivalent Young-subgroup partitions with at most three blocks.
You can instead fix a partition explicitly, e.g. ``--partition 2+3+5``.

The search uses the corrected *state-level* symmetric stabilizer pool from
``symmetric_stabilizer_search_v5.py``.  For every partition it runs

    simultaneous OMP -> heat-bath coordinate annealing -> pair polish.

If an exact numerical span is found, the selected (Q,l,A,b) terms are solved
and verified exactly over Q(i), using the unnormalised Dicke target

    sum_{|x|=k} |x>.

Examples
--------
python search_Dicke.py --n 6 --k 2 --rank 3
python search_Dicke.py --n 8 --k 4 --rank 3 --partition 4+4
python search_Dicke.py --n 10 --k 4 --rank 11 --max-groups 3 --scan-all
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from stab_search import (
    SearchResult,
    anneal_decomposition,
    build_symmetric_pool,
    descriptor_to_Q_l_A_b,
    initial_decomposition,
    pair_polish_decomposition,
)


def parse_partition(s: str) -> tuple[int, ...]:
    p = tuple(int(x) for x in s.replace(',', '+').split('+') if x)
    if not p or any(x <= 0 for x in p):
        raise ValueError('partition parts must be positive')
    return p


def integer_partitions_fixed_length(n: int, m: int, lo: int = 1):
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


def candidate_partitions(n: int, max_groups: int) -> list[tuple[int, ...]]:
    parts = []
    for m in range(1, max_groups + 1):
        parts.extend(integer_partitions_fixed_length(n, m))
    # Empirically, moderately unbalanced partitions are often useful.  Prioritise
    # smaller orbit dimensions first; use imbalance only as a tie-break.
    parts.sort(key=lambda p: (
        math.prod(x + 1 for x in p), max(p) - min(p), len(p), p))
    return parts


def dicke_target(pool, k: int) -> np.ndarray:
    """Normalised |D_k^n> in the pool's normalised block-Dicke basis."""
    n = pool.n
    if not (0 <= k <= n):
        raise ValueError('need 0 <= k <= n')
    den = math.comb(n, k)
    v = np.zeros(len(pool.orbit_data.weights), dtype=np.complex128)
    for j, ws in enumerate(pool.orbit_data.weights):
        if sum(ws) == k:
            v[j] = math.sqrt(pool.orbit_data.sizes[j] / den)
    return v[:, None]


def exactify_and_verify(pool, indices: np.ndarray, k: int):
    """Solve exact coefficients over Q(i) for the unnormalised Dicke state."""
    import sympy as sp

    I = sp.I
    roots = (sp.Integer(1), I, -sp.Integer(1), -I)
    terms = []
    for idx in indices:
        Q, l, A, b = descriptor_to_Q_l_A_b(pool, pool.descriptors[int(idx)])
        terms.append((Q.astype(int), l.astype(int),
                     A.astype(int), b.astype(int)))

    # Solve on block-weight orbit representatives.  The pool states are
    # state-level partition-symmetric, so this is sufficient for solving.
    rows = []
    rhs = []
    for rep, ws in zip(pool.orbit_data.representatives, pool.orbit_data.weights):
        z = int(rep)
        row = []
        for Q, l, A, b in terms:
            x = [(z >> i) & 1 for i in range(pool.n)]
            ok = all(sum(int(rr[i]) * x[i] for i in range(pool.n)) % 2 == int(bb)
                     for rr, bb in zip(A, b))
            if not ok:
                row.append(sp.Integer(0))
                continue
            e = sum(int(l[i]) * x[i] for i in range(pool.n))
            for i in range(pool.n):
                for j in range(i + 1, pool.n):
                    e += 2 * int(Q[i, j]) * x[i] * x[j]
            row.append(roots[e & 3])
        rows.append(row)
        rhs.append(sp.Integer(1 if sum(ws) == k else 0))

    M = sp.Matrix(rows)
    y = sp.Matrix(rhs)
    solset = sp.linsolve((M, y))
    if solset is sp.EmptySet or solset == sp.EmptySet:
        raise RuntimeError('selected numerical span did not solve exactly')
    sol = next(iter(solset))
    free = sorted(set().union(*(c.free_symbols for c in sol)), key=str)
    sub = {s: sp.Integer(0) for s in free}
    coeff = [sp.simplify(c.subs(sub)) for c in sol]

    # Independent full 2^n exact verification.
    for z in range(1 << pool.n):
        x = [(z >> i) & 1 for i in range(pool.n)]
        lhs = sp.Integer(0)
        for c, (Q, l, A, b) in zip(coeff, terms):
            ok = all(sum(int(rr[i]) * x[i] for i in range(pool.n)) % 2 == int(bb)
                     for rr, bb in zip(A, b))
            if not ok:
                continue
            e = sum(int(l[i]) * x[i] for i in range(pool.n))
            for i in range(pool.n):
                for j in range(i + 1, pool.n):
                    e += 2 * int(Q[i, j]) * x[i] * x[j]
            lhs += c * roots[e & 3]
        target = sp.Integer(1 if z.bit_count() == k else 0)
        if sp.simplify(lhs - target) != 0:
            raise AssertionError(
                f'exact verification failed at basis index {z}')
    return terms, coeff


def write_certificate(path: str | Path, *, pool, result: SearchResult, k: int,
                      terms=None, coefficients=None):
    path = Path(path)
    lines = [
        'format = symmetric_stabilizer_Dicke_v1',
        f'n = {pool.n}',
        f'k = {k}',
        f'partition = {"+".join(map(str, pool.partition))}',
        f'phase_mode = {pool.phase_mode}',
        f'rank = {result.rank}',
        f'score = {result.score:.16g}',
        f'defect = {result.defect:.12g}',
        f'pool_size = {pool.size}',
        f'search_method = {result.method}',
        f'seed = {result.seed}',
        f'verified_exact = {str(coefficients is not None).lower()}',
        'convention = unnormalized',
        '',
    ]
    if terms is None:
        terms = []
        for idx in result.indices:
            Q, l, A, b = descriptor_to_Q_l_A_b(
                pool, pool.descriptors[int(idx)])
            terms.append((Q, l, A, b))
    for r, (idx, term) in enumerate(zip(result.indices, terms)):
        Q, l, A, b = term
        lines += [f'Stabiliser {r}:', f'pool_id = {int(idx)}']
        if coefficients is not None:
            lines.append(f'coef = {coefficients[r]}')
        lines += [
            'Q = ' + repr(np.asarray(Q, dtype=int).tolist()),
            'l = ' + repr(np.asarray(l, dtype=int).tolist()),
            'A = ' + repr(np.asarray(A, dtype=int).tolist()),
            'b = ' + repr(np.asarray(b, dtype=int).tolist()),
            '',
        ]
    path.write_text('\n'.join(lines))


def search_partition(partition, *, k, rank, args):
    t0 = time.perf_counter()
    pool = build_symmetric_pool(
        partition, phase_mode=args.phase_mode, verbose=args.verbose)
    T = dicke_target(pool, k)
    sel, score = initial_decomposition(pool, T, rank, seed=args.seed,
                                       oversample=args.oversample)
    if args.verbose:
        print(f'  initial defect={1-score:.12g}')
    if (1 - score > args.tol) and not args.greedy_only:
        res = anneal_decomposition(
            pool, T, rank, initial_indices=sel, seed=args.seed,
            sweeps=args.sweeps, reheats=args.reheats,
            batch_size=args.batch_size, max_gram_pool=args.max_gram_pool,
            exact_tol=args.tol, verbose=args.verbose,
        )
    else:
        res = SearchResult(pool.partition, rank, sel, score, 1, pool.size,
                           'initial', args.seed, 0, time.perf_counter() - t0)

    if (not args.no_pair_polish) and (not res.exact):
        psel, pscore = pair_polish_decomposition(
            pool, T, res.indices, candidate_shortlist=args.pair_candidates,
            max_passes=args.pair_passes, exact_tol=args.tol, verbose=args.verbose)
        if pscore > res.score + 1e-12:
            res = SearchResult(pool.partition, rank, psel, pscore, 1, pool.size,
                               res.method + '+pair-polish', args.seed,
                               res.sweeps, time.perf_counter() - t0)
    return pool, res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--rank', '-r', type=int, required=True)
    ap.add_argument('--partition', default=None,
                    help='fix one partition, e.g. 2+3+5; otherwise scan partitions')
    ap.add_argument('--max-groups', type=int, default=3,
                    help='when --partition is omitted, scan up to this many groups (default 3)')
    ap.add_argument('--scan-all', action='store_true',
                    help='do not stop after the first exact hit')
    ap.add_argument(
        '--phase-mode', choices=['complete', 'manifest'], default='complete')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--oversample', type=int, default=2)
    ap.add_argument('--sweeps', type=int, default=400)
    ap.add_argument('--reheats', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=3000)
    ap.add_argument('--max-gram-pool', type=int, default=5000)
    ap.add_argument('--pair-candidates', type=int, default=50)
    ap.add_argument('--pair-passes', type=int, default=4)
    ap.add_argument('--no-pair-polish', action='store_true')
    ap.add_argument('--greedy-only', action='store_true')
    ap.add_argument('--tol', type=float, default=1e-10)
    ap.add_argument('--output', default=None,
                    help='certificate path; for --scan-all exact hits get partition suffixes')
    ap.add_argument('--no-exactify', action='store_true',
                    help='skip SymPy exact coefficient reconstruction')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    if args.n < 1 or not (0 <= args.k <= args.n):
        ap.error('need n>=1 and 0<=k<=n')
    if args.rank < 1:
        ap.error('rank must be positive')

    # Use X^tensor n symmetry only to reduce a redundant search label, not to
    # change the requested certificate.
    if args.partition:
        p = parse_partition(args.partition)
        if sum(p) != args.n:
            ap.error('--partition must sum to n')
        partitions = [p]
    else:
        partitions = candidate_partitions(args.n, args.max_groups)

    print(f'Searching |D_{args.k}^{args.n}> at rank {args.rank}')
    print('partitions:', ', '.join('+'.join(map(str, p)) for p in partitions))

    best = None
    exact_hits = 0
    for ip, part in enumerate(partitions):
        print(
            f'[{ip+1}/{len(partitions)}] partition={"+".join(map(str, part))}', flush=True)
        try:
            pool, res = search_partition(
                part, k=args.k, rank=args.rank, args=args)
        except MemoryError:
            print('  skipped: MemoryError')
            continue
        print(f'  pool={pool.size} method={res.method} score={res.score:.15g} '
              f'defect={res.defect:.12g} exact={res.exact}')
        if best is None or res.score > best[1].score:
            best = (pool, res)

        if res.exact:
            exact_hits += 1
            terms = coeff = None
            if not args.no_exactify:
                print('  exactifying over Q(i) ...', flush=True)
                terms, coeff = exactify_and_verify(pool, res.indices, args.k)
                print('  symbolic verification: PASS')
                print('  coefficients:', ', '.join(str(c) for c in coeff))
            base = args.output or f'Dicke/n{args.n}_k{args.k}_r{args.rank}.txt'
            if args.scan_all:
                pp = ''.join(map(str, part))
                path = str(Path(base).with_name(
                    Path(base).stem + f'_{pp}' + Path(base).suffix))
            else:
                path = base
            write_certificate(path, pool=pool, result=res, k=args.k,
                              terms=terms, coefficients=coeff)
            print('  wrote', Path(path).resolve())
            if not args.scan_all:
                return

    if exact_hits:
        print(f'finished: {exact_hits} exact hit(s)')
    elif best is not None:
        pool, res = best
        print('NO EXACT HIT')
        print(f'best partition={"+".join(map(str, pool.partition))} '
              f'defect={res.defect:.12g} pool={pool.size}')
        if args.output:
            write_certificate(args.output, pool=pool, result=res, k=args.k)
            print('wrote best numerical near-miss to',
                  Path(args.output).resolve())


if __name__ == '__main__':
    main()
