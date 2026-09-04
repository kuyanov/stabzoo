#!/usr/bin/env python3
"""Search universal stabilizer decompositions for general-w states.

Targets
-------
product:
    (|0> + w|1>)^n

even_cat:
    sum_{|x| even} w^|x| |x>

odd_cat:
    sum_{|x| odd} w^|x| |x>

A zero numerical defect means the selected stabilizers span the corresponding
whole family.  For an exact numerical hit this script reconstructs coefficient
polynomials over Q(i), verifies them symbolically, and writes a complete v2
certificate containing coef(w), Q, l, A and b for every term.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from stab_search import (
    build_symmetric_pool, general_w_target_span, initial_decomposition,
    anneal_decomposition, SearchResult,
)
from general_w_exact import canonical_kind, exactify_pool_selection, terms_from_pool, write_witness


def parse_partition(s: str):
    return tuple(int(x) for x in s.replace(',', '+').split('+') if x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--partition', required=True, help='e.g. 2+3+5')
    ap.add_argument('--kind', choices=['product', 'even_cat', 'odd_cat', 'cat'], required=True,
                    help='cat is accepted as an alias for even_cat')
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--phase-mode', choices=['complete', 'manifest'], default='complete',
                    help='complete includes hidden-gauge symmetric stabilizers')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--oversample', type=int, default=4,
                    help='OMP warm start uses rank+oversample atoms, then prunes')
    ap.add_argument('--sweeps', type=int, default=250)
    ap.add_argument('--reheats', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=3000)
    ap.add_argument('--max-gram-pool', type=int, default=5000)
    ap.add_argument('--greedy-only', action='store_true')
    ap.add_argument('--output', default=None, help='certificate .txt output')
    ap.add_argument('--json', default=None, help='optional JSON summary')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    kind = canonical_kind(args.kind)
    part = parse_partition(args.partition)
    pool = build_symmetric_pool(part, phase_mode=args.phase_mode, verbose=True)
    T = general_w_target_span(pool, kind)
    sel, score = initial_decomposition(
        pool, T, args.rank, seed=args.seed, oversample=args.oversample)
    defect = T.shape[1]-score
    print(f'initial: rank={args.rank} score={score:.15g} defect={defect:.12g}')

    if not args.greedy_only and defect > 1e-10:
        res = anneal_decomposition(
            pool, T, args.rank, initial_indices=sel, seed=args.seed,
            sweeps=args.sweeps, reheats=args.reheats, batch_size=args.batch_size,
            max_gram_pool=args.max_gram_pool, verbose=args.verbose,
        )
    else:
        res = SearchResult(pool.partition, args.rank, sel, score, T.shape[1], pool.size,
                           'initial', args.seed, 0, 0.0)

    print(f'final:   rank={res.rank} score={res.score:.15g} defect={res.defect:.12g} '
          f'exact={res.exact} method={res.method}')

    if res.exact:
        print('exactifying over Q(i)[w] ...')
        terms, coeff = exactify_pool_selection(pool, res.indices, kind)
        exact_status = 'true'
        print('symbolic verification: PASS')
    else:
        terms = terms_from_pool(pool, res.indices)
        coeff = None
        exact_status = 'false'
        print('symbolic verification skipped: numerical span is not exact')

    if res.exact:
        out = args.output or f"general_{kind}/n{sum(part)}_r{args.rank}_{''.join(map(str, part))}.txt"
        write_witness(out, target=kind, partition=part, phase_mode=args.phase_mode,
                      terms=terms, coefficients=coeff, metadata={
                          'score': f'{res.score:.16g}', 'defect': f'{res.defect:.12g}',
                          'pool_size': pool.size, 'search_method': res.method,
                          'seed': args.seed, 'verified_exact': exact_status,
                      })
        print('wrote', Path(out).resolve())
        print('SUCCESS')
    else:
        print('FAIL')
    if args.json:
        Path(args.json).write_text(json.dumps({
            'partition': part, 'kind': kind, 'rank': args.rank, 'phase_mode': args.phase_mode,
            'pool_size': pool.size, 'score': res.score, 'defect': res.defect,
            'exact': res.exact, 'indices': [int(x) for x in res.indices],
        }, indent=2)+'\n')


if __name__ == '__main__':
    main()
