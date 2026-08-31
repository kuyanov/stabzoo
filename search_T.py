#!/usr/bin/env python3
"""Search fixed-T product / cat stabilizer decompositions.

The search target uses omega=exp(i*pi/4).  For an exact numerical hit the
selected stabilizers are converted to a complete exact certificate with one
coefficient in Q(sqrt(2),i) per term, then independently verified.

Examples
--------
python search_T.py --partition 4+5 --kind even_cat --rank 9
python search_T.py --partition 2+5 --kind product --rank 9 --sweeps 600 --reheats 10
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from stab_search import (
    build_symmetric_pool, T_target, initial_decomposition,
    anneal_decomposition, pair_polish_decomposition, SearchResult,
)
from T_exact import (
    canonical_t_kind, exactify_t_pool_selection, terms_from_pool, write_t_witness,
)


def parse_partition(s):
    return tuple(int(x) for x in s.replace(',', '+').split('+') if x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--partition', required=True)
    ap.add_argument('--kind', choices=['product', 'even_cat', 'odd_cat', 'cat'], required=True,
                    help='cat is an alias for even_cat')
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument(
        '--phase-mode', choices=['complete', 'manifest'], default='complete')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--oversample', type=int, default=2)
    ap.add_argument('--sweeps', type=int, default=400)
    ap.add_argument('--reheats', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=3000)
    ap.add_argument('--max-gram-pool', type=int, default=5000)
    ap.add_argument('--greedy-only', action='store_true')
    ap.add_argument('--no-pair-polish', action='store_true',
                    help='disable deterministic two-coordinate polish after annealing')
    ap.add_argument('--pair-candidates', type=int, default=40,
                    help='candidate shortlist per removed pair (default: 40)')
    ap.add_argument('--output', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    kind = canonical_t_kind(args.kind)
    core_kind = 'cat' if kind == 'even_cat' else kind
    part = parse_partition(args.partition)
    pool = build_symmetric_pool(part, phase_mode=args.phase_mode, verbose=True)
    T = T_target(pool, core_kind)
    sel, score = initial_decomposition(
        pool, T, args.rank, seed=args.seed, oversample=args.oversample)
    defect = 1-score
    print(f'initial: rank={args.rank} score={score:.15g} defect={defect:.12g}')
    if not args.greedy_only and defect > 1e-10:
        res = anneal_decomposition(pool, T, args.rank, initial_indices=sel, seed=args.seed,
                                   sweeps=args.sweeps, reheats=args.reheats, batch_size=args.batch_size,
                                   max_gram_pool=args.max_gram_pool, verbose=args.verbose)
    else:
        res = SearchResult(pool.partition, args.rank, sel,
                           score, 1, pool.size, 'initial', args.seed, 0, 0.0)
    # Fixed-T instances can have genuine two-coordinate barriers.  For modest
    # pools, run a deterministic pair polish after the one-coordinate annealer.
    if (not args.no_pair_polish) and (not res.exact):
        psel, pscore = pair_polish_decomposition(
            pool, T, res.indices, candidate_shortlist=args.pair_candidates, verbose=args.verbose)
        if pscore > res.score + 1e-12:
            res = SearchResult(pool.partition, args.rank, psel, pscore, 1, pool.size,
                               res.method+'+pair-polish', args.seed, res.sweeps, res.elapsed)
    print(f'final:   rank={res.rank} score={res.score:.15g} defect={res.defect:.12g} '
          f'exact={res.exact} method={res.method}')

    if res.exact:
        print('exactifying over Q(sqrt(2),i) ...')
        terms, coeff = exactify_t_pool_selection(pool, res.indices, kind)
        exact_status = 'true'
        print('symbolic verification: PASS')
    else:
        terms = terms_from_pool(pool, res.indices)
        coeff = None
        exact_status = 'false'
        print('symbolic verification skipped: numerical span is not exact')

    if res.exact:
        out = args.output or f"T_{kind}/n{sum(part)}_r{args.rank}_{''.join(map(str, part))}.txt"
        write_t_witness(out, target=kind, partition=part, phase_mode=args.phase_mode,
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
            'pool_size': pool.size, 'score': res.score, 'defect': res.defect, 'exact': res.exact,
            'indices': [int(x) for x in res.indices],
        }, indent=2)+'\n')


if __name__ == '__main__':
    main()
