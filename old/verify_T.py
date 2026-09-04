#!/usr/bin/env python3
"""Exact verifier for fixed-T stabilizer decomposition certificates.

Examples
--------
python verify_T.py witness.txt
python verify_T.py witness.txt --mode product
python verify_T.py ./results --mode even_cat --glob 'T_*.txt'

For each file the verifier independently:
  1. parses the displayed (Q,l,A,b) stabilizers;
  2. checks state-level symmetry under the declared partition;
  3. solves exact coefficients over Q(sqrt(2),i);
  4. verifies every partition orbit exactly;
  5. compares with the stored coef= lines when present.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import sympy as sp

from general_w_exact import validate_term_symmetry
from T_exact import (
    canonical_t_kind, parse_t_witness, solve_t_coefficients_exact,
    verify_t_coefficients, format_t_expr,
)


def verify_file(path: Path, mode: str | None, show: bool, compare_stored: bool):
    wit = parse_t_witness(path)
    kind = wit.target if mode is None else canonical_t_kind(mode)
    validate_term_symmetry(wit.terms, wit.partition)
    coeff = solve_t_coefficients_exact(wit.terms, wit.partition, kind)
    verify_t_coefficients(wit.terms, wit.partition, kind, coeff)

    stored_ok = None
    if compare_stored:
        have = all(t.coefficient is not None for t in wit.terms)
        if have:
            stored_ok = all(
                sp.simplify(sp.radsimp(t.coefficient - c)) == 0
                for t, c in zip(wit.terms, coeff)
            )
            if not stored_ok:
                try:
                    verify_t_coefficients(wit.terms, wit.partition, kind,
                                          [t.coefficient for t in wit.terms])
                    stored_ok = 'different-but-valid'
                except Exception:
                    stored_ok = False

    print(f"PASS {path}: mode={kind} n={wit.n} rank={wit.rank} "
          f"partition={'+'.join(map(str, wit.partition))}" +
          ('' if stored_ok is None else f' stored_coefficients={stored_ok}'))
    if show:
        for r, c in enumerate(coeff):
            print(f'  c[{r}] = {format_t_expr(c)}')
    return coeff


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path', help='one fixed-T witness .txt file or a folder')
    ap.add_argument('--mode', choices=['product', 'even_cat', 'odd_cat'], default=None,
                    help='override mode; default is target recorded in each file')
    ap.add_argument('--glob', default='*.txt', help='folder-mode glob')
    ap.add_argument('--show-coefficients', action='store_true',
                    help='print solved coefficients for every folder entry')
    ap.add_argument('--quiet-coefficients', action='store_true',
                    help='for a single file, suppress the default coefficient listing')
    ap.add_argument('--no-compare-stored', action='store_true',
                    help='do not compare with stored coef= lines')
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_file():
        verify_file(p, args.mode, not args.quiet_coefficients,
                    not args.no_compare_stored)
        return
    if not p.is_dir():
        raise SystemExit(f'not a file or directory: {p}')
    files = sorted(x for x in p.glob(args.glob) if x.is_file())
    if not files:
        raise SystemExit(f'no files matched {args.glob!r} in {p}')
    failed = []
    for f in files:
        try:
            verify_file(f, args.mode, args.show_coefficients,
                        not args.no_compare_stored)
        except Exception as e:
            failed.append((f, e))
            print(f'FAIL {f}: {e}', file=sys.stderr)
    print(f'\nverified {len(files)-len(failed)}/{len(files)} files')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
