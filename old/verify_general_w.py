#!/usr/bin/env python3
"""Exact verifier for general-w stabilizer decomposition certificates.

Examples
--------
python verify_general_w.py witness.txt
python verify_general_w.py witness.txt --mode odd_cat
python verify_general_w.py ./results --mode even_cat
python verify_general_w.py ./results --glob 'general_w_*.txt' --show-coefficients

For each file the verifier:
  1. parses the explicit (Q,l,A,b) stabilizers;
  2. checks each state is invariant under the declared partition;
  3. solves the universal coefficient polynomials exactly over Q(i);
  4. checks every block-weight orbit coefficient-wise as a polynomial in w;
  5. optionally compares the solved coefficients with those stored in the file.

Because step 2 verifies state-level partition symmetry on every computational
basis string, step 4 is equivalent to a full 2^n-amplitude symbolic check.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import sympy as sp

from general_w_exact import (
    canonical_kind, parse_witness, solve_coefficients_exact,
    validate_term_symmetry, verify_coefficients, format_expr,
)


def verify_file(path: Path, mode: str | None, show: bool, compare_stored: bool):
    wit = parse_witness(path)
    kind = wit.target if mode is None else canonical_kind(mode)
    validate_term_symmetry(wit.terms, wit.partition)
    coeff, _, _ = solve_coefficients_exact(wit.terms, wit.partition, kind)
    verify_coefficients(wit.terms, wit.partition, kind, coeff)

    stored_ok = None
    if compare_stored:
        have = all(t.coefficient is not None for t in wit.terms)
        if have:
            stored_ok = all(sp.expand(t.coefficient - c) == 0 for t, c in zip(wit.terms, coeff))
            # A different exact coefficient choice can occur for dependent atoms.
            if not stored_ok:
                try:
                    verify_coefficients(wit.terms, wit.partition, kind,
                                        [t.coefficient for t in wit.terms])
                    stored_ok = "different-but-valid"
                except Exception:
                    stored_ok = False

    print(f"PASS {path}: mode={kind} n={wit.n} rank={wit.rank} partition={'+'.join(map(str,wit.partition))}" +
          ("" if stored_ok is None else f" stored_coefficients={stored_ok}"))
    if show:
        for r, c in enumerate(coeff):
            print(f"  c[{r}] = {format_expr(c)}")
    return coeff


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path', help='one witness .txt file or a folder')
    ap.add_argument('--mode', choices=['product','even_cat','odd_cat'], default=None,
                    help='override target mode; default is the target recorded in each file')
    ap.add_argument('--glob', default='*.txt', help='folder mode glob (default: *.txt)')
    ap.add_argument('--show-coefficients', action='store_true',
                    help='print solved coefficient polynomials for every file')
    ap.add_argument('--quiet-coefficients', action='store_true',
                    help='for a single file, suppress the default coefficient listing')
    ap.add_argument('--no-compare-stored', action='store_true',
                    help='do not compare solved coefficients with coef= lines in the file')
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_file():
        verify_file(p, args.mode, not args.quiet_coefficients, not args.no_compare_stored)
        return
    if not p.is_dir():
        raise SystemExit(f"not a file or directory: {p}")
    files = sorted(x for x in p.glob(args.glob) if x.is_file())
    if not files:
        raise SystemExit(f"no files matched {args.glob!r} in {p}")
    failed = []
    for f in files:
        try:
            verify_file(f, args.mode, args.show_coefficients, not args.no_compare_stored)
        except Exception as e:
            failed.append((f, e))
            print(f"FAIL {f}: {e}", file=sys.stderr)
    print(f"\nverified {len(files)-len(failed)}/{len(files)} files")
    if failed:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
