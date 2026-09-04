#!/usr/bin/env python3
"""Exact coefficient reconstruction and verification for fixed-T witnesses.

The certificate convention uses *unnormalized* stabilizer amplitudes

    s_r(x) = (-1)^(x^T Q_r x) i^(l_r^T x) [A_r x = b_r],

and unnormalized fixed-T targets with

    omega = exp(i*pi/4) = (1+i)/sqrt(2).

Thus all exact coefficients live in Q(sqrt(2), i).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import ast
import re

import numpy as np
import sympy as sp

from general_w_exact import (
    ExactTerm, term_amplitude, validate_term_symmetry,
    orbit_weights, orbit_representative, parse_partition,
)

I = sp.I
SQRT2 = sp.sqrt(2)
OMEGA = (sp.Integer(1) + I) / SQRT2


def canonical_t_kind(kind: str) -> str:
    k = kind.strip().lower().replace('-', '_')
    aliases = {
        'cat': 'even_cat', 'even': 'even_cat', 'evencat': 'even_cat',
        'odd': 'odd_cat', 'oddcat': 'odd_cat', 'prod': 'product',
    }
    k = aliases.get(k, k)
    if k not in {'product', 'even_cat', 'odd_cat'}:
        raise ValueError(f'unknown T target kind {kind!r}')
    return k


def target_amplitude(weight: int, kind: str) -> sp.Expr:
    kind = canonical_t_kind(kind)
    if kind == 'product':
        return sp.expand(OMEGA ** int(weight))
    if kind == 'even_cat':
        return sp.Integer(0) if (weight & 1) else sp.expand(OMEGA ** int(weight))
    return sp.Integer(0) if not (weight & 1) else sp.expand(OMEGA ** int(weight))


@dataclass
class TWitness:
    path: Path | None
    target: str
    n: int
    partition: tuple[int, ...]
    rank: int
    phase_mode: str
    terms: list[ExactTerm]
    metadata: dict[str, str]


def _parse_list(text: str):
    return ast.literal_eval(text.strip())


def _parse_expr(text: str) -> sp.Expr | None:
    t = text.strip()
    if t in {'?', 'UNAVAILABLE', 'NONE', 'None'}:
        return None
    return sp.simplify(sp.sympify(
        t.replace('^', '**'),
        locals={'I': I, 'sqrt': sp.sqrt, 'sqrt2': SQRT2, 'omega': OMEGA},
    ))


def parse_t_witness(path: str | Path) -> TWitness:
    path = Path(path)
    lines = path.read_text().splitlines()
    metadata: dict[str, str] = {}
    terms: list[ExactTerm] = []
    cur: dict[str, object] | None = None

    def finish():
        nonlocal cur
        if cur is None:
            return
        missing = {'Q', 'l', 'A', 'b'} - set(cur)
        if missing:
            raise ValueError(
                f'{path}: incomplete stabilizer, missing {sorted(missing)}')
        terms.append(ExactTerm(
            pool_id=cur.get('pool_id'),
            Q=cur['Q'], l=cur['l'], A=cur['A'], b=cur['b'],
            support_size=cur.get('support_size'), coefficient=cur.get('coef'),
        ))
        cur = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if re.fullmatch(r'Stabiliser\s+\d+:', line):
            finish()
            cur = {}
            continue
        if '=' not in line:
            continue
        key, value = (x.strip() for x in line.split('=', 1))
        if cur is None:
            metadata[key] = value
        else:
            if key == 'pool_id':
                cur[key] = int(value)
            elif key in {'Q', 'l', 'A', 'b'}:
                cur[key] = _parse_list(value)
            elif key == 'support_size':
                cur[key] = int(value)
            elif key == 'coef':
                cur[key] = _parse_expr(value)
            else:
                cur[key] = value
    finish()

    if metadata.get('format') != 'symmetric_stabilizer_T_v2':
        raise ValueError(f'{path}: unsupported/missing format marker')
    target = canonical_t_kind(metadata['target'])
    partition = parse_partition(metadata['partition'])
    n = int(metadata.get('n', sum(partition)))
    rank = int(metadata['rank'])
    if n != sum(partition):
        raise ValueError(f'{path}: n={n} != sum(partition)={sum(partition)}')
    if rank != len(terms):
        raise ValueError(f'{path}: rank={rank} but parsed {len(terms)} terms')
    return TWitness(path, target, n, partition, rank,
                    metadata.get('phase_mode', 'unknown'), terms, metadata)


def _orbit_matrix(terms: Sequence[ExactTerm], partition: Sequence[int]):
    weights = orbit_weights(partition)
    reps = [orbit_representative(partition, ws) for ws in weights]
    V = sp.Matrix([[term_amplitude(t, x) for t in terms] for x in reps])
    return weights, V


def solve_t_coefficients_exact(terms: Sequence[ExactTerm], partition: Sequence[int], kind: str):
    """Solve one exact coefficient per stabilizer over Q(sqrt(2), i)."""
    kind = canonical_t_kind(kind)
    weights, V = _orbit_matrix(terms, partition)
    target = sp.Matrix([target_amplitude(sum(ws), kind) for ws in weights])

    _, pivot_cols = V.rref()
    piv = list(map(int, pivot_cols))
    if not piv:
        raise ValueError('empty stabilizer span')
    Vr = V[:, piv]
    _, pivot_rows = Vr.T.rref()
    prows = list(map(int, pivot_rows))
    if len(prows) != len(piv):
        raise AssertionError('failed to choose an independent square minor')
    S = Vr.extract(prows, range(len(piv)))
    tr = target.extract(prows, [0])
    cr = S.inv() * tr
    residual = Vr * cr - target
    if any(sp.simplify(z) != 0 for z in residual):
        raise ValueError(
            'selected stabilizers do not exactly span the requested fixed-T target')

    coeff = [sp.Integer(0)] * len(terms)
    for j, col in enumerate(piv):
        coeff[col] = sp.simplify(sp.radsimp(cr[j, 0]))
    return coeff


def verify_t_coefficients(terms: Sequence[ExactTerm], partition: Sequence[int], kind: str,
                          coefficients: Sequence[sp.Expr]) -> None:
    kind = canonical_t_kind(kind)
    weights, V = _orbit_matrix(terms, partition)
    for i, ws in enumerate(weights):
        lhs = sp.simplify(sum(V[i, r] * coefficients[r]
                          for r in range(len(terms))))
        rhs = target_amplitude(sum(ws), kind)
        diff = sp.simplify(sp.radsimp(lhs - rhs))
        if diff != 0:
            raise ValueError(f'orbit {ws}: fixed-T amplitude mismatch: {diff}')


def format_t_expr(expr: sp.Expr | None) -> str:
    if expr is None:
        return '?'
    e = sp.simplify(sp.radsimp(expr))
    return str(e).replace('**', '^')


def _list_repr(x) -> str:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return repr(x)


def terms_from_pool(pool, indices: Iterable[int]) -> list[ExactTerm]:
    from stab_search import descriptor_to_Q_l_A_b
    out = []
    for idx in map(int, indices):
        d = pool.descriptors[idx]
        Q, l, A, b = descriptor_to_Q_l_A_b(pool, d)
        out.append(ExactTerm(
            idx, Q.astype(int).tolist(), l.astype(int).tolist(),
            A.astype(int).tolist(), b.astype(
                int).tolist(), int(d.support_size), None,
        ))
    return out


def exactify_t_pool_selection(pool, indices: Iterable[int], kind: str):
    terms = terms_from_pool(pool, indices)
    validate_term_symmetry(terms, pool.partition)
    coeff = solve_t_coefficients_exact(terms, pool.partition, kind)
    verify_t_coefficients(terms, pool.partition, kind, coeff)
    return terms, coeff


def write_t_witness(path: str | Path, *, target: str, partition: Sequence[int],
                    phase_mode: str, terms: Sequence[ExactTerm],
                    coefficients: Sequence[sp.Expr] | None,
                    metadata: dict[str, object] | None = None) -> None:
    target = canonical_t_kind(target)
    n = sum(partition)
    lines = [
        'format = symmetric_stabilizer_T_v2',
        f'target = {target}',
        f'n = {n}',
        f"partition = {'+'.join(map(str, partition))}",
        f'phase_mode = {phase_mode}',
        f'rank = {len(terms)}',
        'omega = (1+I)/sqrt(2) = exp(I*pi/4)',
        'convention = unnormalized',
        'term(x) = coef * (-1)^(x^T Q x) * i^(l^T x) * [A x = b]',
    ]
    if metadata:
        reserved = {'format', 'target', 'n', 'partition',
                    'phase_mode', 'rank', 'omega', 'convention'}
        for k, v in metadata.items():
            if k not in reserved:
                lines.append(f'{k} = {v}')
    lines.append('')
    for r, term in enumerate(terms):
        coef = None if coefficients is None else coefficients[r]
        lines += [
            f'Stabiliser {r}:',
            f'pool_id = {term.pool_id if term.pool_id is not None else -1}',
            f'coef = {format_t_expr(coef)}',
            f'Q = {_list_repr(term.Q)}',
            f'l = {_list_repr(term.l)}',
            f'A = {_list_repr(term.A)}',
            f'b = {_list_repr(term.b)}',
            f'support_size = {term.support_size if term.support_size is not None else -1}',
            '',
        ]
    Path(path).write_text('\n'.join(lines))
