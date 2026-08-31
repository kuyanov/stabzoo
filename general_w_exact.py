#!/usr/bin/env python3
"""Exact coefficient reconstruction and verification for general-w witnesses.

The text format used here is intentionally simple and strict.  Stabilizers are
*unnormalized* computational-basis functions

    s_r(x) = (-1)^(x^T Q_r x) i^(l_r^T x) [A_r x = b_r],

so all amplitudes lie in {0, +/-1, +/-i}.  Consequently every universal
coefficient polynomial is over Q(i), and exact reconstruction requires no
square roots.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence
import ast
import re

import numpy as np
import sympy as sp

W = sp.Symbol("w")
I = sp.I
ROOTS4 = (sp.Integer(1), I, sp.Integer(-1), -I)


def canonical_kind(kind: str) -> str:
    k = kind.strip().lower().replace("-", "_")
    aliases = {
        "cat": "even_cat", "even": "even_cat", "evencat": "even_cat",
        "odd": "odd_cat", "oddcat": "odd_cat",
        "prod": "product",
    }
    k = aliases.get(k, k)
    if k not in {"product", "even_cat", "odd_cat"}:
        raise ValueError(f"unknown target kind {kind!r}")
    return k


def target_degrees(n: int, kind: str) -> tuple[int, ...]:
    kind = canonical_kind(kind)
    if kind == "product":
        return tuple(range(n + 1))
    parity = 0 if kind == "even_cat" else 1
    return tuple(k for k in range(n + 1) if (k & 1) == parity)


def parse_partition(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.replace(",", "+").split("+") if x.strip())


def orbit_weights(partition: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    return tuple(product(*[range(m + 1) for m in partition]))


def orbit_representative(partition: Sequence[int], weights: Sequence[int]) -> int:
    x = 0
    off = 0
    for m, wt in zip(partition, weights):
        x |= ((1 << int(wt)) - 1) << off
        off += int(m)
    return x


def block_weights_of_x(x: int, partition: Sequence[int]) -> tuple[int, ...]:
    out = []
    off = 0
    for m in partition:
        mask = ((1 << m) - 1) << off
        out.append((x & mask).bit_count())
        off += m
    return tuple(out)


@dataclass
class ExactTerm:
    pool_id: int | None
    Q: list[list[int]]
    l: list[int]
    A: list[list[int]]
    b: list[int]
    support_size: int | None = None
    coefficient: sp.Expr | None = None


@dataclass
class GeneralWWitness:
    path: Path | None
    target: str
    n: int
    partition: tuple[int, ...]
    rank: int
    phase_mode: str
    terms: list[ExactTerm]
    metadata: dict[str, str]


def term_amplitude(term: ExactTerm, x: int) -> sp.Expr:
    bits = [(x >> i) & 1 for i in range(len(term.l))]
    for row, rhs in zip(term.A, term.b):
        if (sum(int(a) * bits[i] for i, a in enumerate(row)) & 1) != int(rhs):
            return sp.Integer(0)
    e = sum(int(term.l[i]) * bits[i] for i in range(len(bits)))
    Q = term.Q
    for i in range(len(bits)):
        if not bits[i]:
            continue
        for j in range(i + 1, len(bits)):
            if bits[j] and int(Q[i][j]):
                e += 2
    return ROOTS4[e & 3]


def _parse_list(text: str):
    obj = ast.literal_eval(text.strip())
    return obj


def _parse_expr(text: str) -> sp.Expr | None:
    t = text.strip()
    if t in {"?", "UNAVAILABLE", "NONE", "None"}:
        return None
    return sp.expand(sp.sympify(t.replace("^", "**"), locals={"w": W, "I": I}))


def parse_witness(path: str | Path) -> GeneralWWitness:
    """Parse the v2 strict witness format written by search_general_w.py."""
    path = Path(path)
    lines = path.read_text().splitlines()
    metadata: dict[str, str] = {}
    terms: list[ExactTerm] = []
    cur: dict[str, object] | None = None

    def finish():
        nonlocal cur
        if cur is None:
            return
        required = {"Q", "l", "A", "b"}
        missing = required - set(cur)
        if missing:
            raise ValueError(
                f"{path}: incomplete stabilizer, missing {sorted(missing)}")
        terms.append(ExactTerm(
            pool_id=cur.get("pool_id"),
            Q=cur["Q"], l=cur["l"], A=cur["A"], b=cur["b"],
            support_size=cur.get("support_size"), coefficient=cur.get("coef"),
        ))
        cur = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"Stabiliser\s+\d+:", line):
            finish()
            cur = {}
            continue
        if "=" not in line:
            continue
        key, value = (x.strip() for x in line.split("=", 1))
        if cur is None:
            metadata[key] = value
        else:
            if key == "pool_id":
                cur[key] = int(value)
            elif key in {"Q", "l", "A", "b"}:
                cur[key] = _parse_list(value)
            elif key == "support_size":
                cur[key] = int(value)
            elif key == "coef":
                cur[key] = _parse_expr(value)
            else:
                cur[key] = value
    finish()

    if metadata.get("format") != "symmetric_stabilizer_general_w_v2":
        raise ValueError(f"{path}: unsupported/missing format marker")
    target = canonical_kind(metadata["target"])
    partition = parse_partition(metadata["partition"])
    n = int(metadata.get("n", sum(partition)))
    rank = int(metadata["rank"])
    if n != sum(partition):
        raise ValueError(f"{path}: n={n} != sum(partition)={sum(partition)}")
    if rank != len(terms):
        raise ValueError(f"{path}: rank={rank} but parsed {len(terms)} terms")
    return GeneralWWitness(path, target, n, partition, rank,
                           metadata.get("phase_mode", "unknown"), terms, metadata)


def _sympy_orbit_matrix(terms: Sequence[ExactTerm], partition: Sequence[int]):
    weights = orbit_weights(partition)
    reps = [orbit_representative(partition, ws) for ws in weights]
    rows = [[term_amplitude(t, x) for t in terms] for x in reps]
    return weights, sp.Matrix(rows)


def _target_layer_matrix(weights: Sequence[Sequence[int]], n: int, kind: str):
    deg = target_degrees(n, kind)
    B = sp.zeros(len(weights), len(deg))
    pos = {k: j for j, k in enumerate(deg)}
    for i, ws in enumerate(weights):
        k = sum(ws)
        if k in pos:
            B[i, pos[k]] = 1
    return deg, B


def solve_coefficients_exact(terms: Sequence[ExactTerm], partition: Sequence[int],
                             kind: str):
    """Solve c_r(w) exactly over Q(i), returning one polynomial per term.

    If the selected atoms are linearly dependent, free atoms are assigned zero
    coefficient.  Exact verification of the full target span is performed.
    """
    n = sum(partition)
    weights, V = _sympy_orbit_matrix(terms, partition)
    deg, B = _target_layer_matrix(weights, n, kind)

    # Reduce to independent atom columns, then choose independent orbit rows.
    _, pivot_cols = V.rref()
    piv = list(map(int, pivot_cols))
    if not piv:
        raise ValueError("empty stabilizer span")
    Vr = V[:, piv]
    _, pivot_rows = Vr.T.rref()
    prows = list(map(int, pivot_rows))
    if len(prows) != len(piv):
        raise AssertionError("failed to choose an independent square minor")
    S = Vr.extract(prows, range(len(piv)))
    Br = B.extract(prows, range(B.cols))
    Xr = S.inv() * Br
    D = Vr * Xr - B
    if any(sp.simplify(z) != 0 for z in D):
        raise ValueError(
            "selected stabilizers do not exactly span the requested target family")

    X = sp.zeros(len(terms), len(deg))
    for i, col in enumerate(piv):
        for j in range(len(deg)):
            X[col, j] = sp.simplify(Xr[i, j])
    coeff = []
    for r in range(len(terms)):
        p = sum(X[r, j] * W**deg[j] for j in range(len(deg)))
        coeff.append(sp.expand(p))
    return coeff, X, deg


def validate_term_symmetry(terms: Sequence[ExactTerm], partition: Sequence[int]) -> None:
    """Check every displayed gauge represents a state invariant under the partition."""
    n = sum(partition)
    weights = orbit_weights(partition)
    rep_amp = {
        ws: [term_amplitude(t, orbit_representative(partition, ws))
             for t in terms]
        for ws in weights
    }
    for x in range(1 << n):
        ws = block_weights_of_x(x, partition)
        expected = rep_amp[ws]
        for r, t in enumerate(terms):
            got = term_amplitude(t, x)
            if got != expected[r]:
                raise ValueError(
                    f"term {r} is not partition-symmetric: x={x}, weights={ws}, "
                    f"amplitude={got}, representative amplitude={expected[r]}"
                )


def verify_coefficients(terms: Sequence[ExactTerm], partition: Sequence[int], kind: str,
                        coefficients: Sequence[sp.Expr]) -> None:
    """Exact coefficient-wise polynomial verification on all partition orbits."""
    weights, V = _sympy_orbit_matrix(terms, partition)
    n = sum(partition)
    deg = target_degrees(n, kind)
    all_deg = tuple(range(n + 1))
    # Compare every coefficient of w^k, not sampled w values.
    for i, ws in enumerate(weights):
        lhs = sp.expand(sum(V[i, r] * coefficients[r]
                        for r in range(len(terms))))
        wt = sum(ws)
        allowed = wt in deg
        rhs = W**wt if allowed else sp.Integer(0)
        diff = sp.simplify(sp.expand(lhs - rhs))
        if diff != 0:
            raise ValueError(f"orbit {ws}: polynomial mismatch: {diff}")


def format_expr(expr: sp.Expr | None) -> str:
    if expr is None:
        return "?"
    return str(sp.expand(expr)).replace("**", "^")


def _list_repr(x) -> str:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return repr(x)


def write_witness(path: str | Path, *, target: str, partition: Sequence[int],
                  phase_mode: str, terms: Sequence[ExactTerm],
                  coefficients: Sequence[sp.Expr] | None,
                  metadata: dict[str, object] | None = None) -> None:
    target = canonical_kind(target)
    n = sum(partition)
    lines = [
        "format = symmetric_stabilizer_general_w_v2",
        f"target = {target}",
        f"n = {n}",
        f"partition = {'+'.join(map(str, partition))}",
        f"phase_mode = {phase_mode}",
        f"rank = {len(terms)}",
        "convention = unnormalized",
        "term(x) = coef(w) * (-1)^(x^T Q x) * i^(l^T x) * [A x = b]",
    ]
    if metadata:
        for k, v in metadata.items():
            if k in {"format", "target", "n", "partition", "phase_mode", "rank", "convention"}:
                continue
            lines.append(f"{k} = {v}")
    lines.append("")
    for r, term in enumerate(terms):
        coef = None if coefficients is None else coefficients[r]
        lines += [
            f"Stabiliser {r}:",
            f"pool_id = {term.pool_id if term.pool_id is not None else -1}",
            f"coef = {format_expr(coef)}",
            f"Q = {_list_repr(term.Q)}",
            f"l = {_list_repr(term.l)}",
            f"A = {_list_repr(term.A)}",
            f"b = {_list_repr(term.b)}",
            f"support_size = {term.support_size if term.support_size is not None else -1}",
            "",
        ]
    Path(path).write_text("\n".join(lines))


def terms_from_pool(pool, indices: Iterable[int]) -> list[ExactTerm]:
    # Local import avoids making this helper a dependency of the core module.
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


def exactify_pool_selection(pool, indices: Iterable[int], kind: str):
    terms = terms_from_pool(pool, indices)
    validate_term_symmetry(terms, pool.partition)
    coeff, X, deg = solve_coefficients_exact(terms, pool.partition, kind)
    verify_coefficients(terms, pool.partition, kind, coeff)
    return terms, coeff
