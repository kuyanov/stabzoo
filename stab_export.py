"""Export exact or numerical stabiliser decompositions as QlAb JSON.

The exported atoms use the unnormalised convention

    (-1) ** (x.T @ Q @ x) * i ** (l.T @ x) * [A @ x == b],

where the exponents and constraints are evaluated modulo two and four as
described in the JSON metadata.  Any phase introduced while converting the
compact symmetric representation is absorbed into the atom coefficient.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from stab_core import canonical_cs, gen_rrefs, orbit_weights, target_in_orbit_basis


@dataclass(frozen=True)
class QlAbTerm:
    """One unnormalised stabiliser atom and its emitted-pool index."""

    pool_index: int
    Q: np.ndarray
    l: np.ndarray
    A: np.ndarray
    b: np.ndarray
    support_size: int


def _gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the reduced row-echelon form and pivot columns over GF(2)."""
    result = np.asarray(matrix, dtype=np.uint8).copy() & 1
    rows, columns = result.shape
    pivots: list[int] = []
    row = 0
    for column in range(columns):
        candidates = np.flatnonzero(result[row:, column])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        if pivot != row:
            result[[row, pivot]] = result[[pivot, row]]
        mask = result[:, column].astype(bool)
        mask[row] = False
        result[mask] ^= result[row]
        pivots.append(column)
        row += 1
        if row == rows:
            break
    return result, pivots


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert a nonsingular square matrix over GF(2)."""
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    n, columns = matrix.shape
    if n != columns:
        raise ValueError("matrix must be square")
    augmented = np.concatenate([matrix.copy(), np.eye(n, dtype=np.uint8)], axis=1)
    for column in range(n):
        candidates = np.flatnonzero(augmented[column:, column])
        if not len(candidates):
            raise ValueError("matrix is singular over GF(2)")
        pivot = column + int(candidates[0])
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        mask = augmented[:, column].astype(bool)
        mask[column] = False
        augmented[mask] ^= augmented[column]
    return augmented[:, n:]


def _gf2_left_inverse(matrix: np.ndarray) -> np.ndarray:
    """Return R satisfying R @ matrix == identity over GF(2)."""
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    rows, columns = matrix.shape
    if columns == 0:
        return np.zeros((0, rows), dtype=np.uint8)
    _, independent_rows = _gf2_rref(matrix.T)
    if len(independent_rows) != columns:
        raise ValueError("compact support basis does not have full column rank")
    selected = np.asarray(independent_rows, dtype=np.intp)
    result = np.zeros((columns, rows), dtype=np.uint8)
    result[:, selected] = _gf2_inverse(matrix[selected])
    return result


def _gf2_nullspace_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a row basis for the right nullspace of a binary matrix."""
    matrix = np.asarray(matrix, dtype=np.uint8) & 1
    reduced, pivots = _gf2_rref(matrix)
    pivot_set = set(pivots)
    free = [column for column in range(matrix.shape[1]) if column not in pivot_set]
    result = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    for output_row, free_column in enumerate(free):
        result[output_row, free_column] = 1
        for input_row, pivot in enumerate(pivots):
            result[output_row, pivot] = reduced[input_row, free_column]
    return result


def _decode_core_phase(d: int, phase_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode a row index of ``core_phase_table(d)`` into ``l`` and ``Q``."""
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    q_code = phase_index & ((1 << len(pairs)) - 1)
    l_code = phase_index >> len(pairs)
    linear = ((l_code >> (2 * np.arange(d))) & 3).astype(np.int8)
    quadratic = np.zeros((d, d), dtype=np.uint8)
    for bit, (i, j) in enumerate(pairs):
        quadratic[i, j] = (q_code >> bit) & 1
    return linear, quadratic


def _compact_to_qlab(
    ns: tuple[int, ...],
    active: np.ndarray,
    c: np.ndarray,
    basis: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    internal_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Convert one compact symmetric atom to QlAb and a global i-phase."""
    k = len(ns)
    n = sum(ns)
    d = basis.shape[1]
    offsets = np.cumsum(np.asarray((0,) + ns[:-1], dtype=np.int64))

    # Read the core bit from block parity (active) or the first bit (inactive).
    physical_to_core = np.zeros((k, n), dtype=np.uint8)
    for block, (offset, size) in enumerate(zip(offsets, ns)):
        start = int(offset)
        if active[block]:
            physical_to_core[block, start : start + size] = 1
        else:
            physical_to_core[block, start] = 1

    annihilator = _gf2_nullspace_rows(basis.T)
    A_core = (annihilator @ physical_to_core & 1).astype(np.int8)
    b_core = (annihilator @ c.astype(np.uint8) & 1).astype(np.int8)

    equality_rows: list[np.ndarray] = []
    for block, (offset, size) in enumerate(zip(offsets, ns)):
        if active[block]:
            continue
        start = int(offset)
        for bit in range(1, size):
            row = np.zeros(n, dtype=np.int8)
            row[start] = 1
            row[start + bit] = 1
            equality_rows.append(row)
    if equality_rows:
        A = np.vstack((A_core, equality_rows)).astype(np.int8, copy=False)
        b = np.concatenate((b_core, np.zeros(len(equality_rows), dtype=np.int8)))
    else:
        A, b = A_core, b_core

    left_inverse = _gf2_left_inverse(basis)
    affine_map = (left_inverse @ physical_to_core & 1).astype(np.uint8)
    affine_offset = (left_inverse @ c.astype(np.uint8) & 1).astype(np.uint8)

    Q = np.zeros((n, n), dtype=np.uint8)
    l = np.zeros(n, dtype=np.int8)
    if d:
        signs = 1 - 2 * affine_offset.astype(np.int64)
        l[:] = ((linear.astype(np.int64) * signs) @ affine_map & 3).astype(np.int8)

        odd_linear = (linear & 1).astype(np.int64)
        xor_correction = (
            affine_map.T.astype(np.int64)
            @ (odd_linear[:, None] * affine_map.astype(np.int64))
        ) & 1
        Q ^= np.triu(xor_correction.astype(np.uint8), k=1)

        map64 = affine_map.astype(np.int64)
        quadratic64 = quadratic.astype(np.int64)
        offset64 = affine_offset.astype(np.int64)
        substituted = (map64.T @ quadratic64 @ map64) & 1
        Q ^= np.triu((substituted ^ substituted.T).astype(np.uint8), k=1)
        cross = (map64.T @ (((quadratic64 + quadratic64.T) @ offset64) & 1)) & 1
        diagonal = np.diag(substituted).astype(np.uint8) ^ cross.astype(np.uint8)
        Q[np.diag_indices(n)] ^= diagonal

    active_blocks = np.flatnonzero(active)
    for q_bit, block in zip(internal_q, active_blocks):
        if not q_bit:
            continue
        start = int(offsets[block])
        rows, columns = np.triu_indices(ns[block], k=1)
        Q[start + rows, start + columns] ^= 1

    global_phase = (
        int(linear.astype(np.int64) @ affine_offset.astype(np.int64))
        + 2
        * int(
            affine_offset.astype(np.int64)
            @ quadratic.astype(np.int64)
            @ affine_offset.astype(np.int64)
        )
    ) & 3
    return Q.astype(np.int8), l, A, b, global_phase


def qlab_terms_from_indices(
    ns: Sequence[int],
    indices: Sequence[int],
) -> list[QlAbTerm]:
    """Decode selected emitted-pool indices and convert them to QlAb atoms."""
    partition = tuple(map(int, ns))
    requested = sorted({int(index) for index in indices})
    if len(requested) != len(indices):
        raise ValueError("a decomposition cannot contain repeated pool indices")
    if requested and requested[0] < 0:
        raise IndexError("pool indices must be nonnegative")

    found: dict[int, QlAbTerm] = {}
    next_requested = 0
    emitted = 0
    k = len(partition)
    for d in range(k + 1):
        phase_count = 4**d * 2 ** (d * (d - 1) // 2)
        for BT in gen_rrefs(k, d):
            basis = BT.T
            for c in canonical_cs(BT):
                for mode_tuple in product(range(3), repeat=k):
                    batch_end = emitted + phase_count
                    while (
                        next_requested < len(requested)
                        and requested[next_requested] < batch_end
                    ):
                        pool_index = requested[next_requested]
                        linear, quadratic = _decode_core_phase(d, pool_index - emitted)
                        modes = np.asarray(mode_tuple, dtype=np.int8)
                        active = modes != 0
                        internal_q = np.maximum(modes[active] - 1, 0)
                        Q, l, A, b, _global_phase = _compact_to_qlab(
                            partition,
                            active,
                            c,
                            basis,
                            linear,
                            quadratic,
                            internal_q,
                        )
                        support_dimension = d + sum(
                            partition[a] - 1 for a in np.flatnonzero(active)
                        )
                        found[pool_index] = QlAbTerm(
                            pool_index,
                            Q,
                            l,
                            A,
                            b,
                            1 << support_dimension,
                        )
                        next_requested += 1
                    emitted = batch_end
                    if next_requested == len(requested):
                        return [found[int(index)] for index in indices]
    missing = requested[next_requested:]
    raise IndexError(f"pool indices outside the emitted dictionary: {missing}")


def _orbit_representatives(ns: tuple[int, ...]) -> np.ndarray:
    """Return one little-endian physical bit vector per block-weight orbit."""
    weights = orbit_weights(ns)
    representatives = np.zeros((len(weights), sum(ns)), dtype=np.uint8)
    offset = 0
    for block, size in enumerate(ns):
        for row, weight in enumerate(weights[:, block]):
            representatives[row, offset : offset + int(weight)] = 1
        offset += size
    return representatives


def _qlab_orbit_codes(ns: tuple[int, ...], terms: Sequence[QlAbTerm]) -> np.ndarray:
    """Evaluate QlAb atoms on orbit representatives as 0 or 1+p for i**p."""
    x = _orbit_representatives(ns)
    encoded = np.zeros((len(x), len(terms)), dtype=np.int8)
    for column, term in enumerate(terms):
        if len(term.b):
            valid = np.all(((x @ term.A.T) & 1) == term.b, axis=1)
        else:
            valid = np.ones(len(x), dtype=bool)
        linear = (x.astype(np.int64) @ term.l.astype(np.int64)) & 3
        quadratic = (
            np.einsum("bi,ij,bj->b", x, term.Q, x, optimize=True).astype(np.int64) & 1
        )
        encoded[valid, column] = 1 + ((linear[valid] + 2 * quadratic[valid]) & 3)
    return encoded


def _sympy_orbit_matrix(codes: np.ndarray) -> Any:
    """Convert encoded Gaussian-unit amplitudes to an exact SymPy matrix."""
    import sympy as sp

    phases = (sp.Integer(0), sp.Integer(1), sp.I, -sp.Integer(1), -sp.I)
    return sp.Matrix([[phases[int(value)] for value in row] for row in codes])


def _solve_exact(matrix: Any, target: Any) -> Any:
    """Solve an overdetermined exact system, assigning zero to free atoms."""
    import sympy as sp

    # Search normally returns independent atoms.  Test that cheaper case first:
    # RREF of the short-wide transpose simultaneously selects independent rows.
    _, pivot_rows = matrix.T.rref()
    rows = list(map(int, pivot_rows))
    if len(rows) == matrix.cols:
        pivots = list(range(matrix.cols))
        independent = matrix
    else:
        _, pivot_columns = matrix.rref()
        pivots = list(map(int, pivot_columns))
        if not pivots:
            raise ValueError("the selected stabiliser span is empty")
        independent = matrix[:, pivots]
        _, pivot_rows = independent.T.rref()
        rows = list(map(int, pivot_rows))
        if len(rows) != len(pivots):
            raise ValueError("failed to select an independent square minor")
    square = independent.extract(rows, range(len(pivots)))
    rhs = target.extract(rows, range(target.cols))
    reduced = square.inv() * rhs
    if any(sp.simplify(value) != 0 for value in independent * reduced - target):
        raise ValueError("selected stabilisers do not exactly span the target")
    solution = sp.zeros(matrix.cols, target.cols)
    for reduced_row, original_column in enumerate(pivots):
        solution[original_column, :] = reduced[reduced_row, :]
    return solution


def _h_box_counts(ns: tuple[int, ...], m: int) -> np.ndarray:
    """Count full contiguous H-boxes for every orbit of a refining partition."""
    if m <= 0 or sum(ns) % m:
        raise ValueError("H-box size must be positive and divide n")
    weights = orbit_weights(ns)
    sizes = np.asarray(ns)
    counts = np.zeros(len(weights), dtype=np.int16)
    position = 0
    current: list[int] = []
    for block, size in enumerate(ns):
        if position // m != (position + size - 1) // m:
            raise ValueError("partition crosses an H-box boundary")
        current.append(block)
        position += size
        if position % m == 0:
            columns = np.asarray(current, dtype=np.intp)
            counts += np.all(weights[:, columns] == sizes[columns], axis=1)
            current = []
    return counts


def _exact_coefficients(
    ns: tuple[int, ...],
    codes: np.ndarray,
    target_spec: Mapping[str, Any],
) -> tuple[list[str], str] | None:
    """Return verified exact coefficient expressions when the target permits it."""
    import sympy as sp

    kind = str(target_spec["type"])
    weights = orbit_weights(ns).sum(axis=1)
    matrix = _sympy_orbit_matrix(codes)
    if kind in {"T_product", "T_cat"}:
        omega = (1 + sp.I) / sp.sqrt(2)
        parity = None if kind == "T_product" else int(target_spec.get("parity", 0))
        target = sp.Matrix(
            [
                omega ** int(weight)
                if parity is None or int(weight) % 2 == parity
                else 0
                for weight in weights
            ]
        )
        solution = _solve_exact(matrix, target)
        expressions = [
            sp.sstr(sp.radsimp(sp.simplify(solution[row, 0])))
            for row in range(solution.rows)
        ]
        return expressions, "Q(sqrt(2), I)"

    if kind == "Dicke":
        excitation = int(target_spec["k"])
        target = sp.Matrix([int(weight == excitation) for weight in weights])
        solution = _solve_exact(matrix, target)
        expressions = [
            sp.sstr(sp.simplify(solution[row, 0])) for row in range(solution.rows)
        ]
        return expressions, "Q(I)"

    if kind == "H_product" and target_spec.get("family") is False:
        parameter = sp.sympify(str(target_spec["a"]), locals={"I": sp.I})
        real, imaginary = sp.expand_complex(parameter).as_real_imag()
        if not (real.is_Rational and imaginary.is_Rational):
            return None
        exponents = _h_box_counts(ns, int(target_spec["m"]))
        target = sp.Matrix([parameter ** int(exponent) for exponent in exponents])
        solution = _solve_exact(matrix, target)
        expressions = [
            sp.sstr(sp.simplify(solution[row, 0])) for row in range(solution.rows)
        ]
        return expressions, "Q(I)"

    if kind in {"phase_product", "phase_cat", "H_product"} and target_spec.get(
        "family"
    ):
        default_symbol = "a" if kind == "H_product" else "w"
        symbol = sp.Symbol(str(target_spec.get("symbol", default_symbol)))
        if kind == "phase_product":
            degrees = list(range(sum(ns) + 1))
            exponents = weights
        elif kind == "phase_cat":
            parity = int(target_spec.get("parity", 0))
            degrees = list(range(parity, sum(ns) + 1, 2))
            exponents = weights
        else:
            m = int(target_spec["m"])
            copies = sum(ns) // m
            if int(target_spec["copies"]) != copies:
                raise ValueError("H-box copy count is inconsistent with n and m")
            degrees = list(range(copies + 1))
            exponents = _h_box_counts(ns, m)
        degree_to_column = {degree: column for column, degree in enumerate(degrees)}
        target = sp.zeros(len(weights), len(degrees))
        for row, exponent in enumerate(exponents):
            if int(exponent) in degree_to_column:
                target[row, degree_to_column[int(exponent)]] = 1
        solution = _solve_exact(matrix, target)
        expressions = []
        for row in range(solution.rows):
            polynomial = sum(
                solution[row, column] * symbol**degree
                for column, degree in enumerate(degrees)
            )
            expressions.append(sp.sstr(sp.expand(polynomial)))
        return expressions, f"Q(I)[{symbol}]"
    return None


def _numeric_coefficients(
    ns: tuple[int, ...],
    codes: np.ndarray,
    target: np.ndarray,
    target_spec: Mapping[str, Any],
) -> list[str]:
    """Compute stable decimal coefficient expressions for a fixed target."""
    phase_values = np.asarray([0, 1, 1j, -1, -1j], dtype=np.complex128)
    matrix = phase_values[codes]
    multiplicities = np.ones(len(codes), dtype=np.float64)
    weights = orbit_weights(ns)
    from math import comb

    for block, size in enumerate(ns):
        multiplicities *= np.fromiter(
            (comb(size, int(weight)) for weight in weights[:, block]),
            dtype=np.float64,
            count=len(weights),
        )
    roots = np.sqrt(multiplicities)
    weighted_matrix = matrix * roots[:, None]
    target = np.asarray(target, dtype=np.complex128)
    if target.ndim == 1:
        target = target_in_orbit_basis(ns, target)
    elif target.ndim == 2:
        target = np.column_stack(
            [
                target_in_orbit_basis(ns, target[:, column])
                for column in range(target.shape[1])
            ]
        )
    else:
        raise ValueError("target must be a vector or matrix of target columns")
    weighted_target = target * roots if target.ndim == 1 else target * roots[:, None]
    coefficients, *_ = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=1e-12)

    def expression(value: complex) -> str:
        real = 0.0 if abs(value.real) < 5e-15 else float(value.real)
        imag = 0.0 if abs(value.imag) < 5e-15 else float(value.imag)
        return f"({real:.17g}) + ({imag:.17g})*I"

    if coefficients.ndim == 1:
        return [expression(value) for value in coefficients]

    symbol = str(target_spec.get("symbol", "w"))
    if target_spec["type"] == "phase_cat":
        parity = int(target_spec.get("parity", 0))
        degrees = list(range(parity, sum(ns) + 1, 2))
    else:
        degrees = list(range(coefficients.shape[1]))
    result = []
    for row in coefficients:
        terms = []
        for value, degree in zip(row, degrees):
            if abs(value) < 5e-15:
                continue
            coefficient = expression(value)
            if degree == 0:
                suffix = ""
            elif degree == 1:
                suffix = f"*{symbol}"
            else:
                suffix = f"*{symbol}**{degree}"
            terms.append(f"({coefficient}){suffix}")
        result.append(" + ".join(terms) if terms else "0")
    return result


def save_decomposition_json(
    path: str | Path,
    *,
    partition: Sequence[int],
    indices: Sequence[int],
    target: np.ndarray,
    target_spec: Mapping[str, Any],
    residual: float,
) -> None:
    """Write a self-describing QlAb decomposition certificate to JSON."""
    ns = tuple(map(int, partition))
    terms = qlab_terms_from_indices(ns, indices)
    codes = _qlab_orbit_codes(ns, terms)
    try:
        exact = _exact_coefficients(ns, codes, target_spec)
    except ValueError:
        exact = None
    if exact is None:
        coefficients = _numeric_coefficients(ns, codes, target, target_spec)
        coefficient_ring = "complex decimal"
        coefficients_exact = False
    else:
        coefficients, coefficient_ring = exact
        coefficients_exact = True

    term_records = []
    for term, coefficient in zip(terms, coefficients):
        term_records.append(
            {
                "pool_index": term.pool_index,
                "coefficient": coefficient,
                "Q": term.Q.astype(int).tolist(),
                "l": term.l.astype(int).tolist(),
                "A": term.A.astype(int).tolist(),
                "b": term.b.astype(int).tolist(),
                "support_size": term.support_size,
            }
        )

    document = {
        "format": "stabzoo.QlAb.v1",
        "n": sum(ns),
        "partition": list(ns),
        "rank": len(terms),
        "search_residual": float(residual),
        "target": dict(target_spec),
        "coefficient_ring": coefficient_ring,
        "coefficients_exact": coefficients_exact,
        "convention": {
            "decomposition": "target(x) = sum_j coefficient_j * stabiliser_j(x)",
            "stabiliser": "(-1)^(x^T Q x) * i^(l^T x) * [A x = b]",
            "arithmetic": "Q,A,b,x over GF(2); l^T x modulo 4",
            "quadratic": "x^T Q x = sum_{u<=v} Q[u][v] x[u] x[v] modulo 2",
            "normalization": "QlAb stabilisers are unnormalised",
            "bit_order": (
                "blocks are contiguous in partition order; bit 0 is the first "
                "bit of the first block"
            ),
            "coefficient_syntax": (
                "SymPy expression syntax using I, sqrt, exp, pi, and target symbols"
            ),
        },
        "terms": term_records,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
