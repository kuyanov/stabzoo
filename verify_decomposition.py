#!/usr/bin/env python3
"""Verify a ``stabzoo.QlAb.v1`` decomposition using only its JSON file."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

_PHASES = np.asarray([0, 1, 1j, -1, -1j], dtype=np.complex128)
_EXPRESSION_CHARACTERS = re.compile(r"^[0-9A-Za-z+*/().\-\s]*$")
_EXPRESSION_NAMES = {"I", "sqrt", "exp", "pi", "w", "e", "E"}
_NUMERICAL_TOLERANCE = 1e-10


class VerificationError(ValueError):
    """Raised when a decomposition certificate is malformed or incorrect."""


def _require(condition: bool, message: str) -> None:
    """Raise a readable verification error when ``condition`` is false."""
    if not condition:
        raise VerificationError(message)


def _integer(value: Any, name: str) -> int:
    """Read an integer while rejecting booleans and inexact JSON numbers."""
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return int(value)


def _integer_array(value: Any, name: str, ndim: int) -> np.ndarray:
    """Read a rectangular integer array of the requested dimension."""
    try:
        array = np.asarray(value)
    except ValueError as error:
        raise VerificationError(f"{name} must be rectangular") from error
    _require(array.ndim == ndim, f"{name} must have {ndim} dimensions")
    _require(
        array.size == 0
        or (np.issubdtype(array.dtype, np.integer) and array.dtype != np.bool_),
        f"{name} must contain integers",
    )
    return array.astype(np.int64, copy=False)


def _parse_expression(
    text: Any,
    symbols: dict[str, Any],
    name: str = "coefficient",
) -> sp.Expr:
    """Parse the deliberately small symbolic-expression grammar used by JSON."""
    _require(
        isinstance(text, str) and bool(text.strip()),
        f"{name} must be a nonempty string",
    )
    _require(
        bool(_EXPRESSION_CHARACTERS.fullmatch(text)),
        f"unsafe {name} expression: {text!r}",
    )
    names = set(re.findall(r"[A-Za-z]+", text))
    _require(
        names <= _EXPRESSION_NAMES,
        f"unsupported names in {name}: {sorted(names - _EXPRESSION_NAMES)}",
    )
    try:
        expression = sp.sympify(
            text,
            locals={
                "I": sp.I,
                "sqrt": sp.sqrt,
                "exp": sp.exp,
                "pi": sp.pi,
                **symbols,
            },
        )
    except (SyntaxError, TypeError, ValueError, sp.SympifyError) as error:
        raise VerificationError(f"invalid {name} expression: {text!r}") from error
    allowed_symbols = set(symbols.values())
    _require(
        expression.free_symbols <= allowed_symbols,
        f"{name} contains undeclared symbols: {expression.free_symbols - allowed_symbols}",
    )
    return expression


def _gf2_rank(matrix: np.ndarray) -> int:
    """Return the rank of a binary matrix over GF(2)."""
    work = np.asarray(matrix, dtype=np.uint8).copy() & 1
    row = 0
    for column in range(work.shape[1]):
        candidates = np.flatnonzero(work[row:, column])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        if pivot != row:
            work[[row, pivot]] = work[[pivot, row]]
        mask = work[:, column].astype(bool)
        mask[row] = False
        work[mask] ^= work[row]
        row += 1
        if row == work.shape[0]:
            break
    return row


def _basis_data(
    partition: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all bit strings, their orbit IDs, and orbit representatives."""
    n = sum(partition)
    integers = np.arange(1 << n, dtype=np.uint64)
    bits = ((integers[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int64)

    block_weights = []
    representative_indices = np.zeros(
        int(np.prod(np.asarray(partition) + 1)), dtype=np.intp
    )
    offset = 0
    for size in partition:
        block_weights.append(bits[:, offset : offset + size].sum(axis=1))
        offset += size
    orbit_ids = np.ravel_multi_index(
        tuple(block_weights), tuple(size + 1 for size in partition)
    )

    weights = np.indices(tuple(size + 1 for size in partition), dtype=np.int16)
    weights = weights.reshape(len(partition), -1).T
    for row, block_tuple in enumerate(weights):
        offset = 0
        index = 0
        for size, weight in zip(partition, block_tuple):
            index |= ((1 << int(weight)) - 1) << offset
            offset += size
        representative_indices[row] = index
    return bits, orbit_ids, representative_indices


def _validate_term(
    record: Any,
    term_number: int,
    n: int,
    bits: np.ndarray,
    orbit_ids: np.ndarray,
    representative_indices: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Validate one QlAb atom and return its encoded orbit amplitudes."""
    prefix = f"term {term_number}"
    _require(isinstance(record, dict), f"{prefix} must be an object")
    for field in ("coefficient", "Q", "l", "A", "b", "support_size"):
        _require(field in record, f"{prefix} is missing {field!r}")

    Q = _integer_array(record["Q"], f"{prefix}.Q", 2)
    l = _integer_array(record["l"], f"{prefix}.l", 1)
    b = _integer_array(record["b"], f"{prefix}.b", 1)
    _require(Q.shape == (n, n), f"{prefix}.Q must have shape ({n}, {n})")
    _require(l.shape == (n,), f"{prefix}.l must have length {n}")
    _require(np.all((Q == 0) | (Q == 1)), f"{prefix}.Q must be binary")
    _require(np.all(np.tril(Q, k=-1) == 0), f"{prefix}.Q must be upper triangular")
    _require(np.all((0 <= l) & (l <= 3)), f"{prefix}.l must have entries in Z_4")
    _require(np.all((b == 0) | (b == 1)), f"{prefix}.b must be binary")

    raw_A = record["A"]
    if raw_A == []:
        A = np.zeros((0, n), dtype=np.int64)
    else:
        A = _integer_array(raw_A, f"{prefix}.A", 2)
        _require(A.shape[1] == n, f"{prefix}.A must have {n} columns")
    _require(
        A.shape[0] == len(b), f"{prefix}.A and {prefix}.b have incompatible shapes"
    )
    _require(np.all((A == 0) | (A == 1)), f"{prefix}.A must be binary")

    rank_A = _gf2_rank(A)
    augmented = np.column_stack((A, b))
    consistent = _gf2_rank(augmented) == rank_A
    expected_support = (1 << (n - rank_A)) if consistent else 0
    support_size = _integer(record["support_size"], f"{prefix}.support_size")
    _require(
        support_size == expected_support,
        f"{prefix}.support_size is {support_size}, expected {expected_support}",
    )

    if len(b):
        valid = np.all(((bits @ A.T) & 1) == b, axis=1)
    else:
        valid = np.ones(len(bits), dtype=bool)
    linear = (bits @ l) & 3
    quadratic = np.einsum("bi,ij,bj->b", bits, Q, bits, optimize=True) & 1
    encoded = np.zeros(len(bits), dtype=np.int8)
    encoded[valid] = 1 + ((linear[valid] + 2 * quadratic[valid]) & 3)
    _require(
        int(np.count_nonzero(encoded)) == support_size,
        f"{prefix} has the wrong number of nonzero amplitudes",
    )

    orbit_codes = encoded[representative_indices]
    _require(
        np.array_equal(encoded, orbit_codes[orbit_ids]),
        f"{prefix} is not invariant under the stated partition",
    )
    return orbit_codes, record["coefficient"]


def _orbit_weights(partition: tuple[int, ...]) -> np.ndarray:
    """Return the block weights in the JSON format's canonical orbit order."""
    shape = tuple(size + 1 for size in partition)
    return np.indices(shape, dtype=np.int16).reshape(len(partition), -1).T


def _target_exact(
    partition: tuple[int, ...],
    target: dict[str, Any],
    symbol: sp.Symbol,
) -> list[sp.Expr] | None:
    """Construct exact target orbit amplitudes, or return None if numerical."""
    kind = target.get("type")
    total_weights = _orbit_weights(partition).sum(axis=1)
    if kind in {"T_product", "T_cat"}:
        omega = _parse_expression(target.get("omega"), {}, "target.omega")
        canonical_omega = (1 + sp.I) / sp.sqrt(2)
        _require(
            sp.simplify(omega - canonical_omega) == 0,
            "target.omega is not the T-state phase",
        )
        parity = (
            None
            if kind == "T_product"
            else _integer(target.get("parity"), "target.parity")
        )
        _require(parity is None or parity in (0, 1), "target.parity must be 0 or 1")
        return [
            omega ** int(weight)
            if parity is None or int(weight) % 2 == parity
            else sp.Integer(0)
            for weight in total_weights
        ]
    if kind == "Dicke":
        excitation = _integer(target.get("k"), "target.k")
        _require(0 <= excitation <= sum(partition), "target.k must lie between 0 and n")
        return [sp.Integer(int(weight == excitation)) for weight in total_weights]
    if kind in {"phase_product", "phase_cat"} and target.get("family") is True:
        parity = None
        if kind == "phase_cat":
            parity = _integer(target.get("parity"), "target.parity")
            _require(parity in (0, 1), "target.parity must be 0 or 1")
        return [
            symbol ** int(weight)
            if parity is None or int(weight) % 2 == parity
            else sp.Integer(0)
            for weight in total_weights
        ]
    return None


def _target_numeric(partition: tuple[int, ...], target: dict[str, Any]) -> np.ndarray:
    """Construct numerical target amplitudes for a fixed general phase."""
    kind = target.get("type")
    _require(
        kind in {"phase_product", "phase_cat"},
        f"unsupported numerical target type: {kind!r}",
    )
    _require(
        target.get("family") is False, "numerical phase target must set family=false"
    )
    phase = target.get("phase")
    _require(
        isinstance(phase, (int, float)) and not isinstance(phase, bool),
        "target.phase must be a number",
    )
    _require(np.isfinite(float(phase)), "target.phase must be finite")
    phase_value = np.exp(1j * float(phase))
    w_expression = _parse_expression(target.get("w"), {}, "target.w")
    w = complex(sp.N(w_expression, 18))
    _require(
        abs(w - phase_value) <= 1e-14,
        "target.w is inconsistent with target.phase",
    )
    weights = _orbit_weights(partition).sum(axis=1)
    values = w**weights
    if kind == "phase_cat":
        parity = _integer(target.get("parity"), "target.parity")
        _require(parity in (0, 1), "target.parity must be 0 or 1")
        values = np.where((weights & 1) == parity, values, 0.0)
    return values.astype(np.complex128)


def _verify_exact(
    orbit_codes: np.ndarray,
    coefficient_text: Sequence[str],
    target_values: Sequence[sp.Expr],
    coefficient_symbols: dict[str, Any],
) -> None:
    """Verify the decomposition identity exactly on every partition orbit."""
    coefficients = [
        _parse_expression(text, coefficient_symbols) for text in coefficient_text
    ]
    phases = (sp.Integer(0), sp.Integer(1), sp.I, -sp.Integer(1), -sp.I)
    for orbit, expected in enumerate(target_values):
        actual = sum(
            coefficients[term] * phases[int(orbit_codes[orbit, term])]
            for term in range(len(coefficients))
        )
        difference = sp.simplify(sp.expand(actual - expected))
        _require(difference == 0, f"orbit {orbit} has symbolic residual {difference}")


def _verify_numeric(
    partition: tuple[int, ...],
    orbit_codes: np.ndarray,
    coefficient_text: Sequence[str],
    target_values: np.ndarray,
) -> tuple[float, float]:
    """Verify a numerical fixed-phase decomposition and return two errors."""
    coefficients = np.asarray(
        [complex(sp.N(_parse_expression(text, {}), 18)) for text in coefficient_text]
    )
    actual = _PHASES[orbit_codes] @ coefficients
    difference = actual - target_values
    weights = _orbit_weights(partition)
    multiplicities = np.ones(len(weights), dtype=np.float64)
    for block, size in enumerate(partition):
        multiplicities *= np.fromiter(
            (comb(size, int(weight)) for weight in weights[:, block]),
            dtype=np.float64,
            count=len(weights),
        )
    l2_error = float(np.sqrt(np.sum(multiplicities * np.abs(difference) ** 2)))
    max_error = float(np.max(np.abs(difference), initial=0.0))
    _require(
        max_error <= _NUMERICAL_TOLERANCE,
        f"numerical decomposition error {max_error:.6g} exceeds {_NUMERICAL_TOLERANCE:g}",
    )
    return l2_error, max_error


def verify_document(document: Any) -> tuple[bool, float, float]:
    """Validate a decoded JSON certificate and verify its decomposition."""
    _require(isinstance(document, dict), "top-level JSON value must be an object")
    _require(document.get("format") == "stabzoo.QlAb.v1", "unsupported format marker")
    n = _integer(document.get("n"), "n")
    _require(n > 0, "n must be positive")
    partition_value = document.get("partition")
    _require(
        isinstance(partition_value, list) and partition_value,
        "partition must be a nonempty list",
    )
    partition = tuple(
        _integer(size, f"partition[{i}]") for i, size in enumerate(partition_value)
    )
    _require(all(size > 0 for size in partition), "partition sizes must be positive")
    _require(sum(partition) == n, "partition must sum to n")
    _require(n <= 20, "refusing exhaustive symmetry verification for n > 20")

    records = document.get("terms")
    _require(isinstance(records, list), "terms must be a list")
    rank = _integer(document.get("rank"), "rank")
    _require(
        rank == len(records),
        f"rank is {rank}, but the file contains {len(records)} terms",
    )
    _require(rank > 0, "decomposition must contain at least one term")
    target = document.get("target")
    _require(isinstance(target, dict), "target must be an object")
    search_residual = document.get("search_residual")
    _require(
        isinstance(search_residual, (int, float))
        and not isinstance(search_residual, bool)
        and np.isfinite(float(search_residual))
        and float(search_residual) >= 0.0,
        "search_residual must be a finite nonnegative number",
    )

    bits, orbit_ids, representative_indices = _basis_data(partition)
    columns = []
    coefficient_text = []
    for term_number, record in enumerate(records):
        codes, coefficient = _validate_term(
            record,
            term_number,
            n,
            bits,
            orbit_ids,
            representative_indices,
        )
        columns.append(codes)
        coefficient_text.append(coefficient)
    orbit_codes = np.column_stack(columns)

    symbol_name = str(target.get("symbol", "w"))
    _require(symbol_name == "w", "the v1 coefficient symbol must be 'w'")
    symbol = sp.Symbol(symbol_name)
    exact_target = _target_exact(partition, target, symbol)
    declared_exact = document.get("coefficients_exact")
    _require(isinstance(declared_exact, bool), "coefficients_exact must be Boolean")
    if exact_target is not None:
        _require(
            declared_exact, "an exact target must have exact symbolic coefficients"
        )
        uses_w = target.get("type") in {"phase_product", "phase_cat"}
        expected_ring = (
            "Q(I)[w]"
            if uses_w
            else (
                "Q(sqrt(2), I)"
                if target.get("type") in {"T_product", "T_cat"}
                else "Q(I)"
            )
        )
        _require(
            document.get("coefficient_ring") == expected_ring,
            f"coefficient_ring must be {expected_ring!r}",
        )
        coefficient_symbols = {"w": symbol} if uses_w else {}
        _verify_exact(orbit_codes, coefficient_text, exact_target, coefficient_symbols)
        return True, 0.0, 0.0

    _require(
        not declared_exact,
        "fixed numerical phase coefficients must be marked non-exact",
    )
    _require(
        document.get("coefficient_ring") == "complex decimal",
        "numerical coefficient_ring must be 'complex decimal'",
    )
    numerical_target = _target_numeric(partition, target)
    l2_error, max_error = _verify_numeric(
        partition, orbit_codes, coefficient_text, numerical_target
    )
    return False, l2_error, max_error


def argument_parser() -> argparse.ArgumentParser:
    """Build the verifier's one-argument command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path, help="QlAb decomposition JSON file")
    return parser


def main() -> None:
    """Load, validate, and verify one decomposition certificate."""
    parser = argument_parser()
    args = parser.parse_args()
    try:
        document = json.loads(args.json_file.read_text(encoding="utf-8"))
        exact, l2_error, max_error = verify_document(document)
    except (OSError, json.JSONDecodeError, VerificationError) as error:
        parser.exit(1, f"FAILED: {error}\n")

    partition = "+".join(map(str, document["partition"]))
    if exact:
        print(
            f"VERIFIED exact decomposition: n={document['n']} "
            f"partition={partition} rank={document['rank']}"
        )
    else:
        print(
            f"VERIFIED numerical decomposition: n={document['n']} "
            f"partition={partition} rank={document['rank']} "
            f"l2_error={l2_error:.6g} max_error={max_error:.6g}"
        )


if __name__ == "__main__":
    main()
