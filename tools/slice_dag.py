#!/usr/bin/env python3
"""Parse and partition the generated primordial chemistry DAGs.

The emitted Mojo is deliberately ordinary source: each warp function contains
the transitive closure of its outputs in original topological order.  The
report separately identifies profitable exchange candidates for the GPU
shared-memory implementation.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TEMP_RE = re.compile(r"\bx[0-9]+_[0-9]+\b")
DEF_RE = re.compile(r"^var\s+(x[0-9]+_[0-9]+)\s*=", re.DOTALL)
FUNCTION_RE = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
MANIFEST = {
    "rhs_specie": (119, 14),
    "rhs_eint": (162, 1),
    "jac_nuc": (1394, 225),
}


class DagError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Statement:
    text: str
    line: int


@dataclasses.dataclass(frozen=True)
class Definition:
    name: str
    statement: Statement
    dependencies: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Output:
    key: tuple[int, ...]
    statement: Statement
    dependencies: tuple[str, ...]
    kind: str = ""


@dataclasses.dataclass
class FunctionDag:
    name: str
    prelude: tuple[Statement, ...]
    definitions: tuple[Definition, ...]
    outputs: tuple[Output, ...]

    @property
    def by_name(self) -> dict[str, Definition]:
        return {definition.name: definition for definition in self.definitions}


def _delimiter_delta(line: str) -> int:
    depth = 0
    quote = ""
    escaped = False
    for char in line:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    return depth


def function_body(source: str, name: str) -> tuple[list[str], int]:
    lines = source.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = FUNCTION_RE.match(line)
        if match and match.group(1) == name:
            start = index + 1
            break
    if start is None:
        raise DagError(f"missing function {name}")
    end = start
    while end < len(lines):
        line = lines[end]
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        end += 1
    return lines[start:end], start + 1


def collect_statements(body: list[str], first_line: int) -> list[Statement]:
    statements: list[Statement] = []
    current: list[str] = []
    current_line = first_line
    depth = 0
    for offset, raw in enumerate(body):
        if not raw.strip():
            continue
        if not current:
            if not raw.startswith("    "):
                raise DagError(f"line {first_line + offset}: malformed indentation")
            current_line = first_line + offset
        current.append(raw[4:] if len(current) == 0 else raw)
        depth += _delimiter_delta(raw)
        if depth < 0:
            raise DagError(f"line {first_line + offset}: unmatched closing delimiter")
        if depth == 0:
            statements.append(Statement("\n".join(current), current_line))
            current = []
    if current:
        raise DagError(f"line {current_line}: unterminated statement")
    return statements


def _dependencies(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(TEMP_RE.findall(text)))


def _parse_output(function: str, statement: Statement) -> Output | None:
    text = statement.text.strip()
    if function == "rhs_specie" and text.startswith("vset(ydot,"):
        match = re.match(r"vset\(ydot,\s*(\d+)\s*,", text)
        if not match:
            raise DagError(f"line {statement.line}: malformed ydot output")
        return Output(
            (int(match.group(1)),),
            statement,
            _dependencies(text),
            function,
        )
    if function == "rhs_eint" and text.startswith("return "):
        return Output((14,), statement, _dependencies(text), function)
    if function == "jac_nuc" and text.startswith("mset(jac,"):
        match = re.match(r"mset\(jac,\s*(\d+)\s*,\s*(\d+)\s*,", text)
        if not match:
            raise DagError(f"line {statement.line}: malformed Jacobian output")
        return Output(
            (int(match.group(1)), int(match.group(2))),
            statement,
            _dependencies(text),
            function,
        )
    return None


def parse_function(source: str, name: str, *, pin_manifest: bool = True) -> FunctionDag:
    body, first_line = function_body(source, name)
    statements = collect_statements(body, first_line)
    definitions: list[Definition] = []
    outputs: list[Output] = []
    prelude: list[Statement] = []
    seen: set[str] = set()

    for statement in statements:
        text = statement.text.strip()
        definition_match = DEF_RE.match(text)
        if definition_match:
            name_token = definition_match.group(1)
            if name_token in seen:
                raise DagError(f"line {statement.line}: duplicate definition {name_token}")
            expression = text[definition_match.end() :]
            dependencies = _dependencies(expression)
            for dependency in dependencies:
                if dependency not in seen:
                    raise DagError(
                        f"line {statement.line}: unresolved or use-before-definition "
                        f"{dependency}"
                    )
            definitions.append(Definition(name_token, statement, dependencies))
            seen.add(name_token)
            continue

        output = _parse_output(name, statement)
        if output is not None:
            for dependency in output.dependencies:
                if dependency not in seen:
                    raise DagError(
                        f"line {statement.line}: unresolved output dependency {dependency}"
                    )
            outputs.append(output)
            continue

        if not definitions and text == "var T = state.T":
            prelude.append(statement)
            continue
        raise DagError(f"line {statement.line}: unsupported statement in {name}: {text[:60]}")

    dag = FunctionDag(name, tuple(prelude), tuple(definitions), tuple(outputs))
    if pin_manifest:
        expected = MANIFEST[name]
        actual = (len(definitions), len(outputs))
        if actual != expected:
            raise DagError(
                f"{name} manifest drift: expected {expected[0]}/{expected[1]}, "
                f"found {actual[0]}/{actual[1]}"
            )
    return dag


def transitive_closure(dag: FunctionDag, output: Output) -> set[str]:
    by_name = dag.by_name
    closure: set[str] = set()
    pending = list(output.dependencies)
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        pending.extend(by_name[name].dependencies)
    return closure


def merge_base_dags(dags: Iterable[FunctionDag]) -> FunctionDag:
    """Merge Jacobian and RHS DAGs with deterministic structural CSE."""
    definitions: list[Definition] = []
    outputs: list[Output] = []
    expression_names: dict[str, str] = {}

    for dag in dags:
        local_names: dict[str, str] = {}
        for definition in dag.definitions:
            text = definition.statement.text.strip()
            match = DEF_RE.match(text)
            if match is None:
                raise DagError(
                    f"line {definition.statement.line}: malformed base definition"
                )
            expression = TEMP_RE.sub(
                lambda token: local_names[token.group(0)],
                text[match.end() :].strip(),
            )
            expression_key = re.sub(r"\s+", " ", expression).strip()
            name = expression_names.get(expression_key)
            if name is None:
                name = f"x99_{len(definitions)}"
                expression_names[expression_key] = name
                definitions.append(
                    Definition(
                        name,
                        Statement(
                            f"var {name} = {expression}",
                            definition.statement.line,
                        ),
                        _dependencies(expression),
                    )
                )
            local_names[definition.name] = name

        for output in dag.outputs:
            statement = TEMP_RE.sub(
                lambda token: local_names[token.group(0)],
                output.statement.text,
            )
            outputs.append(
                Output(
                    output.key,
                    Statement(statement, output.statement.line),
                    _dependencies(statement),
                    output.kind or dag.name,
                )
            )

    return FunctionDag("base", (), tuple(definitions), tuple(outputs))


def assign_outputs(dag: FunctionDag, warps: int) -> list[list[Output]]:
    assigned: list[list[Output]] = [[] for _ in range(warps)]
    for output in dag.outputs:
        if dag.name == "jac_nuc" or output.kind == "jac_nuc":
            warp = min(warps - 1, output.key[0] * warps // 15)
        else:
            warp = output.key[0] % warps
        assigned[warp].append(output)
    return assigned


def operation_cost(statement: str) -> int:
    expensive = 0
    for call, weight in (("exp(", 25), ("log(", 25), ("sqrt(", 20), ("cbrt(", 20)):
        expensive += statement.count(call) * weight
    arithmetic = sum(statement.count(operator) for operator in ("+", "-", "*", "/"))
    return max(1, expensive + arithmetic)


def subtree_costs(dag: FunctionDag) -> dict[str, int]:
    costs: dict[str, int] = {}
    for definition in dag.definitions:
        costs[definition.name] = operation_cost(definition.statement.text) + sum(
            costs[dependency] for dependency in definition.dependencies
        )
    return costs


@dataclasses.dataclass
class Partition:
    outputs: list[list[Output]]
    closures: list[set[str]]
    exchanges: set[str]
    exchange_bytes: int


def partition(
    dag: FunctionDag, warps: int, threshold: int, batch_cells: int
) -> Partition:
    outputs = assign_outputs(dag, warps)
    closures: list[set[str]] = []
    claims: dict[str, int] = defaultdict(int)
    for warp_outputs in outputs:
        closure: set[str] = set()
        for output in warp_outputs:
            closure.update(transitive_closure(dag, output))
        closures.append(closure)
        for name in closure:
            claims[name] += 1
    costs = subtree_costs(dag)
    exchanges = {
        name
        for name, count in claims.items()
        if count >= 2 and costs[name] >= threshold
    }
    # One Float64 per batch cell for each exchanged scalar.
    exchange_bytes = len(exchanges) * 8 * batch_cells
    return Partition(outputs, closures, exchanges, exchange_bytes)


def ordered_exchange_names(
    dag: FunctionDag, exchanges: set[str]
) -> list[str]:
    return [
        definition.name
        for definition in dag.definitions
        if definition.name in exchanges
    ]


def closure_with_exchange_inputs(
    dag: FunctionDag,
    outputs: Iterable[Output],
    exchanges: set[str],
) -> set[str]:
    closure: set[str] = set()
    pending = [dependency for output in outputs for dependency in output.dependencies]
    while pending:
        name = pending.pop()
        if name in exchanges or name in closure:
            continue
        closure.add(name)
        pending.extend(dag.by_name[name].dependencies)
    return closure


def _indent_statement(text: str) -> str:
    return "\n".join("    " + line.lstrip() if index == 0 else line for index, line in enumerate(text.splitlines()))


def emit_function(
    dag: FunctionDag,
    warp: int,
    closure: set[str],
    outputs: Iterable[Output],
) -> str:
    if dag.name == "jac_nuc":
        signature = (
            f"def jac_slice_{warp}(state: BurnState, mut jac: MatTensor, "
            "X: SpeciesTensor, z: Float64):"
        )
    elif dag.name == "rhs_specie":
        signature = (
            f"def rhs_specie_slice_{warp}(state: BurnState, mut ydot: VecTensor, "
            "X: SpeciesTensor, z: Float64):"
        )
    else:
        signature = (
            f"def rhs_eint_slice_{warp}(state: BurnState, X: SpeciesTensor, "
            "z: Float64) -> Float64:"
        )
    lines = [signature]
    for statement in dag.prelude:
        lines.append(_indent_statement(statement.text))
    for definition in dag.definitions:
        if definition.name in closure:
            lines.append(_indent_statement(definition.statement.text))
    output_list = list(outputs)
    if output_list:
        lines.extend(_indent_statement(output.statement.text) for output in output_list)
    elif dag.name == "rhs_eint":
        lines.append("    return 0.0")
    else:
        lines.append("    pass")
    return "\n".join(lines)


def emit_file(
    path: Path,
    source_hash: str,
    dags: list[FunctionDag],
    partitions: list[Partition],
    warps: int,
    threshold: int,
) -> None:
    header = (
        "# Generated by tools/slice_dag.py; do not edit.\n"
        f"# input_sha256={source_hash} warps={warps} threshold={threshold}\n\n"
        "from std.math import abs, cbrt, exp, log\n"
        "from reproducer import BurnState, MatTensor, SpeciesTensor, VecTensor\n"
        "from reproducer import Log10, Pi, mset, portable_mul, powi_m3, powi_m5, powi_m7\n"
        "from reproducer import portable_sqrt, portable_sqrt as sqrt, truthy, vget, vset\n\n"
    )
    functions: list[str] = []
    for dag, dag_partition in zip(dags, partitions):
        for warp in range(warps):
            functions.append(
                emit_function(
                    dag,
                    warp,
                    dag_partition.closures[warp],
                    dag_partition.outputs[warp],
                )
            )
    path.write_text(header + "\n\n".join(functions) + "\n")


def _replace_shared_operands(
    text: str,
    slots: dict[str, int],
    boolean_names: set[str],
    scratch_offset: int,
    exchange_slots: dict[str, int] | None = None,
) -> str:
    exchange_slots = exchange_slots or {}

    def replace_temp(match: re.Match[str]) -> str:
        name = match.group(0)
        if name in exchange_slots:
            return f"exchange[cell * ExchangeSlots + {exchange_slots[name]}]"
        slot = scratch_offset + slots[name]
        value = f"scratch[cell * ScratchSlots + {slot}]"
        return f"truthy({value})" if name in boolean_names else value

    text = TEMP_RE.sub(replace_temp, text)
    return re.sub(
        r"vget\(X,\s*(\d+)\s*\)",
        lambda match: f"inputs[cell * 15 + {match.group(1)}]",
        text,
    )


def _emit_shared_output(
    dag: FunctionDag,
    output: Output,
    slots: dict[str, int],
    boolean_names: set[str],
    scratch_offset: int,
    exchange_slots: dict[str, int] | None = None,
) -> str:
    text = _replace_shared_operands(
        output.statement.text.strip(),
        slots,
        boolean_names,
        scratch_offset,
        exchange_slots,
    )
    kind = output.kind or dag.name
    jacobian = "jacobian" if dag.name == "base" else "outputs"
    rhs = "rhs" if dag.name == "base" else "outputs"
    if kind == "jac_nuc":
        match = re.match(
            r"mset\(jac,\s*(\d+)\s*,\s*(\d+)\s*,\s*(.*)\)\s*$",
            text,
            re.DOTALL,
        )
        if match is None:
            raise DagError(f"line {output.statement.line}: malformed shared J output")
        row, column, expression = match.groups()
        return (
            f"{jacobian}[cell * 225 + {row} * 15 + {column}] = "
            f"{expression}"
        )
    if kind == "rhs_specie":
        match = re.match(
            r"vset\(ydot,\s*(\d+)\s*,\s*(.*)\)\s*$",
            text,
            re.DOTALL,
        )
        if match is None:
            raise DagError(f"line {output.statement.line}: malformed shared RHS output")
        component, expression = match.groups()
        return f"{rhs}[cell * 15 + {component}] = {expression}"
    if not text.startswith("return "):
        raise DagError(f"line {output.statement.line}: malformed shared energy output")
    return f"{rhs}[cell * 15 + 14] = {text[len('return '):]}"


@dataclasses.dataclass
class SharedSchedule:
    ordered: list[Definition]
    slots: dict[str, int]
    outputs_after: dict[int, list[Output]]
    peak_slots: int


@dataclasses.dataclass
class RegionSchedule:
    ordered: list[Definition]
    outputs_after: dict[int, list[Output]]
    regions: list[tuple[int, int]]
    slots: dict[str, int]
    boolean_names: set[str]
    boundary_live: list[int]
    peak_slots: int


def expression_is_boolean(expression: str, boolean_names: set[str]) -> bool:
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return False

    def is_boolean(value: ast.AST) -> bool:
        if isinstance(value, (ast.BoolOp, ast.Compare)):
            return True
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not):
            return True
        if isinstance(value, ast.Constant) and isinstance(value.value, bool):
            return True
        if isinstance(value, ast.Name):
            return value.id in boolean_names
        if isinstance(value, ast.Call):
            return isinstance(value.func, ast.Name) and value.func.id == "truthy"
        if isinstance(value, ast.IfExp):
            return is_boolean(value.body) and is_boolean(value.orelse)
        return False

    return is_boolean(node)


def make_shared_schedule(
    dag: FunctionDag,
    closure: set[str],
    outputs: Iterable[Output],
) -> SharedSchedule:
    ordered = [
        definition for definition in dag.definitions if definition.name in closure
    ]
    output_list = list(outputs)
    positions = {
        definition.name: index for index, definition in enumerate(ordered)
    }
    outputs_after: dict[int, list[Output]] = defaultdict(list)
    use_counts = {definition.name: 0 for definition in ordered}
    for definition in ordered:
        for dependency in definition.dependencies:
            if dependency in use_counts:
                use_counts[dependency] += 1
    for output in output_list:
        ready_after = max(
            (
                positions[dependency]
                for dependency in output.dependencies
                if dependency in positions
            ),
            default=-1,
        )
        outputs_after[ready_after].append(output)
        for dependency in output.dependencies:
            if dependency in use_counts:
                use_counts[dependency] += 1

    slots: dict[str, int] = {}
    available: list[int] = []
    next_slot = 0
    live = 0
    peak = 0
    released: set[str] = set()

    def release(name: str) -> None:
        nonlocal live
        use_counts[name] -= 1
        if use_counts[name] == 0:
            available.append(slots[name])
            available.sort(reverse=True)
            live -= 1
            released.add(name)

    for index, definition in enumerate(ordered):
        if available:
            slots[definition.name] = available.pop()
        else:
            slots[definition.name] = next_slot
            next_slot += 1
        live += 1
        peak = max(peak, live)

        for dependency in definition.dependencies:
            if dependency in use_counts:
                release(dependency)
        for output in outputs_after.get(index, []):
            for dependency in output.dependencies:
                if dependency in use_counts:
                    release(dependency)
        if use_counts[definition.name] == 0 and definition.name not in released:
            available.append(slots[definition.name])
            available.sort(reverse=True)
            live -= 1
            released.add(definition.name)

    if live != 0:
        raise DagError(f"shared schedule for {dag.name} leaves {live} live values")
    return SharedSchedule(ordered, slots, dict(outputs_after), peak)


def make_region_schedule(
    dag: FunctionDag,
    closure: set[str],
    outputs: Iterable[Output],
    max_region_definitions: int,
) -> RegionSchedule:
    if max_region_definitions <= 0:
        raise DagError("shared region size must be positive")
    ordered = [
        definition for definition in dag.definitions if definition.name in closure
    ]
    positions = {
        definition.name: index for index, definition in enumerate(ordered)
    }
    output_list = list(outputs)
    outputs_after: dict[int, list[Output]] = defaultdict(list)
    last_use = dict(positions)
    for index, definition in enumerate(ordered):
        for dependency in definition.dependencies:
            if dependency in last_use:
                last_use[dependency] = max(last_use[dependency], index)
    for output in output_list:
        ready_after = max(
            (
                positions[dependency]
                for dependency in output.dependencies
                if dependency in positions
            ),
            default=-1,
        )
        outputs_after[ready_after].append(output)
        for dependency in output.dependencies:
            if dependency in last_use:
                last_use[dependency] = max(last_use[dependency], ready_after)

    live_after = [
        sum(
            positions[name] <= cut < last_use[name]
            for name in positions
        )
        for cut in range(max(0, len(ordered) - 1))
    ]
    regions: list[tuple[int, int]] = []
    start = 0
    while len(ordered) - start > max_region_definitions:
        first_cut = start + max_region_definitions // 2 - 1
        last_cut = min(
            start + max_region_definitions - 1, len(ordered) - 2
        )
        cut = min(
            range(first_cut, last_cut + 1),
            key=lambda candidate: (live_after[candidate], -candidate),
        )
        regions.append((start, cut))
        start = cut + 1
    if start < len(ordered):
        regions.append((start, len(ordered) - 1))

    cuts = [end for _, end in regions[:-1]]
    boundary_values = [
        [
            name
            for name in positions
            if positions[name] <= cut < last_use[name]
        ]
        for cut in cuts
    ]
    intervals: dict[str, tuple[int, int]] = {}
    for boundary, names in enumerate(boundary_values):
        for name in names:
            if name in intervals:
                intervals[name] = (intervals[name][0], boundary)
            else:
                intervals[name] = (boundary, boundary)
    starts: dict[int, list[str]] = defaultdict(list)
    ends: dict[int, list[str]] = defaultdict(list)
    for name, (first, last) in intervals.items():
        starts[first].append(name)
        ends[last].append(name)
    slots: dict[str, int] = {}
    available: list[int] = []
    next_slot = 0
    for boundary in range(len(cuts)):
        for name in sorted(starts[boundary], key=lambda item: positions[item]):
            if available:
                slots[name] = available.pop()
            else:
                slots[name] = next_slot
                next_slot += 1
        for name in ends[boundary]:
            available.append(slots[name])
        available.sort(reverse=True)

    boolean_names: set[str] = set()
    for definition in ordered:
        match = DEF_RE.match(definition.statement.text.strip())
        if match is None:
            raise DagError(
                f"line {definition.statement.line}: malformed shared definition"
            )
        expression = definition.statement.text.strip()[match.end() :].strip()
        if expression_is_boolean(expression, boolean_names):
            boolean_names.add(definition.name)
    boundary_live = [len(names) for names in boundary_values]
    return RegionSchedule(
        ordered,
        dict(outputs_after),
        regions,
        slots,
        boolean_names,
        boundary_live,
        max(boundary_live, default=0),
    )


def emit_shared_function(
    dag: FunctionDag,
    warp: int,
    closure: set[str],
    outputs: Iterable[Output],
    scratch_offset: int,
    max_region_definitions: int,
    exchange_slots: dict[str, int] | None = None,
) -> tuple[str, RegionSchedule]:
    exchange_slots = exchange_slots or {}
    schedule = make_region_schedule(
        dag, closure, outputs, max_region_definitions
    )

    def replace_region_operands(text: str, local_names: set[str]) -> str:
        def replace_temp(match: re.Match[str]) -> str:
            name = match.group(0)
            if name in local_names:
                return name
            if name in exchange_slots:
                return (
                    "exchange[cell * ExchangeSlots + "
                    f"{exchange_slots[name]}]"
                )
            if name not in schedule.slots:
                raise DagError(
                    f"{dag.name} warp {warp}: {name} is neither local nor persisted"
                )
            value = (
                "scratch[cell * ScratchSlots + "
                f"{scratch_offset + schedule.slots[name]}]"
            )
            return f"truthy({value})" if name in schedule.boolean_names else value

        text = TEMP_RE.sub(replace_temp, text)
        return re.sub(
            r"vget\(X,\s*(\d+)\s*\)",
            lambda match: f"inputs[cell * 15 + {match.group(1)}]",
            text,
        )

    def emit_region_output(output: Output, local_names: set[str]) -> str:
        text = replace_region_operands(output.statement.text.strip(), local_names)
        kind = output.kind or dag.name
        jacobian = "jacobian" if dag.name == "base" else "outputs"
        rhs = "rhs" if dag.name == "base" else "outputs"
        if kind == "jac_nuc":
            match = re.match(
                r"mset\(jac,\s*(\d+)\s*,\s*(\d+)\s*,\s*(.*)\)\s*$",
                text,
                re.DOTALL,
            )
            if match is None:
                raise DagError(
                    f"line {output.statement.line}: malformed shared J output"
                )
            row, column, expression = match.groups()
            return (
                f"{jacobian}[cell * 225 + {row} * 15 + {column}] = "
                f"{expression}"
            )
        if kind == "rhs_specie":
            match = re.match(
                r"vset\(ydot,\s*(\d+)\s*,\s*(.*)\)\s*$",
                text,
                re.DOTALL,
            )
            if match is None:
                raise DagError(
                    f"line {output.statement.line}: malformed shared RHS output"
                )
            component, expression = match.groups()
            return f"{rhs}[cell * 15 + {component}] = {expression}"
        if not text.startswith("return "):
            raise DagError(
                f"line {output.statement.line}: malformed shared energy output"
            )
        return f"{rhs}[cell * 15 + 14] = {text[len('return '):]}"

    functions: list[str] = []
    for region_index, (start, end) in enumerate(schedule.regions):
        helper_name = f"_{dag.name}_slice_{warp}_region_{region_index}"
        output_arguments = (
            "jacobian: SharedPointer, rhs: SharedPointer, "
            if dag.name == "base"
            else "outputs: SharedPointer, "
        )
        lines = [
            "@no_inline",
            f"def {helper_name}(inputs: SharedPointer, {output_arguments}",
            "    exchange: SharedPointer, scratch: SharedPointer, ",
            "    cell: Int, z: Float64):",
        ]
        region_text = "\n".join(
            definition.statement.text
            for definition in schedule.ordered[start : end + 1]
        )
        region_text += "\n" + "\n".join(
            output.statement.text
            for ready, ready_outputs in schedule.outputs_after.items()
            if start <= ready <= end
            for output in ready_outputs
        )
        if re.search(r"\bT\b", region_text):
            lines.append("    var T = inputs[cell * 15 + 14]")
        local_names: set[str] = set()
        for index in range(start, end + 1):
            definition = schedule.ordered[index]
            match = DEF_RE.match(definition.statement.text.strip())
            if match is None:
                raise DagError(
                    f"line {definition.statement.line}: malformed shared definition"
                )
            expression = replace_region_operands(
                definition.statement.text.strip()[match.end() :].strip(),
                local_names,
            )
            lines.append(f"    var {definition.name} = {expression}")
            local_names.add(definition.name)
            for output in schedule.outputs_after.get(index, []):
                lines.append("    " + emit_region_output(output, local_names))
        if region_index < len(schedule.regions) - 1:
            cut = end
            for name in sorted(
                local_names, key=lambda item: schedule.slots.get(item, -1)
            ):
                if name not in schedule.slots:
                    continue
                needed_later = any(
                    name in definition.dependencies
                    for definition in schedule.ordered[cut + 1 :]
                ) or any(
                    name in output.dependencies
                    for ready, ready_outputs in schedule.outputs_after.items()
                    if ready > cut
                    for output in ready_outputs
                )
                if not needed_later:
                    continue
                value = (
                    f"(1.0 if {name} else 0.0)"
                    if name in schedule.boolean_names
                    else name
                )
                lines.append(
                    "    scratch[cell * ScratchSlots + "
                    f"{scratch_offset + schedule.slots[name]}] = {value}"
                )
        functions.append("\n".join(lines))

    signature = f"def {dag.name}_slice_{warp}_shared("
    signature += "inputs: SharedPointer, "
    if dag.name == "base":
        signature += "jacobian: SharedPointer, rhs: SharedPointer, "
    else:
        signature += "outputs: SharedPointer, "
    signature += "exchange: SharedPointer, "
    signature += "scratch: SharedPointer, cell: Int, z: Float64):"
    wrapper = [signature]
    empty_locals: set[str] = set()
    for output in schedule.outputs_after.get(-1, []):
        wrapper.append("    " + emit_region_output(output, empty_locals))
    for region_index in range(len(schedule.regions)):
        call = (
            f"    _{dag.name}_slice_{warp}_region_{region_index}("
            "inputs, "
        )
        call += "jacobian, rhs, " if dag.name == "base" else "outputs, "
        call += "exchange, "
        wrapper.append(call + "scratch, cell, z)")
    if len(wrapper) == 1:
        wrapper.append("    pass")
    functions.append("\n".join(wrapper))
    return "\n\n".join(functions), schedule


def emit_lifetime_shared_function(
    dag: FunctionDag,
    warp: int,
    closure: set[str],
    outputs: Iterable[Output],
    scratch_offset: int,
    exchange_slots: dict[str, int] | None = None,
) -> tuple[str, SharedSchedule]:
    exchange_slots = exchange_slots or {}
    schedule = make_shared_schedule(dag, closure, outputs)
    boolean_names: set[str] = set()
    signature = f"def {dag.name}_slice_{warp}_shared("
    signature += "inputs: SharedPointer, "
    if dag.name == "base":
        signature += "jacobian: SharedPointer, rhs: SharedPointer, "
    else:
        signature += "outputs: SharedPointer, "
    signature += "exchange: SharedPointer, "
    signature += "scratch: SharedPointer, cell: Int, z: Float64):"
    lines = [signature, "    var T = inputs[cell * 15 + 14]"]
    for output in schedule.outputs_after.get(-1, []):
        lines.append(
            "    "
            + _emit_shared_output(
                dag,
                output,
                schedule.slots,
                boolean_names,
                scratch_offset,
                exchange_slots,
            )
        )
    for index, definition in enumerate(schedule.ordered):
        match = DEF_RE.match(definition.statement.text.strip())
        if match is None:
            raise DagError(
                f"line {definition.statement.line}: malformed shared definition"
            )
        original_expression = definition.statement.text.strip()[
            match.end() :
        ].strip()
        expression = _replace_shared_operands(
            original_expression,
            schedule.slots,
            boolean_names,
            scratch_offset,
            exchange_slots,
        )
        if expression_is_boolean(original_expression, boolean_names):
            boolean_names.add(definition.name)
            expression = f"(1.0 if {expression} else 0.0)"
        lines.append(
            "    scratch[cell * ScratchSlots + "
            f"{scratch_offset + schedule.slots[definition.name]}] = "
            f"{expression}"
        )
        for output in schedule.outputs_after.get(index, []):
            lines.append(
                "    "
                + _emit_shared_output(
                    dag,
                    output,
                    schedule.slots,
                    boolean_names,
                    scratch_offset,
                    exchange_slots,
                )
            )
    if not schedule.ordered and not schedule.outputs_after:
        lines.append("    pass")
    return "\n".join(lines), schedule


def emit_exchange_producer(
    dag: FunctionDag,
    exchange_slots: dict[str, int],
) -> tuple[str, SharedSchedule]:
    names = sorted(exchange_slots, key=exchange_slots.__getitem__)
    outputs = [
        Output(
            (exchange_slots[name],),
            Statement("", dag.by_name[name].statement.line),
            (name,),
        )
        for name in names
    ]
    closure: set[str] = set()
    for output in outputs:
        closure.update(transitive_closure(dag, output))
    schedule = make_shared_schedule(dag, closure, outputs)
    boolean_names: set[str] = set()
    lines = [
        f"def {dag.name}_exchange_shared(",
        "    inputs: SharedPointer, exchange: SharedPointer, ",
        "    scratch: SharedPointer, cell: Int, z: Float64):",
    ]
    if not schedule.ordered:
        lines.append("    pass")
        return "\n".join(lines), schedule
    lines.append("    var T = inputs[cell * 15 + 14]")
    for index, definition in enumerate(schedule.ordered):
        match = DEF_RE.match(definition.statement.text.strip())
        if match is None:
            raise DagError(
                f"line {definition.statement.line}: malformed shared definition"
            )
        original_expression = definition.statement.text.strip()[
            match.end() :
        ].strip()
        expression = _replace_shared_operands(
            original_expression,
            schedule.slots,
            boolean_names,
            0,
        )
        if expression_is_boolean(original_expression, boolean_names):
            boolean_names.add(definition.name)
            expression = f"(1.0 if {expression} else 0.0)"
        lines.append(
            "    scratch[cell * ScratchSlots + "
            f"{schedule.slots[definition.name]}] = {expression}"
        )
        for output in schedule.outputs_after.get(index, []):
            name = output.dependencies[0]
            lines.append(
                "    exchange[cell * ExchangeSlots + "
                f"{output.key[0]}] = scratch[cell * ScratchSlots + "
                f"{schedule.slots[name]}]"
            )
    return "\n".join(lines), schedule


def emit_shared_file(
    path: Path,
    source_hash: str,
    dags: list[FunctionDag],
    partitions: list[Partition],
    warps: int,
    threshold: int,
    max_region_definitions: int,
    concurrent_warps: int,
    warp_order: tuple[int, ...],
) -> dict[str, object]:
    if concurrent_warps <= 0 or concurrent_warps > warps:
        raise DagError("shared DAG wave width must be between 1 and warps")
    if len(warp_order) != warps or sorted(warp_order) != list(range(warps)):
        raise DagError("shared DAG warp order must be a permutation of all warps")
    exchange_maps: list[dict[str, int]] = []
    next_exchange_slot = 0
    for dag, dag_partition in zip(dags, partitions):
        names = ordered_exchange_names(dag, dag_partition.exchanges)
        exchange_map = {
            name: next_exchange_slot + index for index, name in enumerate(names)
        }
        exchange_maps.append(exchange_map)
        next_exchange_slot += len(names)
    effective_closures: list[list[set[str]]] = []
    for dag, dag_partition in zip(dags, partitions):
        effective_closures.append(
            [
                closure_with_exchange_inputs(
                    dag, dag_partition.outputs[warp], dag_partition.exchanges
                )
                for warp in range(warps)
            ]
        )
    warp_slots = [0 for _ in range(warps)]
    region_counts = [0 for _ in range(warps)]
    boundary_live: list[list[int]] = [[] for _ in range(warps)]
    for dag_index, (dag, dag_partition) in enumerate(zip(dags, partitions)):
        for warp in range(warps):
            closure = effective_closures[dag_index][warp]
            if max_region_definitions == 0:
                schedule = make_shared_schedule(
                    dag,
                    closure,
                    dag_partition.outputs[warp],
                )
            else:
                schedule = make_region_schedule(
                    dag,
                    closure,
                    dag_partition.outputs[warp],
                    max_region_definitions,
                )
            warp_slots[warp] = max(warp_slots[warp], schedule.peak_slots)
            if schedule.peak_slots >= warp_slots[warp]:
                if isinstance(schedule, RegionSchedule):
                    region_counts[warp] = len(schedule.regions)
                    boundary_live[warp] = schedule.boundary_live
                else:
                    region_counts[warp] = 1 if schedule.ordered else 0
                    boundary_live[warp] = []
    scratch_offsets = [0 for _ in range(warps)]
    wave_slots: list[int] = []
    wave_warps = [warp_order[:concurrent_warps]]
    if concurrent_warps < warps:
        wave_warps.append(warp_order[concurrent_warps:])
    for logical_warps in wave_warps:
        offset = 0
        for warp in logical_warps:
            scratch_offsets[warp] = offset
            offset += warp_slots[warp]
        wave_slots.append(offset)
    producer_functions: list[str] = []
    producer_reports: list[dict[str, object]] = []
    producer_peak_slots = 0
    for dag, exchange_map in zip(dags, exchange_maps):
        function, schedule = emit_exchange_producer(dag, exchange_map)
        producer_functions.append(function)
        producer_peak_slots = max(producer_peak_slots, schedule.peak_slots)
        producer_reports.append(
            {
                "function": dag.name,
                "definitions": len(schedule.ordered),
                "scratch_slots": schedule.peak_slots,
                "exchange_names": sorted(exchange_map, key=exchange_map.__getitem__),
            }
        )
    scratch_slots = max([*wave_slots, producer_peak_slots])
    header = (
        "# Generated by tools/slice_dag.py; do not edit.\n"
        f"# input_sha256={source_hash} warps={warps} threshold={threshold} "
        f"region_defs={max_region_definitions} wave_warps={concurrent_warps} "
        f"warp_order={','.join(map(str, warp_order))}\n\n"
        "from std.gpu.memory import AddressSpace\n"
        "from std.math import abs, cbrt, exp, log\n"
        "from reproducer import Log10, Pi, portable_mul, powi_m3, powi_m5, powi_m7\n"
        "from reproducer import portable_sqrt, portable_sqrt as sqrt, truthy\n\n"
        "comptime SharedPointer = UnsafePointer[\n"
        "    Scalar[DType.float64], MutUntrackedOrigin,\n"
        "    address_space=AddressSpace.SHARED,\n"
        "]\n"
        f"comptime ScratchSlots = {scratch_slots}\n\n"
        f"comptime ExchangeSlots = {next_exchange_slot}\n\n"
        f"comptime ConcurrentWarps = {concurrent_warps}\n\n"
    )
    functions = producer_functions
    for dag_index, (dag, dag_partition, exchange_map) in enumerate(
        zip(dags, partitions, exchange_maps)
    ):
        for warp in range(warps):
            closure = effective_closures[dag_index][warp]
            if max_region_definitions == 0:
                function, _ = emit_lifetime_shared_function(
                    dag,
                    warp,
                    closure,
                    dag_partition.outputs[warp],
                    scratch_offsets[warp],
                    exchange_map,
                )
            else:
                function, _ = emit_shared_function(
                    dag,
                    warp,
                    closure,
                    dag_partition.outputs[warp],
                    scratch_offsets[warp],
                    max_region_definitions,
                    exchange_map,
                )
            functions.append(function)
    path.write_text(header + "\n\n".join(functions) + "\n")
    return {
        "scratch_slots": scratch_slots,
        "warp_slots": warp_slots,
        "scratch_offsets": scratch_offsets,
        "wave_slots": wave_slots,
        "max_region_definitions": max_region_definitions,
        "concurrent_warps": concurrent_warps,
        "warp_order": list(warp_order),
        "wave_warps": [list(wave) for wave in wave_warps],
        "region_counts": region_counts,
        "boundary_live": boundary_live,
        "exchange_slots": next_exchange_slot,
        "exchange_producers": producer_reports,
    }


CUDA_CALL_NAMES = {
    "portable_mul": "structured_mul",
    "portable_sqrt": "structured_sqrt",
    "truthy": "structured_truthy",
    "abs": "fabs",
}
CUDA_NAME_MAP = {
    "Log10": "kLog10",
    "Pi": "kPi",
    "ScratchSlots": "kScratchSlots",
    "ExchangeSlots": "kExchangeSlots",
}


def cuda_expression(expression: str) -> str:
    """Translate the generated Mojo expression subset to CUDA C++."""
    try:
        root = ast.parse(expression, mode="eval").body
    except SyntaxError as error:
        raise DagError(f"cannot parse CUDA expression: {expression[:80]}") from error

    def emit(node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if node.value is True:
                return "true"
            if node.value is False:
                return "false"
            if isinstance(node.value, (int, float)):
                return repr(node.value)
        if isinstance(node, ast.Name):
            return CUDA_NAME_MAP.get(node.id, node.id)
        if isinstance(node, ast.Subscript):
            return f"{emit(node.value)}[{emit(node.slice)}]"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = CUDA_CALL_NAMES.get(node.func.id, node.func.id)
            return f"{function}({', '.join(emit(argument) for argument in node.args)})"
        if isinstance(node, ast.BinOp):
            operators = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }
            operator = operators.get(type(node.op))
            if operator is not None:
                return f"({emit(node.left)} {operator} {emit(node.right)})"
        if isinstance(node, ast.UnaryOp):
            operators = {ast.USub: "-", ast.UAdd: "+", ast.Not: "!"}
            operator = operators.get(type(node.op))
            if operator is not None:
                return f"({operator}{emit(node.operand)})"
        if isinstance(node, ast.BoolOp):
            operator = " && " if isinstance(node.op, ast.And) else " || "
            return "(" + operator.join(emit(value) for value in node.values) + ")"
        if isinstance(node, ast.Compare):
            operators = {
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }
            comparisons: list[str] = []
            left = node.left
            for operator_node, right in zip(node.ops, node.comparators):
                operator = operators.get(type(operator_node))
                if operator is None:
                    break
                comparisons.append(f"({emit(left)} {operator} {emit(right)})")
                left = right
            else:
                return "(" + " && ".join(comparisons) + ")"
        if isinstance(node, ast.IfExp):
            return f"({emit(node.test)} ? {emit(node.body)} : {emit(node.orelse)})"
        raise DagError(f"unsupported CUDA expression node: {ast.dump(node)}")

    return emit(root)


def emit_cuda_shared_header(
    mojo_path: Path,
    cuda_path: Path,
    namespace: str,
) -> None:
    """Translate a generated shared-memory Mojo DAG file to CUDA source."""
    source_lines = mojo_path.read_text().splitlines()
    constants: dict[str, int] = {}
    for line in source_lines:
        match = re.match(
            r"^comptime (ScratchSlots|ExchangeSlots|ConcurrentWarps) = (\d+)$",
            line,
        )
        if match:
            constants[match.group(1)] = int(match.group(2))
    if set(constants) != {"ScratchSlots", "ExchangeSlots", "ConcurrentWarps"}:
        raise DagError(f"missing shared constants in {mojo_path}")

    output = [
        "// Generated by tools/slice_dag.py; do not edit.",
        "#pragma once",
        "",
        f"namespace {namespace} {{",
        f"inline constexpr int kScratchSlots = {constants['ScratchSlots']};",
        f"inline constexpr int kExchangeSlots = {constants['ExchangeSlots']};",
        f"inline constexpr int kConcurrentWarps = {constants['ConcurrentWarps']};",
        "",
    ]
    index = 0
    emitted_functions = 0
    while index < len(source_lines):
        line = source_lines[index]
        if not line.startswith("def "):
            index += 1
            continue
        no_inline = (
            index > 0 and source_lines[index - 1].strip() == "@no_inline"
        )
        header = line.strip()
        while not header.endswith("):"):
            index += 1
            if index >= len(source_lines):
                raise DagError(f"unterminated function signature in {mojo_path}")
            header += " " + source_lines[index].strip()
        match = re.match(r"def\s+(\w+)\s*\((.*)\):$", header)
        if match is None:
            raise DagError(f"malformed function signature in {mojo_path}: {header}")
        function_name, arguments_text = match.groups()
        argument_names = re.findall(r"\b([a-z_][a-z0-9_]*)\s*:", arguments_text)
        arguments: list[str] = []
        for argument in argument_names:
            if argument == "cell":
                arguments.append("int cell")
            elif argument == "z":
                arguments.append("double z")
            elif argument == "inputs":
                arguments.append("const double* __restrict__ inputs")
            else:
                arguments.append(f"double* __restrict__ {argument}")
        function_attribute = "__noinline__" if no_inline else "__forceinline__"
        output.append(
            f"__device__ {function_attribute} void "
            f"{function_name}({', '.join(arguments)}) {{"
        )
        index += 1
        while index < len(source_lines) and (
            not source_lines[index] or source_lines[index].startswith("    ")
        ):
            statement = source_lines[index].strip()
            index += 1
            if not statement or statement == "pass":
                continue
            if statement.startswith("var "):
                statement = statement[len("var ") :]
                left, separator, right = statement.partition(" = ")
                if not separator:
                    raise DagError(f"malformed CUDA local in {mojo_path}: {statement}")
                output.append(f"    const double {left} = {cuda_expression(right)};")
                continue
            left, separator, right = statement.partition(" = ")
            if not separator:
                output.append(f"    {cuda_expression(statement)};")
                continue
            left = left.replace("ScratchSlots", "kScratchSlots").replace(
                "ExchangeSlots", "kExchangeSlots"
            )
            output.append(f"    {cuda_expression(left)} = {cuda_expression(right)};")
        output.extend(["}", ""])
        emitted_functions += 1
    if emitted_functions == 0:
        raise DagError(f"no shared functions found in {mojo_path}")
    output.extend([f"}}  // namespace {namespace}", ""])
    cuda_path.write_text("\n".join(output))


def make_report(
    source_hash: str,
    dags: list[FunctionDag],
    partitions: list[Partition],
    warps: int,
    batch_cells: int,
    threshold: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "input_sha256": source_hash,
        "warps": warps,
        "batch_cells": batch_cells,
        "threshold": threshold,
        "functions": {},
    }
    function_reports: dict[str, object] = {}
    for dag, dag_partition in zip(dags, partitions):
        costs = subtree_costs(dag)
        warp_reports = []
        for warp in range(warps):
            closure = dag_partition.closures[warp]
            warp_reports.append(
                {
                    "warp": warp,
                    "outputs": [list(output.key) for output in dag_partition.outputs[warp]],
                    "definitions": len(closure),
                    "weighted_operations": sum(costs[name] for name in closure),
                    "estimated_peak_live_temporaries": make_shared_schedule(
                        dag, closure, dag_partition.outputs[warp]
                    ).peak_slots,
                }
            )
        function_reports[dag.name] = {
            "definitions": len(dag.definitions),
            "outputs": len(dag.outputs),
            "exchanged_nodes": sorted(dag_partition.exchanges),
            "recomputed_nodes": sorted(
                set().union(*dag_partition.closures) - dag_partition.exchanges
            ),
            "exchange_bytes": dag_partition.exchange_bytes,
            "warps": warp_reports,
        }
    result["functions"] = function_reports
    result["exchange_bytes"] = sum(part.exchange_bytes for part in partitions)
    return result


def generate(
    input_path: Path,
    output_dir: Path,
    *,
    warps: int = 8,
    batch_cells: int = 32,
    threshold: int = 400,
    max_exchange_bytes: int = 8192,
    shared_region_definitions: int = 0,
    shared_wave_warps: int = 3,
    shared_warp_order: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if warps <= 0:
        raise DagError("warps must be positive")
    if batch_cells <= 0:
        raise DagError("batch cells must be positive")
    if shared_warp_order is None:
        if warps == 8:
            shared_warp_order = (7, 1, 0, 2, 3, 4, 5, 6)
        else:
            shared_warp_order = tuple(range(warps))
    source_bytes = input_path.read_bytes()
    source = source_bytes.decode("utf-8")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    dags = {
        name: parse_function(source, name) for name in MANIFEST
    }
    partitions = {
        name: partition(dag, warps, threshold, batch_cells)
        for name, dag in dags.items()
    }
    base_dag = merge_base_dags(
        (dags["jac_nuc"], dags["rhs_specie"], dags["rhs_eint"])
    )
    base_partition = partition(base_dag, warps, threshold, batch_cells)
    total_exchange = sum(part.exchange_bytes for part in partitions.values())
    if total_exchange > max_exchange_bytes:
        raise DagError(
            f"exchange arena requires {total_exchange} bytes, exceeds "
            f"{max_exchange_bytes}"
        )
    if base_partition.exchange_bytes > max_exchange_bytes:
        raise DagError(
            f"fused base exchange arena requires "
            f"{base_partition.exchange_bytes} bytes, exceeds "
            f"{max_exchange_bytes}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    emit_file(
        output_dir / "slices_jac.mojo",
        source_hash,
        [dags["jac_nuc"]],
        [partitions["jac_nuc"]],
        warps,
        threshold,
    )
    jac_shared = emit_shared_file(
        output_dir / "slices_jac_shared.mojo",
        source_hash,
        [dags["jac_nuc"]],
        [partitions["jac_nuc"]],
        warps,
        threshold,
        shared_region_definitions,
        shared_wave_warps,
        shared_warp_order,
    )
    rhs_shared = emit_shared_file(
        output_dir / "slices_rhs_shared.mojo",
        source_hash,
        [dags["rhs_specie"], dags["rhs_eint"]],
        [partitions["rhs_specie"], partitions["rhs_eint"]],
        warps,
        threshold,
        shared_region_definitions,
        shared_wave_warps,
        shared_warp_order,
    )
    base_shared = emit_shared_file(
        output_dir / "slices_base_shared.mojo",
        source_hash,
        [base_dag],
        [base_partition],
        warps,
        threshold,
        shared_region_definitions,
        shared_wave_warps,
        shared_warp_order,
    )
    emit_cuda_shared_header(
        output_dir / "slices_base_shared.mojo",
        output_dir / "slices_base_shared.cuh",
        "structured_cuda_base",
    )
    emit_cuda_shared_header(
        output_dir / "slices_rhs_shared.mojo",
        output_dir / "slices_rhs_shared.cuh",
        "structured_cuda_rhs",
    )
    emit_file(
        output_dir / "slices_rhs.mojo",
        source_hash,
        [dags["rhs_specie"], dags["rhs_eint"]],
        [partitions["rhs_specie"], partitions["rhs_eint"]],
        warps,
        threshold,
    )
    report = make_report(
        source_hash,
        list(dags.values()),
        list(partitions.values()),
        warps,
        batch_cells,
        threshold,
    )
    report["shared_scratch"] = {
        "jacobian": jac_shared,
        "rhs": rhs_shared,
        "base": base_shared,
    }
    report["fused_base"] = {
        "input_definitions": sum(len(dag.definitions) for dag in dags.values()),
        "definitions": len(base_dag.definitions),
        "eliminated_definitions": (
            sum(len(dag.definitions) for dag in dags.values())
            - len(base_dag.definitions)
        ),
        "outputs": len(base_dag.outputs),
        "exchange_bytes": base_partition.exchange_bytes,
    }
    report["generated_source_bytes"] = {
        name: (output_dir / name).stat().st_size
        for name in (
            "slices_jac.mojo",
            "slices_rhs.mojo",
            "slices_jac_shared.mojo",
            "slices_rhs_shared.mojo",
            "slices_base_shared.mojo",
            "slices_rhs_shared.cuh",
            "slices_base_shared.cuh",
        )
    }
    (output_dir / "slice_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=Path("reproducer.mojo"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--batch-cells", type=int, default=32)
    parser.add_argument("--threshold", type=int, default=400)
    parser.add_argument("--max-exchange-bytes", type=int, default=8192)
    parser.add_argument("--shared-region-definitions", type=int, default=0)
    parser.add_argument("--shared-wave-warps", type=int, default=3)
    parser.add_argument(
        "--shared-warp-order", default="7,1,0,2,3,4,5,6"
    )
    args = parser.parse_args()
    try:
        report = generate(
            args.input,
            args.output_dir,
            warps=args.warps,
            batch_cells=args.batch_cells,
            threshold=args.threshold,
            max_exchange_bytes=args.max_exchange_bytes,
            shared_region_definitions=args.shared_region_definitions,
            shared_wave_warps=args.shared_wave_warps,
            shared_warp_order=tuple(
                int(warp) for warp in args.shared_warp_order.split(",")
            ),
        )
    except DagError as error:
        parser.error(str(error))
    print(json.dumps({"exchange_bytes": report["exchange_bytes"]}))


if __name__ == "__main__":
    main()
