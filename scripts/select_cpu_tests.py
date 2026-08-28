#!/usr/bin/env python3

"""List test modules that are safe to import in the CPU CI environment."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

NON_CPU_MARKERS = frozenset(
    {
        "cuda",
        "integration",
        "multi_npu",
        "nccl",
        "npu",
        "sglang",
        "vllm",
    }
)


def _pytest_markers(node: ast.AST) -> set[str]:
    markers: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute):
            continue
        mark = child.value
        if not isinstance(mark, ast.Attribute) or mark.attr != "mark":
            continue
        if isinstance(mark.value, ast.Name) and mark.value.id == "pytest":
            markers.add(child.attr)
    return markers


def module_markers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    markers: set[str] = set()
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in targets
        ):
            continue
        markers.update(_pytest_markers(statement.value))
    return markers


def select_cpu_test_files(
    tests_root: Path, ignored_paths: tuple[Path, ...] = ()
) -> list[Path]:
    ignored_paths = tuple(path.resolve() for path in ignored_paths)
    selected: list[Path] = []
    for path in sorted(tests_root.rglob("test_*.py")):
        resolved_path = path.resolve()
        if any(resolved_path.is_relative_to(ignored) for ignored in ignored_paths):
            continue
        if module_markers(path).isdisjoint(NON_CPU_MARKERS):
            selected.append(path)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    parser.add_argument("--ignore", action="append", type=Path, default=[])
    args = parser.parse_args()

    for path in select_cpu_test_files(args.tests_root, tuple(args.ignore)):
        print(path)


if __name__ == "__main__":
    main()
