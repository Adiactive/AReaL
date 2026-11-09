"""Disable implicit CWD binds in the pinned Gym SWE command builder."""

import argparse
import ast
from pathlib import Path


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_build_apptainer_command"
    ]
    if len(methods) != 1:
        raise ValueError("Expected one pinned Gym _build_apptainer_command method")
    method = methods[0]
    lines = source.splitlines(keepends=True)
    body = "".join(lines[method.lineno - 1 : method.end_lineno])
    original = (
        "apptainer exec --writable-tmpfs --cleanenv --pid "
        "--no-mount home,tmp,bind-paths "
    )
    isolated = original.rstrip() + ",cwd "
    if body.count("--no-mount") != 1 or (
        body.count(original) + body.count(isolated) != 1
    ):
        raise ValueError("Pinned Gym Apptainer command changed; refusing to patch")
    if isolated in body:
        return source
    # An implicit writable parent mount can hide explicit read-only dependencies.
    body = body.replace(original, isolated)
    result = (
        "".join(lines[: method.lineno - 1]) + body + "".join(lines[method.end_lineno :])
    )
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    original = args.source.read_text()
    try:
        updated = patch_source(original)
    except (SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    if updated != original:
        args.source.write_text(updated)


if __name__ == "__main__":
    main()
