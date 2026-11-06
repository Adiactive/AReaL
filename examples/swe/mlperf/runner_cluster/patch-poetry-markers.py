"""Fix build-tool marker evaluation without changing the benchmark lock or kernel."""

import argparse
import ast
import difflib
import hashlib
import inspect
import json
from pathlib import Path

ORIGINAL_VALIDATION = "return all(m.validate(environment) for m in self._markers)"
FIXED_VALIDATION = (
    "# SWE runner: evaluate OS guards before platform_release comparisons.\n"
    "        return all(m.validate(environment) for m in sorted(\n"
    '            self._markers, key=lambda marker: "platform_release" in str(marker)\n'
    "        ))"
)
BASE_PACKAGING_PIN = "$miniforge_dir/bin/python -m pip install -q 'packaging==26.0'"
SETUP_FIX = (
    "# Keep Conda/Poetry packaging dependencies as resolved by mamba.\n"
    '"$miniforge_dir/bin/python" "$SWE_POETRY_MARKER_FIX" '
    '--provenance-dir "$SWE_BUILD_PROVENANCE_DIR"'
)


def prepare_setup(path: Path) -> None:
    """Run the fix after upstream installs Poetry, before it runs make/poetry."""
    original = path.read_text()
    if SETUP_FIX in original:
        return
    if original.count(BASE_PACKAGING_PIN) != 1:
        raise ValueError("Unexpected OpenHands setup script; refusing to patch it")
    path.write_text(original.replace(BASE_PACKAGING_PIN, SETUP_FIX))


def patched_source(source: str) -> str:
    """Change only conjunction evaluation order; retain each original condition."""
    if FIXED_VALIDATION in source:
        return source
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MultiMarker"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate"
    )
    lines = source.splitlines(keepends=True)
    body = "".join(lines[method.lineno - 1 : method.end_lineno])
    if body.count(ORIGINAL_VALIDATION) != 1:
        raise ValueError("Unexpected Poetry MultiMarker.validate; refusing to patch it")
    replacement = body.replace(ORIGINAL_VALIDATION, FIXED_VALIDATION)
    result = "".join(
        lines[: method.lineno - 1] + [replacement] + lines[method.end_lineno :]
    )
    ast.parse(result)
    return result


def patch_installed_poetry(provenance_dir: Path) -> None:
    from importlib.metadata import version

    from poetry.core.version import markers

    path = Path(inspect.getfile(markers))
    original = path.read_text()
    modified = patched_source(original)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    if modified != original:
        path.write_text(modified)
        (provenance_dir / "poetry-core-markers.patch").write_text(
            "".join(
                difflib.unified_diff(
                    original.splitlines(True),
                    modified.splitlines(True),
                    fromfile="a/poetry/core/version/markers.py",
                    tofile="b/poetry/core/version/markers.py",
                )
            )
        )
    (provenance_dir / "poetry-core-markers.json").write_text(
        json.dumps(
            {
                "poetry_core_version": version("poetry-core"),
                "marker_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fix": "evaluate OS guards before platform_release in conjunctions",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--setup-script", type=Path)
    action.add_argument("--provenance-dir", type=Path)
    args = parser.parse_args()
    if args.setup_script:
        prepare_setup(args.setup_script)
    else:
        patch_installed_poetry(args.provenance_dir)


if __name__ == "__main__":
    main()
