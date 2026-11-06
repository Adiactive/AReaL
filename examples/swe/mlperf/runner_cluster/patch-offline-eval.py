"""Expose baked wheels only to the pinned Gym R2E evaluator command."""

import argparse
import ast
import shlex
from pathlib import Path


def patch_source(source: str, wheel_dir: str) -> str:
    if not Path(wheel_dir).is_absolute() or any(c in wheel_dir for c in ",\n\r\0"):
        raise ValueError(
            "Wheel directory must be absolute and contain no commas or newlines"
        )
    mount = "--mount " + shlex.quote(f"type=bind,src={wheel_dir},dst={wheel_dir},ro")
    environment = (
        " ".join(
            "--env " + shlex.quote(value)
            for value in (
                "UV_NO_INDEX=true",
                f"UV_FIND_LINKS={wheel_dir}",
                "UV_OFFLINE=true",
            )
        )
        + " "
    )
    mount_block = (
        "        # BEGIN MLPerf offline R2E wheels\n"
        '        if command.mode == "eval" and "R2E-Gym" in data_point["dataset_name"]:\n'
        f"            mount_args.append({mount!r})\n"
        "        # END MLPerf offline R2E wheels\n\n"
    )
    env_block = (
        "        # BEGIN MLPerf offline R2E environment\n"
        '        if command.mode == "eval" and "R2E-Gym" in data_point["dataset_name"]:\n'
        f"            env_args += {environment!r}\n"
        "        # END MLPerf offline R2E environment\n"
    )
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
    if "# BEGIN MLPerf offline R2E" in body:
        if body.count(mount_block) == body.count(env_block) == 1:
            return source
        raise ValueError(
            "Existing offline evaluator patch differs; refusing to overwrite"
        )
    anchors = ('        mount_str = " ".join(mount_args)\n', '        env_args = ""\n')
    if any(body.count(anchor) != 1 for anchor in anchors):
        raise ValueError("Pinned Gym command builder changed; refusing to patch")
    body = body.replace(anchors[0], mount_block + anchors[0])
    body = body.replace(anchors[1], anchors[1] + env_block)
    result = (
        "".join(lines[: method.lineno - 1]) + body + "".join(lines[method.end_lineno :])
    )
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--wheel-dir", required=True)
    args = parser.parse_args()
    original = args.source.read_text()
    try:
        updated = patch_source(original, args.wheel_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if updated != original:
        args.source.write_text(updated)


if __name__ == "__main__":
    main()
