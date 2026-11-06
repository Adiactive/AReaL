#!/usr/bin/env python3
"""Resolve host launch defaults from generated JSON-formatted runner YAML.

This host-only helper uses the Python 3.8 standard library. Its stdout contains
NUL-delimited variable/value pairs for start.sh, never executable shell text.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    for option in (
        "role",
        "service",
        "node-ip",
        "ray-head-address",
        "ray-port-base",
        "ray-worker-ports",
        "object-store-memory",
        "port",
        "bind-address",
        "sif-dir",
        "state-dir",
        "results-dir",
        "image",
        "name",
        "env-file",
        "cpus",
        "memory",
        "shm-size",
    ):
        parser.add_argument("--" + option)
    preflight = parser.add_mutually_exclusive_group()
    preflight.add_argument("--preflight")
    preflight.add_argument(
        "--no-preflight", dest="preflight", action="store_const", const=""
    )
    dry_run = parser.add_mutually_exclusive_group()
    dry_run.add_argument("--dry-run", action="store_true", default=None)
    dry_run.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    return parser.parse_args()


def text_value(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(field + " must be a nonempty string without NUL characters")
    return value


def port_value(value: object, field: str) -> str:
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(field + " must be an integer port")
    number = int(value)
    if not 1 <= number <= 65535:
        raise ValueError(field + " must be between 1 and 65535")
    return str(number)


def runner_text_value(value: object, field: str) -> str:
    value = text_value(value, "swe_runner." + field)
    if (field.endswith("_dir") or field == "env_file") and (
        not Path(value).is_absolute()
        or any(character in value for character in (",", "\n", "\r"))
        or "${" in value
    ):
        raise ValueError(
            "swe_runner."
            + field
            + " must be an absolute host path without commas, newlines or interpolation"
        )
    if field == "image" and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*", value):
        raise ValueError("swe_runner.image must be a Docker image name or digest")
    if field == "container_name" and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError("swe_runner.container_name must be a Docker container name")
    if field == "cpus" and (
        not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value)
        or not re.search(r"[1-9]", value)
    ):
        raise ValueError("swe_runner.cpus must be a positive number")
    if field in ("memory", "shm_size") and (
        not re.fullmatch(r"[0-9]+[bkmgBKMG]?", value) or not re.search(r"[1-9]", value)
    ):
        raise ValueError(
            "swe_runner." + field + " must be a positive Docker memory size"
        )
    if field == "preflight" and (
        Path(value).is_absolute()
        or ".." in Path(value).parts
        or value == "."
        or any(character in value for character in ("\n", "\r"))
        or "${" in value
    ):
        raise ValueError(
            "swe_runner.preflight must be a relative SIF path without '..'"
        )
    if field in ("node_ip", "bind_address"):
        try:
            address = ipaddress.IPv4Address(value)
            if address.is_unspecified or address.is_multicast or address.is_reserved:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                "swe_runner." + field + " must be a reachable IPv4 address"
            ) from exc
    if field == "ray_worker_ports":
        try:
            lower, upper = (int(port_value(part, field)) for part in value.split("-"))
            if lower > upper:
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                "swe_runner.ray_worker_ports must be an increasing port range"
            ) from exc
    return value


def launch_defaults(config: dict, args: argparse.Namespace) -> dict[str, str]:
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    runner = config.get("swe_runner", {})
    if not isinstance(runner, dict):
        raise ValueError("swe_runner must be an object")
    defaults = {}
    for field in (
        "image",
        "sif_dir",
        "state_dir",
        "results_dir",
        "cpus",
        "memory",
        "shm_size",
    ):
        if field in runner:
            value = runner_text_value(runner[field], field)
            if getattr(args, field) is None:
                defaults[field] = value
    preflight = (
        args.preflight if args.preflight is not None else runner.get("preflight", "")
    )
    defaults["preflight"] = (
        runner_text_value(preflight, "preflight") if preflight != "" else ""
    )
    if "dry_run" in runner:
        if type(runner["dry_run"]) is not bool:
            raise ValueError("swe_runner.dry_run must be a boolean")
        if args.dry_run is None:
            defaults["dry_run"] = "true" if runner["dry_run"] else "false"
    # Offline checks use image, mounts and resource limits without starting services.
    if preflight:
        return defaults
    if "container_name" in runner:
        value = runner_text_value(runner["container_name"], "container_name")
        if args.name is None:
            defaults["container_name"] = value
    if "ray_worker_ports" in runner:
        value = runner_text_value(runner["ray_worker_ports"], "ray_worker_ports")
        if args.ray_worker_ports is None:
            defaults["ray_worker_ports"] = value
    if "object_store_memory" in runner:
        value = runner["object_store_memory"]
        if type(value) is not int or value < 80 * 1024 * 1024:
            raise ValueError(
                "swe_runner.object_store_memory must be at least 80 MiB, in bytes"
            )
        if args.object_store_memory is None:
            defaults["object_store_memory"] = str(value)

    ray_address = config.get("ray_head_node_address")
    # Preserve the original single-host invocation for configurations predating
    # host metadata, even if their Gym configuration attaches to external Ray.
    legacy_bind = "swe_runner" not in config and args.bind_address is not None
    configured_role = runner.get("role", "")
    if configured_role not in ("", "standalone", "head", "worker"):
        raise ValueError("swe_runner.role must be standalone, head, worker or empty")
    configured_role = configured_role or (
        "head" if ray_address and not legacy_bind else "standalone"
    )
    role = args.role or configured_role
    defaults["role"] = role
    # Changing the node's role must not reuse a different node's local identity.
    for field in ("node_ip", "bind_address"):
        explicit = getattr(args, field)
        if explicit is not None:
            defaults[field] = explicit
        elif role == configured_role and runner.get(field, "") != "":
            defaults[field] = runner_text_value(runner[field], field)
    if role == "worker":
        if args.service not in (None, "standalone") or args.env_file:
            raise ValueError(
                "Workers run Ray only: omit --env-file and --service areal."
            )
        defaults["service"] = "standalone"
    else:
        if args.env_file is None and runner.get("env_file", "") != "":
            defaults["env_file"] = runner_text_value(runner["env_file"], "env_file")
        try:
            model = config["policy_model"]["responses_api_models"]["vllm_model"]
            swe = config["swe_agents"]["responses_api_agents"]["swe_agents"]
            service = "areal" if model["entrypoint"] == "areal_app.py" else "standalone"
            if service == "areal":
                api_port = port_value(
                    urlsplit(config["policy_base_url"]).port, "policy_base_url port"
                )
            else:
                api_port = port_value(swe["port"], "SWE API port")
        except (KeyError, TypeError) as exc:
            raise ValueError("use the generated runner configuration") from exc
        require_match(args, "service", service)
        if role == "head":
            require_match(args, "port", api_port)
        defaults["service"] = service
        defaults["container_port"] = api_port
        if args.port is None:
            defaults["port"] = api_port
        if role == "standalone" and "bind_address" not in defaults:
            try:
                defaults["bind_address"] = text_value(
                    config["head_server"]["host"], "head_server.host"
                )
            except (KeyError, TypeError) as exc:
                raise ValueError("use the generated runner configuration") from exc

    if role == "standalone" and ray_address and not legacy_bind:
        raise ValueError("cluster configuration requires --role head or --role worker")
    if role in ("head", "worker"):
        if ray_address:
            ray_address = text_value(ray_address, "ray_head_node_address")
            try:
                _, ray_port = ray_address.rsplit(":", 1)
                ray_port = port_value(ray_port, "ray_head_node_address port")
            except ValueError as exc:
                raise ValueError("ray_head_node_address must be IPv4:PORT") from exc
            require_match(args, "ray_head_address", ray_address)
            require_match(args, "ray_port_base", ray_port)
            defaults["ray_head_address"] = ray_address
            defaults["ray_port_base"] = ray_port
        if role == "head":
            try:
                head_ip = text_value(config["head_server"]["host"], "head_server.host")
            except (KeyError, TypeError) as exc:
                raise ValueError("use the generated runner configuration") from exc
            require_match(args, "node_ip", head_ip)
            if defaults.get("node_ip", head_ip) != head_ip:
                raise ValueError("swe_runner.node_ip conflicts with head_server.host")
            defaults["node_ip"] = head_ip
    return defaults


def require_match(args: argparse.Namespace, field: str, configured: str) -> None:
    explicit = getattr(args, field)
    if explicit is not None and explicit != configured:
        raise ValueError(
            "--" + field.replace("_", "-") + " conflicts with configuration"
        )


def main() -> int:
    args = parse_args()
    try:
        with args.config.open(encoding="utf-8") as source:
            defaults = launch_defaults(json.load(source), args)
        for field, value in defaults.items():
            sys.stdout.buffer.write((field + "\0" + value + "\0").encode("utf-8"))
    except (OSError, ValueError) as exc:
        sys.stderr.write("Runner launch configuration failed: " + str(exc) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
