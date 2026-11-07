#!/usr/bin/env python3
"""Write standalone or AReaL runner configuration without installing Gym.

The JSON output is also valid YAML. Fields follow Gym 610a08ab plus the
MLPerf reference patch; OpenHands follows the reference's 0d766ad0 pin.
Standalone mode uses a fixed model endpoint. AReaL mode loads the adapter's
configuration helper from the build input or the unified image.
"""

import argparse
import importlib.util
import ipaddress
import json
import math
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

OPENHANDS_COMMIT = "0d766ad06b2be64a42e6f0175b9ebcc4a06599d9"
GYM_ROOT = "/opt/nemo-gym"


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def port_number(value: str) -> int:
    result = int(value)
    if not 1 <= result <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return result


def advertised_host(value: str) -> str:
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as exc:
        raise argparse.ArgumentTypeError("use an explicit IPv4 address") from exc
    if address.is_unspecified or address.is_multicast or address.is_reserved:
        raise argparse.ArgumentTypeError("use a routable or loopback IPv4 address")
    return str(address)


def ray_address(value: str) -> str:
    try:
        host, port = value.split(":")
        host = advertised_host(host)
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("use Ray head IPv4:PORT") from exc
    return f"{host}:{int(port)}"


def container_path(value: str) -> str:
    # The upstream runner embeds mount paths in shell commands without quoting.
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise argparse.ArgumentTypeError(
            "use an absolute container path with letters, digits, '/', '.', '_' or '-'"
        )
    if ".." in Path(value).parts:
        raise argparse.ArgumentTypeError("container paths must not contain '..'")
    return value.rstrip("/")


def host_path(value: str) -> str:
    # Host paths are metadata here, not paths inside the generator container.
    if (
        not Path(value).is_absolute()
        or any(character in value for character in (",", "\n", "\r", "\0"))
        or "${" in value
    ):
        raise argparse.ArgumentTypeError(
            "use an absolute host path without commas, newlines or interpolation"
        )
    return value


def runner_image(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*", value):
        raise argparse.ArgumentTypeError("use a Docker image name or digest")
    return value


def container_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError(
            "container name must start with a letter or digit and contain only letters, digits, '.', '_' or '-'"
        )
    return value


def cpu_limit(value: str) -> str:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) or not re.search(
        r"[1-9]", value
    ):
        raise argparse.ArgumentTypeError("CPU limit must be a positive number")
    return value


def memory_size(value: str) -> str:
    if not re.fullmatch(r"[0-9]+[bkmgBKMG]?", value) or not re.search(r"[1-9]", value):
        raise argparse.ArgumentTypeError(
            "memory size must be a positive integer with optional b, k, m or g suffix"
        )
    return value


def preflight_path(value: str) -> str:
    if value and (
        Path(value).is_absolute()
        or ".." in Path(value).parts
        or value == "."
        or any(character in value for character in ("\n", "\r", "\0"))
        or "${" in value
    ):
        raise argparse.ArgumentTypeError(
            "preflight must be a relative SIF path without '..' or interpolation"
        )
    return value


def worker_port_range(value: str) -> str:
    try:
        lower, upper = (port_number(part) for part in value.split("-"))
        if lower > upper:
            raise ValueError
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "use an increasing port range such as 20000-20127"
        ) from exc
    return f"{lower}-{upper}"


def model_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid model endpoint URL or port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in value)
    ):
        raise argparse.ArgumentTypeError(
            "use an internal HTTP(S) base URL without credentials, query or fragment"
        )
    if not parsed.path.rstrip("/").endswith("/v1"):
        raise argparse.ArgumentTypeError(
            "model base URL must include its '/v1' API prefix; do not include '/chat/completions'"
        )
    return value.rstrip("/")


def build_config(args: argparse.Namespace) -> dict:
    api_key = f"${{oc.env:{args.api_key_env},unused}}"
    config = {
        "default_host": "127.0.0.1",
        "use_absolute_ip": False,
        "head_server": {"host": "127.0.0.1", "port": args.registry_port},
        "skip_venv_if_present": True,
        "policy_base_url": args.model_base_url,
        "policy_api_key": api_key,
        "policy_model_name": args.model_name,
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {
                    "entrypoint": "app.py",
                    "host": "127.0.0.1",
                    "port": args.model_port,
                    "base_url": "${policy_base_url}",
                    "api_key": "${policy_api_key}",
                    "model": "${policy_model_name}",
                    # AReaL records original tokens through its own inference proxy.
                    # Standalone qualification needs only observations and reward.
                    "return_token_id_information": False,
                    "uses_reasoning_parser": args.uses_reasoning_parser,
                    "sequential_reasoning_allowed": False,
                }
            }
        },
        "swe_agents": {
            "responses_api_agents": {
                "swe_agents": {
                    "entrypoint": "app.py",
                    "host": args.host,
                    "port": args.port,
                    "agent_config": f"{GYM_ROOT}/responses_api_agents/swe_agents/configs/oh_config.toml",
                    "agent_framework_repo": "https://github.com/sdevare-nv/nv-OpenHands.git",
                    "agent_framework_commit": OPENHANDS_COMMIT,
                    "agent_max_turns": args.max_turns,
                    "agent_prompt_overrides": [
                        {
                            "user_prompt_template": f"{GYM_ROOT}/responses_api_agents/swe_agents/prompts/openhands/user_prompt.j2",
                            "system_prompt_template": f"{GYM_ROOT}/responses_api_agents/swe_agents/prompts/openhands/system_prompt.j2",
                            "agent_cls": "CodeActAgent",
                            "diversify_tool_names": False,
                        }
                    ],
                    "model_server": {
                        "name": "policy_model",
                        "type": "responses_api_models",
                    },
                    "container_formatter": [
                        f"{args.sif_dir}/r2egym/{{instance_id}}.sif",
                        f"{args.sif_dir}/{{instance_id}}.sif",
                    ],
                    "dataset_path": None,
                    "base_results_dir": args.results_dir,
                    "concurrency": args.concurrency,
                    "swebench_agent_timeout": args.agent_timeout,
                    "swebench_tests_timeout": args.test_timeout,
                    "command_exec_timeout": args.command_timeout,
                    "apptainer_memory_limit_mb": args.memory_limit_mb,
                    "openhands_preflight": False,
                    "openhands_should_log": False,
                    "debug": False,
                }
            }
        },
    }
    if args.ray_head_address:
        config["ray_head_node_address"] = args.ray_head_address
    if args.advertised_host:
        config["default_host"] = args.advertised_host
        config["head_server"]["host"] = args.advertised_host
        config["policy_model"]["responses_api_models"]["vllm_model"]["host"] = (
            args.advertised_host
        )
        config["swe_agents"]["responses_api_agents"]["swe_agents"]["host"] = (
            args.advertised_host
        )
    if args.chat_template_thinking != "server-default":
        config["policy_model"]["responses_api_models"]["vllm_model"][
            "chat_template_kwargs"
        ] = {"enable_thinking": args.chat_template_thinking == "enabled"}
    if args.service == "areal":
        helper_path = args.integration_dir / "configure_runner.py"
        spec = importlib.util.spec_from_file_location(
            "areal_runner_config", helper_path
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load adapter configuration helper: {helper_path}")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        config = helper.configure(
            config,
            args.port,
            args.gym_port,
            advertised_host=args.advertised_host or "127.0.0.1",
            ray_head_address=args.ray_head_address,
        )
    if args.host_sif_dir is not None:
        config["swe_runner"] = {
            "image": args.runner_image
            or (
                "mlperf-swe-runner:cluster"
                if args.ray_head_address
                else "mlperf-swe-runner:local"
            ),
            "sif_dir": args.host_sif_dir,
            "state_dir": args.host_state_dir,
            "results_dir": args.host_results_dir,
            "ray_worker_ports": args.ray_worker_ports or "20000-20127",
            "object_store_memory": args.object_store_memory or 1073741824,
            "container_name": args.container_name or "mlperf-swe-runner",
            "cpus": args.cpus or "48",
            "memory": args.memory or "400g",
            "shm_size": args.shm_size or "16g",
            "preflight": args.preflight or "",
            "dry_run": bool(args.launch_dry_run),
        }
        for key, value in (
            ("role", args.role),
            ("node_ip", args.node_ip),
            ("bind_address", args.bind_address),
            ("env_file", args.runtime_env_file),
        ):
            if value is not None:
                config["swe_runner"][key] = value
        if args.service == "areal":
            config["swe_runner"]["episode_timeout"] = (
                args.episode_timeout if args.episode_timeout is not None else 3600
            )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="site.yaml to write")
    parser.add_argument(
        "--service", choices=("standalone", "areal"), default="standalone"
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=8080,
        help="service listen port: AReaL bridge or standalone SWE API; match start.sh --port for --role head (default: 8080)",
    )
    parser.add_argument(
        "--gym-port",
        type=port_number,
        help="loopback SWE API port in AReaL mode only (default: 8083)",
    )
    parser.add_argument(
        "--registry-port",
        type=port_number,
        default=8081,
        help="Gym registry port (default: 8081)",
    )
    parser.add_argument(
        "--model-port",
        type=port_number,
        default=8082,
        help="Gym model adapter port (default: 8082)",
    )
    parser.add_argument(
        "--advertised-host",
        type=advertised_host,
        help="head IPv4 reachable by all Ray workers",
    )
    parser.add_argument(
        "--ray-head-address", type=ray_address, help="existing Ray cluster IPv4:PORT"
    )
    parser.add_argument(
        "--integration-dir",
        type=Path,
        default=Path("/opt/areal-gym")
        if Path("/opt/areal-gym").is_dir()
        else Path(__file__).resolve().parent.parent,
        help="directory containing configure_runner.py; only needed for host-side AReaL config generation",
    )
    parser.add_argument(
        "--model-base-url",
        type=model_base_url,
        help="your self-hosted model API base, including /v1 (no external default)",
    )
    parser.add_argument("--model-name")
    parser.add_argument(
        "--api-key-env",
        default="MLPERF_MODEL_API_KEY",
        help="environment variable resolved inside the runner; its value is never written here",
    )
    parser.add_argument("--sif-dir", type=container_path, default="/srv/mlperf/sif")
    parser.add_argument(
        "--results-dir", type=container_path, default="/srv/swe-results"
    )
    parser.add_argument(
        "--host-sif-dir", type=host_path, help="host SIF mount for start.sh"
    )
    parser.add_argument(
        "--host-state-dir", type=host_path, help="node-local host state for start.sh"
    )
    parser.add_argument(
        "--host-results-dir", type=host_path, help="shared host results for start.sh"
    )
    parser.add_argument("--runner-image", type=runner_image)
    parser.add_argument("--ray-worker-ports", type=worker_port_range)
    parser.add_argument("--object-store-memory", type=positive_int)
    parser.add_argument(
        "--container-name", type=container_name, help="saved service container name"
    )
    parser.add_argument("--cpus", type=cpu_limit, help="saved Docker CPU limit")
    parser.add_argument("--memory", type=memory_size, help="saved Docker memory limit")
    parser.add_argument(
        "--shm-size", type=memory_size, help="saved Docker shared-memory limit"
    )
    parser.add_argument(
        "--preflight",
        type=preflight_path,
        help="relative SIF path; empty starts the service",
    )
    parser.add_argument("--role", choices=("standalone", "head", "worker"))
    parser.add_argument("--node-ip", type=advertised_host)
    parser.add_argument("--bind-address", type=advertised_host)
    parser.add_argument(
        "--runtime-env-file",
        type=host_path,
        help="host runtime credentials path; never read here",
    )
    parser.add_argument(
        "--launch-dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="save whether start.sh prints its command without launching",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="standalone SWE API bind address; --advertised-host takes precedence",
    )
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--max-turns", type=positive_int, default=30)
    parser.add_argument("--agent-timeout", type=positive_int, default=1800)
    parser.add_argument(
        "--episode-timeout",
        type=positive_float,
        help="saved AReaL bridge Gym request deadline in seconds (default: 3600); requires host launch paths",
    )
    parser.add_argument("--test-timeout", type=positive_int, default=60)
    parser.add_argument("--command-timeout", type=positive_int, default=60)
    parser.add_argument("--memory-limit-mb", type=positive_int, default=32768)
    parser.add_argument(
        "--uses-reasoning-parser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match whether the upstream model server separates reasoning_content",
    )
    parser.add_argument(
        "--chat-template-thinking",
        choices=("disabled", "enabled", "server-default"),
        default="disabled",
        help=(
            "reference Qwen recipe disables thinking; choose server-default for "
            "model servers that do not support chat_template_kwargs"
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output file"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate arguments and generated JSON without writing or contacting services",
    )
    args = parser.parse_args()
    host_paths = (args.host_sif_dir, args.host_state_dir, args.host_results_dir)
    if any(value is not None for value in host_paths) or any(
        value is not None
        for value in (
            args.runner_image,
            args.ray_worker_ports,
            args.object_store_memory,
            args.container_name,
            args.cpus,
            args.memory,
            args.shm_size,
            args.preflight,
            args.role,
            args.node_ip,
            args.bind_address,
            args.runtime_env_file,
            args.launch_dry_run,
            args.episode_timeout,
        )
    ):
        if not all(value is not None for value in host_paths):
            parser.error(
                "launch settings require --host-sif-dir, --host-state-dir and --host-results-dir together"
            )
    if args.role in {"head", "worker"} and not args.ray_head_address:
        parser.error("--role head/worker requires --ray-head-address")
    if args.role == "standalone" and args.ray_head_address:
        parser.error("--role standalone requires omitting --ray-head-address")
    if args.role == "standalone" and args.advertised_host:
        parser.error("--role standalone uses --bind-address, not --advertised-host")
    if args.role == "worker" and not args.node_ip:
        parser.error("--role worker requires --node-ip")
    if args.role != "worker" and args.ray_head_address and args.node_ip:
        if args.node_ip != args.advertised_host:
            parser.error("head --node-ip must equal --advertised-host")
    if (
        args.object_store_memory is not None
        and args.object_store_memory < 80 * 1024 * 1024
    ):
        parser.error("--object-store-memory must be at least 80 MiB, in bytes")
    ports = [args.port, args.registry_port, args.model_port]
    if args.service == "areal":
        args.gym_port = args.gym_port if args.gym_port is not None else 8083
        ports.append(args.gym_port)
    elif args.gym_port is not None:
        parser.error(
            "--gym-port requires --service areal; use --port for the standalone SWE API"
        )
    if len(set(ports)) != len(ports):
        parser.error("API ports must be distinct")
    if args.service == "standalone":
        if not args.model_base_url or not args.model_name:
            parser.error("standalone mode requires --model-base-url and --model-name")
    else:
        if args.model_base_url or args.model_name:
            parser.error(
                "AReaL mode receives model sessions from the bridge; omit fixed model arguments"
            )
        args.model_base_url = f"http://127.0.0.1:{args.port}/v1"
        args.model_name = "areal-routed"
        if not (args.integration_dir / "configure_runner.py").is_file():
            parser.error(
                "AReaL mode requires --integration-dir or execution inside the unified image"
            )
    if args.ray_head_address and (
        not args.advertised_host
        or ipaddress.ip_address(args.advertised_host).is_loopback
    ):
        parser.error("--ray-head-address requires a non-loopback --advertised-host")
    if args.ray_head_address and args.host_sif_dir is not None:
        head_ip, head_port = args.ray_head_address.split(":")
        if head_ip != args.advertised_host:
            parser.error("Ray head IP must equal --advertised-host")
        if int(head_port) > 65527:
            parser.error("Ray head port must leave room for eight additional ports")
        lower, upper = map(int, (args.ray_worker_ports or "20000-20127").split("-"))
        control_ports = range(int(head_port), int(head_port) + 9)
        if any(lower <= port <= upper for port in control_ports):
            parser.error("worker ports overlap the Ray control ports")
        if any(port in control_ports or lower <= port <= upper for port in ports):
            parser.error("API service ports overlap the Ray ports")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env):
        parser.error("--api-key-env must be an environment variable name")
    if not args.model_name.strip() or "${" in args.model_name:
        parser.error("--model-name must be nonempty and cannot contain interpolation")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", args.host):
        parser.error("--host must be an IP address or hostname")
    return args


def main() -> int:
    args = parse_args()
    try:
        config = build_config(args)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Cannot generate configuration: {exc}\n")
        return 2
    encoded = json.dumps(config, indent=2, allow_nan=False) + "\n"
    json.loads(encoded)
    if args.check_only:
        sys.stdout.write(
            "Configuration valid; no files written and no network requests made.\n"
        )
        return 0
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if args.force else os.O_EXCL)
    try:
        with os.fdopen(os.open(args.output, flags, 0o600), "w") as output:
            output.write(encoded)
    except OSError as exc:
        sys.stderr.write(
            f"Cannot write configuration ({type(exc).__name__}); check path or use --force.\n"
        )
        return 2
    sys.stdout.write(
        f"Wrote {args.output}; public API port {args.port}, concurrency {args.concurrency}. "
        f"Service mode: {args.service}.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
