#!/usr/bin/env python3
"""Supervise a private Ray node and, on the head, the sandbox API processes."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

RUNTIME_DIR = Path("/srv/swe-runtime")


def node_address(value: str) -> str:
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as exc:
        raise argparse.ArgumentTypeError("use an IPv4 address") from exc
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        raise argparse.ArgumentTypeError("use this node's reachable private-network IP")
    return str(address)


def positive_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def port_number(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return number


def head_address(value: str) -> str:
    try:
        host, port = value.split(":")
        return f"{node_address(host)}:{port_number(port)}"
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "use a reachable IPv4 address and port"
        ) from exc


def worker_ports(value: str) -> tuple[int, int]:
    try:
        lower, upper = (port_number(part) for part in value.split("-"))
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "use a port range such as 20000-20127"
        ) from exc
    if lower > upper:
        raise argparse.ArgumentTypeError("worker port range must be increasing")
    return lower, upper


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("standalone", "head", "worker"), required=True
    )
    parser.add_argument(
        "--service", choices=("standalone", "areal"), default="standalone"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--node-ip", type=node_address)
    parser.add_argument("--ray-head-address", type=head_address)
    parser.add_argument(
        "--cpus",
        type=positive_number,
        default=48,
        help="positive CPU budget; cluster roles require a whole number for Ray",
    )
    parser.add_argument("--ray-port-base", type=port_number, default=16379)
    parser.add_argument("--ray-worker-ports", type=worker_ports, default=(20000, 20127))
    parser.add_argument("--object-store-memory", type=int, default=1073741824)
    parser.add_argument("--port", type=port_number, default=8080)
    parser.add_argument("--startup-timeout", type=positive_number, default=180)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.object_store_memory < 80 * 1024 * 1024:
        parser.error("--object-store-memory must be at least 80 MiB, in bytes")
    if args.role != "standalone" and (not args.node_ip or not args.ray_head_address):
        parser.error("cluster roles require --node-ip and --ray-head-address")
    if args.role != "standalone":
        if args.cpus != int(args.cpus):
            parser.error(
                "cluster --cpus must be a whole number (Ray --num-cpus is an integer)"
            )
        args.cpus = int(args.cpus)
    if args.role == "standalone" and (args.node_ip or args.ray_head_address):
        parser.error("use --role head or worker for Ray cluster options")
    if args.role == "worker":
        if args.config or args.service != "standalone":
            parser.error(
                "workers run Ray only; do not supply --config or --service areal"
            )
    elif not args.config:
        parser.error("--config is required for the API service")
    if args.ray_port_base > 65527:
        parser.error("--ray-port-base must leave room for eight additional ports")
    if (
        args.role == "head"
        and args.ray_head_address != f"{args.node_ip}:{args.ray_port_base}"
    ):
        parser.error("head address must equal --node-ip:--ray-port-base")
    lower, upper = args.ray_worker_ports
    if any(
        lower <= port <= upper
        for port in range(args.ray_port_base, args.ray_port_base + 9)
    ):
        parser.error("worker ports overlap the Ray control ports")
    return args


def config_service_ports(args: argparse.Namespace) -> list[int]:
    """Reject accidental localhost callbacks before launching a cluster head."""
    if args.role == "worker":
        return []
    with args.config.open(encoding="utf-8") as source:
        config = json.load(source)
    try:
        registry = config["head_server"]
        model = config["policy_model"]["responses_api_models"]["vllm_model"]
        swe = config["swe_agents"]["responses_api_agents"]["swe_agents"]
        ports = [port_number(str(server["port"])) for server in (registry, model, swe)]
        if args.service == "areal":
            bridge = urlsplit(config["policy_base_url"])
            if bridge.port != args.port:
                raise ValueError("bridge port in configuration must match --port")
            if model["entrypoint"] != "areal_app.py":
                raise ValueError(
                    "--service areal requires configure_runner.py configuration"
                )
            ports.append(args.port)
        elif swe["port"] != args.port:
            raise ValueError("SWE API port in configuration must match --port")
        elif model["entrypoint"] == "areal_app.py":
            raise ValueError("AReaL configuration requires --service areal")
        if args.role == "head":
            if config.get("ray_head_node_address") != args.ray_head_address:
                raise ValueError(
                    "configuration ray_head_node_address must match --ray-head-address"
                )
            if any(server["host"] != args.node_ip for server in (registry, model)):
                raise ValueError(
                    "cluster registry and model hosts must equal --node-ip"
                )
            allowed_swe_hosts = {args.node_ip}
            if args.service == "areal":
                allowed_swe_hosts.add("127.0.0.1")
            if swe["host"] not in allowed_swe_hosts:
                raise ValueError("configure the SWE API host for this head node")
            if swe["base_results_dir"] != "/srv/swe-results":
                raise ValueError("cluster base_results_dir must be /srv/swe-results")
            if any(
                "/srv/mlperf/sif/" not in path for path in swe["container_formatter"]
            ):
                raise ValueError("cluster SIF paths must use /srv/mlperf/sif")
        elif config.get("ray_head_node_address"):
            raise ValueError("cluster configuration requires --role head")
    except KeyError as exc:
        raise ValueError("use the generated runner configuration") from exc
    if len(set(ports)) != len(ports):
        raise ValueError("each API service requires a distinct port")
    return ports


def ray_ports(args: argparse.Namespace) -> list[int]:
    ports = list(range(args.ray_port_base + 1, args.ray_port_base + 7))
    if args.role == "head":
        ports += [args.ray_port_base, args.ray_port_base + 7, args.ray_port_base + 8]
    lower, upper = args.ray_worker_ports
    return ports + list(range(lower, upper + 1))


def check_ports(args: argparse.Namespace, service_ports: list[int]) -> None:
    """Preflight only; Ray owns the sockets once startup begins."""
    ports = ray_ports(args) + service_ports
    if len(set(ports)) != len(ports):
        raise ValueError("API service ports overlap the Ray ports")
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise RuntimeError(
                    f"Port {port} is unavailable; no services were started"
                ) from exc
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((args.node_ip, 0))


def ray_command(args: argparse.Namespace) -> list[str]:
    base = args.ray_port_base
    lower, upper = args.ray_worker_ports
    command = [
        "ray",
        "start",
        f"--node-ip-address={args.node_ip}",
        f"--num-cpus={args.cpus}",
        "--num-gpus=0",
        f"--object-store-memory={args.object_store_memory}",
        f"--node-manager-port={base + 1}",
        f"--object-manager-port={base + 2}",
        f"--runtime-env-agent-port={base + 3}",
        f"--dashboard-agent-grpc-port={base + 4}",
        f"--dashboard-agent-listen-port={base + 5}",
        f"--metrics-export-port={base + 6}",
        f"--min-worker-port={lower}",
        f"--max-worker-port={upper}",
        "--disable-usage-stats",
        "--block",
    ]
    if args.role == "head":
        command += [
            "--head",
            f"--port={base}",
            "--include-dashboard=false",
            f"--dashboard-port={base + 8}",
            f"--ray-client-server-port={base + 7}",
            f"--temp-dir={RUNTIME_DIR / 'ray'}",
        ]
    else:
        command.append(f"--address={args.ray_head_address}")
    return command


def service_command(args: argparse.Namespace) -> list[str]:
    if args.service == "areal":
        return [sys.executable, "/opt/areal-gym/serve.py", "--config", str(args.config)]
    return ["gym", "env", "start", f"+config_paths=[{args.config}]"]


def wait_for_ray(args: argparse.Namespace, process: subprocess.Popen) -> None:
    probe = (
        "import ray,sys; "
        "ray.init(address=sys.argv[1], logging_level='ERROR'); "
        "ready=any(n.get('Alive') and n.get('NodeManagerAddress')==sys.argv[2] for n in ray.nodes()); "
        "ray.shutdown(); sys.exit(0 if ready else 1)"
    )
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Ray exited during startup; inspect its preceding logs")
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe, args.ray_head_address, args.node_ip],
                capture_output=True,
                timeout=min(15, max(0.1, deadline - time.monotonic())),
                check=False,
            )
            if result.returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1)
    raise RuntimeError(
        "Ray node registration timed out; inspect connectivity and Ray logs"
    )


def stop_processes(processes: list[subprocess.Popen]) -> None:
    # Never use `ray stop`: only signal process groups owned by this container.
    for process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process in reversed(processes):
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def supervise(args: argparse.Namespace) -> None:
    processes = []
    try:
        if args.role != "standalone":
            processes.append(
                subprocess.Popen(ray_command(args), start_new_session=True)
            )
            wait_for_ray(args, processes[0])
        if args.role != "worker":
            processes.append(
                subprocess.Popen(service_command(args), start_new_session=True)
            )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError("A runner process exited; the container will stop")
    finally:
        stop_processes(processes)


def main() -> int:
    args = parse_args()
    try:
        ports = config_service_ports(args)
        if args.role != "standalone" and set(ports) & set(ray_ports(args)):
            raise ValueError("API service ports overlap the Ray ports")
        if args.validate_only:
            return 0
        os.umask(0o077)
        if args.role != "standalone":
            check_ports(args, ports)
            (RUNTIME_DIR / "ray").mkdir(parents=True, exist_ok=True)
            os.environ["RAY_TMPDIR"] = str(RUNTIME_DIR / "ray")

        def shutdown(signum, _frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        supervise(args)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"Runner startup failed: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
