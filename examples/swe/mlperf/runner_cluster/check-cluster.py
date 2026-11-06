#!/usr/bin/env python3
"""Qualify every sandbox node without running an agent or generating model tokens."""

import argparse
import hashlib
import importlib
import ipaddress
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

SIF_ROOT = Path("/srv/mlperf/sif")
RESULTS_ROOT = Path("/srv/swe-results")
GYM_ROOT = Path("/opt/nemo-gym")
AGENT_PYTHON = GYM_ROOT / "responses_api_agents/swe_agents/.venv/bin/python"
MANIFEST = Path("/opt/swe-runner/runtime-manifest.json")
ADAPTER_MANIFEST = Path("/opt/swe-runner/provenance/areal-adapter.sha256")


def relative_sif(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".sif"
        or any(character in value for character in (",", "\n", "\r", "\0"))
    ):
        raise argparse.ArgumentTypeError(
            "--sif must be a relative .sif path without '..'"
        )
    return str(path)


def node_ip(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a private node IP address") from exc
    if address.is_unspecified or address.is_loopback or address.is_multicast:
        raise argparse.ArgumentTypeError(
            "Use a routable node IP, not a wildcard/loopback"
        )
    return str(address)


def ray_address(value: str) -> str:
    try:
        parsed = urlsplit("//" + value)
        valid = (
            parsed.hostname
            and parsed.port
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        valid = False
    if not valid or parsed.path or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("Use the existing Ray head IP:port")
    node_ip(parsed.hostname)
    return value


def callback_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in ("http", "https") and parsed.hostname
        parsed.port
    except ValueError:
        valid = False
    if (
        not valid
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise argparse.ArgumentTypeError(
            "Use an HTTP(S) server root URL without credentials"
        )
    return value.rstrip("/") + "/"


def select_nodes(nodes: list[dict], expected: list[str]) -> list[dict]:
    if len(set(expected)) != len(expected):
        raise ValueError("--expected-node values must be unique")
    alive = [node for node in nodes if node.get("Alive")]
    addresses = [node["NodeManagerAddress"] for node in alive]
    if len(set(addresses)) != len(addresses):
        raise ValueError("Multiple live Ray nodes share one IP; inspect the cluster")
    missing = sorted(set(expected) - set(addresses))
    extra = sorted(set(addresses) - set(expected))
    if missing or extra:
        raise ValueError(f"Ray node mismatch: missing={missing}, unexpected={extra}")
    by_address = {node["NodeManagerAddress"]: node for node in alive}
    return [by_address[address] for address in expected]


def health_probe(url: str, timeout: float) -> int:
    # Internal callbacks must work directly, independently of host proxy variables.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        if response.status != 200 or response.geturl() != url:
            raise ValueError(f"Health probe did not return direct HTTP 200: {url}")
        return response.status


def run_command(command: list[str], timeout: float) -> str:
    """Bound subprocess output and terminate its process group on cancellation."""
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                pass
            finally:
                # A child can ignore TERM even after the main subprocess exits.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            raise
        output.seek(0, os.SEEK_END)
        output.seek(max(0, output.tell() - 8192))
        tail = output.read().decode("utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(f"{command[0]} exited {process.returncode}:\n{tail}")
        return tail


def diagnose_node(config: dict) -> dict:
    # Imports stay inside the task so the CLI and its unit tests do not need Ray.
    import ray

    actual_id = ray.get_runtime_context().get_node_id()
    if actual_id != config["node_id"]:
        raise ValueError("Ray did not execute the diagnostic on the requested node")
    node = next(node for node in ray.nodes() if node["NodeID"] == actual_id)
    if node["NodeManagerAddress"] != config["node_ip"]:
        raise ValueError("Ray task node IP differs from the expected node IP")
    marker_dir = Path(config["marker_dir"])
    if (marker_dir / "challenge").read_text() != config["challenge"]:
        raise ValueError("Worker cannot read the head's shared-results challenge")
    sif = (SIF_ROOT / config["sif"]).resolve(strict=True)
    if not sif.is_relative_to(SIF_ROOT.resolve()) or not sif.is_file():
        raise ValueError("SIF must resolve to a file within /srv/mlperf/sif")
    header = run_command(["apptainer", "sif", "header", str(sif)], timeout=30)
    architecture = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64"}.get(
        platform.machine()
    )
    match = re.search(r"^Primary Architecture:\s+(\S+)", header, re.MULTILINE)
    if architecture is None or match is None or match[1] != architecture:
        raise ValueError(
            "SIF architecture does not match the native worker architecture"
        )
    wrapper = importlib.import_module("responses_api_agents.swe_agents.app")
    if not hasattr(wrapper, "SWEBenchWrapper"):
        raise ValueError("Installed Gym module has no SWEBenchWrapper")
    # Parsing as well as hashing catches a missing or invalid provenance artifact.
    manifest = MANIFEST.read_bytes()
    json.loads(manifest)
    report = {
        "node_id": actual_id,
        "node_ip": node["NodeManagerAddress"],
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "ray": ray.__version__,
        "architecture": architecture,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "adapter_manifest_sha256": hashlib.sha256(
            ADAPTER_MANIFEST.read_bytes()
        ).hexdigest(),
        "wrapper_sha256": hashlib.sha256(
            Path(wrapper.__file__).read_bytes()
        ).hexdigest(),
        "sif_path": str(sif),
        "sif_size": sif.stat().st_size,
        "health": {
            "registry": health_probe(config["head_url"], timeout=10),
            "model": health_probe(config["model_url"], timeout=10),
        },
        "ray_logical_cpus": node.get("Resources", {}).get("CPU", 0),
        "preflight_passed": False,
    }
    if config["preflight"]:
        report["preflight_tail"] = run_command(
            ["bash", "/opt/swe-runner/preflight.sh", str(sif)],
            timeout=config["timeout"],
        )
        report["preflight_passed"] = True
    marker = marker_dir / f"{actual_id}.json"
    write_report(marker, {"challenge": config["challenge"], "report": report})
    return report


def validate_reports(
    reports: list[dict], nodes: list[dict], marker_dir: Path, challenge: str
) -> None:
    expected = {node["NodeID"]: node["NodeManagerAddress"] for node in nodes}
    received = [report["node_id"] for report in reports]
    if len(received) != len(set(received)) or set(received) != set(expected):
        raise ValueError("Missing, duplicated, or unexpected node diagnostic result")
    for report in reports:
        if report["node_ip"] != expected[report["node_id"]]:
            raise ValueError("Node diagnostic IP does not match the expected node")
        marker = json.loads((marker_dir / f"{report['node_id']}.json").read_text())
        if marker != {"challenge": challenge, "report": report}:
            raise ValueError(
                "Worker's shared-results response does not match its Ray result"
            )
    fields = (
        "python",
        "ray",
        "architecture",
        "manifest_sha256",
        "adapter_manifest_sha256",
        "wrapper_sha256",
        "sif_path",
        "sif_size",
    )
    for field in fields:
        if len({report[field] for report in reports}) != 1:
            raise ValueError(
                f"Workers differ in {field}; use the same runner image and SIF mount"
            )


def write_report(path: Path, report: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", type=ray_address, required=True)
    parser.add_argument("--expected-node", type=node_ip, action="append", required=True)
    parser.add_argument("--sif", type=relative_sif, required=True)
    parser.add_argument(
        "--head-url", type=callback_url, required=True, help="Gym registry URL"
    )
    parser.add_argument(
        "--model-url", type=callback_url, required=True, help="Gym model adapter URL"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run offline SIF preflight on every node",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="Task deadline in seconds (default: 900)",
    )
    parser.add_argument(
        "--output", type=Path, help="Write a new owner-only JSON report"
    )
    args = parser.parse_args()
    if not 30 <= args.timeout <= 3600:
        parser.error("--timeout must be between 30 and 3600 seconds")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new report path")
    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except ImportError:
        parser.error(
            "Run this tool inside the runner image using its Gym Python environment"
        )
    pending = []
    marker_dir = None
    nodes = []
    report = {"ok": False, "nodes": []}
    try:
        # An explicit address never starts a new cluster (unlike address='local').
        ray.init(address=args.ray_address, logging_level="ERROR")
        nodes = select_nodes(ray.nodes(), args.expected_node)
        marker_dir = Path(tempfile.mkdtemp(prefix=".cluster-check-", dir=RESULTS_ROOT))
        challenge = uuid.uuid4().hex
        (marker_dir / "challenge").write_text(challenge)
        task = ray.remote(diagnose_node)
        deadline = time.monotonic() + args.timeout
        for node in nodes:
            config = {
                "node_id": node["NodeID"],
                "node_ip": node["NodeManagerAddress"],
                "sif": args.sif,
                "marker_dir": str(marker_dir),
                "challenge": challenge,
                "head_url": args.head_url,
                "model_url": args.model_url,
                "preflight": args.preflight,
                "timeout": max(1, args.timeout - 40),
            }
            pending.append(
                task.options(
                    num_cpus=0.1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node["NodeID"], soft=False
                    ),
                    runtime_env={
                        "py_executable": str(AGENT_PYTHON),
                        "env_vars": {"PYTHONPATH": str(GYM_ROOT)},
                    },
                ).remote(config)
            )
        reports = ray.get(pending, timeout=max(1, deadline - time.monotonic()))
        pending = []
        validate_reports(reports, nodes, marker_dir, challenge)
        report = {
            "ok": True,
            "nodes": reports,
            "limits": "Ray CPU values are scheduling metadata; this check does not test CPU/RAM enforcement.",
            "remaining": "Real agent generation, grading, concurrent attempts, and cancellation are not tested.",
        }
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_errors = []
        for task in pending:
            # Graceful cancellation lets the diagnostic terminate its subprocess group.
            try:
                ray.cancel(task, force=False)
            except Exception as exc:
                cleanup_errors.append(f"Ray task cancellation: {exc}")
        if pending:
            try:
                _, unfinished = ray.wait(pending, num_returns=len(pending), timeout=10)
                if unfinished:
                    cleanup_errors.append(
                        "Some diagnostic tasks did not acknowledge cancellation"
                    )
            except Exception as exc:
                cleanup_errors.append(f"Ray task cancellation wait: {exc}")
        if marker_dir:
            # Never recursively remove an arbitrary shared path or other users' files.
            try:
                for name in [
                    "challenge",
                    *(f"{node['NodeID']}.json" for node in nodes),
                ]:
                    (marker_dir / name).unlink(missing_ok=True)
                marker_dir.rmdir()
            except OSError:
                cleanup_errors.append(
                    f"Inspect leftover diagnostic directory: {marker_dir}"
                )
        if cleanup_errors:
            report["cleanup_warnings"] = cleanup_errors
        ray.shutdown()
    print(json.dumps(report, indent=2))
    if args.output:
        write_report(args.output, report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
