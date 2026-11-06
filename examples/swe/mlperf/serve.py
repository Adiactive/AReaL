#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Start Gym on the cluster head, then expose the AReaL bridge."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

if __package__:
    from .configure_runner import (
        validate_ipv4_host,
        validate_ports,
        validate_ray_address,
    )
else:
    from configure_runner import (
        validate_ipv4_host,
        validate_ports,
        validate_ray_address,
    )


def service_settings(config: dict) -> tuple[str, int, int, list[str]]:
    swe = config["swe_agents"]["responses_api_agents"]["swe_agents"]
    model = config["policy_model"]["responses_api_models"]["vllm_model"]
    head = config["head_server"]
    bridge = urlsplit(config["policy_base_url"])
    if (
        bridge.scheme != "http"
        or bridge.hostname != "127.0.0.1"
        or bridge.path != "/v1"
        or bridge.username is not None
        or bridge.password is not None
        or bridge.query
        or bridge.fragment
        or bridge.port is None
        or model["entrypoint"] != "areal_app.py"
        or model["return_token_id_information"] is not False
        or model.get("uses_reasoning_parser") is not False
        or model.get("num_workers") != 1
        or swe["host"] != "127.0.0.1"
        or model["host"] != head["host"]
    ):
        raise ValueError("Use configure_runner.py to prepare the runner configuration")
    validate_ipv4_host(head["host"])
    validate_ports([bridge.port, swe["port"], head["port"], model["port"]])
    if type(swe["concurrency"]) is not int or swe["concurrency"] < 1:
        raise ValueError("Concurrency must be a positive integer")
    if config.get("ray_head_node_address") is not None:
        validate_ray_address(config["ray_head_node_address"])
    urls = [
        f"http://{server['host']}:{server['port']}" for server in (swe, model, head)
    ]
    return urls[0], bridge.port, swe["concurrency"], urls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--episode-timeout", type=float, default=3600)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as source:
        gym_url, bridge_port, concurrency, readiness_urls = service_settings(
            json.load(source)
        )
    processes = []

    def shutdown(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        processes.append(
            subprocess.Popen(
                ["gym", "env", "start", f"+config_paths=[{args.config}]"],
                start_new_session=True,
            )
        )
        deadline = time.monotonic() + args.startup_timeout
        with httpx.Client(trust_env=False, timeout=2) as client:
            while readiness_urls:
                if processes[0].poll() is not None:
                    raise RuntimeError("Gym exited during startup; inspect its logs")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Gym readiness timed out; inspect its logs")
                for url in readiness_urls[:]:
                    try:
                        if client.get(url + "/").is_success:
                            readiness_urls.remove(url)
                    except httpx.HTTPError:
                        pass
                if readiness_urls:
                    time.sleep(1)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("runner_bridge.py")),
                    "--gym-url",
                    gym_url,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(bridge_port),
                    "--concurrency",
                    str(concurrency),
                    "--episode-timeout",
                    str(args.episode_timeout),
                ],
                start_new_session=True,
            )
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError("A runner service exited; inspect its logs")
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
