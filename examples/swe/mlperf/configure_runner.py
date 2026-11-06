#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Adapt the standalone runner's generated JSON site.yaml for AReaL routing."""

import argparse
import copy
import ipaddress
import json
import os
from pathlib import Path


def validate_ipv4_host(host: str) -> str:
    """Require a concrete IPv4 address, not a wildcard, URL or hostname."""
    if not isinstance(host, str):
        raise ValueError("Advertised host must be a concrete IPv4 address")
    try:
        address = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError as exc:
        raise ValueError("Advertised host must be a concrete IPv4 address") from exc
    if address.is_unspecified or address.is_multicast or address.is_reserved:
        raise ValueError("Advertised host must be a concrete unicast IPv4 address")
    return str(address)


def validate_ray_address(address: str) -> str:
    """Validate the Ray GCS host:port form used by the cluster launcher."""
    if not isinstance(address, str):
        raise ValueError("Ray head address must use IPv4:port")
    host, separator, port = address.rpartition(":")
    if not separator or not port.isascii() or not port.isdecimal():
        raise ValueError("Ray head address must use IPv4:port")
    validate_ipv4_host(host)
    if not 1 <= int(port) <= 65535:
        raise ValueError("Ray head port must be in 1..65535")
    return address


def validate_ports(ports: list[int]) -> None:
    if any(type(port) is not int or not 1 <= port <= 65535 for port in ports):
        raise ValueError("Ports must be integers in 1..65535")
    if len(set(ports)) != len(ports):
        raise ValueError("Bridge, Gym, head and model ports must be distinct")


def configure(
    config: dict,
    bridge_port: int,
    gym_port: int,
    *,
    advertised_host: str = "127.0.0.1",
    ray_head_address: str | None = None,
) -> dict:
    result = copy.deepcopy(config)
    swe = result["swe_agents"]["responses_api_agents"]["swe_agents"]
    model = result["policy_model"]["responses_api_models"]["vllm_model"]
    ports = [bridge_port, gym_port, result["head_server"]["port"], model["port"]]
    validate_ports(ports)
    advertised_host = validate_ipv4_host(advertised_host)
    if ray_head_address is not None:
        result["ray_head_node_address"] = ray_head_address
    if result.get("ray_head_node_address") is not None:
        validate_ray_address(result["ray_head_node_address"])
    swe.update(host="127.0.0.1", port=gym_port)
    result["head_server"]["host"] = advertised_host
    result["policy_base_url"] = f"http://127.0.0.1:{bridge_port}/v1"
    result["policy_api_key"] = "unused"
    result["policy_model_name"] = "areal-routed"
    model.update(
        entrypoint="areal_app.py",
        num_workers=1,
        host=advertised_host,
        base_url="${policy_base_url}",
        api_key="${policy_api_key}",
        model="${policy_model_name}",
        return_token_id_information=False,
        is_responses_native=False,
        # AReaL returns reasoning inline in content, not in provider-only fields.
        uses_reasoning_parser=False,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge-port", type=int, default=8080)
    parser.add_argument("--gym-port", type=int, default=8083)
    parser.add_argument(
        "--advertised-host",
        default="127.0.0.1",
        type=validate_ipv4_host,
        help="Registry/model IPv4 reachable from sandbox workers (loopback for one host)",
    )
    parser.add_argument(
        "--ray-head-address",
        type=validate_ray_address,
        help="Existing Ray cluster IPv4:port; preserve the input setting when omitted",
    )
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as source:
        config = configure(
            json.load(source),
            args.bridge_port,
            args.gym_port,
            advertised_host=args.advertised_host,
            ray_head_address=args.ray_head_address,
        )
    encoded = json.dumps(config, indent=2, allow_nan=False) + "\n"
    # Exclusive creation preserves an existing site configuration.
    with os.fdopen(
        os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as out:
        out.write(encoded)


if __name__ == "__main__":
    main()
