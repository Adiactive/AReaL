#!/usr/bin/env python3
"""Submit one supplied MLPerf JSONL task to a standalone SWE runner.

No requests are retried: retrying /run could launch a duplicate episode.
Exit 0 means a complete, unmasked attempt, whether solved or unsolved.
Exit 2 means invalid input, transport failure, masking or an incomplete run.
"""

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


def load_request(path: Path, row_index: int, args: argparse.Namespace) -> dict:
    with path.open(encoding="utf-8") as source:
        rows_seen = 0
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if rows_seen == row_index:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at line {line_number}") from exc
                break
            rows_seen += 1
        else:
            raise ValueError("requested row is beyond the end of the JSONL file")
    if not isinstance(row, dict):
        raise ValueError("JSONL task must be an object")
    params = row.get("responses_create_params")
    if not isinstance(params, dict):
        raise ValueError("task must contain a responses_create_params object")
    metadata = params.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("responses_create_params.metadata must be an object")
    for key in ("instance_id", "dataset_name", "instance_dict"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata.{key} must be a nonempty string")
    try:
        instance = json.loads(metadata["instance_dict"])
    except json.JSONDecodeError as exc:
        raise ValueError(
            "metadata.instance_dict must contain a JSON-encoded object"
        ) from exc
    if not isinstance(instance, dict):
        raise ValueError("metadata.instance_dict must contain a JSON-encoded object")
    if not isinstance(params.get("input"), (str, list)):
        raise ValueError("responses_create_params.input must be a string or list")
    if "model" not in params and args.model:
        params["model"] = args.model
    if not isinstance(params.get("model"), str) or not params["model"].strip():
        raise ValueError("task needs responses_create_params.model or --model")
    # The runner indexes temperature and top_p directly. Preserve every supplied
    # generation setting and fill only fields omitted from the dataset row.
    params.setdefault("temperature", args.temperature)
    params.setdefault("top_p", args.top_p)
    params.setdefault("max_output_tokens", args.max_output_tokens)
    for key, minimum, maximum in (("temperature", 0, 2), ("top_p", 0, 1)):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key} is outside the supported range")
    if params["top_p"] == 0:
        raise ValueError("top_p must be greater than zero")
    limit = params["max_output_tokens"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_output_tokens must be a positive integer")
    return row


def classify_result(result: object) -> tuple[dict, int]:
    if not isinstance(result, dict):
        return {"status": "invalid_result"}, 2
    config = result.get("instance_config")
    config = config if isinstance(config, dict) else {}
    summary = {
        "reward": result.get("reward"),
        "resolved": result.get("resolved"),
        "patch_exists": result.get("patch_exists"),
        "mask_sample": config.get("mask_sample", result.get("mask_sample", False)),
        "agent_error_kind": result.get("agent_error_kind"),
        "agent_timed_out": result.get("agent_timed_out"),
        "eval_timed_out": result.get("eval_timed_out"),
    }
    # Only publish known scalar fields. Do not echo provider errors, instance
    # config, prompts, generated code, or model credentials to the terminal.
    allowed_errors = {"max_iteration", "context_window", "stuck_in_loop", "other"}
    error_kind = summary["agent_error_kind"]
    if error_kind is not None and (
        not isinstance(error_kind, str) or error_kind not in allowed_errors
    ):
        summary["agent_error_kind"] = "unrecognized_error"
    for key in (
        "resolved",
        "patch_exists",
        "mask_sample",
        "agent_timed_out",
        "eval_timed_out",
    ):
        if summary[key] is not None and not isinstance(summary[key], bool):
            return {"status": "invalid_result"}, 2
    reward = summary["reward"]
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
    ):
        return {"status": "invalid_result"}, 2
    response = result.get("response")
    output = response.get("output") if isinstance(response, dict) else None
    if summary["agent_timed_out"] or summary["eval_timed_out"]:
        summary["status"] = "timed_out"
    elif summary["mask_sample"]:
        summary["status"] = "masked"
    elif summary["agent_error_kind"]:
        summary["status"] = "agent_ended_with_error"
    elif not isinstance(output, list) or not output or summary["resolved"] is None:
        summary["status"] = "incomplete"
    else:
        summary["status"] = "resolved" if summary["resolved"] else "unresolved"
        return summary, 0
    return summary, 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument(
        "--row", type=int, default=0, help="zero-based nonempty JSONL row"
    )
    parser.add_argument(
        "--runner-url", help="internal runner root, e.g. http://cpu-host:8080"
    )
    parser.add_argument(
        "--output", type=Path, help="new full-result JSON file (mode 0600)"
    )
    parser.add_argument("--model", help="fill model only when absent from the task")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="fill missing temperature"
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="fill missing top_p")
    parser.add_argument(
        "--max-output-tokens", type=int, default=32768, help="fill missing token limit"
    )
    parser.add_argument(
        "--timeout", type=float, default=7200, help="HTTP wait timeout in seconds"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the selected task offline; do not send or save anything",
    )
    args = parser.parse_args()
    if args.row < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("row must be nonnegative and timeout must be finite and positive")
    if not args.check_only and (not args.runner_url or not args.output):
        parser.error(
            "--runner-url and --output are required unless --check-only is used"
        )
    if args.runner_url:
        try:
            url = urlsplit(args.runner_url)
            url.port
        except ValueError:
            parser.error("invalid runner endpoint URL or port")
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
            or any(character.isspace() for character in args.runner_url)
        ):
            parser.error(
                "runner URL must be an internal HTTP(S) root without credentials or path"
            )
    return args


def main() -> int:
    args = parse_args()
    try:
        body = load_request(args.jsonl, args.row, args)
        encoded = json.dumps(body, allow_nan=False).encode("utf-8")
    except (OSError, ValueError) as exc:
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        sys.stderr.write(f"Task validation failed: {message}\n")
        return 2
    if args.check_only:
        sys.stdout.write(
            "Task valid; original fields preserved, missing generation defaults filled. No network request made.\n"
        )
        return 0
    # Reserve output before submitting, so a path error cannot lose the result
    # after an expensive episode. O_EXCL also prevents accidental replacement.
    try:
        output_fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        sys.stderr.write(
            f"Cannot create result file ({type(exc).__name__}); no request sent.\n"
        )
        return 2
    try:
        with os.fdopen(output_fd, "w") as output:
            request = urllib.request.Request(
                args.runner_url.rstrip("/") + "/run",
                data=encoded,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # Internal traffic must not be sent through an inherited web proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=args.timeout) as response:
                result = json.load(response)
            output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except urllib.error.HTTPError as exc:
        sys.stderr.write(
            f"Runner returned HTTP {exc.code}; response body suppressed. No retry attempted.\n"
        )
        return 2
    except TimeoutError:
        sys.stderr.write(
            "HTTP wait timed out; the remote attempt may still be running. No retry attempted.\n"
        )
        return 2
    except (OSError, ValueError, urllib.error.URLError) as exc:
        sys.stderr.write(
            f"Runner request/result failed ({type(exc).__name__}); no retry attempted.\n"
        )
        return 2
    summary, status = classify_result(result)
    sys.stdout.write(json.dumps(summary, allow_nan=False) + "\n")
    sys.stdout.write(
        f"Full result saved to {args.output} with owner-only permissions.\n"
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
