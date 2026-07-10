#!/usr/bin/env python3
"""PR-13 scoring-host probe (memo §8.1).

    <venv>/bin/python -m harness.ctx.host_probe > runtime.json

Runs on the machine that will execute the sealed replay and reports the
facts the scoring manifest cannot know from the repository: which Python
and which library versions will produce the numbers, what accelerator is
present, and — the load-bearing one — whether every artifact pinned in
`consumer_pin.json` is present on this host and byte-exact.

Emits no timestamp: the scoring manifest must be reproducible, so the
same host in the same state must yield the same digest.

Loads no weights, renders nothing, calls no model.
"""

import json
import os
import platform
import socket
import sys

from harness.ctx.seal_consumer import (
    SEAL_DIR,
    _expected,
    _pin,
    _sha_file,
    chat_template_sha256,
)

LIBRARIES = ("torch", "transformers", "tokenizers", "safetensors", "accelerate")


def _versions():
    out = {}
    for name in LIBRARIES:
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def _accelerator():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"kind": "cpu"}
        props = torch.cuda.get_device_properties(0)
        return {"kind": "cuda", "name": props.name,
                "total_bytes": props.total_memory,
                "capability": f"{props.major}.{props.minor}",
                "count": torch.cuda.device_count()}
    except Exception as exc:
        return {"kind": "unknown", "error": repr(exc)}


def _seal(seal_dir=None):
    seal_dir = seal_dir or SEAL_DIR
    pin = _pin()
    present, missing, bad = [], [], []
    for name, want in sorted(_expected(pin).items()):
        path = os.path.join(seal_dir, name)
        if not os.path.exists(path):
            missing.append(name)
        elif _sha_file(path) == want:
            present.append(name)
        else:
            bad.append(name)
    report = {"pin_id": pin["pin_id"],
              "repository_id": pin["artifact"]["repository_id"],
              "revision": pin["artifact"]["revision"],
              "verified": present, "missing": missing, "sha_mismatch": bad,
              "replay_ready": not missing and not bad}
    if report["replay_ready"]:
        got = chat_template_sha256(seal_dir)
        want = pin["artifact"]["chat_template_sha256"]
        report["chat_template_sha256"] = got
        if got != want:
            report["replay_ready"] = False
            report["chat_template_mismatch"] = {"got": got, "want": want}
    return report


def probe(seal_dir=None):
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "libraries": _versions(),
        "accelerator": _accelerator(),
        "seal": _seal(seal_dir),
    }


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2, sort_keys=True))
