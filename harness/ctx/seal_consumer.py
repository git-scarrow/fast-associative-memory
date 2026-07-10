#!/usr/bin/env python3
"""PR-13 consumer seal — fetch/verify the pinned Qwen3-8B artifacts
(memo §8.1; artifact-level pin in harness/ctx/policy/consumer_pin.json).

Usage:
    python harness/ctx/seal_consumer.py fetch-tokenizer   # small files
    python harness/ctx/seal_consumer.py verify            # verify sealed dir

Artifacts land in harness/ctx/sealed/qwen3-8b/ (gitignored — the pin
IS the committed record; bytes are re-verifiable anywhere). ``verify``
checks every present file against the pinned sha256 and reports what
is still missing for the replay (the five weight shards, ~16 GB, are
fetched on the scoring host and MUST verify before the first
evaluation render — the §8.1 selection-timing kill).

Deterministic; network access only in ``fetch-tokenizer``.
"""

import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_PATH = os.path.join(HERE, "policy", "consumer_pin.json")
SEAL_DIR = os.path.join(HERE, "sealed", "qwen3-8b")


def _pin():
    with open(PIN_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected(pin):
    exp = {}
    for group in ("tokenizer_sha256", "config_sha256", "weights_sha256",
                  "index_sha256"):
        exp.update(pin["artifact"][group])
    return exp


def chat_template_sha256(seal_dir=None):
    """The non-thinking mechanism named in the family-level pin lives in
    the chat template; hash it directly rather than trusting the enclosing
    file's sha to stand for it."""
    path = os.path.join(seal_dir or SEAL_DIR, "tokenizer_config.json")
    with open(path, "r", encoding="utf-8") as fh:
        template = json.load(fh)["chat_template"]
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def fetch_tokenizer():
    pin = _pin()
    repo = pin["artifact"]["repository_id"]
    rev = pin["artifact"]["revision"]
    os.makedirs(SEAL_DIR, exist_ok=True)
    small = {**pin["artifact"]["tokenizer_sha256"],
             **pin["artifact"]["config_sha256"],
             **pin["artifact"]["index_sha256"]}
    for name, want in sorted(small.items()):
        dest = os.path.join(SEAL_DIR, name)
        if os.path.exists(dest) and _sha_file(dest) == want:
            print(f"ok (cached)  {name}")
            continue
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
        urllib.request.urlretrieve(url, dest)
        got = _sha_file(dest)
        if got != want:
            os.remove(dest)
            raise SystemExit(f"SEAL FAILURE: {name} sha {got} != pinned {want}")
        print(f"ok (fetched) {name}")


def verify():
    pin = _pin()
    expected = _expected(pin)
    present, missing, bad = [], [], []
    for name, want in sorted(expected.items()):
        path = os.path.join(SEAL_DIR, name)
        if not os.path.exists(path):
            missing.append(name)
        elif _sha_file(path) == want:
            present.append(name)
        else:
            bad.append(name)
    # replay_ready means EVERY pinned artifact is present and byte-exact.
    # (An earlier form only looked for missing weight shards, so a missing
    # model.safetensors.index.json — which a sharded load needs — would
    # have reported ready. Amendment A-2 pins the index and this checks it.)
    report = {"sealed_ok": present, "missing": missing, "sha_mismatch": bad,
              "replay_ready": not bad and not missing}
    if not missing and not bad:
        report["chat_template_sha256"] = chat_template_sha256()
    print(json.dumps(report, indent=2))
    if bad:
        raise SystemExit("SEAL FAILURE: sha mismatch — do not render")
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"fetch-tokenizer": fetch_tokenizer, "verify": verify}[cmd]()
