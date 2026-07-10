"""Tests for the committed, sealed PR-13 scoring manifest (memo §8/§11).

The scoring manifest is the artifact the sealed runner demands before it
will touch a real consumer. These tests check that it is sealed, that it
pins the exact query manifest, that it carries every field the run
deliverable requires, and that the runner's real-render gate opens for it
and for nothing else. No consumer is constructed; nothing is rendered.
"""

import hashlib
import json
import os
import re

import pytest

from harness.ctx import replay

MANIFESTS = os.path.join("harness", "ctx", "manifests")
SCORING_PATH = os.path.join(MANIFESTS, "scoring_manifest_v1.json")
QUERY_PATH = os.path.join(MANIFESTS, "query_manifest_v1.json")


@pytest.fixture(scope="module")
def scoring():
    with open(SCORING_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def query_manifest():
    with open(QUERY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


class _Tripwire:
    is_real = True
    pin_id = "tripwire"

    def generate(self, prompt, max_new_tokens=256):
        raise AssertionError("a real render was attempted")


def test_scoring_manifest_is_sealed_and_pins_the_query_manifest(
        scoring, query_manifest):
    replay.verify_scoring_manifest(scoring, query_manifest)
    assert scoring["query_manifest_sha256"] == query_manifest["manifest_sha256"]


def test_scoring_manifest_seal_detects_tampering(scoring, query_manifest):
    tampered = json.loads(json.dumps(scoring))
    tampered["precision"] = "float16"
    with pytest.raises(replay.SealError):
        replay.verify_scoring_manifest(tampered, query_manifest)


def test_it_records_the_exact_code_commit_and_source_hashes(scoring):
    assert re.fullmatch(r"[0-9a-f]{40}", scoring["code"]["commit"])
    assert scoring["code"]["worktree_clean"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", scoring["code"]["ctx_source_sha256"])
    files = scoring["code"]["files"]
    for required in ("harness/ctx/compile.py", "harness/ctx/replay.py",
                     "harness/ctx/output_contract.py",
                     "harness/ctx/policy/disposition_policy_v1.json",
                     "harness/ctx/policy/consumer_pin.json",
                     "harness/ctx/policy/sample_v1.json",
                     "harness/ctx/prompts/single_turn_v1.txt"):
        assert required in files, required
        assert re.fullmatch(r"[0-9a-f]{64}", files[required])


def test_recorded_source_hashes_match_the_files_on_disk(scoring):
    for path, want in scoring["code"]["files"].items():
        with open(path, "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == want, path


def test_it_records_the_sample_manifest_and_salt(scoring):
    assert scoring["sample"]["salt"] == "pr13-sample-v1"
    assert scoring["sample"]["cap_per_cell"] == 256
    assert re.fullmatch(r"[0-9a-f]{64}",
                        scoring["sample"]["sample_policy_sha256"])
    assert scoring["sample"]["rows"] == 20304


def test_it_records_model_tokenizer_config_and_index_hashes(scoring):
    c = scoring["consumer"]
    assert c["repository_id"] == "Qwen/Qwen3-8B"
    assert len(c["revision"]) == 40
    assert len(c["weights_sha256"]) == 5
    assert len(c["tokenizer_sha256"]) == 4
    assert len(c["config_sha256"]) == 2
    assert len(c["index_sha256"]) == 1
    assert "model.safetensors.index.json" in c["index_sha256"]
    for group in ("weights_sha256", "tokenizer_sha256", "config_sha256",
                  "index_sha256"):
        assert all(re.fullmatch(r"[0-9a-f]{64}", v)
                   for v in c[group].values()), group


def test_it_records_prompt_and_chat_template_hashes(scoring):
    assert set(scoring["prompts"]) == {"single_turn_v1.txt", "none_arm_v1.txt"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", v)
               for v in scoring["prompts"].values())
    assert re.fullmatch(r"[0-9a-f]{64}",
                        scoring["consumer"]["chat_template_sha256"])


def test_recorded_prompt_hashes_match_the_sealed_templates(scoring):
    assert scoring["prompts"] == replay.prompt_shas()


def test_it_records_runtime_and_library_versions(scoring):
    rt = scoring["runtime"]
    assert rt["host"] and rt["platform"] and rt["python"]
    libs = rt["libraries"]
    for required in ("torch", "transformers", "tokenizers", "safetensors"):
        assert libs[required], required
    assert rt["accelerator"]["kind"] in ("cuda", "cpu")


def test_it_records_bfloat16_greedy_and_the_registered_limits(scoring):
    assert scoring["precision"] == "bfloat16"
    assert scoring["quantization"] == "none"
    d = scoring["decoding"]
    assert d["strategy"] == "greedy"
    assert d["do_sample"] is False
    assert d["temperature"] == 0.0
    assert d["enable_thinking"] is False
    assert d["max_new_tokens"] == 256
    limits = scoring["limits"]
    assert limits["input_budgets"] == [800, 1500]
    assert limits["raw_native_budget"] == 1500
    assert limits["max_new_tokens"] == 256
    assert scoring["arm_plan"] == [list(a) for a in replay.ARM_PLAN]


def test_it_records_the_parser_version(scoring):
    p = scoring["parser"]
    assert p["contract_id"] == "pr13-consumer-output-contract"
    assert p["contract_version"] == "1.0"
    for key in ("contract_sha256", "schema_sha256", "module_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", p[key]), key


def test_it_reports_the_consumer_as_sealed_on_the_scoring_host(scoring):
    assert scoring["consumer"]["replay_ready"] is True
    assert scoring["consumer"]["verified_on_host"]


# --- the gate it exists to open ---------------------------------------------

def test_the_real_render_gate_opens_only_with_both_sealed_manifests(
        scoring, query_manifest):
    replay.assert_render_permitted(_Tripwire(), query_manifest, scoring)

    with pytest.raises(replay.SealError):
        replay.assert_render_permitted(_Tripwire(), query_manifest, None)

    unsealed = {k: v for k, v in query_manifest.items()
                if k not in ("sealed", "manifest_sha256")}
    with pytest.raises(replay.SealError):
        replay.assert_render_permitted(_Tripwire(), unsealed, scoring)


def test_a_scoring_manifest_reporting_an_unsealed_consumer_is_refused(
        scoring, query_manifest):
    stale = json.loads(json.dumps(scoring))
    stale["consumer"]["replay_ready"] = False
    stale = replay.seal_manifest(stale)
    with pytest.raises(replay.SealError, match="replay_ready"):
        replay.verify_scoring_manifest(stale, query_manifest)
