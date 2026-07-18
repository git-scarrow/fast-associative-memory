"""Canonical input seals for reproducible five-arm runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from . import ARM_NAMES, CONTEXT_BUDGET_TOKENS
from . import scoring as _scoring_module
from .context import POLICY_VERSION
from .models import MemoryQuestion, MemoryRecord
from .retrievers import (
    CAMIndexSettings,
    ExemplarCAMRetriever,
    FAMRetriever,
    TensorLike,
)
from .scoring import SCORING_VERSION

__all__ = [
    "MANIFEST_VERSION",
    "SCORING_VERSION",
    "EVIDENCE_CLASSES",
    "PHASE_B_SCORING_REFUSAL",
    "canonical_json",
    "policy_sha256",
    "scoring_module_sha256",
    "validate_retriever_settings",
    "build_cam_indexes",
    "seal_manifest",
    "load_manifest",
    "verify_manifest",
]

MANIFEST_VERSION = "memory-eval-manifest-v3"

#: A manifest must say what it is. ``plumbing`` is the only class Phase A may
#: seal. ``scoring-run`` remains recognized so audit verification can diagnose
#: externally constructed envelopes, but public creation/execution is closed.
EVIDENCE_CLASSES = ("plumbing", "scoring-run")

PHASE_B_SCORING_REFUSAL = (
    "Phase B provenance/reconciliation envelope is not implemented; "
    "Phase A cannot seal or execute a scoring-run"
)

RETRIEVER_SETTING_KEYS = frozenset(
    {
        "candidate_k",
        "cam_max_entries",
        "cam_prototype_k",
        "cam_vigilance",
        "cam_hebb_lr",
        "cam_key_lr",
        "cam_ema_beta",
        "cam_inference_temp",
        "cam_use_bfloat16",
        "cam_adaptive_eviction",
        "cam_use_lfu",
        "cam_dynamic_vigilance",
        "cam_retrieval_floor",
        "cam_retrieval_truncation",
        "cam_nstp",
        "cam_sleep",
        "cam_ingest_order",
        "exemplar_write_mode",
        "fam_write_mode",
    }
)


def validate_retriever_settings(
    settings: Mapping[str, Any], *, record_count: int
) -> CAMIndexSettings:
    """Validate the complete, closed CAM treatment and return live settings.

    The schema deliberately has no optional omissions: inactive policies are
    sealed as explicit ``None``/``False`` values, and both retrieval widths are
    mandatory. That makes an old caller or a newly introduced runtime default
    fail closed instead of changing the treatment silently.
    """
    seen = set(settings)
    missing = sorted(RETRIEVER_SETTING_KEYS - seen)
    unknown = sorted(seen - RETRIEVER_SETTING_KEYS)
    problems: list[str] = []
    if missing:
        problems.append(f"missing retriever settings: {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown retriever settings: {', '.join(unknown)}")
    if problems:
        raise ValueError("; ".join(problems))

    for name in ("candidate_k", "cam_max_entries", "cam_prototype_k"):
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    if settings["cam_max_entries"] != record_count:
        raise ValueError(
            f"cam_max_entries {settings['cam_max_entries']} must equal "
            f"record count {record_count}"
        )
    if settings["cam_prototype_k"] > settings["cam_max_entries"]:
        raise ValueError("cam_prototype_k must not exceed cam_max_entries")

    for name in (
        "cam_vigilance",
        "cam_hebb_lr",
        "cam_key_lr",
        "cam_ema_beta",
        "cam_inference_temp",
    ):
        value = settings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if not 0 <= settings["cam_vigilance"] <= 1:
        raise ValueError("cam_vigilance must lie in [0, 1]")
    for name in ("cam_hebb_lr", "cam_key_lr"):
        if not 0 <= settings[name] <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if not 0 <= settings["cam_ema_beta"] <= 1:
        raise ValueError("cam_ema_beta must lie in [0, 1]")
    if settings["cam_inference_temp"] <= 0:
        raise ValueError("cam_inference_temp must be positive")

    for name in (
        "cam_use_bfloat16",
        "cam_adaptive_eviction",
        "cam_use_lfu",
        "cam_sleep",
    ):
        if not isinstance(settings[name], bool):
            raise ValueError(f"{name} must be a boolean")

    for name in (
        "cam_dynamic_vigilance",
        "cam_retrieval_floor",
        "cam_retrieval_truncation",
        "cam_nstp",
    ):
        if settings[name] is not None:
            raise ValueError(f"{name} must be explicitly null")
    if settings["cam_sleep"] is not False:
        raise ValueError("cam_sleep must be explicitly false")

    fixed_treatment = {
        "cam_vigilance": 0.85,
        "cam_hebb_lr": 0.1,
        "cam_key_lr": 0.05,
        "cam_ema_beta": 0.05,
        "cam_inference_temp": 0.05,
        "cam_use_bfloat16": False,
        "cam_adaptive_eviction": False,
        "cam_use_lfu": True,
    }
    for name, expected in fixed_treatment.items():
        if settings[name] != expected:
            raise ValueError(f"{name} must be exactly {expected!r}")

    fixed = {
        "cam_ingest_order": "manifest-record-order",
        "exemplar_write_mode": "allocate-only",
        "fam_write_mode": "condense",
    }
    for name, expected in fixed.items():
        if settings[name] != expected:
            raise ValueError(f"{name} must be {expected!r}")

    return CAMIndexSettings(
        max_entries=settings["cam_max_entries"],
        prototype_k=settings["cam_prototype_k"],
        vigilance=settings["cam_vigilance"],
        hebb_lr=settings["cam_hebb_lr"],
        key_lr=settings["cam_key_lr"],
        ema_beta=settings["cam_ema_beta"],
        inference_temp=settings["cam_inference_temp"],
        use_bfloat16=settings["cam_use_bfloat16"],
        adaptive_eviction=settings["cam_adaptive_eviction"],
        use_lfu=settings["cam_use_lfu"],
    )


def build_cam_indexes(
    records: Sequence[MemoryRecord],
    record_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
) -> tuple[ExemplarCAMRetriever, FAMRetriever]:
    """Build the matched E0/F0 indexes from one validated treatment."""
    settings = validate_retriever_settings(
        retriever_settings, record_count=len(records)
    )
    return (
        ExemplarCAMRetriever(records, record_embeddings, settings=settings),
        FAMRetriever(records, record_embeddings, settings=settings),
    )


def input_fingerprints(
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
) -> dict[str, str]:
    """Fingerprint the sealed INPUTS only, independent of the treatment.

    Separating this from the protocol lets a caller ask "are the inputs the
    sealed ones?" without also re-supplying the policy and consumer — so a
    drifted treatment is reported by its own named check rather than being
    swallowed as a generic fingerprint mismatch.
    """
    return {
        "records": _fingerprint([asdict(record) for record in records]),
        "questions": _fingerprint([asdict(question) for question in questions]),
        "record_embeddings": _fingerprint(_embedding_value(record_embeddings)),
        "query_embeddings": _fingerprint(_embedding_value(query_embeddings)),
    }


def policy_sha256(policy: Mapping[str, Any]) -> str:
    """Fingerprint the disposition policy by CONTENT, not by path.

    The policy is the governed arms' treatment. Revision 1 of the seal left it
    entirely outside the manifest, so an edit between seal and execution ran
    silently under a passing verify.
    """
    return _fingerprint(dict(policy))


def scoring_module_sha256() -> str:
    """Fingerprint the scorer's source. ``scoring_version`` is only a label;
    without this, a post-seal edit to score_rows ships under the sealed label.
    """
    return sha256(Path(_scoring_module.__file__).read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def seal_manifest(
    path: str | Path,
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    *,
    evidence_class: str,
    policy: Mapping[str, Any] | None = None,
    consumer_pin: str | None = None,
    registration: Mapping[str, Any] | None = None,
    registration_memo_path: str | Path | None = None,
) -> dict[str, Any]:
    """Seal a Phase-A plumbing manifest; reject scoring-run creation.

    ``evidence_class`` stays mandatory so no caller can silently downgrade a
    real run to rehearsal status. Phase B must implement provenance and source
    reconciliation before this public API may create confirmatory evidence.
    """
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(
            f"evidence_class must be one of {list(EVIDENCE_CLASSES)}, got {evidence_class!r}"
        )

    if evidence_class == "scoring-run":
        # Phase A deliberately lacks the source-to-normalized provenance and
        # reconciliation envelope required for confirmatory evidence.  Pins,
        # policy bytes, a registration, and index attestations are necessary
        # but cannot substitute for that missing boundary.
        raise RuntimeError(PHASE_B_SCORING_REFUSAL)

    validate_retriever_settings(retriever_settings, record_count=len(records))

    sealed_memo_path: str | None = None
    if evidence_class == "plumbing":
        if registration is not None:
            raise ValueError("a plumbing seal cannot carry a registration")
        if registration_memo_path is not None:
            raise ValueError("a plumbing seal cannot carry a registration memo path")
    else:  # pragma: no cover - scoring-run is refused above in Phase A
        raise AssertionError("unreachable Phase A scoring seal")

    manifest = _manifest_body(
        records,
        questions,
        record_embeddings,
        query_embeddings,
        retriever_settings,
        evidence_class=evidence_class,
        policy=policy,
        consumer_pin=consumer_pin,
        registration=registration,
        registration_memo_path=sealed_memo_path,
    )
    manifest["manifest_sha256"] = _fingerprint(manifest)
    Path(path).write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("sealed") is not True:
        raise RuntimeError("experiment manifest is not sealed")
    claimed = manifest.get("manifest_sha256")
    if not isinstance(claimed, str):
        raise RuntimeError("experiment manifest has no seal digest")
    body = dict(manifest)
    del body["manifest_sha256"]
    if _fingerprint(body) != claimed:
        raise RuntimeError("experiment manifest seal digest mismatch")
    return manifest


def verify_manifest(
    path: str | Path,
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    *,
    evidence_class: str,
    policy: Mapping[str, Any] | None = None,
    consumer_pin: str | None = None,
    registration: Mapping[str, Any] | None = None,
    registration_memo_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(path)
    expected = _manifest_body(
        records,
        questions,
        record_embeddings,
        query_embeddings,
        retriever_settings,
        evidence_class=evidence_class,
        policy=policy,
        consumer_pin=consumer_pin,
        registration=registration,
        registration_memo_path=(
            None
            if registration_memo_path is None
            else str(Path(registration_memo_path).resolve())
        ),
    )
    if manifest.get("fingerprints") != expected["fingerprints"]:
        raise RuntimeError("manifest fingerprint mismatch")
    if manifest.get("protocol") != expected["protocol"]:
        raise RuntimeError("manifest fingerprint mismatch: protocol")
    actual_body = dict(manifest)
    del actual_body["manifest_sha256"]
    if actual_body != expected:
        extra = sorted(set(actual_body) - set(expected))
        missing = sorted(set(expected) - set(actual_body))
        differing = sorted(
            key
            for key in set(actual_body) & set(expected)
            if actual_body[key] != expected[key]
        )
        details: list[str] = []
        if extra:
            details.append(f"unexpected top-level fields: {', '.join(extra)}")
        if missing:
            details.append(f"missing top-level fields: {', '.join(missing)}")
        if differing:
            details.append(f"differing top-level fields: {', '.join(differing)}")
        raise RuntimeError("manifest body mismatch: " + "; ".join(details))
    return manifest


def _manifest_body(
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    *,
    evidence_class: str,
    policy: Mapping[str, Any] | None,
    consumer_pin: str | None,
    registration: Mapping[str, Any] | None,
    registration_memo_path: str | None,
) -> dict[str, Any]:
    _scoring_module.validate_raw_with_invariant(records)
    validate_retriever_settings(retriever_settings, record_count=len(records))
    records_value = [asdict(record) for record in records]
    questions_value = [asdict(question) for question in questions]
    record_vectors = _embedding_value(record_embeddings)
    query_vectors = _embedding_value(query_embeddings)
    protocol = {
        "arms": list(ARM_NAMES),
        "context_budget_tokens": CONTEXT_BUDGET_TOKENS,
        "evidence_class": evidence_class,
        "scoring_version": SCORING_VERSION,
        # The scorer's source, not just its label (G-I2, scorer identity).
        "scoring_module_sha256": scoring_module_sha256(),
        "policy_version": POLICY_VERSION,
        # The governed arms' treatment (G-I2).
        "policy_sha256": None if policy is None else policy_sha256(policy),
        "consumer_pin": consumer_pin,
        "registration": None if registration is None else dict(registration),
        "retriever_settings": dict(retriever_settings),
    }
    if evidence_class == "scoring-run":
        exemplar, fam = build_cam_indexes(
            records, record_embeddings, retriever_settings
        )
        protocol["registration_memo_path"] = registration_memo_path
        protocol["index_attestations"] = {
            "exemplar": asdict(exemplar.attestation),
            "fam": asdict(fam.attestation),
        }
    fingerprints = {
        "records": _fingerprint(records_value),
        "questions": _fingerprint(questions_value),
        "record_embeddings": _fingerprint(record_vectors),
        "query_embeddings": _fingerprint(query_vectors),
        "protocol": _fingerprint(protocol),
    }
    return {
        "manifest_version": MANIFEST_VERSION,
        "sealed": True,
        "record_count": len(records_value),
        "question_count": len(questions_value),
        "protocol": protocol,
        "fingerprints": fingerprints,
    }


def _embedding_value(embeddings: Mapping[str, TensorLike]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for item_id in sorted(embeddings):
        vector = torch.as_tensor(embeddings[item_id], dtype=torch.float32).detach().cpu()
        if vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
            raise ValueError(f"invalid embedding for {item_id}")
        result[str(item_id)] = [float(value) for value in vector.tolist()]
    return result


def _fingerprint(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()
