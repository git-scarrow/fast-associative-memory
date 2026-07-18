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
from .preregistration import validate_registration
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

#: A seal must say what it is. ``plumbing`` cannot carry a registration and is
#: never admissible as evidence; ``scoring-run`` must carry a complete one
#: (gate G-I1) and binds the treatment (gate G-I2).
EVIDENCE_CLASSES = ("plumbing", "scoring-run")

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
    if settings["cam_adaptive_eviction"] is not False:
        raise ValueError("cam_adaptive_eviction must be explicitly false")
    if settings["cam_sleep"] is not False:
        raise ValueError("cam_sleep must be explicitly false")

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
    """Seal inputs and, for a scoring run, the treatment that acts on them.

    ``evidence_class`` is mandatory and self-labelling: a ``plumbing`` seal
    refuses to carry a registration, and a ``scoring-run`` seal refuses to
    exist without a complete one. There is no default, because the dangerous
    mistake is sealing a real run as though it were a rehearsal.
    """
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(
            f"evidence_class must be one of {list(EVIDENCE_CLASSES)}, got {evidence_class!r}"
        )

    validate_retriever_settings(retriever_settings, record_count=len(records))

    sealed_memo_path: str | None = None
    if evidence_class == "plumbing":
        if registration is not None:
            raise ValueError("a plumbing seal cannot carry a registration")
        if registration_memo_path is not None:
            raise ValueError("a plumbing seal cannot carry a registration memo path")
    else:
        sealed_memo_path = _refuse_unless_registrable(
            policy,
            consumer_pin,
            registration,
            registration_memo_path,
        )

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


def _refuse_unless_registrable(
    policy: Mapping[str, Any] | None,
    consumer_pin: str | None,
    registration: Mapping[str, Any] | None,
    registration_memo_path: str | Path | None,
) -> str | None:
    """Gate G-I1 + G-I2 at seal time. Reports every reason at once."""
    problems: list[str] = []
    if policy is None:
        problems.append("scoring-run seal requires the disposition policy (G-I2)")
    if not consumer_pin:
        problems.append("scoring-run seal requires a consumer_pin (G-I2)")
    if registration is None:
        problems.append("scoring-run seal requires a registration block (G-I1)")
    else:
        problems.extend(
            f"registration: {error}" for error in validate_registration(registration)
        )
    memo_path: Path | None = None
    if registration_memo_path is None:
        problems.append("scoring-run seal requires a registration memo path (G-I1)")
    else:
        memo_path = Path(registration_memo_path).resolve()
        try:
            memo_digest = sha256(memo_path.read_bytes()).hexdigest()
        except OSError as exc:
            problems.append(
                f"registration memo path {str(memo_path)!r} is unreadable: {exc}"
            )
        else:
            registered_digest = (
                registration.get("memo_sha256") if registration is not None else None
            )
            if registered_digest != memo_digest:
                problems.append(
                    "registration memo_sha256 does not match the bytes at the "
                    f"registered memo path: registered={registered_digest!r}, "
                    f"actual={memo_digest!r}"
                )
    if problems:
        raise RuntimeError(
            "refusing to seal a scoring run:\n  - " + "\n  - ".join(problems)
        )
    assert memo_path is not None
    return str(memo_path)


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
