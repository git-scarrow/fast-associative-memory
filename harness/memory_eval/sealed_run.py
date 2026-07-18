"""Preflight and construct the only admissible sealed scoring runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .ledger import MemoryLedger
from .manifest import (
    MANIFEST_VERSION,
    build_cam_indexes,
    input_fingerprints,
    load_manifest,
    policy_sha256,
    scoring_module_sha256,
    validate_retriever_settings,
)
from .models import MemoryQuestion, MemoryRecord
from .preregistration import validate_registration
from .retrievers import ExemplarCAMRetriever, FAMRetriever, TensorLike
from .runner import FiveArmRunner
from .scoring import SCORING_VERSION


@dataclass(frozen=True, slots=True)
class Check:
    gate: str
    name: str
    passed: bool
    detail: str


class PreflightFailed(RuntimeError):
    """Raised before runner construction and therefore before generation."""

    def __init__(self, checks: Sequence[Check]) -> None:
        self.checks = tuple(checks)
        self.failures = tuple(check for check in self.checks if not check.passed)
        lines = "\n  - ".join(
            f"[{check.gate}] {check.name}: {check.detail}"
            for check in self.failures
        )
        super().__init__(f"pre-execution verification BLOCKED:\n  - {lines}")


def preflight(
    manifest_path: str | Path,
    *,
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    policy: Mapping[str, Any],
    consumer: Any,
) -> tuple[Check, ...]:
    """Rebuild the sealed treatment and report every confirmatory gate.

    A plumbing seal is readable, but deliberately fails the evidence-class,
    registration, memo, and index-attestation gates. It can never yield a
    confirmatory mechanism or application verdict through this entry point.
    """
    checks: list[Check] = []
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001 - the report preserves the cause
        return (
            Check("G-I2", "manifest integrity", False, _exception_detail(exc)),
        )

    protocol_value = manifest.get("protocol")
    protocol = protocol_value if isinstance(protocol_value, Mapping) else {}

    checks.append(
        Check(
            "G-I2",
            "manifest version",
            manifest.get("manifest_version") == MANIFEST_VERSION,
            f"sealed {manifest.get('manifest_version')!r}, expected {MANIFEST_VERSION!r}",
        )
    )

    sealed_inputs = {
        key: value
        for key, value in manifest.get("fingerprints", {}).items()
        if key != "protocol"
    }
    try:
        live_inputs = input_fingerprints(
            records, questions, record_embeddings, query_embeddings
        )
        drifted = sorted(
            key for key, value in live_inputs.items() if sealed_inputs.get(key) != value
        )
        checks.append(
            Check(
                "G-I2",
                "input integrity",
                not drifted,
                "records, questions and embeddings match the seal"
                if not drifted
                else f"drifted: {', '.join(drifted)}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("G-I2", "input integrity", False, _exception_detail(exc)))

    sealed_settings_value = protocol.get("retriever_settings")
    sealed_settings = (
        sealed_settings_value if isinstance(sealed_settings_value, Mapping) else {}
    )
    settings_errors: list[str] = []
    try:
        validate_retriever_settings(
            sealed_settings, record_count=manifest.get("record_count")
        )
    except Exception as exc:  # noqa: BLE001
        settings_errors.append(f"sealed schema: {_exception_detail(exc)}")
    try:
        validate_retriever_settings(retriever_settings, record_count=len(records))
    except Exception as exc:  # noqa: BLE001
        settings_errors.append(f"live schema: {_exception_detail(exc)}")
    if dict(sealed_settings) != dict(retriever_settings):
        settings_errors.append(
            f"sealed {dict(sealed_settings)!r} vs live {dict(retriever_settings)!r}"
        )
    checks.append(
        Check(
            "G-I2",
            "treatment settings",
            not settings_errors,
            "complete closed schema matches the seal"
            if not settings_errors
            else "; ".join(settings_errors),
        )
    )

    evidence_class = protocol.get("evidence_class")
    checks.append(
        Check(
            "G-I2",
            "evidence class",
            evidence_class == "scoring-run",
            f"sealed as {evidence_class!r}; only 'scoring-run' is confirmatory",
        )
    )

    registration_value = protocol.get("registration")
    registration = (
        registration_value if isinstance(registration_value, Mapping) else None
    )
    if registration is None:
        checks.append(
            Check("G-I1", "registration complete", False, "no registration block sealed")
        )
    else:
        registration_errors = validate_registration(registration)
        checks.append(
            Check(
                "G-I1",
                "registration complete",
                not registration_errors,
                "clean" if not registration_errors else "; ".join(registration_errors),
            )
        )

    memo_path = protocol.get("registration_memo_path")
    memo_error: str | None = None
    actual_memo_digest: str | None = None
    if evidence_class != "scoring-run":
        memo_error = "plumbing seals carry no registered memo"
    elif not isinstance(memo_path, str) or not memo_path:
        memo_error = "no registration memo path sealed"
    else:
        try:
            actual_memo_digest = sha256(Path(memo_path).read_bytes()).hexdigest()
        except OSError as exc:
            memo_error = f"registered memo is unreadable: {exc}"
    registered_memo_digest = (
        registration.get("memo_sha256") if registration is not None else None
    )
    if memo_error is None and registered_memo_digest != actual_memo_digest:
        memo_error = (
            f"registered {_short(registered_memo_digest)} vs "
            f"actual {_short(actual_memo_digest)}"
        )
    checks.append(
        Check(
            "G-I1",
            "registration memo",
            memo_error is None,
            f"memo bytes match {_short(actual_memo_digest)}"
            if memo_error is None
            else memo_error,
        )
    )

    sealed_policy = protocol.get("policy_sha256")
    try:
        live_policy = policy_sha256(policy)
        policy_matches = sealed_policy == live_policy
        policy_detail = (
            f"sealed {_short(sealed_policy)} vs live {_short(live_policy)}"
        )
    except Exception as exc:  # noqa: BLE001
        policy_matches = False
        policy_detail = _exception_detail(exc)
    checks.append(Check("G-I2", "policy binding", policy_matches, policy_detail))

    sealed_pin = protocol.get("consumer_pin")
    live_pin = getattr(consumer, "pin_id", None)
    checks.append(
        Check(
            "G-I2",
            "consumer pin",
            bool(sealed_pin) and sealed_pin == live_pin,
            f"sealed {sealed_pin!r} vs live {live_pin!r}",
        )
    )

    sealed_scorer = protocol.get("scoring_module_sha256")
    live_scorer = scoring_module_sha256()
    checks.append(
        Check(
            "G-I2",
            "scorer identity",
            sealed_scorer == live_scorer,
            f"sealed {_short(sealed_scorer)} vs live {_short(live_scorer)} "
            f"(version {protocol.get('scoring_version')!r} / {SCORING_VERSION!r})",
        )
    )

    exemplar: ExemplarCAMRetriever | None = None
    fam: FAMRetriever | None = None
    rebuild_error: str | None = None
    try:
        exemplar, fam = build_cam_indexes(
            records, record_embeddings, sealed_settings
        )
    except Exception as exc:  # noqa: BLE001
        rebuild_error = _exception_detail(exc)

    attestations_value = protocol.get("index_attestations")
    attestations = (
        attestations_value if isinstance(attestations_value, Mapping) else {}
    )
    checks.append(
        _attestation_check(
            "exemplar index",
            attestations.get("exemplar"),
            exemplar,
            rebuild_error,
        )
    )
    checks.append(
        _attestation_check(
            "FAM index",
            attestations.get("fam"),
            fam,
            rebuild_error,
        )
    )

    if fam is None:
        active = False
        activity_detail = rebuild_error or "FAM index was not rebuilt"
    else:
        attestation = fam.attestation
        active = attestation.merged > 0 and attestation.key_drifted_merges > 0
        activity_detail = (
            f"merged={attestation.merged}, "
            f"key_drifted_merges={attestation.key_drifted_merges}"
        )
    checks.append(
        Check("G-M0", "mechanism activity", active, activity_detail)
    )

    integrity = False
    integrity_detail = rebuild_error or "indexes were not rebuilt"
    if exemplar is not None and fam is not None:
        e0 = exemplar.attestation
        f0 = fam.attestation
        integrity = (
            sealed_settings.get("cam_max_entries") == len(records)
            and e0.written == len(records)
            and f0.written == len(records)
            and e0.allocated == len(records)
            and e0.prototype_count == len(records)
            and e0.dropped == e0.evicted == f0.dropped == f0.evicted == 0
        )
        integrity_detail = (
            f"capacity={sealed_settings.get('cam_max_entries')!r}, "
            f"records={len(records)}, E0 prototypes={e0.prototype_count}, "
            f"drops/evictions={e0.dropped}/{e0.evicted}/{f0.dropped}/{f0.evicted}"
        )
    checks.append(
        Check(
            "G-I2",
            "provenance and capacity integrity",
            integrity,
            integrity_detail,
        )
    )
    return tuple(checks)


def build_sealed_run(
    manifest_path: str | Path,
    *,
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    policy: Mapping[str, Any],
    consumer: Any,
) -> FiveArmRunner:
    """Build both live CAM arms exclusively from a passing sealed protocol."""
    checks = preflight(
        manifest_path,
        records=records,
        questions=questions,
        record_embeddings=record_embeddings,
        query_embeddings=query_embeddings,
        retriever_settings=retriever_settings,
        policy=policy,
        consumer=consumer,
    )
    if any(not check.passed for check in checks):
        raise PreflightFailed(checks)

    protocol = load_manifest(manifest_path)["protocol"]
    sealed_settings = protocol["retriever_settings"]
    exemplar, fam = build_cam_indexes(records, record_embeddings, sealed_settings)
    return FiveArmRunner(
        ledger=MemoryLedger(records),
        exemplar_retriever=exemplar,
        fam_retriever=fam,
        consumer=consumer,
        candidate_k=sealed_settings["candidate_k"],
        policy=dict(policy),
    )


def _attestation_check(
    name: str,
    sealed: Any,
    rebuilt: ExemplarCAMRetriever | FAMRetriever | None,
    rebuild_error: str | None,
) -> Check:
    if rebuilt is None:
        return Check("G-I2", name, False, rebuild_error or "index was not rebuilt")
    actual = asdict(rebuilt.attestation)
    if not isinstance(sealed, Mapping):
        return Check("G-I2", name, False, "no confirmatory attestation sealed")
    differing = sorted(
        key
        for key in set(sealed) | set(actual)
        if sealed.get(key) != actual.get(key)
    )
    return Check(
        "G-I2",
        name,
        not differing,
        "every attestation field matches"
        if not differing
        else f"differing fields: {', '.join(differing)}",
    )


def _short(digest: Any) -> str:
    return "<none>" if not isinstance(digest, str) else digest[:12]


def _exception_detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
