"""Phase-A audit preflight and explicitly non-confirmatory plumbing runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from .ledger import MemoryLedger
from .manifest import (
    MANIFEST_VERSION,
    PHASE_B_SCORING_REFUSAL,
    build_cam_indexes,
    load_manifest,
    policy_sha256,
    scoring_module_sha256,
    verify_manifest,
)
from .models import MemoryQuestion, MemoryRecord
from .preregistration import PreflightReceipt, validate_registration
from .retrievers import (
    ExemplarCAMRetriever,
    FAMRetriever,
    IndexBuildAttestation,
    TensorLike,
)
from .runner import FiveArmRunner
from .scoring import SCORING_VERSION


@dataclass(frozen=True, slots=True)
class Check:
    gate: str
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RebuiltIndexAttestations:
    exemplar: IndexBuildAttestation
    fam: IndexBuildAttestation


@dataclass(frozen=True, slots=True)
class PlumbingRunBundle:
    """Verified but permanently non-confirmatory execution plumbing."""

    manifest_sha256: str
    evidence_class: Literal["plumbing"]
    admissible: Literal[False]
    receipt: PreflightReceipt
    rebuilt_attestations: RebuiltIndexAttestations
    runner: FiveArmRunner


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


def build_plumbing_run(
    manifest_path: str | Path,
    *,
    records: Sequence[MemoryRecord],
    questions: Sequence[MemoryQuestion],
    record_embeddings: Mapping[str, TensorLike],
    query_embeddings: Mapping[str, TensorLike],
    retriever_settings: Mapping[str, Any],
    policy: Mapping[str, Any],
    consumer: Any,
) -> PlumbingRunBundle:
    """Rebuild a verified manifest-v3 plumbing bundle without evidence claims.

    Plumbing seals bind inputs and treatment but intentionally contain no
    registration or sealed index attestations. The bundled runner is only an
    execution fixture; its immutable receipt cannot authorize a confirmatory
    verdict.
    """
    manifest = load_manifest(manifest_path)
    protocol_value = manifest.get("protocol")
    if not isinstance(protocol_value, Mapping):
        raise RuntimeError("plumbing manifest has no protocol block")
    protocol = protocol_value
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise RuntimeError(
            "plumbing run requires "
            f"{MANIFEST_VERSION!r}, got {manifest.get('manifest_version')!r}"
        )
    if protocol.get("evidence_class") != "plumbing":
        raise RuntimeError("build_plumbing_run requires a plumbing evidence class")

    forbidden: list[str] = []
    if protocol.get("registration") is not None:
        forbidden.append("registration")
    for name in ("registration_memo_path", "index_attestations"):
        if name in protocol:
            forbidden.append(name)
    if forbidden:
        raise RuntimeError(
            "plumbing manifest must not carry confirmatory blocks: "
            + ", ".join(forbidden)
        )

    live_pin = getattr(consumer, "pin_id", None)
    if not isinstance(live_pin, str) or not live_pin:
        raise RuntimeError("plumbing consumer must expose a non-empty pin_id")
    verified = verify_manifest(
        manifest_path,
        records,
        questions,
        record_embeddings,
        query_embeddings,
        retriever_settings,
        evidence_class="plumbing",
        policy=policy,
        consumer_pin=live_pin,
    )
    sealed_settings = verified["protocol"]["retriever_settings"]
    exemplar, fam = build_cam_indexes(records, record_embeddings, sealed_settings)
    runner = FiveArmRunner(
        ledger=MemoryLedger(records),
        exemplar_retriever=exemplar,
        fam_retriever=fam,
        consumer=consumer,
        candidate_k=sealed_settings["candidate_k"],
        policy=dict(policy),
        # Bind run() to the VERIFIED query-side fingerprints: a bundle runner
        # refuses foreign questions/embeddings even under this passing digest
        # (the review executed both; see runner.run()).
        sealed_query_fingerprints={
            "questions": verified["fingerprints"]["questions"],
            "query_embeddings": verified["fingerprints"]["query_embeddings"],
        },
    )
    digest = verified["manifest_sha256"]
    receipt = PreflightReceipt(
        manifest_sha256=digest,
        evidence_class="plumbing",
        passed=True,
        confirmatory=False,
    )
    return PlumbingRunBundle(
        manifest_sha256=digest,
        evidence_class="plumbing",
        admissible=False,
        receipt=receipt,
        rebuilt_attestations=RebuiltIndexAttestations(
            exemplar=exemplar.attestation,
            fam=fam.attestation,
        ),
        runner=runner,
    )


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
    """Audit a manifest while keeping Phase-A confirmatory execution closed."""
    checks: list[Check] = []
    try:
        loaded = load_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001 - preserve exact audit cause
        return (Check("G-I2", "manifest integrity", False, _exception_detail(exc)),)

    raw_protocol_value = loaded.get("protocol")
    raw_protocol = (
        raw_protocol_value if isinstance(raw_protocol_value, Mapping) else {}
    )
    declared_evidence = raw_protocol.get("evidence_class")
    declared_registration_value = raw_protocol.get("registration")
    declared_registration = (
        declared_registration_value
        if isinstance(declared_registration_value, Mapping)
        else None
    )
    declared_memo = raw_protocol.get("registration_memo_path")
    live_pin = getattr(consumer, "pin_id", None)

    try:
        verified = verify_manifest(
            manifest_path,
            records,
            questions,
            record_embeddings,
            query_embeddings,
            retriever_settings,
            evidence_class=(
                declared_evidence if isinstance(declared_evidence, str) else ""
            ),
            policy=policy,
            consumer_pin=live_pin,
            registration=declared_registration,
            registration_memo_path=(
                declared_memo if isinstance(declared_memo, str) else None
            ),
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(
            Check("G-I2", "full manifest binding", False, _exception_detail(exc))
        )
        checks.append(
            Check(
                "G-I6",
                "Phase B provenance/reconciliation envelope",
                False,
                PHASE_B_SCORING_REFUSAL,
            )
        )
        return tuple(checks)

    checks.append(
        Check(
            "G-I2",
            "full manifest binding",
            True,
            "all envelope fields, counts, protocol identity, and fingerprints match",
        )
    )
    # No raw protocol field is used beyond this point.  Execution/audit values
    # come exclusively from the full verify result above.
    protocol = verified["protocol"]
    sealed_settings = protocol["retriever_settings"]
    evidence_class = protocol["evidence_class"]
    registration_value = protocol.get("registration")
    registration = (
        registration_value if isinstance(registration_value, Mapping) else None
    )

    checks.append(
        Check(
            "G-I2",
            "evidence class",
            evidence_class == "scoring-run",
            f"sealed as {evidence_class!r}; only 'scoring-run' is confirmatory",
        )
    )
    registration_errors = list(
        ("no registration block sealed",)
        if registration is None
        else validate_registration(registration)
    )
    if registration is not None:
        for field in ("candidate_k", "cam_prototype_k"):
            if registration.get(field) != sealed_settings.get(field):
                registration_errors.append(
                    f"{field} registration {registration.get(field)!r} does not "
                    f"match verified treatment {sealed_settings.get(field)!r}"
                )
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
    live_policy = policy_sha256(policy)
    checks.append(
        Check(
            "G-I2",
            "policy binding",
            sealed_policy == live_policy,
            f"sealed {_short(sealed_policy)} vs live {_short(live_policy)}",
        )
    )
    sealed_pin = protocol.get("consumer_pin")
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
        exemplar, fam = build_cam_indexes(records, record_embeddings, sealed_settings)
    except Exception as exc:  # noqa: BLE001
        rebuild_error = _exception_detail(exc)
    attestations_value = protocol.get("index_attestations")
    attestations = (
        attestations_value if isinstance(attestations_value, Mapping) else {}
    )
    checks.append(
        _attestation_check("exemplar index", attestations.get("exemplar"), exemplar, rebuild_error)
    )
    checks.append(
        _attestation_check("FAM index", attestations.get("fam"), fam, rebuild_error)
    )

    if fam is None:
        active = False
        activity_detail = rebuild_error or "FAM index was not rebuilt"
    else:
        active = fam.attestation.merged > 0 and fam.attestation.key_drifted_merges > 0
        activity_detail = (
            f"merged={fam.attestation.merged}, "
            f"key_drifted_merges={fam.attestation.key_drifted_merges}"
        )
    checks.append(Check("G-M0", "mechanism activity", active, activity_detail))

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
        Check("G-I2", "provenance and capacity integrity", integrity, integrity_detail)
    )
    checks.append(
        Check(
            "G-I6",
            "Phase B provenance/reconciliation envelope",
            False,
            PHASE_B_SCORING_REFUSAL,
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
    """Phase-A hard stop: no manifest can produce an admissible runner."""
    raise RuntimeError(PHASE_B_SCORING_REFUSAL)


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
