"""Deterministic persistence, protocol validation, and lock protection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel

from app.benchmark.fingerprints import canonical_sha256
from app.dataset.models import (
    HANDBOOK_VERSION,
    ONTOLOGY_VERSION,
    AdjudicationArtifact,
    DatasetAnnotation,
    IntakeRegistry,
    LockOverrideRecord,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_json_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"JSON document not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        return model_type.model_validate(json.load(stream))


def write_json_model(document: BaseModel, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(
            document.model_dump(mode="json"),
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        stream.write("\n")
    return destination


def load_annotation(path: str | Path) -> DatasetAnnotation:
    document = read_json_model(path, DatasetAnnotation)
    validate_annotation_protocol(document)
    if document.locked and document.annotation_hash != annotation_content_hash(
        document
    ):
        raise ValueError("LOCKED_ANNOTATION_HASH_MISMATCH")
    return document


def load_registry(path: str | Path) -> IntakeRegistry:
    return read_json_model(path, IntakeRegistry)


def load_adjudication(path: str | Path) -> AdjudicationArtifact:
    artifact = read_json_model(path, AdjudicationArtifact)
    if artifact.ontology_version != ONTOLOGY_VERSION:
        raise ValueError(
            "ONTOLOGY_VERSION_MISMATCH: adjudication uses "
            f"{artifact.ontology_version!r}; expected {ONTOLOGY_VERSION!r}"
        )
    if artifact.handbook_version != HANDBOOK_VERSION:
        raise ValueError(
            "HANDBOOK_VERSION_MISMATCH: adjudication uses "
            f"{artifact.handbook_version!r}; expected {HANDBOOK_VERSION!r}"
        )
    if document_sha256(artifact.annotation_a) != artifact.annotation_a_hash:
        raise ValueError("ADJUDICATION_ANNOTATION_A_HASH_MISMATCH")
    if document_sha256(artifact.annotation_b) != artifact.annotation_b_hash:
        raise ValueError("ADJUDICATION_ANNOTATION_B_HASH_MISMATCH")
    if artifact.locked:
        digest = canonical_sha256(
            artifact.model_dump(mode="json", exclude={"adjudication_hash"})
        )
        if artifact.adjudication_hash != digest:
            raise ValueError("LOCKED_ADJUDICATION_HASH_MISMATCH")
    return artifact


def validate_annotation_protocol(document: DatasetAnnotation) -> None:
    if document.ontology_version != ONTOLOGY_VERSION:
        raise ValueError(
            "ONTOLOGY_VERSION_MISMATCH: annotation uses "
            f"{document.ontology_version!r}; expected {ONTOLOGY_VERSION!r}. "
            "Migrate or re-annotate before mixing protocols."
        )
    if document.handbook_version != HANDBOOK_VERSION:
        raise ValueError(
            "HANDBOOK_VERSION_MISMATCH: annotation uses "
            f"{document.handbook_version!r}; expected {HANDBOOK_VERSION!r}. "
            "Migrate or re-annotate before mixing protocols."
        )


def annotation_content_hash(document: DatasetAnnotation) -> str:
    return canonical_sha256(
        document.model_dump(mode="json", exclude={"annotation_hash"})
    )


def lock_annotation(
    document: DatasetAnnotation,
    *,
    locked_at: datetime | None = None,
) -> DatasetAnnotation:
    timestamp = locked_at or datetime.now(timezone.utc)
    candidate = document.model_copy(
        update={"locked": True, "locked_at": timestamp, "annotation_hash": None}
    )
    return candidate.model_copy(
        update={"annotation_hash": annotation_content_hash(candidate)}
    )


def save_annotation(
    document: DatasetAnnotation,
    path: str | Path,
    *,
    override_lock: bool = False,
    override_reason: str | None = None,
    timestamp: datetime | None = None,
) -> DatasetAnnotation:
    validate_annotation_protocol(document)
    destination = Path(path)
    saved = document
    if destination.is_file():
        existing = load_annotation(destination)
        changed = existing.model_dump(mode="json") != document.model_dump(mode="json")
        if existing.locked and changed:
            if not override_lock:
                raise PermissionError(
                    "annotation is locked; use an explicit audited override"
                )
            if override_reason is None or len(override_reason.strip()) < 3:
                raise ValueError("locked annotation override requires a reason")
            action: Literal["override_edit", "unlock", "relock"] = (
                "override_edit" if document.locked else "unlock"
            )
            record = LockOverrideRecord(
                timestamp=timestamp or datetime.now(timezone.utc),
                action=action,
                reason=override_reason.strip(),
            )
            saved = document.model_copy(
                update={
                    "lock_override_history": [
                        *document.lock_override_history,
                        record,
                    ],
                    "annotation_hash": None,
                }
            )
            if saved.locked:
                saved = lock_annotation(saved, locked_at=saved.locked_at)
    if saved.locked and saved.annotation_hash != annotation_content_hash(saved):
        saved = lock_annotation(saved, locked_at=saved.locked_at)
    write_json_model(saved, destination)
    return saved


def document_sha256(document: BaseModel) -> str:
    return canonical_sha256(document.model_dump(mode="json"))
