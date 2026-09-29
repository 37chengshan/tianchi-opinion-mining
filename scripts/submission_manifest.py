#!/usr/bin/env python3
"""Create a local-only manifest and upload queue for validated submissions.

The script deliberately has no browser, network, credential, or submission
integration.  It records an already validated CSV as a reproducible local
candidate and can later record a leaderboard value that a human or another
authorized process has supplied locally.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping


UTF8_BOM = b"\xef\xbb\xbf"
MANIFEST_NAME = "manifest.csv"
QUEUE_NAME = "queue.json"
MANIFEST_FIELDS = (
    "candidate_id",
    "source_csv",
    "candidate_csv",
    "candidate_copy",
    "encoding",
    "bom",
    "utf8_no_bom",
    "rows",
    "bytes",
    "sha256",
    "test_ids",
    "test_id_count",
    "test_ids_sha256",
    "oof_json",
    "config_json",
    "config_hash",
    "oof_sha256",
    "config_sha256",
    "local_score",
    "local_score_metric",
    "local_score_source",
    "leaderboard_score",
    "status",
    "created_at_utc",
    "updated_at_utc",
)

# The first version of this local queue had a header and did not include the
# test-id/evidence fields.  It is accepted on read so an existing queue can be
# rewritten into the current headerless format without losing scored status.
LEGACY_MANIFEST_FIELDS = (
    "candidate_id", "source_csv", "candidate_csv", "candidate_copy",
    "encoding", "bom", "utf8_no_bom", "rows", "bytes", "sha256",
    "oof_json", "config_json", "config_hash", "local_score",
    "local_score_metric", "local_score_source", "leaderboard_score",
    "status", "created_at_utc", "updated_at_utc",
)


class ManifestError(ValueError):
    """Raised when a local manifest input is incomplete or inconsistent."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a file without transforming its bytes."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_file(path: str | Path, label: str) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ManifestError(f"{label} does not exist or is not a file: {file_path}")
    return file_path


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number is not allowed: {value}")


def _read_json(path: str | Path, label: str) -> Any:
    file_path = _absolute_file(path, label)
    try:
        # Accept an input JSON BOM for convenience.  Generated JSON is always
        # written without a BOM; the CSV input has a stricter check below.
        text = file_path.read_bytes().decode("utf-8-sig")
        return json.loads(text, parse_constant=_reject_json_constant)
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{label} is not valid UTF-8: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{label} is not valid JSON: {file_path}: {exc}") from exc
    except ValueError as exc:
        raise ManifestError(f"{label} contains an unsupported JSON value: {file_path}: {exc}") from exc


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically for a semantic config hash."""

    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"JSON payload cannot be canonicalized: {exc}") from exc
    return text.encode("utf-8")


def config_hash(payload: Any) -> str:
    """Return the SHA256 hash of canonical JSON rather than formatting bytes."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _inspect_csv(path: Path, expected_test_ids: Iterable[int] | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw.startswith(UTF8_BOM):
        raise ManifestError(f"CSV must be UTF-8 without a BOM: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"CSV is not valid UTF-8: {path}") from exc

    rows = 0
    test_ids: set[int] = set()
    for line_no, row in enumerate(csv.reader(io.StringIO(text, newline="")), start=1):
        if len(row) != 5:
            raise ManifestError(f"CSV line {line_no} must contain five fields")
        try:
            test_ids.add(int(row[0]))
        except ValueError as exc:
            raise ManifestError(f"CSV line {line_no} has an invalid test id: {row[0]!r}") from exc
        rows += 1
    if not test_ids:
        raise ManifestError(f"CSV contains no test IDs: {path}")
    expected = None if expected_test_ids is None else {int(value) for value in expected_test_ids}
    if expected is not None and test_ids != expected:
        missing = sorted(expected - test_ids)
        extra = sorted(test_ids - expected)
        raise ManifestError(f"CSV test-id coverage mismatch; missing={missing[:5]} extra={extra[:5]}")
    ids_text = ",".join(str(value) for value in sorted(test_ids))
    return {
        "rows": rows,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "test_ids": ids_text,
        "test_id_count": len(test_ids),
        "test_ids_sha256": hashlib.sha256(ids_text.encode("ascii")).hexdigest(),
    }


def _as_finite_score(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ManifestError(f"{label} must be a finite number")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{label} must be a finite number") from exc
    if not math.isfinite(score):
        raise ManifestError(f"{label} must be a finite number")
    return score


def _score_from_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("f1", "local_score", "value"):
            if key in value:
                try:
                    return _as_finite_score(value[key], f"score field {key}")
                except ManifestError:
                    continue
        return None
    if value is None or isinstance(value, bool):
        return None
    try:
        return _as_finite_score(value, "local score")
    except ManifestError:
        return None


def _find_score(payload: Any) -> tuple[float, str] | None:
    """Find a named local score without scanning arbitrary candidate values."""

    if not isinstance(payload, Mapping):
        return None

    for key in ("local_score", "f1"):
        if key in payload:
            value = _score_from_value(payload[key])
            if value is not None:
                return value, key

    for container_key in (
        "score",
        "metrics",
        "evaluation",
        "oof_score",
        "local",
        "full_oof_after_shrink",
        "lofo_oof",
    ):
        if container_key not in payload:
            continue
        container = payload[container_key]
        if isinstance(container, Mapping) and "f1" in container:
            value = _score_from_value(container["f1"])
            if value is not None:
                return value, f"{container_key}.f1"
        value = _score_from_value(container)
        if value is not None:
            return value, container_key
    return None


def extract_local_score(payload: Any) -> float | None:
    """Extract an F1-like local score from a known OOF/report JSON shape."""

    found = _find_score(payload)
    return found[0] if found else None


def _resolve_local_score(
    oof_payload: Any,
    config_payload: Any,
    explicit_score: Any | None,
) -> tuple[float, str]:
    if explicit_score is not None:
        return _as_finite_score(explicit_score, "--local-score"), "--local-score"

    for label, payload in (("oof_json", oof_payload), ("config_json", config_payload)):
        found = _find_score(payload)
        if found is not None:
            value, location = found
            return value, f"{label}:{location}"
    raise ManifestError(
        "no local score found in OOF/config JSON; provide --local-score "
        "instead of leaving the manifest score unspecified"
    )


def _score_text(score: float) -> str:
    return format(float(score), ".17g")


def _candidate_id_is_safe(candidate_id: str) -> bool:
    return bool(candidate_id) and candidate_id not in {".", ".."} and not any(char in candidate_id for char in "/\\")


def _copy_exact(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_sha256:
            raise ManifestError(f"candidate copy already exists with different bytes: {destination}")
        return

    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise ManifestError(f"candidate copy changed while being copied: {source}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw.startswith(UTF8_BOM):
        raise ManifestError(f"manifest must be UTF-8 without a BOM: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"manifest is not valid UTF-8: {path}") from exc
    if not text.strip():
        return []

    reader = csv.reader(io.StringIO(text, newline=""))
    raw_rows = list(reader)
    if not raw_rows:
        return []
    first = raw_rows[0]
    if first and first[0] == "candidate_id":
        fieldnames = tuple(first)
        required = {"candidate_id", "sha256", "status"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ManifestError(f"manifest is missing required columns {missing}: {path}")
        data_rows = raw_rows[1:]
        dict_rows = [dict(zip(fieldnames, row)) for row in data_rows]
    else:
        data_rows = raw_rows
        dict_rows = []
        for index, row in enumerate(data_rows, start=1):
            if len(row) == len(MANIFEST_FIELDS):
                names = MANIFEST_FIELDS
            elif len(row) == len(LEGACY_MANIFEST_FIELDS):
                names = LEGACY_MANIFEST_FIELDS
            else:
                raise ManifestError(f"manifest row {index} has {len(row)} fields; expected {len(MANIFEST_FIELDS)}")
            dict_rows.append(dict(zip(names, row)))
    return [{field: str(row.get(field) or "") for field in MANIFEST_FIELDS} for row in dict_rows]


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            for row in rows:
                writer.writerow([row.get(field, "") for field in MANIFEST_FIELDS])
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_queue(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "queue": []}
    payload = _read_json(path, "queue JSON")
    if isinstance(payload, list):
        payload = {"schema_version": 1, "queue": payload}
    if not isinstance(payload, Mapping) or not isinstance(payload.get("queue"), list):
        raise ManifestError(f"queue JSON must contain a top-level queue list: {path}")
    queue = []
    for index, item in enumerate(payload["queue"]):
        if not isinstance(item, Mapping) or not item.get("candidate_id"):
            raise ManifestError(f"queue item {index} has no candidate_id: {path}")
        queue.append(dict(item))
    output = dict(payload)
    output["schema_version"] = 1
    output["queue"] = queue
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _merge_existing(new: dict[str, Any], old: Mapping[str, Any] | None) -> dict[str, Any]:
    if old is None:
        return new
    merged = dict(new)
    if old.get("created_at_utc"):
        merged["created_at_utc"] = old["created_at_utc"]
    old_status = str(old.get("status") or "")
    if old_status and old_status != "pending":
        merged["status"] = old_status
        if old.get("leaderboard_score") not in (None, ""):
            merged["leaderboard_score"] = old["leaderboard_score"]
        if old.get("updated_at_utc"):
            merged["updated_at_utc"] = old["updated_at_utc"]
    return merged


def _upsert(items: list[dict[str, Any]], item: dict[str, Any], key: str = "candidate_id") -> dict[str, Any]:
    matches = [index for index, existing in enumerate(items) if existing.get(key) == item.get(key)]
    if len(matches) > 1:
        raise ManifestError(f"duplicate {key} in local records: {item.get(key)}")
    if not matches:
        items.append(item)
        return item
    merged = _merge_existing(item, items[matches[0]])
    items[matches[0]] = merged
    return merged


def _queue_item(record: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    return {
        "candidate_id": record["candidate_id"],
        "source_csv": record["source_csv"],
        "candidate_csv": record["candidate_csv"],
        "upload_path": record["candidate_csv"],
        "candidate_copy": record["candidate_copy"] == "true",
        "encoding": record["encoding"],
        "bom": record["bom"] == "true",
        "rows": int(record["rows"]),
        "bytes": int(record["bytes"]),
        "sha256": record["sha256"],
        "test_ids": record["test_ids"],
        "test_id_count": int(record["test_id_count"]),
        "test_ids_sha256": record["test_ids_sha256"],
        "oof_json": record["oof_json"],
        "config_json": record["config_json"],
        "config_hash": record["config_hash"],
        "oof_sha256": record["oof_sha256"],
        "config_sha256": record["config_sha256"],
        "local_score": float(record["local_score"]),
        "local_score_metric": record["local_score_metric"],
        "local_score_source": record["local_score_source"],
        "leaderboard_score": None if record["leaderboard_score"] == "" else float(record["leaderboard_score"]),
        "status": record["status"],
        "manifest_path": str(manifest_path),
        "created_at_utc": record["created_at_utc"],
        "updated_at_utc": record["updated_at_utc"] or None,
    }


def create_candidate_manifest(
    csv_path: str | Path,
    oof_json: str | Path,
    config_json: str | Path,
    output_dir: str | Path,
    *,
    candidate_dir: str | Path | None = None,
    candidate_id: str | None = None,
    local_score: float | str | None = None,
    expected_test_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Create or update one local candidate, manifest row, and queue item.

    ``candidate_dir`` is optional.  When omitted, ``output_dir/candidates`` is
    used only if that directory already exists.  The input CSV is never
    rewritten; a candidate copy is byte-for-byte and receives the same hash.
    """

    source = _absolute_file(csv_path, "CSV")
    oof_path = _absolute_file(oof_json, "OOF JSON")
    config_path = _absolute_file(config_json, "config JSON")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    oof_payload = _read_json(oof_path, "OOF JSON")
    config_payload = _read_json(config_path, "config JSON")
    score, score_source = _resolve_local_score(oof_payload, config_payload, local_score)
    facts = _inspect_csv(source, expected_test_ids)
    cfg_hash = config_hash(config_payload)
    oof_sha = sha256_file(oof_path)
    config_sha = sha256_file(config_path)

    resolved_id = candidate_id or f"candidate-{facts['sha256'][:12]}-{cfg_hash[:12]}"
    if not _candidate_id_is_safe(resolved_id):
        raise ManifestError(f"candidate_id must be a simple path-safe name: {resolved_id!r}")

    if candidate_dir is None:
        automatic_dir = output / "candidates"
        copy_root = automatic_dir if automatic_dir.is_dir() else None
    else:
        copy_root = Path(candidate_dir).expanduser().resolve()
        copy_root.mkdir(parents=True, exist_ok=True)

    upload_path = source
    copied = False
    if copy_root is not None:
        destination = copy_root / resolved_id / source.name
        _copy_exact(source, destination, facts["sha256"])
        if sha256_file(destination) != facts["sha256"]:
            raise ManifestError(f"candidate copy hash mismatch: {destination}")
        upload_path = destination.resolve()
        copied = True

    created_at = _now_utc()
    record: dict[str, Any] = {
        "candidate_id": resolved_id,
        "source_csv": str(source),
        "candidate_csv": str(upload_path),
        "candidate_copy": "true" if copied else "false",
        "encoding": "UTF-8",
        "bom": "false",
        "utf8_no_bom": "true",
        "rows": str(facts["rows"]),
        "bytes": str(facts["bytes"]),
        "sha256": facts["sha256"],
        "test_ids": facts["test_ids"],
        "test_id_count": str(facts["test_id_count"]),
        "test_ids_sha256": facts["test_ids_sha256"],
        "oof_json": str(oof_path),
        "config_json": str(config_path),
        "config_hash": cfg_hash,
        "oof_sha256": oof_sha,
        "config_sha256": config_sha,
        "local_score": _score_text(score),
        "local_score_metric": "f1",
        "local_score_source": score_source,
        "leaderboard_score": "",
        "status": "pending",
        "created_at_utc": created_at,
        "updated_at_utc": "",
    }

    manifest_path = output / MANIFEST_NAME
    manifest_rows = _read_manifest(manifest_path)
    record = _upsert(manifest_rows, record)
    _write_manifest(manifest_path, manifest_rows)

    queue_path = output / QUEUE_NAME
    queue_payload = _read_queue(queue_path)
    queue_item = _queue_item(record, manifest_path)
    queue_item = _upsert(queue_payload["queue"], queue_item)
    _write_json(queue_path, queue_payload)

    return {
        "candidate_id": resolved_id,
        "manifest_path": str(manifest_path),
        "queue_path": str(queue_path),
        "candidate_csv": str(upload_path),
        "record": record,
        "queue_item": queue_item,
    }


def backfill_leaderboard(
    manifest_path: str | Path,
    queue_path: str | Path,
    candidate_id: str,
    leaderboard_score: float | str,
) -> dict[str, Any]:
    """Record a manually supplied leaderboard score in local files only.

    This function performs no upload and no network operation.  A score is
    written only when the caller explicitly invokes it with a finite numeric
    value; the candidate status then becomes ``scored``.
    """

    score = _as_finite_score(leaderboard_score, "leaderboard score")
    manifest = Path(manifest_path).expanduser().resolve()
    queue_file = Path(queue_path).expanduser().resolve()
    rows = _read_manifest(manifest)
    queue_payload = _read_queue(queue_file)

    manifest_matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    queue_matches = [item for item in queue_payload["queue"] if item.get("candidate_id") == candidate_id]
    if len(manifest_matches) != 1:
        raise ManifestError(f"candidate_id not found exactly once in manifest: {candidate_id}")
    if len(queue_matches) != 1:
        raise ManifestError(f"candidate_id not found exactly once in queue: {candidate_id}")

    manifest_row = manifest_matches[0]
    queue_item = queue_matches[0]
    if queue_item.get("sha256") != manifest_row.get("sha256"):
        raise ManifestError(f"manifest/queue SHA256 mismatch for candidate: {candidate_id}")

    updated_at = _now_utc()
    manifest_row["leaderboard_score"] = _score_text(score)
    manifest_row["status"] = "scored"
    manifest_row["updated_at_utc"] = updated_at
    queue_item["leaderboard_score"] = score
    queue_item["status"] = "scored"
    queue_item["updated_at_utc"] = updated_at

    _write_manifest(manifest, rows)
    _write_json(queue_file, queue_payload)
    return {
        "candidate_id": candidate_id,
        "leaderboard_score": score,
        "status": "scored",
        "manifest_path": str(manifest),
        "queue_path": str(queue_file),
    }


# A descriptive alias for callers that prefer the explicit function name.
backfill_leaderboard_score = backfill_leaderboard


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="submission_manifest.py",
        description="Create a local-only candidate manifest and pending upload queue.",
    )
    parser.add_argument("csv_path", nargs="?", help="already validator-approved submission CSV")
    parser.add_argument("oof_json", nargs="?", help="local OOF or OOF report JSON")
    parser.add_argument("config_json", nargs="?", help="local model/threshold config JSON")
    parser.add_argument("--csv", dest="csv_option", help="alias for the CSV positional argument")
    parser.add_argument("--oof", "--oof-json", dest="oof_option", help="alias for the OOF positional argument")
    parser.add_argument("--config", "--config-json", dest="config_option", help="alias for the config positional argument")
    parser.add_argument("--output-dir", default=None, help="directory receiving manifest.csv and queue.json")
    parser.add_argument(
        "--candidate-dir",
        default=None,
        help="optional directory for a byte-for-byte candidate copy; otherwise use output-dir/candidates only if it exists",
    )
    parser.add_argument("--candidate-id", default=None, help="optional path-safe candidate name")
    parser.add_argument("--local-score", default=None, help="explicit local OOF score when JSON has no recognized score field")
    return parser


def _backfill_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="submission_manifest.py backfill",
        description="Record a manually supplied leaderboard score in local manifest files.",
    )
    parser.add_argument("--manifest", required=True, help="path to manifest.csv")
    parser.add_argument("--queue", required=True, help="path to queue.json")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--leaderboard-score", required=True, help="finite score copied from a verified leaderboard result")
    return parser


def _run_create(argv: list[str]) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)
    csv_path = args.csv_option or args.csv_path
    oof_json = args.oof_option or args.oof_json
    config_json = args.config_option or args.config_json
    if not csv_path or not oof_json or not config_json:
        parser.error("CSV, OOF JSON, and config JSON are required")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(csv_path).expanduser().resolve().parent / "online_loop")
    result = create_candidate_manifest(
        csv_path,
        oof_json,
        config_json,
        output_dir,
        candidate_dir=args.candidate_dir,
        candidate_id=args.candidate_id,
        local_score=args.local_score,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _run_backfill(argv: list[str]) -> int:
    parser = _backfill_parser()
    args = parser.parse_args(argv)
    result = backfill_leaderboard(
        args.manifest,
        args.queue,
        args.candidate_id,
        args.leaderboard_score,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] in {"backfill", "backfill-score"}:
            return _run_backfill(arguments[1:])
        if arguments and arguments[0] in {"create", "manifest"}:
            return _run_create(arguments[1:])
        return _run_create(arguments)
    except (ManifestError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
