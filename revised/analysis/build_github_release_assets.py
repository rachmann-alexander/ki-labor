#!/usr/bin/env python3
"""Prepare reproducibly packaged, GitHub-compatible release assets for the paper data.

The default mode is read-only and validates the source layout.  ``--prepare``
writes only small manifests and checksums with an explicit PENDING marker.
``--build-assets`` creates and member-hash-verifies the seven ZIP files. It
requires a clean tagged Git checkout, its full commit SHA, and a tracked license
to prevent an untraceable or legally ambiguous release from being staged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISED_ROOT = PROJECT_ROOT / "revised"
RELEASE_ROOT = REVISED_ROOT / "github_release"
UPLOAD_ROOT = RELEASE_ROOT / "upload"

RELEASE_TAG = "electronics-revision-data-v1"
REPOSITORY_URL = "https://github.com/weissbeck-lucas/ki-labor"
PENDING = "PENDING_BUILD"


@dataclass(frozen=True)
class Campaign:
    asset_name: str
    source_dir: str
    role: str
    expected_files: int
    expected_bytes: int


CAMPAIGNS: tuple[Campaign, ...] = (
    Campaign(
        "electronics_campaign_fixed-order_2026-03-13.zip",
        "Ergebnisse_13.03.2026 (feste Reihenfolge)",
        "fixed-order campaign",
        231,
        143_630_169,
    ),
    Campaign(
        "electronics_campaign_random-order_2026-03-23.zip",
        "Ergebnisse_23.03.2026 (zufällige Reihenfolge)",
        "random-order campaign",
        225,
        171_598_483,
    ),
    Campaign(
        "electronics_campaign_masterjetson_2026-07-07.zip",
        "Ergebnisse_MasterJetson_07.07.2026",
        "MasterJetson small-batch audit campaign",
        291,
        274_377_330,
    ),
    Campaign(
        "electronics_campaign_clone1_2026-07-13.zip",
        "Ergebnisse_Clone1_13.07.2026",
        "primary physical-board block",
        507,
        446_481_868,
    ),
    Campaign(
        "electronics_campaign_clone2_2026-07-13.zip",
        "Ergebnisse_Clone2_13.07.2026",
        "primary physical-board block",
        507,
        431_573_652,
    ),
    Campaign(
        "electronics_campaign_clone3_2026-07-13.zip",
        "Ergebnisse_Clone3_13.07.2026",
        "primary physical-board block",
        507,
        412_776_337,
    ),
)

ANALYSIS_ASSET = "electronics_analysis_and_supplement.zip"
EXPECTED_SOURCE_FILES = 2_282
EXPECTED_CAMPAIGN_FILES = 2_268
EXPECTED_CAMPAIGN_BYTES = 1_880_437_839
EXPECTED_PLOTCODE_FILES = 14
EXPECTED_PLOTCODE_BYTES = 97_690
EXPECTED_TOTAL_SOURCE_BYTES = 1_880_535_529
EXPECTED_PYTHON = (3, 11, 9)

ANALYSIS_DIRECTORIES = (
    "analysis",
    "generated",
    "figures",
    "supplementary",
    "Definitions",
)
ANALYSIS_TOP_LEVEL_FILES = (
    "main.tex",
)
EXCLUDED_DIR_NAMES = {"__pycache__", ".git", ".agents", ".codex"}
EXCLUDED_SUFFIXES = {".aux", ".log", ".out", ".pyc", ".pyo", ".zip"}

ZIP_ASSET_NAMES = tuple(campaign.asset_name for campaign in CAMPAIGNS) + (ANALYSIS_ASSET,)
STAGING_METADATA_NAMES = {
    "README.md",
    "RELEASE_MANIFEST.json",
    "analysis_outputs_sha256.csv",
    "release_manifest.csv",
    "SHA256SUMS.txt",
}
FINAL_UPLOAD_NAMES = STAGING_METADATA_NAMES | set(ZIP_ASSET_NAMES) | {"LICENSE"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def source_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
    )


def filtered_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(directory).parts
        if any(part in EXCLUDED_DIR_NAMES for part in relative_parts[:-1]):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def byte_count(files: Iterable[Path]) -> int:
    return sum(path.stat().st_size for path in files)


def validate_source_layout(*, verify_hashes: bool) -> dict[str, str]:
    missing = [campaign.source_dir for campaign in CAMPAIGNS if not (PROJECT_ROOT / campaign.source_dir).is_dir()]
    if not (PROJECT_ROOT / "Plotcode").is_dir():
        missing.append("Plotcode")
    if missing:
        raise RuntimeError("Missing required source directories: " + ", ".join(missing))

    all_source_files: list[Path] = []
    total_files = 0
    total_bytes = 0
    for campaign in CAMPAIGNS:
        files = source_files(PROJECT_ROOT / campaign.source_dir)
        count = len(files)
        size = byte_count(files)
        if (count, size) != (campaign.expected_files, campaign.expected_bytes):
            raise RuntimeError(
                f"Source mismatch for {campaign.source_dir!r}: got {count} files / {size} bytes; "
                f"expected {campaign.expected_files} / {campaign.expected_bytes}."
            )
        total_files += count
        total_bytes += size
        all_source_files.extend(files)

    plotcode = source_files(PROJECT_ROOT / "Plotcode")
    plotcode_size = byte_count(plotcode)
    if (len(plotcode), plotcode_size) != (EXPECTED_PLOTCODE_FILES, EXPECTED_PLOTCODE_BYTES):
        raise RuntimeError(
            f"Plotcode mismatch: got {len(plotcode)} files / {plotcode_size} bytes; "
            f"expected {EXPECTED_PLOTCODE_FILES} / {EXPECTED_PLOTCODE_BYTES}."
        )
    total_files += len(plotcode)
    total_bytes += plotcode_size
    all_source_files.extend(plotcode)

    if total_files != EXPECTED_SOURCE_FILES or total_bytes != EXPECTED_TOTAL_SOURCE_BYTES:
        raise RuntimeError(
            f"Combined source mismatch: got {total_files} files / {total_bytes} bytes; "
            f"expected {EXPECTED_SOURCE_FILES} / {EXPECTED_TOTAL_SOURCE_BYTES}."
        )

    inventory = REVISED_ROOT / "supplementary" / "audit" / "file_inventory_sha256.csv"
    with inventory.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    inventory_bytes = sum(int(row["bytes"]) for row in rows)
    if len(rows) != EXPECTED_SOURCE_FILES or inventory_bytes != EXPECTED_TOTAL_SOURCE_BYTES:
        raise RuntimeError(
            f"Authoritative inventory mismatch: got {len(rows)} rows / {inventory_bytes} bytes."
        )

    actual_paths = {path.relative_to(PROJECT_ROOT).as_posix(): path for path in all_source_files}
    inventory_hashes: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        relative = row["path"]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
            raise RuntimeError(f"Unsafe path in source inventory: {relative!r}")
        if relative in inventory_hashes:
            raise RuntimeError(f"Duplicate path in source inventory: {relative!r}")
        path = actual_paths.get(relative)
        if path is None:
            raise RuntimeError(f"Inventory path is absent from the source tree: {relative!r}")
        expected_size = int(row["bytes"])
        if path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Inventory size mismatch for {relative!r}: got {path.stat().st_size}; expected {expected_size}."
            )
        expected_hash = row["sha256"].lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise RuntimeError(f"Invalid SHA-256 value in source inventory for {relative!r}.")
        inventory_hashes[relative] = expected_hash
        if verify_hashes:
            if index == 1:
                print("Verifying all 2,282 source-file SHA-256 values ...", flush=True)
            observed_hash = sha256(path)
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"Inventory SHA-256 mismatch for {relative!r}: got {observed_hash}; expected {expected_hash}."
                )
            if index % 250 == 0 or index == len(rows):
                print(f"  verified {index}/{len(rows)} files", flush=True)

    missing_from_inventory = sorted(set(actual_paths) - set(inventory_hashes))
    if missing_from_inventory:
        raise RuntimeError(
            "Source files absent from the authoritative inventory: " + ", ".join(missing_from_inventory[:5])
        )
    return inventory_hashes


def validate_upload_allowlist(*, final: bool) -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    entries = list(UPLOAD_ROOT.iterdir())
    directories = [entry.name for entry in entries if entry.is_dir()]
    if directories:
        raise RuntimeError("Unexpected directories in flat upload staging: " + ", ".join(sorted(directories)))
    names = {entry.name for entry in entries if entry.is_file()}
    unexpected = names - FINAL_UPLOAD_NAMES
    if unexpected:
        raise RuntimeError("Unexpected files in upload staging: " + ", ".join(sorted(unexpected)))
    if final:
        missing = FINAL_UPLOAD_NAMES - names
        if missing:
            raise RuntimeError("Final upload staging is incomplete: " + ", ".join(sorted(missing)))


def validate_build_runtime() -> None:
    observed = sys.version_info[:3]
    if observed != EXPECTED_PYTHON:
        raise RuntimeError(
            "Final release packaging requires Python "
            f"{'.'.join(map(str, EXPECTED_PYTHON))}; observed {platform.python_version()}."
        )


def write_output_inventory() -> Path:
    output = UPLOAD_ROOT / "analysis_outputs_sha256.csv"
    candidates: list[tuple[Path, str]] = []
    for subdir in ("generated", "supplementary/audit"):
        directory = REVISED_ROOT / subdir
        for path in filtered_files(directory):
            candidates.append((path, "revised deterministic analysis pipeline"))

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("path", "producer", "bytes", "sha256"))
        for path, producer in sorted(candidates, key=lambda item: item[0].relative_to(PROJECT_ROOT).as_posix()):
            writer.writerow((path.relative_to(PROJECT_ROOT).as_posix(), producer, path.stat().st_size, sha256(path)))
    return output


def write_release_json(commit_sha: str, *, pending: bool, license_path: Path | None = None) -> Path:
    inventory = REVISED_ROOT / "supplementary" / "audit" / "file_inventory_sha256.csv"
    document = {
        "schema_version": 1,
        "release_tag": RELEASE_TAG,
        "repository": REPOSITORY_URL,
        "commit_sha": commit_sha,
        "status": "pending asset build" if pending else "assets built",
        "extract_root": ".",
        "required_top_level_paths": [campaign.source_dir for campaign in CAMPAIGNS]
        + ["Plotcode", "revised", "validation", "README.md", "LICENSE"],
        "source_inventory": {
            "path": "revised/supplementary/audit/file_inventory_sha256.csv",
            "rows": EXPECTED_SOURCE_FILES,
            "total_source_bytes": EXPECTED_TOTAL_SOURCE_BYTES,
            "manifest_bytes": inventory.stat().st_size,
            "manifest_sha256": sha256(inventory),
        },
        "environment": {
            "python": platform.python_version(),
            "requirements": "revised/analysis/requirements.txt",
            "zip_compression": "DEFLATE level 6",
            "zlib_compile_version": zlib.ZLIB_VERSION,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        },
        "pipeline": [
            "python revised/analysis/full_audit.py",
            "python revised/analysis/build_revision_results.py",
            "python revised/analysis/long_run_drift_analysis.py",
        ],
        "expected": {
            "source_files": EXPECTED_SOURCE_FILES,
            "runs": 558,
            "primary_runs": 378,
            "requested_fast_runs": 126,
            "issues_rows": 0,
            "raw_csv_power_max_abs_diff_W": 0.0,
            "raw_csv_elapsed_max_abs_diff_s": 0.0,
        },
        "output_inventory": "validation/analysis_outputs_sha256.csv",
        "license": (
            {"status": "pending author selection"}
            if license_path is None
            else {
                "path": "LICENSE",
                "bytes": license_path.stat().st_size,
                "sha256": sha256(license_path),
            }
        ),
        "redistribution_exclusion": "ImageNet validation images are not included.",
    }
    output = UPLOAD_ROOT / "RELEASE_MANIFEST.json"
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def campaign_entries(campaign: Campaign) -> list[tuple[Path, PurePosixPath]]:
    directory = PROJECT_ROOT / campaign.source_dir
    return [(path, PurePosixPath(path.relative_to(PROJECT_ROOT).as_posix())) for path in source_files(directory)]


def analysis_entries(*, include_license: bool) -> list[tuple[Path, PurePosixPath]]:
    entries: list[tuple[Path, PurePosixPath]] = []
    for path in filtered_files(PROJECT_ROOT / "Plotcode"):
        entries.append((path, PurePosixPath(path.relative_to(PROJECT_ROOT).as_posix())))
    for subdir in ANALYSIS_DIRECTORIES:
        directory = REVISED_ROOT / subdir
        for path in filtered_files(directory):
            entries.append((path, PurePosixPath(path.relative_to(PROJECT_ROOT).as_posix())))
    for filename in ANALYSIS_TOP_LEVEL_FILES:
        path = REVISED_ROOT / filename
        if not path.is_file():
            raise RuntimeError(f"Missing analysis-release file: {path}")
        entries.append((path, PurePosixPath(path.relative_to(PROJECT_ROOT).as_posix())))
    public_readme = UPLOAD_ROOT / "README.md"
    if not public_readme.is_file():
        raise RuntimeError(f"Missing public release README: {public_readme}")
    entries.append((public_readme, PurePosixPath("README.md")))
    for filename in ("RELEASE_MANIFEST.json", "analysis_outputs_sha256.csv"):
        path = UPLOAD_ROOT / filename
        if not path.is_file():
            raise RuntimeError(f"Run --prepare or --build-assets first; missing {path.name}.")
        entries.append((path, PurePosixPath("validation") / filename))
    if include_license:
        license_path = UPLOAD_ROOT / "LICENSE"
        if not license_path.is_file() or license_path.stat().st_size == 0:
            raise RuntimeError("Final analysis asset requires a non-empty upload/LICENSE file.")
        entries.append((license_path, PurePosixPath("LICENSE")))

    entries.sort(key=lambda item: str(item[1]))
    archive_names = [str(item[1]) for item in entries]
    if len(archive_names) != len(set(archive_names)):
        raise RuntimeError("Duplicate archive path in analysis asset.")
    return entries


def deterministic_zip(
    output: Path,
    entries: Sequence[tuple[Path, PurePosixPath]],
    *,
    expected_hashes: dict[str, str] | None = None,
) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    expected_sizes = {str(archive_path): source.stat().st_size for source, archive_path in entries}
    if expected_hashes is None:
        expected_hashes = {str(archive_path): sha256(source) for source, archive_path in entries}
    if set(expected_hashes) != set(expected_sizes):
        raise RuntimeError(f"Expected-hash map does not match the members of {output.name}.")

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
            strict_timestamps=False,
        ) as archive:
            for source, archive_path in entries:
                name = str(archive_path)
                if "\\" in name or name.startswith("/") or ".." in archive_path.parts:
                    raise RuntimeError(f"Unsafe archive name: {name}")
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = 6
                info.external_attr = 0o100644 << 16
                info.flag_bits |= 0x800
                with source.open("rb") as reader, archive.open(info, "w") as writer:
                    shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)

        # Reading every member to EOF validates its CRC; the SHA-256 comparison
        # additionally proves equality with the audited source payload.
        with zipfile.ZipFile(temporary, "r") as archive:
            actual_names = archive.namelist()
            if len(actual_names) != len(set(actual_names)) or set(actual_names) != set(expected_sizes):
                raise RuntimeError(f"Archive member mismatch in {output.name}")
            for info in archive.infolist():
                if info.file_size != expected_sizes[info.filename]:
                    raise RuntimeError(f"Archive size mismatch for {info.filename} in {output.name}")
                with archive.open(info, "r") as reader:
                    observed_hash = sha256_stream(reader)
                if observed_hash != expected_hashes[info.filename]:
                    raise RuntimeError(f"Archive SHA-256 mismatch for {info.filename} in {output.name}")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def asset_rows(*, built: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for campaign in CAMPAIGNS:
        path = UPLOAD_ROOT / campaign.asset_name
        rows.append(
            {
                "asset_name": campaign.asset_name,
                "source_path": campaign.source_dir,
                "archive_root": campaign.source_dir + "/",
                "role": campaign.role,
                "file_count": campaign.expected_files,
                "uncompressed_bytes": campaign.expected_bytes,
                "sha256": sha256(path) if built else PENDING,
                "notes": "Extract into the common release root; original directory name retained.",
            }
        )

    entries = analysis_entries(include_license=built)
    path = UPLOAD_ROOT / ANALYSIS_ASSET
    rows.append(
        {
            "asset_name": ANALYSIS_ASSET,
            "source_path": "Plotcode; public release README and LICENSE; revised analysis/generated/figures/supplementary/Definitions and main.tex",
            "archive_root": "README.md; LICENSE; Plotcode/; revised/; validation/",
            "role": "analysis code, audit outputs, editable figures, manuscript source, license, and validation metadata",
            "file_count": len(entries) if built else PENDING,
            "uncompressed_bytes": sum(source.stat().st_size for source, _ in entries) if built else PENDING,
            "sha256": sha256(path) if built else PENDING,
            "notes": "No raw ImageNet validation images; no LaTeX build artefacts or nested ZIP files.",
        }
    )
    return rows


def write_release_csv(rows: Sequence[dict[str, object]]) -> Path:
    output = UPLOAD_ROOT / "release_manifest.csv"
    fieldnames = (
        "asset_name",
        "source_path",
        "archive_root",
        "role",
        "file_count",
        "uncompressed_bytes",
        "sha256",
        "notes",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_checksums(*, built: bool) -> Path:
    output = UPLOAD_ROOT / "SHA256SUMS.txt"
    if not built:
        output.write_text(
            "# PENDING_BUILD: generated hashes replace this file after --build-assets.\n"
            "# Do not upload this placeholder as a completed checksum record.\n",
            encoding="utf-8",
        )
        return output

    names = [campaign.asset_name for campaign in CAMPAIGNS] + [
        ANALYSIS_ASSET,
        "README.md",
        "LICENSE",
        "release_manifest.csv",
        "RELEASE_MANIFEST.json",
        "analysis_outputs_sha256.csv",
    ]
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for name in names:
            handle.write(f"{sha256(UPLOAD_ROOT / name)}  {name}\n")
    return output


def prepare_metadata(commit_sha: str, *, built: bool) -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    write_output_inventory()
    write_release_json(commit_sha, pending=not built)
    rows = asset_rows(built=built)
    write_release_csv(rows)
    write_checksums(built=built)


def public_payload_sources() -> list[Path]:
    sources: list[Path] = []
    sources.extend(filtered_files(PROJECT_ROOT / "Plotcode"))
    for subdir in ANALYSIS_DIRECTORIES:
        sources.extend(filtered_files(REVISED_ROOT / subdir))
    for filename in ANALYSIS_TOP_LEVEL_FILES:
        sources.append(REVISED_ROOT / filename)
    sources.append(UPLOAD_ROOT / "README.md")
    return sorted(set(sources), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def run_git(checkout: Path, *arguments: str, binary: bool = False):
    command = ["git", "-C", str(checkout), *arguments]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git check failed ({' '.join(arguments)}): {detail}")
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="strict").strip()


def validate_git_binding(
    checkout: Path, commit_sha: str, license_name: str
) -> tuple[Path, dict[str, str], str]:
    checkout = checkout.resolve()
    if not checkout.is_dir():
        raise RuntimeError(f"Repository checkout does not exist: {checkout}")
    top_level = Path(run_git(checkout, "rev-parse", "--show-toplevel")).resolve()
    if top_level != checkout:
        raise RuntimeError(f"--repository-checkout must be the Git top level: {top_level}")

    head = run_git(checkout, "rev-parse", "--verify", "HEAD").lower()
    if head != commit_sha:
        raise RuntimeError(f"Git HEAD is {head}, not the requested release commit {commit_sha}.")
    tag_commit = run_git(checkout, "rev-parse", "--verify", f"refs/tags/{RELEASE_TAG}^{{commit}}").lower()
    if tag_commit != commit_sha:
        raise RuntimeError(f"Tag {RELEASE_TAG!r} points to {tag_commit}, not {commit_sha}.")
    tracked_status = run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if tracked_status:
        raise RuntimeError("The repository contains tracked or untracked changes; use a completely clean tagged checkout.")

    remote = run_git(checkout, "remote", "get-url", "origin").strip()
    normalized_remote = remote.lower().removesuffix(".git").rstrip("/")
    allowed_remotes = {
        "https://github.com/weissbeck-lucas/ki-labor",
        "git@github.com:weissbeck-lucas/ki-labor",
        "ssh://git@github.com/weissbeck-lucas/ki-labor",
    }
    if normalized_remote not in allowed_remotes:
        raise RuntimeError(f"Unexpected origin remote for the data release: {remote!r}")
    remote_refs = run_git(
        checkout,
        "ls-remote",
        "--exit-code",
        "origin",
        f"refs/tags/{RELEASE_TAG}",
        f"refs/tags/{RELEASE_TAG}^{{}}",
    )
    remote_pairs = {}
    for line in remote_refs.splitlines():
        fields = line.split()
        if len(fields) == 2:
            remote_pairs[fields[1]] = fields[0].lower()
    remote_commit = remote_pairs.get(f"refs/tags/{RELEASE_TAG}^{{}}") or remote_pairs.get(
        f"refs/tags/{RELEASE_TAG}"
    )
    if remote_commit != commit_sha:
        raise RuntimeError(
            f"Remote tag {RELEASE_TAG!r} resolves to {remote_commit!r}, not the requested commit {commit_sha}."
        )

    tracked_raw = run_git(checkout, "-c", "core.quotepath=false", "ls-files", "-z", binary=True)
    tracked = {item.decode("utf-8", errors="strict") for item in tracked_raw.split(b"\0") if item}
    payload_hashes: dict[str, str] = {}
    for source in public_payload_sources():
        relative = source.relative_to(PROJECT_ROOT).as_posix()
        if relative not in tracked:
            raise RuntimeError(f"Public analysis payload is not tracked by the tagged commit: {relative}")
        committed_copy = checkout / Path(*PurePosixPath(relative).parts)
        source_hash = sha256(source)
        if not committed_copy.is_file() or sha256(committed_copy) != source_hash:
            raise RuntimeError(f"Tagged-checkout content differs from the staged analysis payload: {relative}")
        payload_hashes[relative] = source_hash

    candidate = Path(license_name)
    license_path = candidate.resolve() if candidate.is_absolute() else (checkout / candidate).resolve()
    try:
        license_relative = license_path.relative_to(checkout).as_posix()
    except ValueError as exc:
        raise RuntimeError("The release license must be inside the tagged repository checkout.") from exc
    if license_relative not in tracked:
        raise RuntimeError(f"Release license is not tracked by the tagged commit: {license_relative}")
    if not license_path.is_file() or license_path.stat().st_size == 0:
        raise RuntimeError(f"Release license is missing or empty: {license_path}")
    return license_path, payload_hashes, sha256(license_path)


def build_assets(
    commit_sha: str,
    inventory_hashes: dict[str, str],
    license_source: Path,
    license_hash: str,
    payload_hashes: dict[str, str],
) -> None:
    validate_upload_allowlist(final=False)
    shutil.copyfile(license_source, UPLOAD_ROOT / "LICENSE")
    if sha256(UPLOAD_ROOT / "LICENSE") != license_hash:
        raise RuntimeError("Copied release license differs from the tagged license.")
    # Metadata must exist before it is included in the analysis ZIP.
    prepare_metadata(commit_sha, built=False)
    for campaign in CAMPAIGNS:
        output = UPLOAD_ROOT / campaign.asset_name
        print(f"Building {output.name} ...", flush=True)
        entries = campaign_entries(campaign)
        member_hashes = {str(archive_path): inventory_hashes[str(archive_path)] for _, archive_path in entries}
        deterministic_zip(output, entries, expected_hashes=member_hashes)
    # Put the final (non-pending) release metadata inside the analysis ZIP.
    write_output_inventory()
    write_release_json(commit_sha, pending=False, license_path=UPLOAD_ROOT / "LICENSE")
    print(f"Building {ANALYSIS_ASSET} ...", flush=True)
    entries = analysis_entries(include_license=True)
    analysis_hashes: dict[str, str] = {}
    for source, archive_path in entries:
        archive_name = str(archive_path)
        if archive_name == "LICENSE":
            analysis_hashes[archive_name] = license_hash
            continue
        relative = source.relative_to(PROJECT_ROOT).as_posix()
        analysis_hashes[archive_name] = payload_hashes[relative] if relative in payload_hashes else sha256(source)
    deterministic_zip(UPLOAD_ROOT / ANALYSIS_ASSET, entries, expected_hashes=analysis_hashes)
    write_release_csv(asset_rows(built=True))
    write_checksums(built=True)
    validate_upload_allowlist(final=True)
    for filename in ("SHA256SUMS.txt", "release_manifest.csv", "RELEASE_MANIFEST.json"):
        if PENDING in (UPLOAD_ROOT / filename).read_text(encoding="utf-8"):
            raise RuntimeError(f"Pending marker remains in final upload file: {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="write small staging manifests with PENDING hashes")
    mode.add_argument("--build-assets", action="store_true", help="create all seven ZIP assets and final hashes")
    parser.add_argument(
        "--commit-sha",
        default="<INSERT_FULL_40_CHARACTER_GIT_COMMIT_SHA>",
        help="full Git commit SHA anchored by the release",
    )
    parser.add_argument(
        "--repository-checkout",
        type=Path,
        help="clean Git top-level checkout whose HEAD and release tag equal --commit-sha",
    )
    parser.add_argument(
        "--license-file",
        default="LICENSE",
        help="tracked license path relative to --repository-checkout (default: LICENSE)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_upload_allowlist(final=False)
    if args.build_assets:
        validate_build_runtime()
        if re.fullmatch(r"[0-9a-fA-F]{40}", args.commit_sha) is None:
            raise RuntimeError("--build-assets requires a full 40-character Git commit SHA.")
        if args.repository_checkout is None:
            raise RuntimeError("--build-assets requires --repository-checkout for Git/tag verification.")
        commit_sha = args.commit_sha.lower()
        license_source, payload_hashes, license_hash = validate_git_binding(
            args.repository_checkout, commit_sha, args.license_file
        )
        inventory_hashes = validate_source_layout(verify_hashes=True)
        build_assets(commit_sha, inventory_hashes, license_source, license_hash, payload_hashes)
        # Recheck local and remote Git state after the potentially long archive build.
        validate_git_binding(args.repository_checkout, commit_sha, args.license_file)
        print(f"Release assets ready in: {UPLOAD_ROOT}")
    elif args.prepare:
        existing_zips = sorted(path.name for path in UPLOAD_ROOT.glob("*.zip"))
        if existing_zips:
            raise RuntimeError("Refusing to overwrite an existing built release with pending metadata: " + ", ".join(existing_zips))
        validate_source_layout(verify_hashes=False)
        prepare_metadata(args.commit_sha, built=False)
        print(f"Staging metadata prepared in: {UPLOAD_ROOT}")
    else:
        validate_source_layout(verify_hashes=False)
        plotcode_files = source_files(PROJECT_ROOT / "Plotcode")
        print(
            "Source layout verified: "
            f"{EXPECTED_CAMPAIGN_FILES} campaign files / {EXPECTED_CAMPAIGN_BYTES} bytes; "
            f"{len(plotcode_files)} Plotcode files / {byte_count(plotcode_files)} bytes; "
            f"{EXPECTED_SOURCE_FILES} files / {EXPECTED_TOTAL_SOURCE_BYTES} bytes total."
        )
        print("No files were written. Use --prepare or --build-assets as documented.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
