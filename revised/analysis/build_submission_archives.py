#!/usr/bin/env python3
"""Build portable MDPI submission archives with POSIX ZIP entry names."""

from __future__ import annotations

import os
import hashlib
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
ANALYSIS_FILES = (
    "full_audit.py",
    "build_revision_results.py",
    "long_run_drift_analysis.py",
    "build_github_release_assets.py",
    "build_submission_archives.py",
    "requirements.txt",
)
BUILD_SUFFIXES = {".aux", ".log", ".out", ".pyc"}
FIGURE_EXPORT_SUFFIXES = {".tex", ".pdf", ".png", ".md"}
CHECKSUM_FILES = (
    "cover_letter.pdf",
    "electronics-4377237_revised.pdf",
    "electronics-4377237_revised_source.zip",
    "electronics-figures.zip",
    "electronics-responses.zip",
    "electronics-supplement.zip",
    "README.md",
    "response_reviewer1.pdf",
    "response_reviewer2.pdf",
)


def add_tree(
    entries: list[tuple[Path, Path]],
    source_dir: Path,
    archive_dir: Path,
    *,
    allowed_suffixes: set[str] | None = None,
) -> None:
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        if source.suffix.lower() in BUILD_SUFFIXES:
            continue
        if allowed_suffixes is not None and source.suffix.lower() not in allowed_suffixes:
            continue
        entries.append((source, archive_dir / source.relative_to(source_dir)))


def add_analysis(entries: list[tuple[Path, Path]]) -> None:
    for name in ANALYSIS_FILES:
        entries.append((ROOT / "analysis" / name, Path("analysis") / name))


def write_zip(destination: Path, entries: list[tuple[Path, Path]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    seen: set[str] = set()
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source, archive_path in sorted(entries, key=lambda item: item[1].as_posix()):
            if not source.is_file():
                raise FileNotFoundError(source)
            name = archive_path.as_posix()
            if name.startswith("/") or ".." in archive_path.parts:
                raise ValueError(f"Unsafe ZIP entry: {name}")
            if name in seen:
                raise ValueError(f"Duplicate ZIP entry: {name}")
            seen.add(name)
            archive.write(source, name)
    os.replace(temporary, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_checksums() -> None:
    destination = SUBMISSION / "SHA256SUMS.txt"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    lines = []
    for name in CHECKSUM_FILES:
        path = SUBMISSION / name
        if not path.is_file():
            raise FileNotFoundError(path)
        lines.append(f"{sha256_file(path)}  {name}\n")
    temporary.write_text("".join(lines), encoding="ascii", newline="\n")
    os.replace(temporary, destination)


def source_entries() -> list[tuple[Path, Path]]:
    entries = [(ROOT / "main.tex", Path("main.tex")), (ROOT / "README.md", Path("README.md"))]
    for directory in ("Definitions", "figures", "generated", "supplementary"):
        add_tree(entries, ROOT / directory, Path(directory))
    add_analysis(entries)
    return entries


def supplement_entries() -> list[tuple[Path, Path]]:
    entries = [(ROOT / "GITHUB_DATA_RELEASE.md", Path("GITHUB_DATA_RELEASE.md"))]
    for directory in ("supplementary", "generated", "figures"):
        add_tree(entries, ROOT / directory, Path(directory))
    add_analysis(entries)
    return entries


def figure_entries() -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    add_tree(
        entries,
        ROOT / "figure_exports",
        Path("figure_exports"),
        allowed_suffixes=FIGURE_EXPORT_SUFFIXES,
    )
    for directory in ("figures", "generated"):
        add_tree(entries, ROOT / directory, Path(directory))
    return entries


def response_entries() -> list[tuple[Path, Path]]:
    names = (
        "response_reviewer1.tex",
        "response_reviewer1.pdf",
        "response_reviewer2.tex",
        "response_reviewer2.pdf",
    )
    return [(ROOT / name, Path(name)) for name in names]


def main() -> None:
    archives = {
        "electronics-4377237_revised_source.zip": source_entries(),
        "electronics-supplement.zip": supplement_entries(),
        "electronics-figures.zip": figure_entries(),
        "electronics-responses.zip": response_entries(),
    }
    for name, entries in archives.items():
        write_zip(SUBMISSION / name, entries)

    copies = {
        ROOT / "main.pdf": SUBMISSION / "electronics-4377237_revised.pdf",
        ROOT / "cover_letter.pdf": SUBMISSION / "cover_letter.pdf",
        ROOT / "response_reviewer1.pdf": SUBMISSION / "response_reviewer1.pdf",
        ROOT / "response_reviewer2.pdf": SUBMISSION / "response_reviewer2.pdf",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)
    shutil.copy2(SUBMISSION / "electronics-supplement.zip", ROOT / "electronics-supplement.zip")
    write_checksums()

    print(
        {
            name: {
                "entries": len(entries),
                "bytes": (SUBMISSION / name).stat().st_size,
            }
            for name, entries in archives.items()
        }
    )


if __name__ == "__main__":
    main()
