from __future__ import annotations

import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import BinaryIO
from urllib.error import URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

ALLOWED_SOURCE_ARCHIVE_EXTENSIONS = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
)
SOURCE_ARCHIVE_FORMAT_HINT = ", ".join(ALLOWED_SOURCE_ARCHIVE_EXTENSIONS)
SOURCE_ARCHIVE_DOWNLOAD_MAX_BYTES = 512 * 1024 * 1024
SOURCE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS = 30


@dataclass
class DownloadedSourceArchive:
    name: str
    file: BinaryIO


def archive_format_from_name(name: str) -> str | None:
    lower_name = name.lower()
    if lower_name.endswith(".zip"):
        return "zip"
    if lower_name.endswith(
        (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
    ):
        return "tar"
    return None


def validate_source_archive_upload(fileobj: BinaryIO, name: str) -> str:
    archive_format = archive_format_from_name(name)
    if archive_format is not None:
        _validate_archive_fileobj(fileobj, archive_format)
        return archive_format

    for candidate_format in ("zip", "tar"):
        try:
            fileobj.seek(0)
            _validate_archive_fileobj(fileobj, candidate_format)
        except ValueError:
            continue
        return candidate_format

    raise ValueError(
        f"source must be one of these archive formats: {SOURCE_ARCHIVE_FORMAT_HINT}"
    )


def download_source_archive(
    url: str,
    *,
    max_bytes: int = SOURCE_ARCHIVE_DOWNLOAD_MAX_BYTES,
    timeout_seconds: int = SOURCE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS,
) -> DownloadedSourceArchive:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Download URL must use http or https.")

    request = Request(url, headers={"User-Agent": "Catchy/0.1"})
    fileobj = SpooledTemporaryFile(max_size=10 * 1024 * 1024, mode="w+b")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError("source archive download is too large")
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("source archive download is too large")
                fileobj.write(chunk)
    except URLError as exc:
        fileobj.close()
        raise ValueError(f"could not download source archive: {exc}") from exc
    except Exception:
        fileobj.close()
        raise

    fileobj.seek(0)
    name = _download_name_from_url(url)
    archive_format = validate_source_archive_upload(fileobj, name)
    fileobj.seek(0)
    return DownloadedSourceArchive(
        name=_normalized_archive_name(name, archive_format),
        file=fileobj,
    )


def safe_extract_archive(archive_path: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        _safe_extract_zip(archive_path, destination)
        return
    if tarfile.is_tarfile(archive_path):
        _safe_extract_tar(archive_path, destination)
        return
    raise ValueError(
        f"source archive is not a supported archive: {SOURCE_ARCHIVE_FORMAT_HINT}"
    )


def _validate_archive_fileobj(fileobj: BinaryIO, archive_format: str) -> None:
    if archive_format == "zip":
        _validate_zip_fileobj(fileobj)
    else:
        _validate_tar_fileobj(fileobj)


def _validate_zip_fileobj(fileobj: BinaryIO) -> None:
    try:
        with zipfile.ZipFile(fileobj) as archive:
            _validate_zip_members(archive.infolist())
    except zipfile.BadZipFile as exc:
        raise ValueError("source archive is not a valid zip file") from exc


def _validate_tar_fileobj(fileobj: BinaryIO) -> None:
    try:
        with tarfile.open(fileobj=fileobj, mode="r:*") as archive:
            _validate_tar_members(archive.getmembers())
    except tarfile.TarError as exc:
        raise ValueError("source archive is not a valid tar archive") from exc


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        _validate_zip_members(archive.infolist())
        _validate_destination_members(
            destination,
            [info.filename for info in archive.infolist()],
        )
        archive.extractall(destination)


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        _validate_tar_members(members)
        _validate_destination_members(destination, [member.name for member in members])
        archive.extractall(destination)


def _validate_zip_members(members: list[zipfile.ZipInfo]) -> None:
    for member in members:
        _validate_member_name(member.filename)
        mode = member.external_attr >> 16
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            raise ValueError(f"archive member cannot be a symlink: {member.filename}")


def _validate_tar_members(members: list[tarfile.TarInfo]) -> None:
    for member in members:
        _validate_member_name(member.name)
        if member.issym() or member.islnk():
            raise ValueError(f"archive member cannot be a link: {member.name}")
        if member.isdev():
            raise ValueError(f"archive member cannot be a device file: {member.name}")


def _validate_member_name(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive member escapes destination: {name}")


def _validate_destination_members(destination: Path, member_names: list[str]) -> None:
    destination_root = destination.resolve()
    for name in member_names:
        target = (destination / name).resolve()
        try:
            target.relative_to(destination_root)
        except ValueError as exc:
            raise ValueError(f"archive member escapes destination: {name}") from exc


def _download_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    return name or "source"


def _normalized_archive_name(name: str, archive_format: str) -> str:
    if archive_format_from_name(name):
        return name
    extension = ".zip" if archive_format == "zip" else ".tar"
    return f"{name}{extension}"
