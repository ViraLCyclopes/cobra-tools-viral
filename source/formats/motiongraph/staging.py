"""Shared staging and verified publication helpers for motiongraph writers."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


def validate_staged_pair(source: Path, output: Path, *, require_output: bool = True):
    """Resolve and validate a read-only source and an existing staged OVL."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file() or source.suffix.lower() != ".ovl":
        raise ValueError(f"Choose an existing source OVL: {source}")
    same = source == output
    if not same and output.exists():
        try:
            same = os.path.samefile(source, output)
        except OSError:
            pass
    if same:
        raise ValueError("Refusing to overwrite the source OVL; use a complete staged family")
    if source.name.lower() != output.name.lower():
        raise ValueError(
            "Source and output OVL basenames must match so the staged OVS/AUX family can reload"
        )
    if not output.parent.is_dir() or (require_output and not output.is_file()):
        raise ValueError("Copy the complete archive family to the stage directory first")
    if require_output:
        source_family = {member.name.lower(): member for member in motiongraph_family(source)}
        staged_family = {member.name.lower(): member for member in motiongraph_family(output)}
        required = set(source_family)
        staged = set(staged_family)
        missing = sorted(required - staged)
        if missing:
            raise ValueError(
                "Staged archive family is incomplete; missing: " + ", ".join(missing)
            )
        source_companions = _companion_hashes(source)
        staged_companions = _companion_hashes(output)
        if staged_companions != source_companions:
            changed = sorted(
                name for name in set(source_companions) | set(staged_companions)
                if source_companions.get(name) != staged_companions.get(name)
            )
            raise ValueError(
                "Staged OVS/AUX companions do not match the source snapshot: "
                + ", ".join(changed)
            )
    return source, output


def motiongraph_family(source: Path) -> list[Path]:
    """Return an OVL, its basename OVS streams, and all hash-named AUX files."""
    source = Path(source).resolve()
    if not source.is_file() or source.suffix.lower() != ".ovl":
        raise ValueError("Choose an existing .ovl file")
    prefix = source.stem.lower() + "."
    members = [candidate for candidate in source.parent.iterdir()
               if candidate.is_file() and (
                   (candidate.name.lower().startswith(prefix)
                    and (candidate.name.lower().endswith(".ovl")
                         or ".ovs" in candidate.name.lower()))
                   or candidate.name.lower().endswith(".aux"))]
    if source not in members:
        members.append(source)
    return sorted(set(members), key=lambda path: path.name.lower())


def copy_motiongraph_family(source: Path, destination: Path) -> list[Path]:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.parent == destination:
        raise ValueError("The stage directory must differ from the source directory")
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for member in motiongraph_family(source):
        target = destination / member.name
        shutil.copy2(member, target)
        outputs.append(target)
    return outputs


def family_hashes(source: Path) -> dict[str, str]:
    return {member.name.lower(): hashlib.sha256(member.read_bytes()).hexdigest()
            for member in motiongraph_family(source)}


def _companion_hashes(source: Path) -> dict[str, str]:
    source = Path(source).resolve()
    return {member.name.lower(): hashlib.sha256(member.read_bytes()).hexdigest()
            for member in motiongraph_family(source) if member != source}


class CandidatePublication:
    """Private same-volume complete-family candidate committed with ``os.replace``."""

    def __init__(self, source: Path, output: Path):
        self._committed = False
        self._temporary = None
        self.source, self.output = validate_staged_pair(source, output)
        self._companions = _companion_hashes(self.source)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="motiongraph-candidate-", dir=self.output.parent)
        self.root = Path(self._temporary.name)
        copy_motiongraph_family(self.source, self.root)
        self.path = self.root / self.source.name

    def commit(self):
        if not self.path.is_file():
            raise ValueError("Motiongraph operation did not produce a candidate OVL")
        candidate = _companion_hashes(self.path)
        destination = _companion_hashes(self.output)
        if candidate != self._companions or destination != self._companions:
            raise ValueError(
                "Candidate or staged OVS/AUX companions changed during a STATIC-only operation"
            )
        os.replace(self.path, self.output)
        self._committed = True
        self._temporary.cleanup()
        return self.output

    def __del__(self):
        if not getattr(self, "_committed", False) and self._temporary is not None:
            try:
                self._temporary.cleanup()
            except Exception:
                pass


@contextmanager
def verified_candidate(source: Path, output: Path):
    """Yield a private complete-family output and publish it atomically on success.

    The caller must write and fully verify the yielded OVL.  Any exception leaves
    both the original source and the previous staged result untouched.
    """
    source, output = validate_staged_pair(source, output)
    companion_before = _companion_hashes(source)
    with tempfile.TemporaryDirectory(prefix="motiongraph-candidate-", dir=output.parent) as raw:
        root = Path(raw)
        copy_motiongraph_family(source, root)
        candidate = root / source.name
        yield candidate
        if not candidate.is_file():
            raise ValueError("Motiongraph operation did not produce a candidate OVL")
        candidate_companions = _companion_hashes(candidate)
        destination_companions = _companion_hashes(output)
        if (candidate_companions != companion_before
                or destination_companions != companion_before):
            raise ValueError(
                "Candidate or staged OVS/AUX companions changed during a STATIC-only operation"
            )
        os.replace(candidate, output)
