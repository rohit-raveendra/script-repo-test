#!/usr/bin/env python3
"""
Idempotent GitLab -> GitHub mirror sync for large repositories.

What it does:
- Keeps a local bare mirror (`git clone --mirror` once, then `git fetch --prune`).
- Syncs only changed refs (branches/tags), including deletions.
- Retries fetch/push with backoff on transient failures and rate-limit style errors.
- Optionally syncs Git LFS objects (`git lfs fetch --all` + `git lfs push --all`).
- Detects GitHub 100MB file-limit errors and can optionally rewrite to LFS.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
from urllib.parse import quote, urlsplit, urlunsplit


class SyncError(RuntimeError):
    """Raised for non-retriable sync failures."""


class LargeFileLimitError(SyncError):
    """Raised when GitHub rejects objects above file-size limit."""


@dataclass
class RefPlan:
    kind: str
    local_count: int
    remote_count: int
    upsert: List[str]
    delete: List[str]


@dataclass
class LargeBlob:
    size_bytes: int
    path: str
    oid: str


def now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


def configure_logging(log_file: Path, verbose: bool) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def redact(text: str, secrets: Iterable[str]) -> str:
    out = text
    for secret in secrets:
        if secret:
            out = out.replace(secret, "***")
    return out


def normalize_git_url(url: str) -> str:
    raw = url.strip()
    if not raw:
        raise ValueError("Repository URL cannot be empty.")

    # git@host:org/repo.git -> https://host/org/repo.git
    ssh_like = re.match(r"^[^@\s]+@([^:\s]+):(.+)$", raw)
    if ssh_like:
        host = ssh_like.group(1)
        path = ssh_like.group(2).lstrip("/")
        return f"https://{host}/{path}"

    if raw.startswith("http://") or raw.startswith("https://"):
        return raw.rstrip("/")

    raise ValueError(f"Unsupported repository URL format: {url}")


def with_basic_auth(url: str, username: str, token: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs are supported for token auth: {url}")
    host = parsed.netloc.split("@", 1)[-1]
    userinfo = f"{quote(username, safe='')}:{quote(token, safe='')}"
    return urlunsplit((parsed.scheme, f"{userinfo}@{host}", parsed.path, parsed.query, parsed.fragment))


def repo_folder_name(repo_url: str) -> str:
    parsed = urlsplit(repo_url)
    name = Path(parsed.path).name or "repo"
    if not name.endswith(".git"):
        name = f"{name}.git"
    return name


def parse_size_to_bytes(text: str) -> int:
    value = text.strip().upper()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)?$", value)
    if not m:
        raise ValueError(f"Invalid size format: {text}")
    num = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    mult = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 * 1024,
        "GB": 1024 * 1024 * 1024,
    }[unit]
    return int(num * mult)


def mb_str(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f}"


def render_large_blob_table(rows: List[LargeBlob]) -> str:
    if not rows:
        return "| Size(MB) | Path | Blob SHA |\n|---:|---|---|\n| 0 | - | - |"

    header = "| Size(MB) | Path | Blob SHA |"
    sep = "|---:|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append(f"| {mb_str(row.size_bytes)} | {row.path} | {row.oid} |")
    return "\n".join(lines)


def render_branch_file_table(branch_files: Dict[str, List[str]]) -> str:
    if not branch_files:
        return "| Branch | Files (>= threshold) |\n|---|---|\n| - | - |"
    header = "| Branch | Files (>= threshold) |"
    sep = "|---|---|"
    lines = [header, sep]
    for branch in sorted(branch_files.keys()):
        files = branch_files[branch]
        files_cell = "<br>".join(files) if files else "-"
        lines.append(f"| {branch} | {files_cell} |")
    return "\n".join(lines)


def scan_large_blobs(
    repo_dir: Path,
    *,
    threshold_bytes: int,
    progress_interval_seconds: float,
) -> List[LargeBlob]:
    rev = subprocess.Popen(
        ["git", "rev-list", "--objects", "--all"],
        cwd=str(repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if rev.stdout is None:
        raise SyncError("Failed to open git rev-list output stream.")

    cat = subprocess.Popen(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)"],
        cwd=str(repo_dir),
        stdin=rev.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    rev.stdout.close()
    if cat.stdout is None:
        raise SyncError("Failed to open git cat-file output stream.")

    start = time.monotonic()
    last_hb = start
    processed = 0
    matches: List[LargeBlob] = []

    for line in cat.stdout:
        processed += 1
        now = time.monotonic()
        if progress_interval_seconds > 0 and (now - last_hb) >= progress_interval_seconds:
            logging.info(
                "large-file scan still running... elapsed=%ss, scanned=%s, matches=%s",
                int(now - start),
                processed,
                len(matches),
            )
            last_hb = now

        parts = line.rstrip("\n").split(" ", 3)
        if len(parts) < 3:
            continue
        oid, obj_type, size_s = parts[0], parts[1], parts[2]
        if obj_type != "blob":
            continue
        try:
            size = int(size_s)
        except ValueError:
            continue
        if size < threshold_bytes:
            continue
        path = parts[3].strip() if len(parts) > 3 else "(no-path)"
        matches.append(LargeBlob(size_bytes=size, path=path, oid=oid))

    cat.wait()
    rev.wait()
    rev_err = (rev.stderr.read() if rev.stderr else "").strip()
    cat_err = (cat.stderr.read() if cat.stderr else "").strip()
    if rev.returncode != 0:
        raise SyncError(f"git rev-list failed ({rev.returncode}): {rev_err}")
    if cat.returncode != 0:
        raise SyncError(f"git cat-file failed ({cat.returncode}): {cat_err}")

    matches.sort(key=lambda x: x.size_bytes, reverse=True)
    return matches


def resolve_refs_for_blob(
    repo_dir: Path,
    blob_oid: str,
    *,
    ref_prefix: str,
    commit_ref_cache: Dict[Tuple[str, str], Set[str]],
    progress_interval_seconds: float,
) -> Set[str]:
    commits_cp = run_cmd(
        ["git", "log", "--all", f"--find-object={blob_oid}", "--pretty=%H"],
        cwd=repo_dir,
        secrets=[],
        check=True,
        progress_label=f"find-object {blob_oid[:10]}",
        progress_interval_seconds=progress_interval_seconds,
    )
    commits = [c.strip() for c in (commits_cp.stdout or "").splitlines() if c.strip()]
    refs: Set[str] = set()
    for commit in commits:
        key = (ref_prefix, commit)
        if key in commit_ref_cache:
            refs.update(commit_ref_cache[key])
            continue
        contains_cp = run_cmd(
            ["git", "for-each-ref", "--format=%(refname:short)", "--contains", commit, ref_prefix],
            cwd=repo_dir,
            secrets=[],
            check=True,
            progress_label=f"branch-contains {commit[:10]}",
            progress_interval_seconds=progress_interval_seconds,
        )
        found = {b.strip() for b in (contains_cp.stdout or "").splitlines() if b.strip()}
        commit_ref_cache[key] = found
        refs.update(found)
    return refs


def build_ref_file_report(
    repo_dir: Path,
    blobs: List[LargeBlob],
    *,
    ref_prefix: str,
    no_ref_marker: str,
    progress_interval_seconds: float,
) -> Dict[str, List[str]]:
    ref_files: Dict[str, Set[str]] = {}
    commit_ref_cache: Dict[Tuple[str, str], Set[str]] = {}
    start = time.monotonic()
    last_hb = start
    for idx, blob in enumerate(blobs, start=1):
        now = time.monotonic()
        if progress_interval_seconds > 0 and (now - last_hb) >= progress_interval_seconds:
            logging.info(
                "branch mapping still running... elapsed=%ss, blobs=%s/%s",
                int(now - start),
                idx - 1,
                len(blobs),
            )
            last_hb = now
        refs = resolve_refs_for_blob(
            repo_dir,
            blob.oid,
            ref_prefix=ref_prefix,
            commit_ref_cache=commit_ref_cache,
            progress_interval_seconds=progress_interval_seconds,
        )
        if not refs:
            refs = {no_ref_marker}
        for ref in refs:
            ref_files.setdefault(ref, set()).add(blob.path)
    return {k: sorted(v) for k, v in ref_files.items()}


def build_branch_file_report(
    repo_dir: Path,
    blobs: List[LargeBlob],
    *,
    progress_interval_seconds: float,
) -> Dict[str, List[str]]:
    return build_ref_file_report(
        repo_dir,
        blobs,
        ref_prefix="refs/heads",
        no_ref_marker="(no-branch-head)",
        progress_interval_seconds=progress_interval_seconds,
    )


def build_tag_file_report(
    repo_dir: Path,
    blobs: List[LargeBlob],
    *,
    progress_interval_seconds: float,
) -> Dict[str, List[str]]:
    return build_ref_file_report(
        repo_dir,
        blobs,
        ref_prefix="refs/tags",
        no_ref_marker="(no-tag-head)",
        progress_interval_seconds=progress_interval_seconds,
    )


def choose_large_file_action(*, can_ignore: bool) -> str:
    if not sys.stdin.isatty():
        logging.warning("No interactive terminal detected. Defaulting large-file action to 'cancel'.")
        return "cancel"
    print("\nLarge files (> threshold) detected.")
    print("Choose next action:")
    print("1) rewrite_to_lfs  - rewrite history and push large files to GitHub LFS")
    if can_ignore:
        print("2) ignore_large_refs - skip affected refs and continue syncing others")
    print("3) cancel - stop now for customer confirmation")
    while True:
        val = input("Enter choice [1/2/3]: ").strip().lower()
        if val in ("1", "rewrite", "rewrite_to_lfs"):
            return "rewrite"
        if can_ignore and val in ("2", "ignore", "ignore_large_refs"):
            return "ignore"
        if val in ("3", "cancel", ""):
            return "cancel"
        if can_ignore:
            print("Please enter 1, 2, or 3.")
        else:
            print("Please enter 1 or 3.")


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None,
    secrets: Iterable[str],
    check: bool = True,
    progress_label: str = "",
    progress_interval_seconds: float = 0.0,
) -> subprocess.CompletedProcess[str]:
    out_fd, out_path = tempfile.mkstemp(prefix="mirror_sync_out_", suffix=".log")
    err_fd, err_path = tempfile.mkstemp(prefix="mirror_sync_err_", suffix=".log")
    os.close(out_fd)
    os.close(err_fd)
    try:
        with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
            proc = subprocess.Popen(
                list(cmd),
                cwd=str(cwd) if cwd else None,
                stdout=out_f,
                stderr=err_f,
            )
            start = time.monotonic()
            last_heartbeat = start
            interval = max(0.0, progress_interval_seconds)
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                now = time.monotonic()
                if progress_label and interval > 0 and (now - last_heartbeat) >= interval:
                    elapsed_s = int(now - start)
                    out_size = Path(out_path).stat().st_size if Path(out_path).exists() else 0
                    err_size = Path(err_path).stat().st_size if Path(err_path).exists() else 0
                    total_mb = (out_size + err_size) / (1024 * 1024)
                    logging.info(
                        "%s still running... elapsed=%ss, output=%.1f MB",
                        progress_label,
                        elapsed_s,
                        total_mb,
                    )
                    last_heartbeat = now
                time.sleep(1.0)

            proc.wait()

        stdout_text = Path(out_path).read_text(encoding="utf-8", errors="replace")
        stderr_text = Path(err_path).read_text(encoding="utf-8", errors="replace")
        cp = subprocess.CompletedProcess(list(cmd), proc.returncode, stdout_text, stderr_text)
    finally:
        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            Path(err_path).unlink(missing_ok=True)
        except Exception:
            pass

    if cp.returncode != 0 and check:
        merged = (cp.stdout or "") + "\n" + (cp.stderr or "")
        merged = redact(merged, secrets)
        raise SyncError(f"Command failed ({cp.returncode}): {' '.join(cmd)}\n{merged.strip()}")
    return cp


def looks_retriable(output: str) -> bool:
    text = output.lower()
    patterns = [
        "secondary rate limit",
        "temporarily blocked from content creation",
        "rate limit",
        "too many requests",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "service unavailable",
        "timed out",
        "connection reset",
        "the remote end hung up unexpectedly",
        "failed to connect",
        "proxy error",
    ]
    return any(p in text for p in patterns)


def has_large_file_rejection(output: str) -> bool:
    text = output.lower()
    return (
        "gh001" in text
        or "exceeds github's file size limit of 100.00 mb" in text
        or "file size limit is 100.00 mb" in text
    )


def run_git_with_retry(
    cmd: Sequence[str],
    *,
    cwd: Path,
    secrets: Iterable[str],
    max_attempts: int,
    initial_backoff_seconds: float,
    backoff_multiplier: float,
    max_backoff_seconds: float,
    progress_interval_seconds: float,
    operation: str,
) -> subprocess.CompletedProcess[str]:
    delay = max(1.0, initial_backoff_seconds)
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        cp = run_cmd(
            cmd,
            cwd=cwd,
            secrets=secrets,
            check=False,
            progress_label=operation,
            progress_interval_seconds=progress_interval_seconds,
        )
        combined = (cp.stdout or "") + "\n" + (cp.stderr or "")
        if cp.returncode == 0:
            return cp

        redacted = redact(combined, secrets).strip()
        last_error = redacted
        if has_large_file_rejection(combined):
            raise LargeFileLimitError(f"{operation} failed due to GitHub file-size limit.\n{redacted}")

        if attempt < max_attempts and looks_retriable(combined):
            logging.warning(
                "%s failed attempt %s/%s; retrying in %.1fs",
                operation,
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
            delay = min(max_backoff_seconds, delay * max(1.0, backoff_multiplier))
            continue
        break

    raise SyncError(f"{operation} failed after retries.\n{last_error}")


def ensure_git_available() -> None:
    cp = subprocess.run(["git", "--version"], capture_output=True, text=True, check=False)
    if cp.returncode != 0:
        raise SyncError("git is not installed or not available in PATH.")


def git_lfs_available() -> bool:
    cp = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True, check=False)
    return cp.returncode == 0


def ensure_mirror_repo(
    mirror_dir: Path,
    source_auth_url: str,
    *,
    secrets: Iterable[str],
    retry_cfg: Dict[str, float],
) -> None:
    if not mirror_dir.exists():
        mirror_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--mirror", source_auth_url, str(mirror_dir)]
        run_git_with_retry(
            cmd,
            cwd=mirror_dir.parent,
            secrets=secrets,
            operation="git clone --mirror",
            **retry_cfg,
        )
        return

    cp = run_cmd(["git", "rev-parse", "--is-bare-repository"], cwd=mirror_dir, secrets=secrets, check=False)
    if cp.returncode != 0 or (cp.stdout or "").strip().lower() != "true":
        raise SyncError(f"Existing mirror path is not a bare repo: {mirror_dir}")


def ensure_remote(repo_dir: Path, name: str, remote_url: str, *, secrets: Iterable[str]) -> None:
    cp = run_cmd(["git", "remote", "get-url", name], cwd=repo_dir, secrets=secrets, check=False)
    if cp.returncode == 0:
        run_cmd(["git", "remote", "set-url", name, remote_url], cwd=repo_dir, secrets=secrets, check=True)
    else:
        run_cmd(["git", "remote", "add", name, remote_url], cwd=repo_dir, secrets=secrets, check=True)


def list_local_refs(repo_dir: Path, prefix: str, *, secrets: Iterable[str]) -> Dict[str, str]:
    cp = run_cmd(
        ["git", "for-each-ref", "--format=%(refname) %(objectname)", prefix],
        cwd=repo_dir,
        secrets=secrets,
        check=True,
    )
    refs: Dict[str, str] = {}
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        refs[parts[0]] = parts[1]
    return refs


def list_remote_refs(repo_dir: Path, remote: str, kind: str, *, secrets: Iterable[str]) -> Dict[str, str]:
    if kind == "heads":
        cmd = ["git", "ls-remote", "--heads", remote]
    elif kind == "tags":
        cmd = ["git", "ls-remote", "--tags", remote]
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    cp = run_cmd(cmd, cwd=repo_dir, secrets=secrets, check=True)
    refs: Dict[str, str] = {}
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        sha, ref = parts
        # Ignore peeled tag helper refs from ls-remote output.
        if ref.endswith("^{}"):
            continue
        refs[ref] = sha
    return refs


def build_ref_plan(kind: str, local_refs: Dict[str, str], remote_refs: Dict[str, str]) -> RefPlan:
    upsert = sorted([ref for ref, sha in local_refs.items() if remote_refs.get(ref) != sha])
    delete = sorted([ref for ref in remote_refs.keys() if ref not in local_refs])
    return RefPlan(
        kind=kind,
        local_count=len(local_refs),
        remote_count=len(remote_refs),
        upsert=upsert,
        delete=delete,
    )


def batched(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def push_ref_batches(
    repo_dir: Path,
    *,
    remote: str,
    refs: List[str],
    is_delete: bool,
    force_upsert: bool,
    batch_size: int,
    dry_run: bool,
    secrets: Iterable[str],
    retry_cfg: Dict[str, float],
    label: str,
) -> int:
    if not refs:
        return 0

    pushed = 0
    batches = batched(refs, batch_size)
    for idx, chunk in enumerate(batches, start=1):
        if is_delete:
            refspecs = [f":{r}" for r in chunk]
        else:
            if force_upsert:
                refspecs = [f"+{r}:{r}" for r in chunk]
            else:
                refspecs = [f"{r}:{r}" for r in chunk]

        cmd = ["git", "push", "--progress"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend([remote, *refspecs])
        operation = f"{label} batch {idx}/{len(batches)}"
        logging.info("%s (%s refs)", operation, len(chunk))
        run_git_with_retry(
            cmd,
            cwd=repo_dir,
            secrets=secrets,
            operation=operation,
            **retry_cfg,
        )
        pushed += len(chunk)
    return pushed


def maybe_sync_lfs(
    repo_dir: Path,
    *,
    remote: str,
    enabled: bool,
    strict_lfs: bool,
    dry_run: bool,
    secrets: Iterable[str],
    retry_cfg: Dict[str, float],
) -> Tuple[bool, str]:
    if not enabled:
        return False, "disabled"
    if not git_lfs_available():
        msg = "git-lfs not installed; skipping LFS sync."
        if strict_lfs:
            raise SyncError(msg)
        logging.warning(msg)
        return False, "git-lfs-not-installed"

    # Default hardening: disable LFS lock verification to avoid /locks/verify timeouts.
    # This is safe for migration use-cases where lock checks are not required.
    run_cmd(
        ["git", "config", "lfs.locksverify", "false"],
        cwd=repo_dir,
        secrets=secrets,
        check=False,
    )
    remote_url_cp = run_cmd(
        ["git", "remote", "get-url", remote],
        cwd=repo_dir,
        secrets=secrets,
        check=False,
    )
    remote_url = (remote_url_cp.stdout or "").strip()
    if remote_url:
        endpoint = remote_url.rstrip("/") + "/info/lfs"
        run_cmd(
            ["git", "config", f"lfs.{endpoint}.locksverify", "false"],
            cwd=repo_dir,
            secrets=secrets,
            check=False,
        )
    logging.info("Configured LFS lock verification disabled for migration push.")

    logging.info("Running Git LFS fetch from origin.")
    try:
        run_git_with_retry(
            ["git", "lfs", "fetch", "--all", "origin"],
            cwd=repo_dir,
            secrets=secrets,
            operation="git lfs fetch --all origin",
            **retry_cfg,
        )
    except SyncError as exc:
        if strict_lfs:
            raise
        logging.warning("LFS fetch failed; continuing because strict LFS is off. %s", exc)
        return False, "lfs-fetch-failed"

    if dry_run:
        logging.info("Dry-run: skipping Git LFS push.")
        return True, "dry-run"

    logging.info("Running Git LFS push to %s.", remote)
    try:
        run_git_with_retry(
            ["git", "lfs", "push", "--all", remote],
            cwd=repo_dir,
            secrets=secrets,
            operation=f"git lfs push --all {remote}",
            **retry_cfg,
        )
        return True, "synced"
    except SyncError as exc:
        if strict_lfs:
            raise
        logging.warning("LFS push failed; continuing because strict LFS is off. %s", exc)
        return True, "lfs-push-failed"


def maybe_migrate_large_files_to_lfs(
    repo_dir: Path,
    *,
    threshold: str,
    secrets: Iterable[str],
    retry_cfg: Dict[str, float],
) -> None:
    if not git_lfs_available():
        raise SyncError("Cannot auto-migrate large files to LFS because git-lfs is not installed.")
    logging.warning(
        "Running history rewrite to move files above %s to LFS. This changes commit SHAs.",
        threshold,
    )
    run_git_with_retry(
        ["git", "lfs", "migrate", "import", "--everything", f"--above={threshold}"],
        cwd=repo_dir,
        secrets=secrets,
        operation="git lfs migrate import",
        **retry_cfg,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Idempotent large-repo mirror sync from GitLab to GitHub.")
    p.add_argument("--gitlab-repo-url", required=True, help="Source GitLab repository URL.")
    p.add_argument("--github-repo-url", required=True, help="Target GitHub repository URL.")
    p.add_argument(
        "--workdir",
        default=str(Path(__file__).resolve().parent / "workdir"),
        help="Directory where local mirror repositories are stored.",
    )
    p.add_argument(
        "--mirror-dir",
        default="",
        help="Optional exact mirror directory path. If omitted, derived from source repo name under --workdir.",
    )
    p.add_argument("--gitlab-token-env", default="GITLAB_TOKEN", help="Env var name for GitLab token.")
    p.add_argument(
        "--github-token-env",
        default="GITHUB_GIT_PUSH_TOKEN",
        help="Env var name for GitHub token used for push/API auth.",
    )
    p.add_argument("--batch-size", type=int, default=100, help="Refs per push batch.")
    p.add_argument("--max-attempts", type=int, default=6, help="Max retry attempts for retriable failures.")
    p.add_argument("--initial-backoff-seconds", type=float, default=5.0, help="Initial retry delay.")
    p.add_argument("--backoff-multiplier", type=float, default=2.0, help="Retry backoff multiplier.")
    p.add_argument("--max-backoff-seconds", type=float, default=300.0, help="Max retry delay.")
    p.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=10.0,
        help="Heartbeat interval for long-running git commands.",
    )
    p.add_argument("--no-lfs", action="store_true", help="Disable Git LFS fetch/push.")
    p.add_argument(
        "--force-lfs-sync",
        action="store_true",
        help="Force LFS fetch/push even when no refs changed in this run.",
    )
    p.add_argument("--strict-lfs", action="store_true", help="Fail run if LFS sync fails.")
    p.add_argument(
        "--migrate-large-files-to-lfs",
        dest="migrate_large_files_to_lfs",
        action="store_true",
        default=True,
        help="Auto-migrate >100MB blobs to LFS on GitHub rejection (default: enabled).",
    )
    p.add_argument(
        "--no-migrate-large-files-to-lfs",
        dest="migrate_large_files_to_lfs",
        action="store_false",
        help="Disable automatic LFS migration fallback on >100MB GitHub rejection.",
    )
    p.add_argument(
        "--lfs-migrate-above",
        default="100MB",
        help="Threshold for --migrate-large-files-to-lfs (example: 100MB).",
    )
    p.add_argument(
        "--no-large-file-report",
        action="store_true",
        help="Disable default terminal table report for blobs above LFS threshold.",
    )
    p.add_argument(
        "--prompt-before-lfs-migrate",
        dest="prompt_before_lfs_migrate",
        action="store_true",
        default=True,
        help="Ask for confirmation before LFS history rewrite (default: enabled).",
    )
    p.add_argument(
        "--no-prompt-before-lfs-migrate",
        dest="prompt_before_lfs_migrate",
        action="store_false",
        help="Do not prompt before LFS history rewrite.",
    )
    p.add_argument("--dry-run", action="store_true", help="Compute and print push plan without changing remote.")
    p.add_argument(
        "--skip-fetch-origin",
        action="store_true",
        help="Skip fetching origin before push planning. Use when caller already fetched and possibly rewrote history.",
    )
    p.add_argument(
        "--force-upsert",
        action="store_true",
        help="Force-update branch/tag refs on GitHub during upsert batches (+src:dst).",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    log_file = script_dir / "results" / "large_repo_migration" / "output" / f"mirror_sync_{now_utc_tag()}.log"
    configure_logging(log_file, args.verbose)

    ensure_git_available()

    gl_token = os.getenv(args.gitlab_token_env, "").strip()
    gh_token = os.getenv(args.github_token_env, "").strip()
    if not gl_token:
        raise SyncError(f"Missing GitLab token. Set env var: {args.gitlab_token_env}")
    if not gh_token:
        raise SyncError(f"Missing GitHub token. Set env var: {args.github_token_env}")

    src_url = normalize_git_url(args.gitlab_repo_url)
    dst_url = normalize_git_url(args.github_repo_url)
    src_auth = with_basic_auth(src_url, "oauth2", gl_token)
    dst_auth = with_basic_auth(dst_url, "x-access-token", gh_token)

    workdir = Path(args.workdir).resolve()
    mirror_dir = Path(args.mirror_dir).resolve() if args.mirror_dir else workdir / repo_folder_name(src_url)

    secrets = {
        gl_token,
        gh_token,
        quote(gl_token, safe=""),
        quote(gh_token, safe=""),
    }

    retry_cfg = {
        "max_attempts": args.max_attempts,
        "initial_backoff_seconds": args.initial_backoff_seconds,
        "backoff_multiplier": args.backoff_multiplier,
        "max_backoff_seconds": args.max_backoff_seconds,
        "progress_interval_seconds": args.progress_interval_seconds,
    }
    threshold_bytes = parse_size_to_bytes(args.lfs_migrate_above)

    logging.info("Starting mirror sync.")
    logging.info("GitLab source: %s", src_url)
    logging.info("GitHub target: %s", dst_url)
    logging.info("Mirror path: %s", mirror_dir)
    logging.info(
        "Using tokens: %s=%s, %s=%s",
        args.gitlab_token_env,
        mask_token(gl_token),
        args.github_token_env,
        mask_token(gh_token),
    )

    ensure_mirror_repo(mirror_dir, src_auth, secrets=secrets, retry_cfg=retry_cfg)
    ensure_remote(mirror_dir, "origin", src_auth, secrets=secrets)
    ensure_remote(mirror_dir, "github", dst_auth, secrets=secrets)

    if args.skip_fetch_origin:
        logging.info("Skipping git fetch origin as requested (--skip-fetch-origin).")
    else:
        run_git_with_retry(
            ["git", "fetch", "origin", "--prune", "--prune-tags", "--tags", "--force"],
            cwd=mirror_dir,
            secrets=secrets,
            operation="git fetch origin",
            **retry_cfg,
        )

    run_git_with_retry(
        ["git", "remote", "prune", "github"],
        cwd=mirror_dir,
        secrets=secrets,
        operation="git remote prune github",
        **retry_cfg,
    )
    if args.force_upsert:
        logging.warning("Force-upsert enabled: branch/tag upserts will overwrite remote refs if SHA differs.")

    migrated_large_files = False
    large_rows: List[LargeBlob] = []
    branch_files: Dict[str, List[str]] = {}
    tag_files: Dict[str, List[str]] = {}
    affected_head_refs: Set[str] = set()
    affected_tag_refs: Set[str] = set()
    ignored_head_refs: Set[str] = set()
    ignored_tag_refs: Set[str] = set()
    no_branch_marker = "(no-branch-head)"
    no_tag_marker = "(no-tag-head)"

    need_initial_large_scan = (not args.no_large_file_report) or args.prompt_before_lfs_migrate
    if need_initial_large_scan:
        logging.info("Scanning repository for blobs >= %s", args.lfs_migrate_above)
        large_rows = scan_large_blobs(
            mirror_dir,
            threshold_bytes=threshold_bytes,
            progress_interval_seconds=args.progress_interval_seconds,
        )
        branch_files = build_branch_file_report(
            mirror_dir,
            large_rows,
            progress_interval_seconds=args.progress_interval_seconds,
        )
        tag_files = build_tag_file_report(
            mirror_dir,
            large_rows,
            progress_interval_seconds=args.progress_interval_seconds,
        )
        affected_head_refs = {
            f"refs/heads/{b}" for b in branch_files.keys() if b and b != no_branch_marker
        }
        affected_tag_refs = {
            f"refs/tags/{t}" for t in tag_files.keys() if t and t != no_tag_marker
        }
        logging.info(
            "Large-file scan complete: found %s blobs >= %s",
            len(large_rows),
            args.lfs_migrate_above,
        )
        if not args.no_large_file_report:
            print("\n=== LARGE FILE REPORT BY BRANCH (BEFORE SYNC) ===")
            print(render_branch_file_table(branch_files))
            print("=== END LARGE FILE REPORT BY BRANCH ===\n")
            print("\n=== LARGE FILE REPORT (BEFORE SYNC) ===")
            print(render_large_blob_table(large_rows))
            print("=== END LARGE FILE REPORT ===\n")

    # Decide large-file strategy before first push (interactive mode).
    if args.prompt_before_lfs_migrate and large_rows and not args.dry_run:
        action = choose_large_file_action(
            can_ignore=bool(affected_head_refs or affected_tag_refs),
        )
        if action == "cancel":
            raise SyncError(
                "Cancelled by user before push. No refs were pushed. "
                "You can re-run after customer confirmation."
            )
        if action == "ignore":
            if not affected_head_refs and not affected_tag_refs:
                raise SyncError(
                    "Ignore option selected, but no affected refs could be resolved from large files."
                )
            ignored_head_refs = set(affected_head_refs)
            ignored_tag_refs = set(affected_tag_refs)
            logging.warning(
                "Pre-push choice=ignore_large_refs. blocked_heads=%s blocked_tags=%s",
                len(ignored_head_refs),
                len(ignored_tag_refs),
            )
        elif action == "rewrite":
            logging.info("Pre-push choice=rewrite_to_lfs. Starting history rewrite now.")
            maybe_migrate_large_files_to_lfs(
                mirror_dir,
                threshold=args.lfs_migrate_above,
                secrets=secrets,
                retry_cfg=retry_cfg,
            )
            migrated_large_files = True
            # Re-scan after rewrite for updated reporting and ref mapping.
            large_rows = scan_large_blobs(
                mirror_dir,
                threshold_bytes=threshold_bytes,
                progress_interval_seconds=args.progress_interval_seconds,
            )
            branch_files = build_branch_file_report(
                mirror_dir,
                large_rows,
                progress_interval_seconds=args.progress_interval_seconds,
            )
            tag_files = build_tag_file_report(
                mirror_dir,
                large_rows,
                progress_interval_seconds=args.progress_interval_seconds,
            )
            affected_head_refs = {
                f"refs/heads/{b}" for b in branch_files.keys() if b and b != no_branch_marker
            }
            affected_tag_refs = {
                f"refs/tags/{t}" for t in tag_files.keys() if t and t != no_tag_marker
            }
            if not args.no_large_file_report:
                print("\n=== LARGE FILE REPORT BY BRANCH (AFTER PRE-PUSH LFS MIGRATE) ===")
                print(render_branch_file_table(branch_files))
                print("=== END LARGE FILE REPORT BY BRANCH ===\n")
                print("\n=== LARGE FILE REPORT (AFTER PRE-PUSH LFS MIGRATE) ===")
                print(render_large_blob_table(large_rows))
                print("=== END LARGE FILE REPORT ===\n")

    sync_attempt = 1
    max_sync_attempts = 5
    while sync_attempt <= max_sync_attempts:
        try:
            local_heads = list_local_refs(mirror_dir, "refs/heads", secrets=secrets)
            remote_heads = list_remote_refs(mirror_dir, "github", "heads", secrets=secrets)
            heads_plan = build_ref_plan("heads", local_heads, remote_heads)

            local_tags = list_local_refs(mirror_dir, "refs/tags", secrets=secrets)
            remote_tags = list_remote_refs(mirror_dir, "github", "tags", secrets=secrets)
            tags_plan = build_ref_plan("tags", local_tags, remote_tags)

            logging.info(
                "Plan heads: local=%s remote=%s upsert=%s delete=%s",
                heads_plan.local_count,
                heads_plan.remote_count,
                len(heads_plan.upsert),
                len(heads_plan.delete),
            )
            logging.info(
                "Plan tags: local=%s remote=%s upsert=%s delete=%s",
                tags_plan.local_count,
                tags_plan.remote_count,
                len(tags_plan.upsert),
                len(tags_plan.delete),
            )

            heads_upsert_refs = [r for r in heads_plan.upsert if r not in ignored_head_refs]
            heads_delete_refs = [r for r in heads_plan.delete if r not in ignored_head_refs]
            tags_upsert_refs = [r for r in tags_plan.upsert if r not in ignored_tag_refs]
            tags_delete_refs = [r for r in tags_plan.delete if r not in ignored_tag_refs]

            if ignored_head_refs or ignored_tag_refs:
                logging.warning(
                    "Ignore mode active: skipped refs heads(upsert=%s delete=%s) tags(upsert=%s delete=%s)",
                    len(heads_plan.upsert) - len(heads_upsert_refs),
                    len(heads_plan.delete) - len(heads_delete_refs),
                    len(tags_plan.upsert) - len(tags_upsert_refs),
                    len(tags_plan.delete) - len(tags_delete_refs),
                )

            heads_upsert = push_ref_batches(
                mirror_dir,
                remote="github",
                refs=heads_upsert_refs,
                is_delete=False,
                force_upsert=args.force_upsert,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                secrets=secrets,
                retry_cfg=retry_cfg,
                label="push heads",
            )
            heads_delete = push_ref_batches(
                mirror_dir,
                remote="github",
                refs=heads_delete_refs,
                is_delete=True,
                force_upsert=args.force_upsert,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                secrets=secrets,
                retry_cfg=retry_cfg,
                label="delete heads",
            )
            tags_upsert = push_ref_batches(
                mirror_dir,
                remote="github",
                refs=tags_upsert_refs,
                is_delete=False,
                force_upsert=args.force_upsert,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                secrets=secrets,
                retry_cfg=retry_cfg,
                label="push tags",
            )
            tags_delete = push_ref_batches(
                mirror_dir,
                remote="github",
                refs=tags_delete_refs,
                is_delete=True,
                force_upsert=args.force_upsert,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                secrets=secrets,
                retry_cfg=retry_cfg,
                label="delete tags",
            )

            lfs_enabled = not args.no_lfs
            should_run_lfs = (heads_upsert + tags_upsert) > 0 or args.force_lfs_sync
            if lfs_enabled and should_run_lfs:
                lfs_done, lfs_status = maybe_sync_lfs(
                    mirror_dir,
                    remote="github",
                    enabled=lfs_enabled,
                    strict_lfs=args.strict_lfs,
                    dry_run=args.dry_run,
                    secrets=secrets,
                    retry_cfg=retry_cfg,
                )
            elif lfs_enabled and not should_run_lfs:
                lfs_done, lfs_status = False, "skipped-no-ref-upserts"
                logging.info("Skipping LFS sync because no branch/tag upserts detected in this run.")
            else:
                lfs_done, lfs_status = False, "disabled"

            logging.info(
                "Sync summary: heads_upserted=%s heads_deleted=%s tags_upserted=%s tags_deleted=%s lfs=%s(%s) migrated_large_files=%s",
                heads_upsert,
                heads_delete,
                tags_upsert,
                tags_delete,
                lfs_done,
                lfs_status,
                migrated_large_files,
            )
            logging.info("Completed successfully. Log file: %s", log_file)
            return 0
        except LargeFileLimitError as exc:
            if args.migrate_large_files_to_lfs and not migrated_large_files:
                logging.warning("%s", exc)
                if not large_rows:
                    large_rows = scan_large_blobs(
                        mirror_dir,
                        threshold_bytes=threshold_bytes,
                        progress_interval_seconds=args.progress_interval_seconds,
                    )
                if not branch_files:
                    branch_files = build_branch_file_report(
                        mirror_dir,
                        large_rows,
                        progress_interval_seconds=args.progress_interval_seconds,
                    )
                if not tag_files:
                    tag_files = build_tag_file_report(
                        mirror_dir,
                        large_rows,
                        progress_interval_seconds=args.progress_interval_seconds,
                    )
                affected_head_refs = {
                    f"refs/heads/{b}" for b in branch_files.keys() if b and b != no_branch_marker
                }
                affected_tag_refs = {
                    f"refs/tags/{t}" for t in tag_files.keys() if t and t != no_tag_marker
                }

                action = "rewrite"
                if args.prompt_before_lfs_migrate:
                    action = choose_large_file_action(
                        can_ignore=bool(affected_head_refs or affected_tag_refs),
                    )
                if action == "cancel":
                    raise SyncError(
                        "LFS rewrite cancelled by user. No history rewrite was done. "
                        "You can re-run after customer confirmation."
                    )
                if action == "ignore":
                    if not affected_head_refs and not affected_tag_refs:
                        raise SyncError(
                            "Ignore option selected, but no affected refs could be resolved from large files."
                        )
                    ignored_head_refs = set(affected_head_refs)
                    ignored_tag_refs = set(affected_tag_refs)
                    logging.warning(
                        "Ignoring affected refs and continuing: blocked_heads=%s blocked_tags=%s",
                        len(ignored_head_refs),
                        len(ignored_tag_refs),
                    )
                    sync_attempt += 1
                    continue
                maybe_migrate_large_files_to_lfs(
                    mirror_dir,
                    threshold=args.lfs_migrate_above,
                    secrets=secrets,
                    retry_cfg=retry_cfg,
                )
                if not args.no_large_file_report:
                    logging.info("Re-scanning blobs after LFS migration rewrite.")
                    post_rows = scan_large_blobs(
                        mirror_dir,
                        threshold_bytes=threshold_bytes,
                        progress_interval_seconds=args.progress_interval_seconds,
                    )
                    post_branch_files = build_branch_file_report(
                        mirror_dir,
                        post_rows,
                        progress_interval_seconds=args.progress_interval_seconds,
                    )
                    post_tag_files = build_tag_file_report(
                        mirror_dir,
                        post_rows,
                        progress_interval_seconds=args.progress_interval_seconds,
                    )
                    logging.info(
                        "Post-migration large-file scan complete: found %s blobs >= %s",
                        len(post_rows),
                        args.lfs_migrate_above,
                    )
                    print("\n=== LARGE FILE REPORT BY BRANCH (AFTER LFS MIGRATE) ===")
                    print(render_branch_file_table(post_branch_files))
                    print("=== END LARGE FILE REPORT BY BRANCH ===\n")
                    print("\n=== LARGE FILE REPORT (AFTER LFS MIGRATE) ===")
                    print(render_large_blob_table(post_rows))
                    print("=== END LARGE FILE REPORT ===\n")
                    logging.info("Post-migration affected tags (>= threshold blobs): %s", len(post_tag_files))
                ignored_head_refs.clear()
                ignored_tag_refs.clear()
                migrated_large_files = True
                sync_attempt += 1
                continue
            raise

    raise SyncError("Sync did not complete successfully.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SyncError as exc:
        logging.error("%s", exc)
        raise SystemExit(1)
