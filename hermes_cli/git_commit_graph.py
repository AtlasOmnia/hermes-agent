"""Keep a checkout's commit-graph fresh: the ``hermes -w`` pass and every ``hermes update``.

Generation numbers in the commit-graph keep ``git tag --merged HEAD`` (the version lookup) and
ancestry checks off a full history walk: on hermes-agent a graph two months behind cost 0.33 s
per lookup, a fresh one 0.04 s. Partial clones have git's own graph writers switched off
(gitlock._TREE0_MAINTENANCE_OFF, #127711), so nothing else refreshes it there. Without Bloom data
the write reads commits only, never trees, so it cannot start that lazy-fetch loop, and it costs
~3 s instead of ~27 s when changed paths would be recomputed.

Also home of the per-clone maintenance slot and the bounded git runner the ``-w`` repack shares,
so the update path imports this module and not all of ``worktree_ops``.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

from hermes_cli._subprocess_compat import (
    IS_WINDOWS,
    NO_LAZY_FETCH_ENV,
    bounded_probe_run,
    kill_process_tree,
    noninteractive_git_env,
    windows_detach_flags_without_breakaway,
    windows_detach_popen_kwargs,
)

logger = logging.getLogger(__name__)

_WRITE = ("commit-graph", "write", "--reachable", "--split", "--no-changed-paths", "--no-progress")
_MAX_AGE = 24 * 3600
_TIMEOUT = 600
_MIN_INTERVAL = 6 * 3600
_LOCK = "hermes-commit-graph.lock"
# A write with no new commits leaves git's files untouched; this stamp records that it succeeded,
# so an idle repo is not rewritten on every slot.
_FRESH_STAMP = "hermes-commit-graph.fresh"
# git deletes its lock and tmp files on SIGTERM; a SIGKILLed writer strands commit-graph-chain.lock
# and tmp_graph_*, and that lock fails every later write until the stale-lock sweep removes it.
_KILL_GRACE = 5.0
_CHILD = ("import sys; sys.path.insert(0, sys.argv[1]); "
          "from hermes_cli.git_commit_graph import refresh_in_child; refresh_in_child(*sys.argv[2:])")


def claim_maintenance_slot(git_dir: Path, lock_name: str, interval: float) -> bool:
    """Exactly one process per clone gets to run a maintenance job per ``interval``.

    The lock file's mtime is the stamp: younger than the interval means another launch is
    running the job (or just tried and timed out) — skip. A stale lock is taken over by ``replace``,
    which only one of N racing processes can win; the O_EXCL create then serializes against a
    process that found no lock at all.
    """
    lock = git_dir / lock_name
    try:
        st = lock.stat()
    except FileNotFoundError:
        pass
    else:
        if time.time() - st.st_mtime < interval:
            return False
        try:
            lock.replace(lock.with_suffix(".stale"))
        except OSError:
            return False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"{os.getpid()}\n")
    lock.with_suffix(".stale").unlink(missing_ok=True)
    return True


def run_bounded_git(repo_root: str, args: Sequence[str], timeout: float, env: Optional[dict] = None) -> Optional[int]:
    """Run a niced ``git *args`` whose whole process tree dies with the timeout or with us.

    Returns the exit code, or None when the timeout killed it; a failure is logged with git's last
    stderr line. ``repack`` forks ``pack-objects``; ``subprocess.run(timeout=)`` killed only the
    parent and left the grandchild packing for days, and a daemon thread's child outlived the CLI the
    same way. A new session/process group + ``atexit`` reaps both cases, SIGTERM first so git can
    delete its own lock and tmp files.
    """
    cmd = ["git", *args]
    if os.name == "posix":
        cmd = ["nice", "-n", "19", *cmd]
        group_kw: dict = {"process_group": 0}
    else:
        group_kw = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, cwd=repo_root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=err, env=env, **group_kw)

        def _reap() -> None:
            if proc.poll() is None:
                kill_process_tree(proc, grace=_KILL_GRACE)

        atexit.register(_reap)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _reap()
            logger.info("git %s exceeded %ds; killed", args[0], timeout)
            return None
        finally:
            atexit.unregister(_reap)
        if rc:
            err.seek(0)
            lines = err.read()[-4096:].decode("utf-8", "replace").strip().splitlines()
            # git's "File exists" ends on advice; the fatal/error line names the cause.
            cause = next((ln for ln in reversed(lines) if ln.startswith(("fatal:", "error:"))), lines[-1] if lines else "no output")
            logger.warning("git %s in %s failed (exit %d): %s", args[0], repo_root, rc, cause)
        return rc


def commit_graph_stale(git_dir: Path) -> bool:
    """No commit-graph, or none written or confirmed current for a day."""
    info = git_dir / "objects" / "info"
    graphs = [p.stat().st_mtime for p in (info / "commit-graph", info / "commit-graphs" / "commit-graph-chain")
              if p.is_file()]
    if not graphs:
        return True
    stamp = git_dir / _FRESH_STAMP
    newest = max(graphs + ([stamp.stat().st_mtime] if stamp.is_file() else []))
    return time.time() - newest > _MAX_AGE


def claim_commit_graph_refresh(git_dir: Path) -> bool:
    return commit_graph_stale(git_dir) and claim_maintenance_slot(git_dir, _LOCK, _MIN_INTERVAL)


def _commit_graph_disabled(repo_root: str) -> bool:
    """``core.commitGraph=false``: the owner turned graphs off in this repo; never write one there."""
    probe = bounded_probe_run(["git", "config", "--type=bool", "--get", "core.commitGraph"],
                              timeout=10, cwd=repo_root, env=noninteractive_git_env())
    return probe is not None and probe.stdout.strip() == "false"


def refresh_commit_graph(repo_root: str) -> None:
    """Clear what a killed writer stranded, then write the graph (bounded, niced, never lazy-fetching)."""
    from hermes_cli.gitlock import OBJECT_STORE_LOCK_NAMES, clear_stale_git_locks, clear_stale_tmp_packs

    if _commit_graph_disabled(repo_root):
        logger.info("core.commitGraph=false in %s; not writing a commit-graph", repo_root)
        return
    root = Path(repo_root)
    # Object-store debris only: ``hermes -w`` runs this in whatever repo it was launched from, where
    # index.lock / HEAD.lock belong to the user's own git (and pgrep can be blind to it).
    clear_stale_git_locks(root, names=OBJECT_STORE_LOCK_NAMES)
    clear_stale_tmp_packs(root)
    logger.info("git commit-graph missing or stale — writing it in background")
    if run_bounded_git(repo_root, _WRITE, _TIMEOUT, env={**noninteractive_git_env(), **NO_LAZY_FETCH_ENV}) == 0:
        (root / ".git" / _FRESH_STAMP).touch()


def refresh_in_child(repo_root: str, log_file: str) -> None:
    """Body of the detached child: a failure is appended to the home's agent.log; never raises.

    A plain handler, not ``setup_logging``: that reads config.yaml and seeds the home's layout,
    side effects a git maintenance child has no business having in a profile mid-update.
    ``delay`` opens the log only when there is a failure to record.
    """
    handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        refresh_commit_graph(repo_root)
    except Exception:
        logger.warning("Background commit-graph refresh of %s failed", repo_root, exc_info=True)


def spawn_commit_graph_refresh(repo_root: Path) -> bool:
    """Start the bounded commit-graph refresh in a detached child when it is due; never waits.

    ``hermes update`` is the one regular git pass an install checkout gets (``hermes -w`` is
    rarely launched from it), so without this its version lookups never get a graph. Detached
    because the write takes seconds and must neither delay the update nor die with it. A spawn that
    fails gives the slot back, so the next update retries instead of waiting out the interval.
    """
    from hermes_constants import get_hermes_home

    git_dir = Path(repo_root) / ".git"
    if not (git_dir / "objects").is_dir() or not claim_commit_graph_refresh(git_dir):
        return False
    code_root = Path(__file__).resolve().parents[1]
    argv = [sys.executable, "-c", _CHILD, str(code_root), str(repo_root), str(get_hermes_home() / "logs" / "agent.log")]
    kwargs: dict = dict(cwd=str(repo_root), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, close_fds=True)
    try:
        try:
            subprocess.Popen(argv, **kwargs, **windows_detach_popen_kwargs())
        except OSError:
            if not IS_WINDOWS:
                raise
            # A job object without BREAKAWAY_OK refuses CREATE_BREAKAWAY_FROM_JOB; retry without it.
            subprocess.Popen(argv, **kwargs, creationflags=windows_detach_flags_without_breakaway())
    except OSError:
        (git_dir / _LOCK).unlink(missing_ok=True)
        raise
    return True
