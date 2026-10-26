"""Regression for #144750: observing a checkout must not invent an updater."""
from concurrent.futures import ThreadPoolExecutor
import errno
import threading

import pytest

from hermes_cli import gateway_runtime, update_lock


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    path = update_lock.checkout_lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    monkeypatch.setattr(update_lock, "checkout_lock_path", lambda *args: path)
    # Production reaches the kernel probe only when a marker exists (even a dead one).
    (tmp_path / update_lock.MARKER_NAME).write_text("malformed", encoding="utf-8")
    return root, path


def test_parallel_real_probe_does_not_fence_gateway(checkout, tmp_path, monkeypatch):
    root, _ = checkout
    acquired, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = update_lock._try_lock
    observer = threading.current_thread()

    def synchronized_try(fd):
        locked = original(fd)
        if threading.current_thread() is not observer and locked:
            acquired.set()
            assert release.wait(10), "second reader never reached the overlapping probe"
        elif threading.current_thread() is observer and not locked:
            release.set()
            assert finished.wait(10), "first reader did not close its real kernel lock"
        return locked

    def first_reader():
        try:
            return update_lock.checkout_lock_held(root)
        finally:
            finished.set()

    monkeypatch.setattr(update_lock, "_try_lock", synchronized_try)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(first_reader)
        try:
            assert acquired.wait(10), "first reader never acquired the real kernel lock"
            assert not gateway_runtime._update_fenced({tmp_path})
        finally:
            release.set()
        assert first.result(timeout=10) is False
    assert (tmp_path / update_lock.MARKER_NAME).is_file(), "discovery must remain read-only"


def test_stable_real_lock_still_fences_and_release_clears(checkout, tmp_path):
    _, path = checkout
    with path.open("rb") as holder:
        assert update_lock._try_lock(holder.fileno())
        try:
            assert gateway_runtime._update_fenced({tmp_path})
        finally:
            update_lock._unlock(holder.fileno())
    assert not gateway_runtime._update_fenced({tmp_path})


def test_unopenable_lock_file_still_fences(checkout, tmp_path, monkeypatch):
    _, path = checkout
    original = update_lock.os.open

    def failing_open(target, *args, **kwargs):
        if target == path:
            raise OSError(errno.EACCES, "probe open failed")
        return original(target, *args, **kwargs)

    monkeypatch.setattr(update_lock.os, "open", failing_open)
    assert gateway_runtime._update_fenced({tmp_path}) is True
