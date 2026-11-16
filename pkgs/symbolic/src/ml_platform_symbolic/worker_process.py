"""Lifecycle of one task-owned worker; never stop unrelated processes."""

import os
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path


def stop_worker(process):
    """Stop the owned POSIX group (or live Windows process tree), then reap."""
    # Timeout handlers may request cleanup before recovering a snapshot. The
    # enclosing context must not repeat a failed/denied stop on exit.
    if getattr(process, "_symbolic_stop_attempted", False):
        error = getattr(process, "_symbolic_stop_error", None)
        if error is not None:
            raise error
        return
    process._symbolic_stop_attempted = True
    try:
        _stop_and_reap(process)
    except Exception as error:
        process._symbolic_stop_error = error
        raise


def _stop_and_reap(process):
    if os.name == "nt":
        if process.poll() is None:
            command = str(Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32/taskkill.exe")
            result = subprocess.run(
                [command, "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=10,
                text=True,
                errors="replace",
            )
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    f"Owned worker {process.pid} cleanup unconfirmed: taskkill exit {result.returncode}: {detail}"
                )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


@contextmanager
def worker_process(command, **options):
    """Cleanup also runs on parent cancellation, not only TimeoutExpired.

    POSIX native children must remain in the assigned session/group. A host
    crash/SIGKILL and intentionally detached children are not Python guarantees.
    """
    process = subprocess.Popen(command, start_new_session=os.name != "nt", **options)
    primary_error = None
    try:
        yield process
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            stop_worker(process)
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"Owned-worker cleanup unconfirmed: {cleanup_error}")
