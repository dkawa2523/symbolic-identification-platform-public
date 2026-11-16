"""One CPU thread setting for isolated engine workers, never a concurrency API."""

import os
from contextlib import contextmanager


def thread_count(options):
    count = options.get("threads", 1)
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if type(count) is not int or not 1 <= count <= available:
        raise ValueError(f"threads must be an integer from 1 to {available} available logical CPUs")
    return count


@contextmanager
def torch_thread_scope(options):
    """Restore process-global torch setting on success/error; quota is separate."""
    count = thread_count(options)
    import torch
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(count)
        yield count
    finally:
        torch.set_num_threads(previous)
