"""Subprocess entrypoint; numerical behavior lives in runner and the selected adapter."""

import sys

from .runner import _worker

if __name__ == "__main__":
    _worker(sys.argv[1], recover_pysr=len(sys.argv) == 3 and sys.argv[2] == "--recover-pysr")
