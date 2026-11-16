"""One task-owned evaluator process. Import/dependency failures are never successful evidence."""

import platform
import sys
import traceback
from importlib import import_module
from pathlib import Path

from ..data import digest, read_json, write_json
from ..domains import initialize_runtime_extensions
from .api import METHODS, load_study


def main():
    job = read_json(sys.argv[1])
    try:
        initialize_runtime_extensions(job.get("runtime_extensions", []))
        context, records, rows, arrays = load_study(job["study"])
        context["study_sha256"] = digest(Path(job["study"]) / "study.json")
        backend = import_module(
            job.get("backend_module", METHODS.get(job["method"], f"{__package__}.backends.unavailable"))
        )
        result = backend.run(context, records, rows, arrays, job["directory"], job["options"])
        result["axis"] = job.get("axis")
        result["backend_source_sha256"] = digest(backend.__file__)
        result["python"] = platform.python_version()
    except ImportError as exc:
        result = {"outcome": "unavailable", "reason": str(exc), "results": []}
    except Exception as exc:
        traceback.print_exc()
        result = {"outcome": "failed", "reason": f"{type(exc).__name__}: {exc}", "results": []}
    write_json(job["directory"] + "/result.json", result)


if __name__ == "__main__":
    main()
