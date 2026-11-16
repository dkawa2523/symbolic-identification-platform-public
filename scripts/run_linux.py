"""Run the common package CLI in a task-owned local Linux image. Standard library only."""

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory


def checked(arguments):
    # A blocking Windows wait delays Python's Ctrl-C handler until Docker exits.
    process = subprocess.Popen(arguments)  # Fixed argument list; no shell or credentials.
    try:
        while True:
            try:
                code = process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException as error:
        # Reap this invocation's client. invoke() separately stops its owned container.
        try:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
        except Exception as cleanup_error:
            error.add_note(f"Owned Docker client cleanup unconfirmed: {cleanup_error}")
        raise
    if code:
        raise subprocess.CalledProcessError(code, arguments)


def main():
    project = Path(__file__).resolve().parents[1]
    profiles = json.loads((project / "containers/profiles.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profiles, default="core")
    parser.add_argument("--build", action="store_true", help="Build the selected pinned runtime first")
    parser.add_argument("--memory", default="10g", help="Container memory cap: positive bytes or integer k/m/g (default 10g)")
    parser.add_argument(
        "--setup-network", action="store_true", help="Explicit asset setup only; forbidden in task mode"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--discovery-only",
        action="store_true",
        help="Compatibility spelling for the default: stop after discovery; use study/compare/choose next",
    )
    mode.add_argument("--compare", action="store_true", help="Compare Selection equations and export PNG/tables; never choose or read Final")
    parser.add_argument("--config", help="Run all configured engines using their runtime families and shared stages")
    parser.add_argument("--run-dir", help="New task directory within output-root; required with config")
    parser.add_argument("--output-root", default="results", help="Writable root: results, build or data; pretrained requires explicit setup")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command following --, e.g. python -m pytest -q")
    args = parser.parse_args()
    if args.compare and not args.config:
        parser.error("--compare requires --config")
    args.memory = args.memory.lower()
    if re.fullmatch(r"[1-9][0-9]*[kmg]?", args.memory) is None:
        parser.error("--memory requires positive bytes or an integer with k/m/g suffix")
    output = (project / args.output_root).resolve()
    allowed = [project / "results", project / "build", project / "data"]
    if args.setup_network and not args.config:
        allowed.append(project / "pretrained")
    if not any(output.is_relative_to(path) for path in allowed):
        raise ValueError("output-root must remain inside results, build or data; pretrained requires explicit --setup-network")
    if args.compare and args.run_dir and not (project / args.run_dir).resolve().parent.parent.is_relative_to(output):
        parser.error("--compare needs its output parent inside output-root for atomic staging; use the default output-root or a nested calculation directory")
    output.mkdir(parents=True, exist_ok=True)
    mount_arguments = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--pull=never",
        "--network",
        "bridge" if args.setup_network else "none",
        "--cpus",
        "4",
        "--memory",
        args.memory,
        "--mount",
        f"type=bind,source={project},target=/work,readonly",
        "--workdir",
        "/work",
    ]
    # Computation and its readable output share one result tree. Development
    # checks use an explicitly selected build tree; inputs remain read-only.
    writable = [output]
    for path in writable:
        if path.resolve() != path:
            raise ValueError("Writable roots must not redirect outside their declared project paths")
        path.mkdir(parents=True, exist_ok=True)
        mount_arguments.extend([
            "--mount", f"type=bind,source={path},target=/work/{path.relative_to(project).as_posix()}"
        ])
    built, images = set(), {}
    execution_file = None
    execution = {"images": images, "cpus": 4, "memory_limit": args.memory, "network": "none", "status": "starting"}

    def persist_execution():
        if execution_file is not None:
            execution_file.write_text(json.dumps(execution, indent=2), encoding="utf-8")

    def invoke(name, command):
        profile = profiles[name]
        if args.build and name not in built:
            checked(
                [
                    "docker",
                    "build",
                    "--file",
                    str(project / profile["dockerfile"]),
                    "--target",
                    profile["target"],
                    "--tag",
                    profile["image"],
                    str(project),
                ]
            )
            built.add(name)
        if name not in images:
            images[name] = subprocess.check_output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", profile["image"]], text=True
            ).strip()
        gpu = ["--gpus", "all"] if profile["gpu"] else []
        persist_execution()
        runtime = project / "runtime"
        runtime.mkdir(exist_ok=True)
        container_temp = TemporaryDirectory(prefix=".docker-", dir=runtime)
        cid_file = Path(container_temp.name) / "container.cid"
        try:
            checked(
                [
                    *mount_arguments,
                    *gpu,
                    "--cidfile",
                    str(cid_file),
                    "--env",
                    f"SYMBOLIC_CONTAINER_IMAGE_ID={images[name]}",
                    images[name],
                    *command,
                ]
            )
        except BaseException as exc:
            # The CLI and the daemon have separate lifetimes. Stop only the
            # container whose ID Docker wrote for this invocation, never all
            # running containers. A denied cleanup is recorded, not retried.
            cleanup = {"identity_file": str(cid_file), "attempted": False}
            if cid_file.is_file():
                cid = cid_file.read_text(encoding="utf-8").strip()
                if re.fullmatch(r"[a-f0-9]{64}", cid):
                    cleanup.update(container_id=cid, attempted=True)
                    try:
                        stopped = subprocess.run(
                            ["docker", "stop", "--time", "5", cid],
                            capture_output=True,
                            text=True,
                            timeout=15,
                            check=False,
                        )
                        cleanup.update(
                            return_code=stopped.returncode,
                            detail=stopped.stderr[-1000:],
                            confirmed=stopped.returncode == 0,
                        )
                        if stopped.returncode:
                            # --rm may already have removed a failed/interrupted container.
                            present = subprocess.run(
                                ["docker", "container", "ls", "--all", "--quiet", "--no-trunc", "--filter", f"id={cid}"],
                                capture_output=True, text=True, timeout=15, check=False,
                            )
                            if present.returncode == 0 and not present.stdout.strip():
                                cleanup.update(confirmed=True, state="already_absent")
                    except Exception as cleanup_error:
                        cleanup.update(confirmed=False, error=str(cleanup_error))
                else:
                    cleanup["error"] = "Invalid owned container identity; no stop attempted"
            execution["container_cleanup"] = cleanup
            persist_execution()
            if cleanup.get("attempted") and not cleanup.get("confirmed"):
                exc.add_note(
                    f"Owned container {cleanup.get('container_id')} cleanup is unconfirmed: "
                    f"{cleanup.get('error') or cleanup.get('detail') or 'Docker state unavailable'}. "
                    + ("See container_cleanup in the execution record." if execution_file else "Inspect this container before restarting.")
                )
            raise
        finally:
            container_temp.cleanup()

    if not args.config:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        invoke(args.profile, command)
        return
    if args.setup_network:
        raise ValueError("Discovery/evaluation task mode always has networking disabled")
    if not args.run_dir or args.command:
        raise ValueError("Task mode needs --run-dir and no trailing command")
    config = (project / args.config).resolve()
    directory = (project / args.run_dir).resolve()
    if args.compare and directory.name != "calculation":
        parser.error("--compare requires --run-dir results/NAME/calculation; outputs are published beside calculation")
    if not config.is_relative_to(project) or not config.is_file() or not directory.is_relative_to(output):
        raise ValueError("Config must be a project file and run-dir must remain within output-root")
    if args.compare and directory.parent.exists() and any(
        p.name not in {"calculation", "comparisons"} or not p.is_dir() or p.is_symlink()
        for p in directory.parent.iterdir()
    ):
        raise ValueError("Comparison output already exists; use a new result directory")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Run directory must be new/empty")
    container_dir = "/work/" + directory.relative_to(project).as_posix()
    common = ["python", "scripts/container_stage.py"]
    directory.mkdir(parents=True, exist_ok=True)
    execution_file = directory / "container_execution.json"
    execution.update(status="running", started_at=datetime.now(UTC).isoformat(), phase="prepare")
    persist_execution()
    try:
        invoke(
            "core",
            [
                *common,
                "prepare",
                "--config",
                "/work/" + config.relative_to(project).as_posix(),
                "--run-dir",
                container_dir,
            ],
        )
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        for engine in state["search"]["engines"]:
            family = next((name for name, info in profiles.items() if engine in info["engines"]), None)
            if family is None:
                raise ValueError(f"No accepted Linux runtime profile for {engine}")
            for seed in state["search"]["seeds"]:
                execution["phase"] = f"discover:{engine}/{seed}"
                persist_execution()
                invoke(
                    family, [*common, "discover", "--run-dir", container_dir, "--engine", engine, "--seed", str(seed)]
                )
        invoke("core", [*common, "summarize", "--run-dir", container_dir])
        execution.update(
            status="discovery_finished", phase="awaiting_explicit_scientific_comparison", final_accessed=False
        )
        if args.compare:
            execution["phase"] = "compare"
            persist_execution()
            invoke("core", [*common, "compare", "--run-dir", container_dir])
            execution.update(status="comparison_finished", phase="selection_comparison", final_accessed=False)
        persist_execution()
        return
    except BaseException as exc:
        execution.update(status="failed", error=str(exc))
        if isinstance(exc, Exception) and (directory / "state.json").exists():
            try:
                invoke("core", [*common, "summarize", "--run-dir", container_dir])
            except Exception as diagnostic:
                execution["diagnostic_summary_error"] = str(diagnostic)
        raise
    finally:
        execution["finished_at"] = datetime.now(UTC).isoformat()
        persist_execution()


if __name__ == "__main__":
    main()
