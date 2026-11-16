"""Assemble an allowlisted internal candidate from prebuilt, pinned wheels. No downloads/publication."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_wheel_payload(archive, source):
    """Require the complete runtime payload, not only the members a wheel happens to contain."""
    expected = {
        "ml_platform_symbolic/" + path.relative_to(source).as_posix(): path
        for path in source.rglob("*")
        if path.is_file() and path.suffix in {".py", ".jl", ".html"}
    }
    members = [
        name
        for name in archive.namelist()
        if name.startswith("ml_platform_symbolic/") and name.endswith((".py", ".jl", ".html"))
    ]
    if len(members) != len(set(members)):
        raise ValueError("Candidate wheel contains duplicate runtime members")
    missing, extra = sorted(expected.keys() - set(members)), sorted(set(members) - expected.keys())
    if missing or extra:
        raise ValueError(f"Candidate wheel/source inventory mismatch: missing={missing}, extra={extra}")
    for name, path in expected.items():
        if archive.read(name) != path.read_bytes():
            raise ValueError(f"Candidate wheel/source mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    release_version = tomllib.loads((project / "pkgs/symbolic/pyproject.toml").read_text())["project"]["version"]
    with zipfile.ZipFile(args.wheel) as archive:
        metadata = Parser().parsestr(
            archive.read(next(n for n in archive.namelist() if n.endswith("/METADATA"))).decode()
        )
        if metadata["Name"] != "ml-platform-symbolic" or metadata["Version"] != release_version:
            raise ValueError("Candidate wheel does not match the source release version")
        validate_wheel_payload(archive, project / "pkgs/symbolic/src/ml_platform_symbolic")
    output = args.output.resolve()
    if not any(output.is_relative_to((project / name).resolve()) for name in ("build",)):
        raise ValueError("Build output must remain inside this project's build directory")
    output.mkdir(parents=True, exist_ok=False)
    wheels = output / "wheelhouse"
    notices = output / "NOTICES"
    wheels.mkdir()
    notices.mkdir()
    packages, requirements = [], []
    inputs = [
        p
        for p in args.wheelhouse.glob("*.whl")
        if not p.name.lower().startswith(("setuptools-", "ml_platform_symbolic-"))
    ]
    inputs.append(args.wheel)
    for src in sorted(inputs):
        with zipfile.ZipFile(src) as archive:
            metadata = Parser().parsestr(
                archive.read(next(n for n in archive.namelist() if n.endswith("/METADATA"))).decode()
            )
            name, version = metadata["Name"], metadata["Version"]
            license_files = []
            for member in archive.namelist():
                if not member.endswith("/") and (
                    "license" in member.lower() or "notice" in member.lower() or "copying" in member.lower()
                ):
                    # Preserve nested bundled-library notices, not only top-level classifiers.
                    if ".." in Path(member).parts or Path(member).is_absolute():
                        raise ValueError("Unsafe wheel license path")
                    path = notices / name / member
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(member))
                    license_files.append(path.relative_to(output).as_posix())
            packages.append(
                {
                    "name": name,
                    "version": version,
                    "file": "wheelhouse/" + src.name,
                    "sha256": sha(src),
                    "requires": metadata.get_all("Requires-Dist", []),
                    "license_expression": metadata.get("License-Expression"),
                    "license_metadata": metadata.get("License"),
                    "license_files": license_files,
                    "source": "paired core commit91aa29a235527b9ecd6fc7f6d17101beb0cfe9a0"
                    if name == "ml-platform-core"
                    else "current local source"
                    if name == "ml-platform-symbolic"
                    else f"https://pypi.org/project/{name}/{version}/",
                }
            )
            requirements.append(f"{name}=={version} --hash=sha256:{sha(src)}")
        shutil.copy2(src, wheels / src.name)
    names = {p["name"].lower().replace("_", "-") for p in packages}
    if len(names) != len(packages) or not {"ml-platform-symbolic", "ml-platform-core"} <= names:
        raise ValueError("Duplicate/missing paired distributions")
    (output / "requirements-lean.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    (output / "THIRD_PARTY.json").write_text(
        json.dumps(
            {
                "bundled": packages,
                "not_bundled": [
                    "native search libraries",
                    "Julia",
                    "GPU runtimes",
                    "checkpoints/indexes",
                    "empirical datasets",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    setup = (project / "docs/user_manual.md").read_text(encoding="utf-8")
    offline = setup[setup.index("## オフライン配布と権利"):]
    (output / "README.md").write_text(
        "# オフライン利用\n\n"
        "通常のsource導入は native-source/docs/user_manual.md を参照してください。\n"
        "このdirectoryでは次の固定wheelhouse手順を使います。\n\n"
        + offline[offline.index("固定wheelhouse"):],
        encoding="utf-8",
    )
    shutil.copy2(project / "scripts/verify_internal_manifest.py", output / "verify_manifest.py")
    examples = output / "examples"
    examples.mkdir(parents=True)
    for name in ("basic.py", "prediction_intervals.py"):
        shutil.copy2(project / "examples" / name, examples / name)
    (output / "Dockerfile.lean").write_text(
        """FROM python@sha256:420310dd2ff7895895f0f1f9d15cae5a95dabceb8f1d6b9a23ef33c2c1c542c3
COPY wheelhouse /wheels
COPY requirements-lean.txt /requirements-lean.txt
RUN python -m pip install --no-index --find-links /wheels --require-hashes -r /requirements-lean.txt && python -m pip check
COPY examples /examples
WORKDIR /app
CMD ["symbolic", "--help"]
""",
        encoding="utf-8",
    )
    native = output / "native-source"
    subprocess.run([sys.executable, str(project / "scripts/export_source.py"), "--output", str(native)], check=True)
    (notices / "INTERNAL_RIGHTS_PENDING.md").write_text(
        "# Rights confirmation pending\n\nNo new license is granted by this file.\n"
        "Confirm permission for internal use, copying and handoff of this project and the paired core.\n"
        "A new OSS license or external publication permission is not being requested.\n"
        "Rights holders are not recorded; the core repository namespace is not proof of legal ownership.\n"
        "Third-party notices are retained as supplied, without legal guarantees. Optional native libraries/assets are not bundled.\n",
        encoding="utf-8",
    )
    (output / "RELEASE_STATUS.json").write_text(
        json.dumps(
            {
                "version": release_version,
                "platform": "Linux x86-64 Python3.11",
                "technical_validation": "pending",
                "rights": "pending_holder_confirmation",
                "distribution_authorized": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "version": release_version,
        "files": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
