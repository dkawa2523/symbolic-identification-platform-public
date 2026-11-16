"""Data-only result handoff. No search, scoring, choice, raw-data or code import.

A detached readout and its explicitly supplied reusable predictor/interval can
move together. This is not relocation of the live discovery/evaluation project.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path

from .data import digest, read_json, write_json

MAX_RESULT_ENTRY_BYTES = 100_000_000


def _bound_prediction(model, view):
    decision = view.get("decision") or {}
    if not decision.get("selected_id"):
        raise ValueError("A reusable predictor needs a displayed frozen selected subject")
    from .hypotheses import require_prediction_purpose

    require_prediction_purpose(decision.get("purpose"))
    metadata = model.metadata
    embedded = metadata.get("decision", {})
    candidate = metadata.get("candidate_id", metadata.get("selected_id", embedded.get("selected_id")))
    if candidate != decision["selected_id"]:
        raise ValueError("Reusable artifact belongs to a different selected candidate")
    allowed = {view.get("decision_sha256"), decision.get("frozen_choice_sha256")} - {None}
    if embedded != decision and metadata.get("decision_sha256") not in allowed:
        raise ValueError("Reusable artifact belongs to a different frozen decision")
    subject = getattr(model, "record", {}).get("subject") or embedded.get("subject")
    if subject != decision.get("subject"):
        raise ValueError("Reusable prediction view/state differs from the displayed subject")
    if hasattr(model, "equation") and model.equation != decision.get("equation"):
        raise ValueError("Reusable equation differs from the frozen displayed equation")
    for key, expected in decision.get("applicability_schema", {}).items():
        if model.schema.get(key) != expected:
            raise ValueError("Reusable prediction applicability differs from the displayed decision")
    if model.schema.get("unit_coordinates") != view["context"].get("unit_coordinates"):
        raise ValueError("Reusable prediction coordinate defaults differ from the displayed study")


def export_result(
    comparison_dir,
    *,
    run_dir,
    decision_dir=None,
    audit_dir=None,
    prediction=None,
    prediction_interval=None,
    coverage=None,
    representative_ids=None,
    supplemental_comparison=None,
):
    """Freeze an already evaluated readout and optional SAME-subject prediction.

    Only explicitly supplied prediction artifacts are copied. Native binaries,
    model assets, extensions and raw CSVs are not implicitly collected/executed.
    The readout still contains stored evaluation values; it is not anonymization.
    """
    from .hypotheses import load_prediction
    from .model import load_model
    from .prediction_intervals import load_prediction_interval
    from .report_data import build_report_view
    from .reporting import normalize_report_data, render_report_data

    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    view = normalize_report_data(
        build_report_view(
            comparison_dir,
            decision_dir=decision_dir,
            audit_dir=audit_dir,
            prediction_interval=prediction_interval,
            coverage=coverage,
            representative_ids=representative_ids,
            supplemental_comparison=supplemental_comparison,
        )
    )
    view["detached_result"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Stored evidence as of export. Paths/exposure are historical references, not live source checks or new scientific confirmation.",
    }
    predictor_path = None
    if prediction is not None:
        path = Path(prediction)
        predictor_path = "prediction.zip" if path.suffix == ".zip" else "prediction.json"
        shutil.copy2(path, directory / predictor_path)
        copied = directory / predictor_path
        model = load_model(copied) if copied.suffix == ".zip" else load_prediction(copied)
        _bound_prediction(model, view)
    if (view.get("decision") or {}).get("interval_sha256") and prediction_interval is None:
        raise ValueError("Calibrated-use handoff must include its assessed prediction_interval explicitly")
    if prediction_interval is not None:
        (directory / "interval").mkdir()
        for name in ("prediction.json", "interval.json"):
            shutil.copy2(Path(prediction_interval) / name, directory / "interval" / name)
        interval = load_prediction_interval(
            directory / "interval",
            decision=view.get("decision"),
            decision_sha256=view.get("decision_sha256"),
            displayed=view.get("interval"),
        )
        _bound_prediction(interval.model, view)
    write_json(directory / "report.json", view)
    render_report_data(view, output=directory / "report.html")
    files = {}
    for path in directory.rglob("*"):
        if path.is_file():
            if path.stat().st_size > MAX_RESULT_ENTRY_BYTES:
                raise ValueError("Result entry exceeds the reader size limit; no complete manifest created")
            files[path.relative_to(directory).as_posix()] = digest(path)
    write_json(
        directory / "result_manifest.json",
        {
            "schema_version": 1,
            "kind": "frozen_equation_result",
            "files": files,
            "predictor": predictor_path,
            "interval": "interval" if prediction_interval is not None else None,
            "study_sha256": view["study_sha256"],
            "comparison_sha256": view["comparison_sha256"],
            "decision_sha256": view.get("decision_sha256"),
            "scope": "Detached stored readout and named predictor only. Original paths remain provenance; no live project reanalysis or native-code relocation.",
            "omitted": [
                "source CSV files (stored evaluation values remain in the readout)",
                "native binaries/assets",
                "extension code",
                "live source bindings",
            ],
        },
    )
    return directory


def load_result(directory, *, load_predictors=True):
    """Verify a local handoff. Readout can avoid initializing predictor runtimes.

    File and calibration/decision identity are checked in both modes. Predictor
    runtime usability is checked only when loading the optional predictors.
    No mode initializes extensions from the artifact.
    """
    from .hypotheses import load_prediction
    from .model import load_model
    from .prediction_intervals import load_prediction_interval, read_interval_evidence

    directory = Path(directory).resolve()
    manifest = read_json(directory / "result_manifest.json")
    if manifest.get("kind") != "frozen_equation_result" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported result package")
    predictor = manifest.get("predictor")
    if predictor not in {None, "prediction.zip", "prediction.json"} or manifest.get("interval") not in {
        None,
        "interval",
    }:
        raise ValueError("Unsupported result prediction layout")
    expected = {"report.json", "report.html"}
    if predictor:
        expected.add(predictor)
    if manifest.get("interval"):
        expected.update({"interval/interval.json", "interval/prediction.json"})
    if set(manifest["files"]) != expected:
        raise ValueError("Result package has unexpected/missing artifact references")
    for name, fingerprint in manifest["files"].items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory)
            or path.stat().st_size > MAX_RESULT_ENTRY_BYTES
        ):
            raise ValueError("Result entry exceeds local data boundary")
        if digest(path) != fingerprint:
            raise ValueError(f"Result artifact changed: {name}")
    view = read_json(directory / "report.json")
    if any(view.get(key) != manifest.get(key) for key in ("study_sha256", "comparison_sha256", "decision_sha256")):
        raise ValueError("Result readout identity differs from its manifest")
    model = None
    if predictor and load_predictors:
        path = directory / predictor
        model = load_model(path) if path.suffix == ".zip" else load_prediction(path)
        _bound_prediction(model, view)
    interval = None
    if (view.get("decision") or {}).get("interval_sha256") and not manifest.get("interval"):
        raise ValueError("Calibrated-use handoff is missing its assessed interval")
    if manifest.get("interval"):
        if view.get("interval") is None:
            raise ValueError("Packaged interval has no displayed evidence")
        read_interval_evidence(
            directory / "interval",
            decision=view.get("decision"),
            decision_sha256=view.get("decision_sha256"),
            displayed=view["interval"],
        )
        if load_predictors:
            interval = load_prediction_interval(
                directory / "interval",
                decision=view.get("decision"),
                decision_sha256=view.get("decision_sha256"),
            )
            _bound_prediction(interval.model, view)
    return {
        "manifest": manifest,
        "view": view,
        "prediction": model,
        "interval": interval,
        "predictors_loaded": load_predictors,
    }
