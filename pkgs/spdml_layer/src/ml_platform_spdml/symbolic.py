"""Optional transport for symbolic RunResult. Copy beside the platform's reports.py.

Call only with an explicitly created local/offline or authorized live Task.
This module owns SDK use; the numerical package never creates a Task.
"""

from pathlib import Path

import pandas as pd


def report_symbolic_result(task, result, *, register_model=False, model_factory=None):
    """Candidate artifacts stay on the Task; only the selected bundle can be registered."""
    from ml_platform_symbolic.data import read_json

    info = read_json(result.artifacts["model_info"])
    task.connect(info["provenance"]["config"], name="Symbolic config")
    logger = task.get_logger()
    for name, path in {**result.artifacts, **result.tables, **result.plots}.items():
        uploaded = task.upload_artifact(name=name, artifact_object=str(Path(path)), wait_on_upload=True)
        if uploaded is False:
            raise RuntimeError(f"Artifact upload failed: {name}")
    for name, value in result.metrics.items():
        scope = "selection" if name.startswith("selection_") else "final"
        logger.report_scalar(title=scope, series=name, value=value, iteration=0)
    for name, path in result.tables.items():
        logger.report_table(title="Symbolic", series=name, iteration=0, table_plot=pd.read_csv(path))
    logger.report_text(f"Selected {info['selected_candidate_id']}: {info['equation']}")
    model = None
    if register_model:
        if model_factory is None:
            from clearml import OutputModel

            model_factory = OutputModel
        model = model_factory(task=task, name="selected equation", framework="Custom")
        model.update_weights(weights_filename=str(result.artifacts["model_bundle"]), auto_delete_file=False)
    logger.flush(wait=True)
    return model


def offline_example(config, run_dir):
    """Explicit offline SDK exercise, containing no live server interaction."""
    from clearml import Task
    from ml_platform_symbolic import run_task

    Task.set_offline(offline_mode=True)
    task = Task.init(
        project_name="symbolic",
        task_name="offline",
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
        auto_resource_monitoring=False,
    )
    try:
        result = run_task(config, run_dir=run_dir)
        report_symbolic_result(task, result)
        return result
    finally:
        task.close()
