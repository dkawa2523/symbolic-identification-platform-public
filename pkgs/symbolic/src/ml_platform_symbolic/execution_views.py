"""Named data-only execution views. Never invent predictions for residual-only relations."""

from copy import deepcopy

from .data import value_digest as _identity


def bind_view(record, specification=None):
    result = deepcopy(record)
    kind = record["kind"]
    defaults = {
        "explicit": "value",
        "ensemble": "weighted_mean",
        "implicit": "projection" if record.get("predictor") else "residual",
        "native_result": "native_only",
    }
    specification = deepcopy(specification or {"name": defaults[kind], "settings": {}})
    if set(specification) != {"name", "settings"} or not isinstance(specification["settings"], dict):
        raise ValueError("An execution view has name and explicit settings")
    name, settings = specification["name"], specification["settings"]
    allowed = {
        "explicit": {"value"},
        "ensemble": {"weighted_mean", "weighted_quantile"},
        "implicit": {"projection", "residual"},
        "native_result": {"native_only"},
    }
    if name not in allowed[kind]:
        raise ValueError("Execution view is unavailable for this native output kind")
    if name == "weighted_quantile":
        if set(settings) != {"q"} or not 0 < settings["q"] < 1:
            raise ValueError("Weighted structural quantile needs q in (0,1); not an observation interval")
    elif settings:
        raise ValueError("This view has no configurable numerical settings")
    if name == "projection" and record.get("predictor") is None:
        raise ValueError("Implicit relation has no declared target-free branch projection")
    state = {
        key: record[key]
        for key in (
            "equation",
            "members",
            "residual",
            "predictor",
            "native_member",
            "target",
            "variables",
            "input_geometry",
            "native_payload",
        )
        if key in record
    }
    result["execution_view"] = specification
    result["subject"] = {
        "relation_ref": record["id"],
        "relation_state_sha256": _identity(state),
        "view": name,
        "settings_ref": _identity(settings),
    }
    return result


def verify_subject(record):
    if "subject" in record or "execution_view" in record:
        expected = bind_view(record, record.get("execution_view"))["subject"]
        if "subject" in record and expected != record["subject"]:
            raise ValueError("Frozen relation/view/settings differ from the recorded execution subject")
