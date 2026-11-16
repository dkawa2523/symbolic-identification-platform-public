from importlib import import_module

ENGINES = {
    "mean": ".dictionary",
    "linear": ".dictionary",
    "dictionary": ".dictionary",
    "known_form": ".known_form",
    "template_enumeration": ".template_enumeration",
    "pysr": ".pysr",
    "physo": ".physo",
    "torchsisso": ".torchsisso",
    "proged": ".proged",
    "parfam": ".parfam",
    "pse": ".pse",
    "flash_ansr": ".flash_ansr",
    "pysips": ".pysips",
    "rasr": ".rasr",
    "e2e": ".e2e",
    "e2e_template": ".e2e_template",
    "flash_template": ".flash_template",
    "rasr_template": ".rasr_template",
}


def register_engine(name, module):
    """Explicit installed adapter registration; no artifact-driven imports."""
    if (
        not isinstance(name, str)
        or not name
        or name in ENGINES
        or not isinstance(module, str)
        or not module
        or module.startswith(".")
    ):
        raise ValueError("New engine needs a unique name and fully qualified installed module")
    native = import_module(module)
    if not callable(getattr(native, "discover", None)) or not callable(getattr(native, "capabilities", None)):
        raise ValueError("Adapter must implement capabilities and discover")
    ENGINES[name] = module


def get_engine(name):
    if name not in ENGINES:
        raise ValueError(f"Unsupported engine {name}; available: {sorted(ENGINES)}")
    return import_module(ENGINES[name], __package__)
