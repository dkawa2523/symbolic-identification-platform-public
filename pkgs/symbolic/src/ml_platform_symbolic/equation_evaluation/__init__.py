"""Solver-neutral equation evaluation. Optional evaluators are imported only on request."""

from .api import create_study, evaluate_equations, load_study

__all__ = ["create_study", "evaluate_equations", "load_study"]
