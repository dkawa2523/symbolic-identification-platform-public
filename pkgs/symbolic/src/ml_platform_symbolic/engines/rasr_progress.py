"""Completed RASR refinement evidence using the common atomic checkpoint boundary."""

import time
from copy import copy, deepcopy

import numpy as np

from .base import DiscoveryResult, checked_candidate


class NativeProgress:
    def __init__(self, request, context, exporter):
        self.request, self.context, self.exporter = request, context, exporter
        self.fitted = None
        self.started = time.monotonic()
        self.candidates, self.rejected = [], []
        self.summary = {
            "phase": "load_started",
            "completed_refinements": 0,
            "phase_seconds": {},
            "timing_basis": "cumulative elapsed seconds since model-load start, including checkpoint/parity overhead",
            "generated": None,
            "parsed": None,
            "checkpoint_scope": "completed native refinement and native final filters; no decoder prefixes or optimizer resume",
            "candidate_order": "original generation cand_id, not final native rank",
        }
        self("load_started")

    def __call__(self, phase, **fields):
        native = fields.pop("candidate", None)
        elapsed = time.monotonic() - self.started
        self.summary.update(phase=phase, elapsed_seconds=elapsed, **fields)
        if phase != "candidate_completed" and phase != "candidate_started":
            self.summary["phase_seconds"][phase] = elapsed
        if phase == "candidate_completed":
            self.summary["completed_refinements"] += 1
            identifier = fields["candidate_id"]
            if native is not None:
                try:
                    equation = self.exporter(native, self.fitted, self.request.variables, self.request.target)
                    oracle = copy(self.fitted)
                    oracle.results_, oracle.best_result_ = [native], native
                    candidate = checked_candidate(
                        "rasr",
                        identifier,
                        equation,
                        self.request,
                        lambda X: np.asarray(oracle.predict(X, nth_best_beam=0)).reshape(-1),
                    )
                    candidate.candidate_id += "/completed-refinement"
                    candidate.native_check.update(
                        native_generation_id=identifier,
                        native_final_filters_passed=True,
                        final_native_rank_available=False,
                    )
                    self.candidates.append(candidate)
                except (ValueError, TypeError, OverflowError) as exc:
                    self.rejected.append({"native_generation_id": identifier, "reason": str(exc)})
            else:
                self.rejected.append(
                    {
                        "native_generation_id": identifier,
                        "reason": "native refinement/final filters produced no candidate",
                    }
                )
        # The existing checkpoint owns atomic serialization and recovery. No second ledger.
        if self.context.checkpoint is not None:
            self.context.checkpoint(
                DiscoveryResult(
                    list(self.candidates),
                    "partial" if self.candidates else "timeout",
                    {
                        **deepcopy(self.summary),
                        "rejected_completed_candidates": deepcopy(self.rejected),
                        "native_execution_status": "partial",
                        "reason": "native search still in progress",
                    },
                )
            )
