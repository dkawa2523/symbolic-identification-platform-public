"""Small semantic results consumed by criteria, independent of analysis library.

Raw native profiles stay in their evaluator artifact. A validated conditional
interval is not a predictive interval or evidence of structural correctness.
"""

from .data import value_digest
from .units import EVIDENCE_COORDINATE_POLICY, unit_spec


def conditional_request(definition, *, confidence_level, observation_model):
    """Frozen meaning of a conditional interval, independent of its library.

    Budgets/scan ranges are not quantity definitions. Known Gaussian sigma is
    canonicalized as a difference, never an absolute-coordinate value.
    """
    import math
    from copy import deepcopy

    if not isinstance(definition, dict) or not definition or not 0 < confidence_level < 1:
        raise ValueError("Interval request needs an explicit quantity definition and confidence in(0,1)")
    noise = deepcopy(observation_model)
    if not isinstance(noise, dict) or not noise.get("family"):
        raise ValueError("Interval request needs explicit observation/noise assumptions")
    if noise["family"] == "iid_gaussian_known_sigma":
        spec = unit_spec(noise.get("unit"))
        sigma = noise.get("sigma")
        if spec is None or not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Gaussian request needs positive known sigma and known unit")
        noise = {
            "family": noise["family"],
            "sigma_canonical": abs(spec.scale) * sigma,
            "dimensions": [str(d) for d in spec.dimensions],
            "angle": str(spec.angle),
            "cycle": str(spec.cycle),
            "conditioning": "fixed_observed_X",
            "independent_unit": "row",
            "data_role": "train",
        }
    return {"definition": deepcopy(definition), "confidence_level": float(confidence_level), "observation_model": noise}


def question_signature(question):
    """Shared comparison/adoption identity. Missing meaning is not equality."""
    request = question.get("request")
    if not isinstance(request, dict) or set(request) != {"definition", "confidence_level", "observation_model"}:
        return None
    if not request["definition"] or not request["observation_model"] or not 0 < request["confidence_level"] < 1:
        return None
    return value_digest(
        {key: question.get(key) for key in ("kind", "estimand", "unit", "coordinate_role")} | {"request": request}
    )


def same_question(left, right):
    signature = question_signature(left)
    return signature is not None and signature == question_signature(right)


def profile_questions(item, *, coordinate_policy):
    """Project verified native profiles into the shared conditional-interval questions."""
    if coordinate_policy != EVIDENCE_COORDINATE_POLICY or item.get("outcome") != "ok":
        return []
    questions = []
    for profile in item.get("native", {}).get("quantities", []):
        endpoints = profile.get("accepted_endpoints", {})
        values = [endpoints.get(side) for side in ("left", "right")]
        verified = (
            profile.get("unit_status") == "pass"
            and profile.get("endpoint_validation", {}).get("all_verified") is True
            and all(v is not None for v in values)
            and set(profile.get("retcodes", {}).values()) == {"Identifiable"}
        )
        questions.append(
            {
                "kind": "conditional_interval",
                "estimand": "conditional_model",
                "quantity": profile["name"],
                "definition": profile.get("definition"),
                "request": profile.get("request"),
                "unit": profile.get("unit"),
                "coordinate_role": profile.get("coordinate_role"),
                "status": "established" if verified else "undetermined",
                "endpoints": values if verified else None,
                "confidence_level": item.get("confidence_level"),
                "assumptions": item.get("meaning"),
                "fit_artifact": item.get("fit_artifact"),
                "fit_sha256": item.get("fit_sha256"),
                "meaning": "conditional on the named fitted structure, noise model, bounds and Train; not source-candidate or structure uncertainty",
            }
        )
    return questions


def resolved_questions(result):
    """Read new common results or the explicit historical profile boundary only."""
    if (
        result.get("result_kind") == "evidence.profile/v1"
        and result.get("coordinate_policy") != EVIDENCE_COORDINATE_POLICY
    ):
        return []  # Historical profile coordinates cannot be certified by payload shape.
    output = []
    for item in result.get("results", []):
        questions = item.get("questions")
        if questions is None and result.get("result_kind") == "evidence.profile/v1":
            questions = profile_questions(item, coordinate_policy=result.get("coordinate_policy"))
        output.extend({"candidate_id": item["id"], **question} for question in questions or [])
    return output


def missing_precision(analyses, candidate_id, requirements, *, estimand="conditional_model"):
    """Criteria matching only. Does not evaluate, refit, or infer interval meaning."""
    questions = [q for a in analyses for q in resolved_questions(a["result"])]
    missing = []
    for name, requirement in requirements.items():
        supported = any(
            q["candidate_id"] == candidate_id
            and q.get("kind") == "conditional_interval"
            and q.get("estimand") == estimand
            and q.get("status") == "established"
            and q.get("quantity") == name
            and same_question(
                q,
                {
                    "kind": "conditional_interval",
                    "estimand": estimand,
                    "unit": requirement["unit"],
                    "coordinate_role": requirement.get("coordinate_role"),
                    "request": requirement.get("request"),
                },
            )
            and q.get("endpoints") is not None
            and 0 <= q["endpoints"][1] - q["endpoints"][0] <= requirement["max_width"]
            for q in questions
        )
        if not supported:
            missing.append(f"Conditional precision/quantity definition/confidence/noise not established: {name}")
    return missing
