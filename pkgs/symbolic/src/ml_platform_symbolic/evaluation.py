"""Selection rule retained for existing explicit benchmark callers."""

def select_row(rows, settings):
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise ValueError("No eligible candidates; see leaderboard reasons")
    best = min(r["rmse"] for r in eligible)
    policy = settings["selection_policy"]
    threshold = None
    if policy == "accuracy":

        def key(r):
            return (r["rmse"], r["n_nodes"], r["n_parameters"], r["candidate_id"])
    else:
        threshold = (
            settings["acceptable_rmse"]
            if policy == "simplicity"
            else best * (1 + settings["relative_tolerance"]) + settings["absolute_tolerance"]
        )
        eligible = [r for r in eligible if r["rmse"] <= threshold]
        if not eligible:
            raise ValueError("No candidate meets acceptable_rmse")

        def key(r):
            return (r["n_nodes"], r["rmse"], r["n_parameters"], r["candidate_id"])

    return min(eligible, key=key), threshold


