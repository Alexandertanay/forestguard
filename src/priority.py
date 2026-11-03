"""
ForestGuard Conservation Priority Engine.

This module creates a transparent 0-100 planning score from model output
and selected environmental / human-pressure indicators. It is a prototype
decision-support heuristic, not a calibrated ecological risk index.
"""

def _number(value, default=0.0):
    try:
        number = float(value)
        if number != number:  # NaN
            return default
        return number
    except (TypeError, ValueError):
        return default


def assess_conservation_priority(input_data, high_risk_probability, risk_level):
    """Return a weighted conservation-priority score and human-readable rationale.

    Weights:
      - model High-risk probability: 40%
      - previous forest loss: 20%
      - fire incidents: 15%
      - tree-cover deficit: 15%
      - proximity to roads / settlements: 10%

    The thresholds and normalization ranges are initial prototype choices.
    """
    high_probability = max(0.0, min(_number(high_risk_probability), 100.0))
    prior_loss = max(0.0, _number(input_data.get("prior_loss_pct")))
    fire_count = max(0.0, _number(input_data.get("fire_count_last_year")))
    tree_cover = max(0.0, min(_number(input_data.get("tree_cover_pct"), 100.0), 100.0))
    road_distance = max(0.0, _number(input_data.get("dist_to_road_km"), 30.0))
    settlement_distance = max(0.0, _number(input_data.get("dist_to_settlement_km"), 35.0))

    components = {
        "Model High-risk probability": high_probability,
        "Previous forest loss": min(prior_loss / 40.0, 1.0) * 100.0,
        "Fire incidents": min(fire_count / 5.0, 1.0) * 100.0,
        "Tree-cover deficit": 100.0 - tree_cover,
        "Road / settlement proximity": (
            (1.0 - min(road_distance / 30.0, 1.0)) * 50.0
            + (1.0 - min(settlement_distance / 35.0, 1.0)) * 50.0
        )
    }

    weighted_points = {
        "Model High-risk probability": components["Model High-risk probability"] * 0.40,
        "Previous forest loss": components["Previous forest loss"] * 0.20,
        "Fire incidents": components["Fire incidents"] * 0.15,
        "Tree-cover deficit": components["Tree-cover deficit"] * 0.15,
        "Road / settlement proximity": components["Road / settlement proximity"] * 0.10
    }

    score = round(sum(weighted_points.values()), 1)

    if score >= 75:
        level = "Critical"
        action = "Prioritize prompt field verification; review recent forest-loss evidence and coordinate an inspection."
    elif score >= 55:
        level = "High"
        action = "Schedule an early field assessment and review satellite imagery for recent changes."
    elif score >= 35:
        level = "Moderate"
        action = "Include this location in the next monitoring cycle and investigate the strongest contributing indicators."
    else:
        level = "Routine"
        action = "Continue routine monitoring and reassess when new environmental or land-use data becomes available."

    reasons = []
    if components["Model High-risk probability"] >= 50:
        reasons.append(f"Elevated model High-risk probability ({high_probability:.1f}%)")
    if prior_loss >= 10:
        reasons.append(f"Previous forest loss is notable ({prior_loss:.1f}%)")
    if fire_count >= 2:
        reasons.append(f"Multiple fire incidents recorded ({fire_count:.0f})")
    if tree_cover <= 40:
        reasons.append(f"Tree cover is relatively low ({tree_cover:.1f}%)")
    if components["Road / settlement proximity"] >= 60:
        reasons.append("Location is relatively close to roads and/or settlements")

    if not reasons:
        reasons.append("No individual indicator crossed the prototype explanation thresholds")

    return {
        "score": score,
        "level": level,
        "components": components,
        "weighted_points": weighted_points,
        "reasons": reasons,
        "recommended_action": action,
        "risk_level": risk_level
    }
