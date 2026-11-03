
def get_conservation_actions(risk_level, input_data):
    """
    Generate conservation recommendations based on predicted risk
    and environmental indicators.
    """

    risk_level = risk_level.strip().lower()
    actions = []
    alerts = []

    if risk_level == "high":
        priority = "URGENT"
        actions = [
            "Arrange an immediate field inspection by forest officials.",
            "Compare recent satellite imagery with historical forest cover.",
            "Investigate possible illegal logging and land clearing.",
            "Increase forest patrols in vulnerable locations.",
            "Prepare a restoration plan if forest degradation is confirmed."
        ]

    elif risk_level == "medium":
        priority = "MODERATE"
        actions = [
            "Schedule regular forest inspections.",
            "Monitor changes in tree cover using satellite imagery.",
            "Assess nearby agricultural and settlement expansion.",
            "Strengthen community awareness and reporting.",
            "Prepare preventive conservation measures."
        ]

    else:
        priority = "ROUTINE"
        actions = [
            "Continue periodic forest-cover monitoring.",
            "Maintain existing conservation measures.",
            "Encourage community participation in forest protection.",
            "Review the region's risk periodically."
        ]

    # Identify specific environmental warning signals
    try:
        fire_count = float(input_data.get("fire_count_last_year", 0))
        if fire_count > 0:
            alerts.append(
                f"{fire_count:g} fire incidents reported: increase fire surveillance."
            )
    except (TypeError, ValueError):
        pass

    try:
        forest_loss = float(input_data.get("prior_loss_pct", 0))
        if forest_loss > 0:
            alerts.append(
                "Previous forest loss detected: investigate the affected locations."
            )
    except (TypeError, ValueError):
        pass

    try:
        tree_cover = float(input_data.get("tree_cover_pct", 0))
        if tree_cover < 30:
            alerts.append(
                "Low tree cover indicator: assess forest degradation and restoration needs."
            )
    except (TypeError, ValueError):
        pass

    try:
        protected = int(float(input_data.get("protected_area", 0)))
        if protected == 1:
            alerts.append(
                "Protected-area status: verify possible encroachment or violations."
            )
        elif protected == 0:
            alerts.append(
                "Check whether additional conservation protection is needed."
            )
    except (TypeError, ValueError):
        pass

    return {
        "priority": priority,
        "actions": actions,
        "alerts": alerts
    }
