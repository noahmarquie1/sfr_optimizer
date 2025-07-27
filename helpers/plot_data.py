plots = {
    "keff": {
        "name": "K-Effective",
        "type": "single",
        "scale": "regular-scale",
    },
    "pkf": {
        "name": "Peaking Factor",
        "type": "single",
        "scale": "regular-scale",
    },
    "reward-comparison": {
        "name": "Normalized Reward Value",
        "type": "multiple",
        "scale": "regular-scale",
        "legend": ["Gaussian Process", "Simulation"],
        "prioritize_first": True,
    },
    "reward-composite": {
        "name": "Normalized Reward Value",
        "type": "multiple",
        "scale": "regular-scale",
        "legend": ["Full", "k_eff", "peaking_factor", "heating_rate"],
        "prioritize_first": True,
    },
    "enrichment-composite": {
        "name": "Enrichments (%)",
        "type": "multiple",
        "scale": "regular-scale",
        "legend": ["enrichment_ring1", "enrichment_ring2", "enrichment_ring3"],
        "prioritize_first": False,
    },
    "enrichment_ring1": {
        "name": "Enrichment Ring 1",
        "type": "single",
        "scale": "regular-scale",
    },
    "enrichment_ring2": {
        "name": "Enrichment Ring 2",
        "type": "single",
        "scale": "regular-scale",
    },
    "enrichment_ring3": {
        "name": "Enrichment Ring 3",
        "type": "single",
        "scale": "regular-scale",
    },
    "heating_rate": {
        "name": "Total Heating Rate",
        "type": "single",
        "scale": "regular-scale",
    },
    "gap_thickness": {
        "name": "Gap Thickness",
        "type": "single",
        "scale": "regular-scale",
    },
    "pin_margin": {
        "name": "Fuel Pin Margin (cm)",
        "type": "single",
        "scale": "regular-scale",
    },
    "fuel_radius": {
        "name": "Fuel Radius (cm)",
        "type": "single",
        "scale": "regular-scale",
    }
}