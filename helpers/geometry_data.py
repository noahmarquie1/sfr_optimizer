geometry = {
    # Needs a 'min_' value for all mutable geometry parameters

    "min_reflector_thickness": 10.0, # Unchanging for now
    "min_reactor_diameter": 100.0, # Unchanging for now
    "min_clad_thickness": 0.05, # Unchanging for now

    "min_fuel_radius": 0.65,
    "min_pin_margin": 0.01,
    "min_enrichment_ring1": 2.0,
    "min_enrichment_ring2": 2.0,
    "min_enrichment_ring3": 2.0,
    "min_gap_thickness": 0.001,

    "max_pin_margin": 0.0,
    "max_enrichment_ring1": 19.0,
    "max_enrichment_ring2": 19.0,
    "max_enrichment_ring3": 19.0,
    "max_gap_thickness": 0.01,

    # Needs a 'default_' value for all non-derived geometry parameters
    "default_fuel_radius": 1.5,         
    "default_reactor_diameter": 100.0,
    "default_pin_margin": 1.0,
    "default_gap_thickness": 0.005,
    "default_clad_thickness": 0.05,
    "default_reflector_thickness": 10.0,
    "default_enrichment_zone1": 5.0,
    "default_enrichment_zone2": 4.0,
    "default_enrichment_zone3": 3.0,

    # Factors needed to instantiate lattice model which are non-mutable
    "reactor_diameter": 100.0,
    "reflector_thickness": 10.0,
    "clad_thickness": 0.05,
    "n_rings": 8,
    "nbin_radial": 20,

    "num_enrichments": 3,

    "mutable_geometry": ["reflector_thickness", "fuel_radius", "clad_thickness", "pin_margin", "gap_thickness"]
}
