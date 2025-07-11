from reactor_setup import *
import argparse

if __name__ == '__main__':
    # Set up command-line arguments for generating/running the model

    # ============Fixed parameters ========
    # The reactor diameter and the number of rings are fixed
    reactor_diameter = 100.0 # cm
    n_rings = 8
    # ============Fixed parameters ========

    # ============Variable parameters ========
    fuel_radius = 2.0 # cm
    clad_thickness = 0.06 # cm
    min_dist_pin2pin = 0.4 # cm
    reflector_thickness = 10.0 # cm

    enrichment_ring1 = 10.0 # %
    enrichment_ring2 = 10.0 # %
    enrichment_ring3 = 5.0 # %
    # ============Variable parameters ========

    # ============Derived parameters ========
    clad_radius = fuel_radius + clad_thickness
    pitch = 2*clad_radius + min_dist_pin2pin

    # verify the reactor diameter is greater than 2*n_rings*pitch + 2*reflector_thickness
    if reactor_diameter < 2*n_rings*pitch + 2*reflector_thickness:
        raise ValueError("Reactor diameter (%.2f cm) must be greater than 2*n_rings*pitch+2*reflector_thickness (%.2f cm), you must change the reflector thickness or lattice parameters" % (reactor_diameter, 2*n_rings*pitch + 2*reflector_thickness))


    # ============Mesh dimension ========
    # mesh dimension is the number of cells in the mesh used to calculate the power distribution
    mesh_dimension = [200, 200]
    # bin resolution for the radial distribution analysis of the heating rate
    nbin_radial = 20
    # ============Mesh dimension ========

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plot_geom', action='store_true', help='Plot XY view of the lattice')
    parser.add_argument('--analyze', action='store_true', help='Analyze results')
    args = parser.parse_args()
    if not args.generate and not args.run and not args.plot_geom and not args.analyze:
        parser.print_help()

    if args.generate or args.run or args.plot_geom or args.analyze:
        # Remove any existing XML files before generating new ones
        remove_xml_files()
        
        model = lattice_model(
            pitch,
            fuel_radius,
            clad_radius,
            reactor_diameter,
            reflector_thickness,
            enrichment_ring1,
            enrichment_ring2,
            enrichment_ring3,
            mesh_dimension
        )
        # model = lattice_model(pitch) 
        model.export_to_xml()
        if args.generate:
            model.export_to_xml()
        if args.run:
            plot_lattice(reactor_diameter)
            model.run()
            analyze_results(mesh_dimension, nbin_radial)
        if args.plot_geom:
            plot_lattice(reactor_diameter)
        if args.analyze:
            analyze_results(mesh_dimension, nbin_radial)