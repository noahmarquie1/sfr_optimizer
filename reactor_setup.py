"""
This script builds a single Sodium cooled reactor core with a 4-ring hexagonal lattice.
Each ring has different fuel compositions
"""

# from math import log10
import glob
import os

import numpy as np
import openmc
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from io import StringIO

# Add this new function to remove XML files
def remove_xml_files():
    """Remove any existing XML files in the current directory."""
    xml_files = glob.glob('*.xml')
    for xml_file in xml_files:
        try:
            os.remove(xml_file)
            print(f"Removed {xml_file}")
        except Exception as e:
            print(f"Error removing {xml_file}: {e}")

def create_mox_fuel(enrichment):
    Pu_fraction = enrichment / 100
    U_fraction = 1 - Pu_fraction

    fuel = openmc.Material(name="MOX Fuel")

    # Uranium component
    fuel.add_nuclide("U238", 0.99 * U_fraction)
    fuel.add_nuclide("U235", 0.01 * U_fraction)

    # Plutonium component
    fuel.add_nuclide("Pu239", 0.7 * Pu_fraction)
    fuel.add_nuclide("Pu240", 0.15 * Pu_fraction)
    fuel.add_nuclide("Pu241", 0.1 * Pu_fraction)
    fuel.add_nuclide("Pu242", 0.05 * Pu_fraction)

    fuel.set_density("g/cm3", 10.4)

    return fuel

def create_spent_fuel():
    spent_fuel = openmc.Material(name='Spent MOX Fuel')

    # Depleted fissile isotopes
    spent_fuel.add_nuclide('U238', 0.870)
    spent_fuel.add_nuclide('U235', 0.005)
    spent_fuel.add_nuclide('Pu239', 0.030)
    spent_fuel.add_nuclide('Pu241', 0.003)

    # Accumulated neutron poisons (fission products)
    spent_fuel.add_nuclide('Sm149', 0.012)  # strong neutron absorber
    spent_fuel.add_nuclide('Nd143', 0.015)
    spent_fuel.add_nuclide('Cs137', 0.008)

    # Minor actinides
    spent_fuel.add_nuclide('Pu240', 0.022)  # also non-fissile
    spent_fuel.add_nuclide('Pu242', 0.010)
    spent_fuel.add_nuclide('Am241', 0.008)
    spent_fuel.add_nuclide('Cm244', 0.002)

    # Set density (adjust as needed)
    spent_fuel.set_density('g/cm3', 10.2)

    return spent_fuel

# Define common materials
clad = openmc.Material(name="HT9 Cladding")
clad.set_density("g/cm3", 7.8)
clad.add_element("Fe", 0.88)
clad.add_element("Cr", 0.11)
clad.add_element("C", 0.01)

# Create sodium coolant material
sodium = openmc.Material(name="Liquid Sodium")
sodium.add_element("Na", 1.0)
sodium.set_density("g/cm3", 0.927)

# Create steel reflector material
steel_reflector = openmc.Material(name='Steel Reflector')
steel_reflector.set_density('g/cm3', 7.9)  # Typical density of steel
steel_reflector.add_element('Fe', 0.95)  # 95% iron
steel_reflector.add_element('C', 0.05)   # 5% carbon (typical for steel)

# Create helium gap material
helium = openmc.Material(name="Helium Gap")
helium.add_element("He", 1.0)
helium.set_density("g/cm3", 0.0001785)  # Density at typical reactor conditions



def fuel_pin(fuel_material,fuel_or,clad_or,gap_or):
# Define surfaces
    fuel_or = openmc.ZCylinder(r=fuel_or, name='Fuel OR')
    clad_or = openmc.ZCylinder(r=clad_or, name='Clad OR')
    gap_or = openmc.ZCylinder(r=gap_or, name='Gap OR')

    """Returns a fuel pin universe with specified fuel material."""
    fuel_cell = openmc.Cell(fill=fuel_material, region=-fuel_or)
    clad_cell = openmc.Cell(fill=clad, region=+fuel_or & -clad_or)
    sodium_cell = openmc.Cell(fill=sodium, region=+clad_or)
    gap_cell = openmc.Cell(fill=helium, region=+gap_or & -clad_or)

    univ = openmc.Universe(name=f'Fuel Pin {fuel_material.name}')
    univ.add_cells([fuel_cell, gap_cell, clad_cell, sodium_cell])
    return univ


def lattice_model(
        pitch=21.42,
        fuel_radius=0.39218,
        clad_radius=0.45720,
        gap_thickness=0.001,
        reactor_diameter=100.0,
        reflector_thickness=5.0,
        enrichment_zone1=5.0,
        enrichment_zone2=4.0,
        enrichment_zone3=3.0,
        mesh_dimension=[100, 100]):
    """Returns a single sodium cooled fuel lattice with hexagonal geometry and n_rings rings."""
    
    model = openmc.model.Model()

    # Create fuel materials for each ring
    fuel_zone1 = create_mox_fuel(enrichment_zone1)
    fuel_zone2 = create_mox_fuel(enrichment_zone2)
    fuel_zone3 = create_mox_fuel(enrichment_zone3)
    fuel_zone4 = create_spent_fuel()

    # Create fuel pin universes for each ring
    fuel_pin_zone1 = fuel_pin(fuel_zone1,fuel_radius,clad_radius,gap_thickness)
    fuel_pin_zone2 = fuel_pin(fuel_zone2,fuel_radius,clad_radius,gap_thickness)
    fuel_pin_zone3 = fuel_pin(fuel_zone3,fuel_radius,clad_radius,gap_thickness)
    fuel_pin_zone4 = fuel_pin(fuel_zone4,fuel_radius,clad_radius,gap_thickness)

    # Create fuel lattice
    lattice = openmc.HexLattice(name='Fuel lattice')
    lattice.center = (0., 0.)
    lattice.pitch = (pitch, )
    
    # Define the lattice pattern
    ring8 = [fuel_pin_zone4] * 42
    ring7 = [fuel_pin_zone3] * 36
    ring6 = [fuel_pin_zone3] * 30
    ring5 = [fuel_pin_zone2] * 24
    ring4 = [fuel_pin_zone2] * 18
    ring3 = [fuel_pin_zone2] * 12
    ring2 = [fuel_pin_zone1] * 6
    ring1 = [fuel_pin_zone1]
    
    # Combine all rings
    lattice.universes = [ring8, ring7, ring6, ring5, ring4, ring3, ring2, ring1]
    n_rings = len(lattice.universes)

    # Create outer universe filled with sodium coolant
    outer_cell = openmc.Cell(fill=sodium)
    outer_universe = openmc.Universe(cells=[outer_cell,])
    lattice.outer = outer_universe

    # Create the core boundary (inner boundary of the reflector)
    core_boundary = openmc.model.HexagonalPrism(
        edge_length=reactor_diameter * 0.5 - reflector_thickness,
        orientation='y'
    )

    # Create the outer boundary (outer boundary of the reflector)
    outer_boundary = openmc.model.HexagonalPrism(
        edge_length=(reactor_diameter * 0.5),
        orientation='y',
        boundary_type='vacuum'
    )

    # Create cells for the core and reflector
    core_cell = openmc.Cell(fill=lattice, region=-core_boundary)
    reflector_cell = openmc.Cell(fill=steel_reflector, region=+core_boundary & -outer_boundary)

    # Create geometry with both core and reflector
    model.geometry = openmc.Geometry([core_cell, reflector_cell])

    # Set simulation parameters
    model.settings.batches = 150
    model.settings.inactive = 50
    model.settings.particles = 10000
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Box((-pitch, -pitch, -1), (pitch, pitch, 1)),
        constraints={'fissionable': True}
    )

    # Mesh for spatial distribution
    mesh = openmc.RegularMesh()
    mesh.dimension = mesh_dimension 
    mesh.lower_left = [-n_rings*pitch, -n_rings*pitch]
    mesh.upper_right = [n_rings*pitch, n_rings*pitch]

    # Create fission tally
    fission_tally = openmc.Tally(name='fission')
    fission_tally.filters = [openmc.MeshFilter(mesh)]
    fission_tally.scores = ['fission-q-recoverable']

    model.tallies.append(fission_tally)

    model.geometry.export_to_xml()

    return model

def analyze_heating_and_capture_rates(sp, mesh_dimension=[100, 100]):
    """Analyze fission and absorption results."""

    # Get the power tally
    fission_tally = sp.get_tally(name='fission')
    
    heating_rates = fission_tally.get_values(scores=['fission-q-recoverable'])

    # Reshape array to 2D for plotting
    heating_rates = heating_rates.reshape(mesh_dimension[0], mesh_dimension[1])

    # plot the fission rates
    plt.figure(figsize=(10, 10))
    plt.imshow(heating_rates, cmap='viridis')
    plt.colorbar(label='Fission heating rate [eV/source]')
    plt.title('Fission heating rate')
    plt.savefig('plot_fission_heating_rate.png')

    tot_heating_rate = np.sum(heating_rates)
    # plot the normalized fission number distribution
    plt.figure(figsize=(10, 10))
    plt.imshow(heating_rates, cmap='viridis')
    plt.colorbar(label='Fission heating rate [eV/source]')
    plt.title('Fission heating rate')
    plt.savefig('plot_fission_heating_rate.png')
    plt.close("all")

    results = {
        "total_heating_rate": tot_heating_rate,
    }
    
    return results

def analyze_radial_distribution(sp, mesh_dimension=[100, 100], n_annuli=20):
    """Analyze the radial distribution of heating rates.
    
    Args:
        sp: Statepoint file
        mesh_dimension: Dimensions of the mesh [nx, ny]
        n_annuli: Number of concentric annuli to divide the reactor into
    
    Assumptions:
        - the mesh is centered on the origin
    """
    # Get the heating rate data
    fission_tally = sp.get_tally(name='fission')
    heating_rates = fission_tally.get_values(scores=['fission-q-recoverable'])
    heating_rates = heating_rates.reshape(mesh_dimension[0], mesh_dimension[1])
    
    # Create coordinate grids
    x = np.linspace(-mesh_dimension[0]/2, mesh_dimension[0]/2, mesh_dimension[0])
    y = np.linspace(-mesh_dimension[1]/2, mesh_dimension[1]/2, mesh_dimension[1])
    X, Y = np.meshgrid(x, y)
    
    # Calculate radial distances from center
    R = np.sqrt(X**2 + Y**2)
    
    # Calculate maximum radius (half of the smaller dimension)
    max_radius = min(mesh_dimension[0], mesh_dimension[1]) / 2
    
    # Create annuli boundaries
    r_bins = np.linspace(0, max_radius, n_annuli + 1)
    
    # Initialize arrays for storing results
    avg_heating_rates = np.zeros(n_annuli)
    r_centers = np.zeros(n_annuli)
    
    # Calculate average heating rate in each annulus
    for i in range(n_annuli):
        mask = (R >= r_bins[i]) & (R < r_bins[i+1])
        if np.any(mask):
            avg_heating_rates[i] = np.mean(heating_rates[mask])
        r_centers[i] = (r_bins[i] + r_bins[i+1]) / 2
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(r_centers, avg_heating_rates, 'bo-', label='Average heating rate')
    plt.xlabel('Radius [mesh units]')
    plt.ylabel('Average heating rate [eV/source]')
    plt.title('Radial Distribution of Heating Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig('radial_heating_distribution.png')
    plt.close("all")
    
    # Calculate flatness metrics
    mean_rate = np.mean(avg_heating_rates)
    # Create mask for points above 10% of mean heating rate
    mask = avg_heating_rates > (0.1 * mean_rate)
    # Calculate peaking factor only for points above threshold
    heating_max = np.max(avg_heating_rates)
    heating_mean = np.mean(avg_heating_rates[mask])
    peaking_factor = (heating_max / heating_mean) ** 2
    
    return peaking_factor

def analyze_results(mesh_dimension=[100, 100], n_annuli=20, verbose=1):
    """Analyze the results of the reactor model."""
    # Load the statepoint file to get the combined k-effective
    sp = openmc.StatePoint('statepoint.150.h5')

    results = analyze_heating_and_capture_rates(sp, mesh_dimension)
    peaking_factor = analyze_radial_distribution(sp, mesh_dimension, n_annuli)

    keff = sp.keff  # This returns a tuple of (mean, std_dev)

    if verbose:
        print("--------------output for optimization------------------")
        print(f"k_eff: {keff.n:.5f} ± {1e5 * keff.s:.0f} [pcm]")
        print(f"Total fission heating rate: {results['total_heating_rate']:.2e} [eV/source]")
        print(f"Peaking factor: {peaking_factor:.3f}")
        print("--------------output for optimization------------------")
        print("")

    return keff, peaking_factor, results["total_heating_rate"]

def plot_lattice(reactor_diameter=100.0, show=True, path_xy='plot_xy', path_yz='plot_yz'):
    """Plot an XY view of the hexagonal lattice lattice."""

    plot_xy = openmc.Plot(plot_id=1)
    plot_xy.filename = path_xy
    plot_xy.basis = 'xy'
    plot_xy.origin = [0, 0, 0]
    plot_xy.width = [reactor_diameter, reactor_diameter]
    plot_xy.pixels = [1000, 1000]
    plot_xy.color_by = 'material'
    plot_xy.colors = {
        # fuel: 'red',
        clad: 'blue',
        sodium: 'yellow'
    }
    plot_xy.background = 'white'  # Add white background for better visibility

    plot_yz = openmc.Plot(plot_id=2)
    plot_yz.filename = path_yz
    plot_yz.basis = 'yz'
    plot_yz.origin = [0, 0, 0]
    plot_yz.width = [reactor_diameter, reactor_diameter]
    plot_yz.pixels = [1000, 1000]
    plot_yz.color_by = 'material'
    plot_yz.colors = {
        # fuel: 'red',
        clad: 'blue',
        sodium: 'yellow'
    }
    # plot_xz.background = 'white'  # Add white background for better visibility
    
    # Create and export the plot
    plot_file = openmc.Plots([plot_xy, plot_yz])
    plot_file.export_to_xml()
    
    # Generate the plot
    openmc.plot_geometry()
    
    # Load and display the plot
    if show:
        img = plt.imread(path_xy + '.png')
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Lead cooled Hexagonal lattice XY View')
        plt.show()


def print_all_plots(path, iteration):

    print("hello")
    sp = openmc.StatePoint('statepoint.150.h5')

    plot_xy = openmc.Plot(plot_id=1)
    plot_xy.filename = path + f"/plot_xy/plot_xy_{iteration}"
    plot_xy.basis = 'xy'
    plot_xy.origin = [0, 0, 0]
    plot_xy.width = [100, 100]
    plot_xy.pixels = [1000, 1000]
    plot_xy.color_by = 'material'
    plot_xy.colors = {
        # fuel: 'red',
        clad: 'blue',
        sodium: 'yellow'
    }
    plot_xy.background = 'white'  # Add white background for better visibility

    plot_yz = openmc.Plot(plot_id=2)
    plot_yz.filename = path + f"/plot_yz/plot_yz_{iteration}"
    plot_yz.basis = 'yz'
    plot_yz.origin = [0, 0, 0]
    plot_yz.width = [100, 100]
    plot_yz.pixels = [1000, 1000]
    plot_yz.color_by = 'material'
    plot_yz.colors = {
        # fuel: 'red',
        clad: 'blue',
        sodium: 'yellow'
    }
    # plot_xz.background = 'white'  # Add white background for better visibility

    # Create and export the plot
    f = StringIO()
    with redirect_stdout(f):
        plot_file = openmc.Plots([plot_xy, plot_yz])
        plot_file.export_to_xml()
        openmc.plot_geometry()

    fission_tally = sp.get_tally(name='fission')

    heating_rates = fission_tally.get_values(scores=['fission-q-recoverable'])

    # Reshape array to 2D for plotting
    heating_rates = heating_rates.reshape(200, 200)

    tot_heating_rate = np.sum(heating_rates)
    # plot the normalized fission number distribution
    plt.figure(figsize=(10, 10))
    plt.imshow(heating_rates, cmap='viridis')
    plt.colorbar(label='Fission heating rate [eV/source]')
    plt.title('Fission heating rate')
    plt.savefig(path + f'/fission_heating_rate/fission_heating_rate_{iteration}.png')
    plt.close("all")