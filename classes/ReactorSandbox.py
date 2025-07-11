# Imports
from reactor_setup import lattice_model, analyze_results
from helpers.geometry_data import geometry

from io import StringIO
import openmc
from contextlib import redirect_stdout
import os
import time
import math

# Class
class ReactorSandbox:
    def __init__(
            self,
            space,
            mesh_dimension=[100, 100],
            nbin_radial=20,
            num_particles=5000,
            num_threads=40,
        ):

        self.space = space
        self.mesh_dimension = mesh_dimension
        self.nbin_radial = nbin_radial
        self.num_particles = num_particles
        self.num_threads = num_threads
        self.current_geometry = {
            "pitch": 21.42,
            "fuel_radius": 1.64, #TEMPORARY RL EXPERIMENTATION geometry["default_fuel_radius"],
            "clad_radius": 0.45720,
            "reactor_diameter": 100.0,
            "min_dist_pin2pin": geometry["default_min_dist_pin2pin"],
            "clad_thickness": geometry["default_clad_thickness"],
            "reflector_thickness": geometry["default_reflector_thickness"],
            "enrichment_zone1": geometry["default_enrichment_zone1"],
            "enrichment_zone2": geometry["default_enrichment_zone2"],
            "enrichment_zone3": geometry["default_enrichment_zone3"],
        }
        self.min_mutable_geometry = {}

        for param in geometry["mutable_geometry"]:
            self.min_mutable_geometry[param] = geometry[f"min_{param}"]


    def check_candidate_viability(self, candidate):
        """
        Takes in a dict of geometry components, and returns whether it is feasible physically in simulation

        Parameters
        ----------
        candidate: Dict of geometry components
        
        Returns
        -------
        Boolean - if candidate is physically viable

        Side Effects
        ------------
        None

        Calls
        -----
        None

        Called By
        ---------
        find_candidate_start()
        find_candidate_training()
        """

        viable = candidate["reactor_diameter"] >= 2*geometry["n_rings"]*candidate["pitch"] + 2*candidate["reflector_thickness"]
        return viable 

    def space_to_geometry(self, candidate):
        """
        Takes in a normalized space and returns a dictionary of corresponding de-normalized values plus additional geometry components

        Parameters
        ----------
        candidate: Tensor of size (1, num dims in self.space), normalized, input to model
        
        Returns
        -------
        dict of denormalized values of all components in self.space plus additional geometry components

        Side Effects
        ------------
        None

        Calls
        -----
        self.denormalize_params()

        Called By
        ---------
        find_candidate_start()
        find_candidate_training()
        """

        candidate_geometry = self.denormalize_params(candidate)

        # Fixing default values
        defaults = [param.split("default_")[1] for param in geometry.keys() if "default" in param]
        for key in defaults:
            if not key in candidate_geometry.keys():
                candidate_geometry[key] = geometry[f"default_{key}"]

        # Fixing derived params
        candidate_geometry["clad_radius"] = candidate_geometry["fuel_radius"] + candidate_geometry["clad_thickness"]
        candidate_geometry["pitch"] = 2*candidate_geometry["clad_radius"] + candidate_geometry["min_dist_pin2pin"]

        # Fixing remaining parameters
        for key, value in self.current_geometry.items():
            if not key in candidate_geometry.keys():
                candidate_geometry[key] = value

        return candidate_geometry

    def denormalize_params(self, x):
        """
        Takes in a normalized space and returns denormalized dict ready for OpenMC model.

        Parameters
        ----------
        x: Tensor of size (6,), normalized, input to model. Assumes: 
         - [Fuel Radius, Min Dist Pin2Pin, Enrichment Ring1, Enrichment Ring2, Enrichment Ring3, Enrichment Ring4]
        
        Returns
        -------
        dict of denormalized values of all components in self.space 
         + additional components:
         2) reflector thickness
         3) clad thickness

        Side Effects
        ------------
        None

        Calls
        -----
        None

        Called By
        ---------
        self.space_to_geometry()
        self.minimal_model_run()
        self.training_model_run()
        self.eval_model_run()
        """
        # Convert x to list
        x = x.tolist()

        # Set Initial Bounds
        n_rods = 2*(geometry["n_rings"] - 1)
        geometry["max_fuel_radius"] = (geometry["reactor_diameter"] - 2*(n_rods)*geometry["clad_thickness"] - 2*geometry["reflector_thickness"] - (n_rods - 1)*geometry["min_min_dist_pin2pin"]) / (2*n_rods)

        denormalized_space = {}

        denormalized_space["fuel_radius"] = x[0] * (geometry[f"max_fuel_radius"] - geometry[f"min_fuel_radius"]) + geometry[f"min_fuel_radius"]
        max_min_dist_pin2pin = (geometry["reactor_diameter"] - 2*n_rods*geometry["clad_thickness"] - 2*n_rods*denormalized_space["fuel_radius"] - 2*geometry["reflector_thickness"]) / (n_rods - 1)

        denormalized_space["min_dist_pin2pin"] = x[1] * (geometry[f"max_min_dist_pin2pin"] - geometry[f"min_min_dist_pin2pin"]) + geometry[f"min_min_dist_pin2pin"]
        
        for i in range(3):
            denormalized_space[f"enrichment_ring{i+1}"] = x[i+2] * (geometry[f"max_enrichment_ring{i+1}"] - geometry[f"min_enrichment_ring{i+1}"]) + geometry[f"min_enrichment_ring{i+1}"]

        denormalized_space["reflector_thickness"] = geometry["reflector_thickness"]
        denormalized_space["clad_thickness"] = geometry["clad_thickness"]
        return denormalized_space

    def normalize(self, denormalized_val, min_val, max_val):
        """
        Normalizes value according to a min and max. Should push values between 0 and 1 for model processing

        Parameters
        ----------
        denormalized_val: float, value to be normalized
        min_val: float
        max_val: float
        
        Returns
        -------
        float: normalized value

        Side Effects
        ------------
        None

        Calls
        -----
        None

        Called By
        ---------
        OptimizerBO.find_candidate_start()
        OptimizerBO.find_candidate_training()
        """

        return (denormalized_val - min_val) / (max_val - min_val)

    def lattice_model(self):
        """
        Defines an OpenMC lattice model according to ReactorSandbox internal state (self.current_geometry)

        Parameters
        ----------
        self
        
        Returns
        -------
        model: lattice_model() - OpenMC

        Side Effects
        ------------
        None

        Calls
        -----
        None

        Called By
        ---------
        self.minimal_model_run()
        """

        model = lattice_model(
            pitch=self.current_geometry["pitch"], 
            fuel_radius=self.current_geometry["fuel_radius"], 
            clad_radius=self.current_geometry["clad_radius"], 
            reactor_diameter=self.current_geometry["reactor_diameter"], 
            reflector_thickness=self.current_geometry["reflector_thickness"], 
            enrichment_zone1=self.current_geometry["enrichment_ring1"], 
            enrichment_zone2=self.current_geometry["enrichment_ring2"], 
            enrichment_zone3=self.current_geometry["enrichment_ring3"], 
            mesh_dimension=self.mesh_dimension,
        )
        return model

    def minimal_model_run(self, x, verbose=0):
        """
        Runs openmc simulation with minimal extra output and provides result - Hopes to maximize time efficiency

        Parameters
        ----------
        x: Tensor of size (1, num dims in self.space), normalized, input to model
        verbose: Boolean, affects printing and plot creation
        
        Returns
        -------
        dict of model output

        Side Effects
        ------------
        self.current_geometry

        Calls
        -----
        self.denormalize_params()
        analyze_results() - helpers.reactor_methods

        Called By
        ---------
        self.training_model_run()
        self.eval_model_run()
        """

        # Define Parameters from x and Define Model
        denormalized_params = self.denormalize_params(x)
        for name in self.current_geometry.keys():
            if name in denormalized_params.keys():
                self.current_geometry[name] = denormalized_params[name]
        
        for i in range(geometry["num_enrichments"]):
            self.current_geometry[f"enrichment_ring{i+1}"] = denormalized_params[f"enrichment_ring{i+1}"]

        self.current_geometry["clad_radius"] = denormalized_params["fuel_radius"] + geometry["clad_thickness"]
        self.current_geometry["pitch"] = 2*self.current_geometry["clad_radius"] + denormalized_params["min_dist_pin2pin"]

        lattice_model = self.lattice_model()

        # Run Model and Retrieve Results
        f = StringIO()
        with redirect_stdout(f):
            lattice_model.run(threads=self.num_threads, particles=self.num_particles)
        
        k_eff, peaking_factor, heating_rate = analyze_results(mesh_dimension=self.mesh_dimension, n_annuli=self.nbin_radial, verbose=verbose)
        info = {
            "k_eff": k_eff,
            "k_eff.n": k_eff.n,
            "k_eff.s": k_eff.s,
            "peaking_factor": peaking_factor,
            "heating_rate": heating_rate,
        }
        return info

    def training_model_run(
            self, 
            x, 
            reward,
            plot_manager=None,
            run_type="TRAINING",
            iteration=0
        ):
        """
        Layers on minimal model run to run openmc model with more terminal feedback about training, returning additional loss information

        Parameters
        ----------
        x: List of length=(num dims in self.space), normalized, input to model
        loss: Function of type (float, float, float, float) -> float - used to measure model 
        plot_manager: PlotManager class instance
        run_type: String
        iteration: int
        
        Returns
        -------
        dict of model output + loss with components

        Side Effects
        ------------
        plot_manager
        self.current_geometry - via minimal_model_run

        Calls
        -----
        self.minimal_model_run()
        plot_manager.add()
        self.denormalize_params()     

        analyze_results() - helpers.reactor_methods : via minimal_model_run

        Called By
        ---------
        OptimizerBO.write_starts()
        OptimizerBO.run_start()
        OptimizerBO.run_epoch()
        """

        denormalized_params = self.denormalize_params(x)
        enrichments = [denormalized_params[f"enrichment_ring{i}"] for i in range(1, 4)]
        print(f"============ {f'Iteration: {iteration}'} ({run_type}) ============")
        pre_info_string = ""
        for i in range(math.ceil((len(list(self.space.keys()))) / 2)):
            if i == 0:
                pre_info_string += "(PRE)  : "
            else:
                pre_info_string += "       : "
            if 2*i+1 < len(list(self.space.keys())):
                current_param_1, current_param_2 = list(self.space.keys())[2*i:2*i+2]
                pre_info_string += f"{current_param_1}: {denormalized_params[current_param_1]:.3f}"
                pre_info_string += f", "
                pre_info_string += f"{current_param_2}: {denormalized_params[current_param_2]:.3f} \n"
            else:
                current_param_1 = list(self.space.keys())[2*i]
                pre_info_string += f"{current_param_1}: {denormalized_params[current_param_1]:.3f}"

        print(pre_info_string)

        info = self.minimal_model_run(x, verbose=0)
        reward_value, keff_term, peak_term, heating_rate_term = reward(keff=info["k_eff"].n, peaking_factor=info["peaking_factor"], heating_rate=info["heating_rate"])
        info.update({
            "reward": reward_value,
            "keff_term": keff_term,
            "pkf_term": peak_term,
            "heating_rate_term": heating_rate_term
        })

        if not plot_manager == None:
            plot_updates = {
                "keff": info["k_eff.n"],
                "pkf": info["peaking_factor"],
                "heating_rate": info["heating_rate"],
                "enrichment-composite": {"enrichment_ring1": enrichments[0], "enrichment_ring2": enrichments[1], "enrichment_ring3": enrichments[2]},
                "fuel_radius": denormalized_params["fuel_radius"],
                "min_dist_pin2pin": denormalized_params["min_dist_pin2pin"],
                "reward-composite": {"Full": reward_value, "k_eff": keff_term, "peaking_factor": peak_term, "heating_rate": heating_rate_term},
            }

            for plot_name in plot_manager.plots.keys():
                if plot_name in plot_updates.keys():
                    plot_manager.add(iteration, {plot_name: plot_updates[plot_name]})
        
        print(f"(POST) : k_eff: {info['k_eff.n']:.5f} ± {1e5*info['k_eff.s']:.0f} [pcm]", f"pkf: {info['peaking_factor']:.3f}", f"reward: {info['reward']:.4f}") 
        print(f"       : total_heating_rate: {info['heating_rate']:.2e}")

        return info

    def eval_model_run(self, x, reward, output_dir, start_time):
        """
        Layers on minimal model run to run openmc model writing information about current model state. Intended use after training

        Parameters
        ----------
        x: Tensor of size (1, num dims in self.space), normalized, input to model
        loss: Function of type (float, float, float, float) -> float - used to measure model 
        plot_manager: PlotManager class instance
        output_dir: String, directory
        start_time: Time
        
        Returns
        -------
        N/A

        Side Effects
        ------------
        plot_manager
        results.txt

        Calls
        -----
        self.minimal_model_run()
        self.denormalize_params()   

        analyze_results() - helpers.reactor_methods : via minimal_model_run

        Called By
        ---------
        OptimizerBO.train()
        """

        denormalized_params = self.denormalize_params(x)
        enrichments = [denormalized_params[f"enrichment_ring{i}"] for i in range(1, 4)]

        info = self.minimal_model_run(x, verbose=1)
        reward_value = reward(keff=info["k_eff"].n, peaking_factor=info["peaking_factor"], heating_rate=info["heating_rate"])[0]
        info.update({"loss": reward_value})

        results_text = "\n============ Results ============\n"
        results_text += "Best parameters found:\n\n"

        results_text += f"Fuel radius: {denormalized_params['fuel_radius']:.2f}cm\n"
        results_text += f"Min Dist Pin2Pin: {denormalized_params['min_dist_pin2pin']:.2f}cm\n"

        results_text += f"Enrichment ring 1: {enrichments[0]:.2f}%\n"
        results_text += f"Enrichment ring 2: {enrichments[1]:.2f}%\n"
        results_text += f"Enrichment ring 3: {enrichments[2]:.2f}%\n"

        results_text += "Final given values\n"
        results_text += f"k_eff: {info['k_eff.n']:.4f} ± {1e5*info['k_eff.s']:.0f} [pcm]\n"
        results_text += f"peaking_factor: {info['peaking_factor']:.3f}\n"
        results_text += f"heating_rate: {info['heating_rate']:.2e}\n"
        results_text += f"loss: {info['loss']:.2e}\n\n"
        time_elapsed = time.time() - start_time
        results_text += f"Total execution time: {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)\n"

        # Write results to file
        print(results_text)
        os.makedirs(f'{output_dir}', exist_ok=True)
        with open(f'{output_dir}/results.txt', 'w') as f:
            f.write(results_text)

