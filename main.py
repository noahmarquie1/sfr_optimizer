from classes.OptimizerBO import OptimizerBO
from classes.OptimizerRL import OptimizerRL
from classes.PlotManager import PlotManager
import warnings
from argparse import ArgumentParser
from helpers.plot_data import plots
import numpy as np

from classes.GPModeler import CSVConfigManager, GPModel
import torch

# Setup
torch.set_default_dtype(torch.float32)
torch.set_default_device("cpu")
torch.set_num_threads(6)
torch.set_num_interop_threads(2)
sim_threads = 4

warnings.filterwarnings('ignore', category=UserWarning, module='openmc.material')
warnings.filterwarnings('ignore', category=UserWarning, module='openmc.mixin')
parser = ArgumentParser()
parser.add_argument("--train_bo", action="store", nargs=2)
parser.add_argument("--train_rl", action="store", nargs=2)
parser.add_argument("--train_rl_gp", action="store", nargs=2)
args = parser.parse_args()

bo_plots = plots
rl_plots = plots

bo_plot_manager = PlotManager(bo_plots)
rl_plot_manager = PlotManager(rl_plots)

# Reward Function
def get_reward(keff, peaking_factor, heating_rate):
    w_keff = 0.45
    keff_term = w_keff * np.exp((-20 * (keff - 1.0) ** 2) if keff <= 1.0 else (-100 * (keff - 1.0) ** 2))

    w_pkf = 0.40
    pkf_term = w_pkf * np.exp(-(peaking_factor - 1) ** 2)

    w_heating_rate = 0.15
    heating_rate_term = w_heating_rate * (1 - np.exp(-2 * heating_rate * 1e-8))

    return keff_term + pkf_term + heating_rate_term, keff_term / w_keff, pkf_term / w_pkf, heating_rate_term / w_heating_rate

# Space Data 
bo_space = { # (Needs all enrichments, but can be run with any combination of geometrical parameters)
    "fuel_radius": 0,
    "pin_margin": 0,
    "gap_thickness": 0,
    "enrichment_ring1": 0,
    "enrichment_ring2": 0,
    "enrichment_ring3": 0,
}

rl_space = { # (Can only be run with all enrichments and no geometrical parameters)
    "fuel_radius": 0,
    "pin_margin": 0,
    "gap_thickness": 0,
    "enrichment_ring1": 0,
    "enrichment_ring2": 0,
    "enrichment_ring3": 0,
}

bo = OptimizerBO(
    space=bo_space, 
    output_dir="output_BO", 
    mesh_dimension=[200, 200],
    num_threads=sim_threads,
    reward=get_reward,
    num_particles=5000,
)

# Helpers
def train_bo(n_starts, epochs):
    bo.train(bo_plot_manager, num_starts=n_starts, epochs=epochs, beta_start=3, beta_end=0.5, make_vids=True)
    bo_plot_manager.save("output_BO")


def train_rl(starts, epochs):
    rl = OptimizerRL(
        space=rl_space,
        output_dir="output_RL",
        plot_manager=rl_plot_manager,
        learning_starts=starts,
        mesh_dimension=[200, 200],
        num_particles = 2000,
        reward=get_reward,
        num_threads=sim_threads,
    )

    rl.train(epochs)
    rl_plot_manager.save("output_RL")


def train_rl_gp(iters, csv_mode):
    gp_train_iters = 500
    csv_path = f"data_RL/gp_data.csv"

    config_manager = CSVConfigManager(space=rl_space, csv_path=csv_path, num_particles=5000, num_threads=40)
    config_manager.create_configs(iters, mode=csv_mode, objective=get_reward)
    x, y = config_manager.csv_to_tensor()

    keff_model = GPModel(x, y[:, 0])
    keff_model.train_rl(gp_train_iters)
    keff_model.save_gp(f"data_RL/keff_gp.pth")

    pkf_model = GPModel(x, y[:, 1])
    pkf_model.train_rl(gp_train_iters)
    pkf_model.save_gp(f"data_RL/pkf_gp.pth")

    heating_rate_model = GPModel(x, y[:, 2])
    heating_rate_model.train_rl(gp_train_iters)
    heating_rate_model.save_gp(f"data_RL/heating_rate_gp.pth")


# Command Line Interaction
if args.train_bo:
    train_bo(int(args.train_bo[0]), int(args.train_bo[1]))
elif args.train_rl:
    train_rl(int(args.train_rl[0]), int(args.train_rl[1]))
elif args.train_rl_gp:
    train_rl_gp(int(args.train_rl_gp[0]), args.train_rl_gp[1])
else:
    print("Please input a valid action.")