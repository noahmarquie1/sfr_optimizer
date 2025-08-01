# Imports
from classes.ReactorSandbox import ReactorSandbox
from helpers.geometry_data import geometry
from helpers.media_funcs import clear_dirs, create_all_videos

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.analytic import UpperConfidenceBound, LogExpectedImprovement
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf

from gpytorch.kernels import MaternKernel
from gpytorch.priors import GammaPrior
from gpytorch.mlls import ExactMarginalLogLikelihood

from reactor_setup import plot_lattice, print_all_plots

import torch
import time

# Helper Functions
def beta_schedule(i, max_iter, beta_start=3.0, beta_end=0.1):
    return beta_start * (beta_end / beta_start) ** (i / max_iter)


# Class
class OptimizerBO(ReactorSandbox):
    def __init__(self, space, output_dir, reward, **kwargs):
        super().__init__(space=space, **kwargs)

        self.train_x = torch.tensor([], dtype=torch.float64)
        self.train_y = torch.tensor([], dtype=torch.float64)
        self.reward = reward
        self.outcome_transform = Standardize(m=1)
        self.output_dir = output_dir
        self.best_idx = 0
        self.best_x = 0
        self.best_y = 0
        self.vid_dirs = [
            "output_BO/media/fission_heating_rate",
            "output_BO/media/plot_xy",
            "output_BO/media/plot_yz",
        ]

        self.acq = None
        self.covar_module = MaternKernel(nu=2.5, lengthscale_prior=GammaPrior(3.0, 6.0))
        self.bounds = torch.tensor([[0] * len(self.space.keys()), [1] * len(self.space.keys())], dtype=torch.float64)

    def choose_gp_candidate(self):
        next_x, _ = optimize_acqf(
            self.acq, 
            self.bounds,
            q=1,
            num_restarts=50,
            raw_samples=200
        )
        return next_x

    def choose_random_candidate(self):
        next_x = torch.rand(1, len(self.space), dtype=torch.float64)
        return next_x

    def run_epoch(self, iteration, beta, plot_manager):
        gp = SingleTaskGP(
            self.train_x, 
            self.train_y, 
            outcome_transform=self.outcome_transform, 
            covar_module=self.covar_module, 
            train_Yvar=torch.tensor([[0.001] for _ in range(len(self.train_x))], dtype=torch.float64)
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        self.acq = UpperConfidenceBound(gp, beta=beta)

        next_x = self.choose_gp_candidate()
        run_type="TRAINING"

        reward_val = self.training_model_run(
            next_x[0],
            self.reward, 
            plot_manager=plot_manager, 
            run_type=run_type,
            iteration=iteration, 
        )["reward"]

        next_y = torch.tensor([[reward_val]], dtype=torch.float64)
        self.train_x = torch.cat([self.train_x, next_x], dim=0)
        self.train_y = torch.cat([self.train_y, next_y], dim=0)

    def run_start(self, x, iteration):
        reward_val = self.training_model_run(x, self.reward, run_type="RANDOM START", iteration=iteration)["reward"]
        next_y = torch.tensor([[reward_val]], dtype=torch.float64)
        self.train_y = torch.cat([self.train_y, next_y], dim=0)

    def train(
        self,
        plot_manager,
        num_starts=1, # Integer 
        epochs=10,
        beta_start=5,
        beta_end=1,
        make_vids=False,
    ):
        iteration = 0
        start_time = time.time()
        clear_dirs(self.vid_dirs)

        for i in range(num_starts):
            self.train_x = torch.cat([self.train_x, self.choose_random_candidate()], dim=0)
        self.train_y = torch.tensor([], dtype=torch.float64)
        for i in range(len(self.train_x)):
            iteration += 1
            self.run_start(self.train_x[i], iteration=iteration)


        for i in range(iteration, epochs):
            beta = beta_schedule(i - iteration, epochs - iteration, beta_start, beta_end)
            self.run_epoch(i+1, beta, plot_manager)
            if make_vids:
                print_all_plots("output_BO/media", i)

        create_all_videos(self.vid_dirs)
        self.eval_model_run(self.get_results()[0], self.reward, self.output_dir, start_time)
    
    def get_results(self):
        self.best_idx = self.train_y.argmax()
        self.best_x =self.train_x[self.best_idx]
        self.best_y = self.train_y.min().item()
        return (self.best_x, self.best_y)