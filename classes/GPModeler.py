from classes.ReactorSandbox import ReactorSandbox

from scipy.stats import qmc

import torch
import gpytorch

import numpy as np
import pandas as pd
import os
from pathlib import Path


# Class
class CSVConfigManager(ReactorSandbox):
    def __init__(
            self,
            space,
            csv_path,
            mesh_dimension=[100, 100],
            nbin_radial=20,
            num_particles=5000,
            num_threads=40,
        ):
        super().__init__(
            space, 
            mesh_dimension=mesh_dimension,
            nbin_radial=nbin_radial,
            num_particles=num_particles,
            num_threads=40,
        )

        self.current_x = []
        self.csv_path = csv_path

        if not os.path.isfile(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            Path(self.csv_path).touch(exist_ok=True)


    def create_config_array(self, num_configs, objective):
        sampler = qmc.LatinHypercube(d=5)
        initial_x = sampler.random(num_configs)

        final_x = np.array([], dtype=np.float64).reshape(0, 5)
        y = np.array([], dtype=np.float64).reshape(0, 3)

        self.update_current_x()
        for config in initial_x:
            if not self.item_in_csv(config):
                final_x = np.vstack([final_x, config])

        for i, config in enumerate(final_x):
            info = self.training_model_run(
                config,
                reward=objective,
                run_type="TRAINING GP",
                iteration=i+1
            )

            next_y_array = np.array([[info["k_eff"].n, info["peaking_factor"], info["heating_rate"] * 1e-8]], dtype=np.float64)
            y = np.vstack([y, next_y_array])
        return final_x, y
        

    def write_to_csv(self, x, y, mode="w"):
        self.update_current_x()
        if mode == "a" and self.current_x.shape[0] == 0:
            mode == "w"
        
        df = pd.DataFrame(np.hstack([x, y]), dtype=np.float64)
        df.to_csv(self.csv_path, header=False if mode == "a" else True, mode=mode, index=False)
    

    def item_in_csv(self, x):
        self.update_current_x()
        in_csv = np.any(np.all(self.current_x == x, axis=1))
        return in_csv
    

    def update_current_x(self):
        try:
            self.current_x = pd.read_csv(self.csv_path, index_col=0, dtype=np.float64)
        except: # If file is empty, we want to write the headers
            df = pd.DataFrame(np.array([]).reshape(0, 8))
            df.to_csv(self.csv_path, header=True, index=False)
            self.current_x = pd.read_csv(self.csv_path, index_col=0, dtype=np.float64)
        self.current_x = self.current_x.to_numpy(dtype=np.float64)[:, :5] if not self.current_x.empty else np.array([]).reshape(0, 5)


    def create_configs(self, num_configs, objective, mode="w"):
        x, y = self.create_config_array(num_configs, objective)
        self.write_to_csv(x, y, mode)


    def csv_to_tensor(self):
        df = pd.read_csv(self.csv_path)
        if df.empty:
            print("CSV is empty")
            return
        x = df.to_numpy()[:, :5]
        y = df.to_numpy()[:, 5:]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super(GPModel, self).__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.x = x
        self.y = y
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def save_gp(self, path):
        torch.save({
            "model_state_dict": self.state_dict(),
            "likelihood_state_dict": self.likelihood.state_dict(),
        }, path)

    def train_rl(self, iters):
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(iters):
            optimizer.zero_grad()
            output = self(self.x)
            loss = -mll(output, self.y).mean()
            loss.backward()
            print("Iter %d/%d - Loss: %.3f - lengthscale: %.3f - noise: %3f" % (
                i + 1, iters, loss.item(),
                self.covar_module.base_kernel.lengthscale.item(),
                self.likelihood.noise.item()
            ))
            optimizer.step()

    




