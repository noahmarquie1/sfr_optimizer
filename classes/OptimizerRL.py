# Imports
from classes.ReactorSandbox import ReactorSandbox
from classes.GPModeler import GPModel, CSVConfigManager

from gymnasium.spaces import Box
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood

import numpy as np
import gymnasium as gym

import time
import math


# Helper Functions
def gp_iter_schedule(current_iter, max_iters, start_amount):
    return start_amount * (-current_iter / max_iters + 1)


# Classes
class ActionNoiseCallback(BaseCallback):
    def __init__(self, optimizer, action_noise, max_iters, verbose: int = 0):
        super().__init__(verbose)
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.action_noise = action_noise

    def _on_step(self):
        sigma = self.optimizer.max_sigma * min(1.0, ((self.max_iters + self.optimizer.learning_starts - self.n_calls - 1) / (self.max_iters)))
        self.action_noise._sigma = sigma * np.ones(self.optimizer.n_actions)
        return True


# Environment Classes
class SingleEpisodeEnv(gym.Env):
    def __init__(self, reward, reactor_sim, training_reactor_sim, plot_manager=None, training_starts=0):
        
        self.action_space = Box(0.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = Box(0.0, 10.0, shape=(3,), dtype=np.float32)

        self.reward = reward
        self.reactor_sim = reactor_sim
        self.training_reactor_sim = training_reactor_sim
        self.plot_manager = plot_manager

        self.iters = 0
        self.training_starts = training_starts

        self.current_layout = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64)
        self.current_solution = []

        self.vid_dirs = [
            "output_BO/media/fission_heating_rate",
            "output_BO/media/plot_xy",
            "output_BO/media/plot_yz",
            "output_BO/media/radial_heating_distribution",
        ]


    def _get_obs(self, verbose=0):
        if verbose:
            info = self.training_reactor_sim(
                self.current_layout,
                self.reward,
                plot_manager = (self.plot_manager if self.iters > self.training_starts else None),
                run_type = "TRAINING" if self.iters > self.training_starts else "RANDOM START",
                iteration = self.iters,
            )
        else:
            info = self.reactor_sim(self.current_layout)

        keff = info["k_eff"].n
        pkf = info["peaking_factor"]
        heating_rate = info["heating_rate"] * 1e-8
        return np.array([keff, pkf, heating_rate], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        self.current_layout = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64)
        obs = 0
        info = {}
        return obs, info

    def step(self, action):
        self.iters += 1
        
        self.current_layout = action
        self.current_solution = self.current_layout
        obs = self._get_obs(verbose=1)
        info = self._get_info()

        reward = self.reward(keff=obs[0], peaking_factor=obs[1], heating_rate=obs[2])[0]
        terminated, truncated = False, False

        return obs, reward, terminated, truncated, info
    
    
# Envs using Gaussian Process
class SingleEpisodeGPEnv(SingleEpisodeEnv):
    def __init__(self, reward, space, reactor_sim, training_reactor_sim, plot_manager=None, training_starts=0):
        super().__init__(reward, reactor_sim, training_reactor_sim, plot_manager, training_starts)

        self.reward = reward
        self.space = space
        self.reactor_frequency = 10

        self.observation_space = Box(0.0, 10.0, shape=(3,), dtype=np.float32)
        self.current_layout = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        config_manager = CSVConfigManager(space=self.space, csv_path="data_RL/gp_data.csv", num_particles=2000)
        x, y = config_manager.csv_to_tensor()

        keff_likelihood = GaussianLikelihood()
        pkf_likelihood = GaussianLikelihood()
        heating_rate_likelihood = GaussianLikelihood()

        self.keff_gp = GPModel(x, y[:, 0], likelihood=keff_likelihood)
        self.pkf_gp = GPModel(x, y[:, 1], likelihood=pkf_likelihood)
        self.heating_rate_gp = GPModel(x, y[:, 2], likelihood=heating_rate_likelihood)

        self.keff_gp.load_state_dict(torch.load("data_RL/keff_gp.pth")["model_state_dict"])
        self.pkf_gp.load_state_dict(torch.load("data_RL/pkf_gp.pth")["model_state_dict"])
        self.heating_rate_gp.load_state_dict(torch.load("data_RL/heating_rate_gp.pth")["model_state_dict"])

        keff_likelihood.load_state_dict(torch.load("data_RL/keff_gp.pth")["likelihood_state_dict"])
        pkf_likelihood.load_state_dict(torch.load("data_RL/pkf_gp.pth")["likelihood_state_dict"])
        heating_rate_likelihood.load_state_dict(torch.load("data_RL/heating_rate_gp.pth")["likelihood_state_dict"])

        self.keff_gp.eval()
        self.pkf_gp.eval()
        self.heating_rate_gp.eval()

        keff_likelihood.eval()
        pkf_likelihood.eval()
        heating_rate_likelihood.eval()
        
    def _get_obs_gaussian(self):
        x = torch.from_numpy(self.current_layout).unsqueeze(0).double()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            keff = self.keff_gp(x).mean.item()
            pkf = self.pkf_gp(x).mean.item()
            heating_rate = self.heating_rate_gp(x).mean.item()

        return np.array([keff, pkf, heating_rate], dtype=np.float32)

    def _get_obs_reactor(self):
        info = self.training_reactor_sim(
            self.current_layout,
            self.reward,
            plot_manager = (self.plot_manager if self.iters > self.training_starts else None),
            run_type = "TRAINING" if self.iters > self.training_starts else "RANDOM START",
            iteration = self.iters,
        )

        keff = info["k_eff"].n
        pkf = info["peaking_factor"]
        heating_rate = info["heating_rate"] * 1e-8
        return np.array([keff, pkf, heating_rate], dtype=np.float32)

    def _is_reactor_step(self, iters):
        if iters % self.reactor_frequency == 0:
            return True
        return False

    def step(self, action):
        self.current_layout = action
        self.current_solution = self.current_layout

        self.iters += 1
        obs = self._get_obs_reactor() if self._is_reactor_step(self.iters) else self._get_obs_gaussian()

        reward, keff_rew, pkf_rew, heating_rate_rew = self.reward(keff=obs[0], peaking_factor=obs[1], heating_rate=obs[2])

        enrichment_ring1 = 17 * (self.current_layout[2]) + 2
        enrichment_ring2 = 17 * (self.current_layout[3]) + 2
        enrichment_ring3 = 17 * (self.current_layout[4]) + 2

        if self.iters > self.training_starts:
            plot_data = {
                "enrichment_ring1": enrichment_ring1,
                "enrichment_ring2": enrichment_ring2, 
                "enrichment_ring3": enrichment_ring3, 
                "enrichment-composite": {"enrichment_ring1": enrichment_ring1, "enrichment_ring2": enrichment_ring2, "enrichment_ring3": enrichment_ring3},
                "fuel_radius": self.current_layout[0],
                "min_dist_pin2pin": self.current_layout[1],
                "keff": obs[0],
                "pkf": obs[1],
                "reward-composite": {"Full": reward, "k_eff": keff_rew, "peaking_factor": pkf_rew, "heating_rate": heating_rate_rew},
                "reward-comparison": {"Gaussian Process": reward},
            }

            self.plot_manager.add(self.iters - self.training_starts, plot_data)

        info = self._get_info()
        terminated, truncated = False, False
        if self.iters % 5 == 0 and not self.iters % 10 == 0:
            print(f"====== Iteration {self.iters} ======")
            print(f"Enrichments: {enrichment_ring1:.2f}, {enrichment_ring2:.2f}, {enrichment_ring3:.2f}")
            print(f"Reward: {reward}\n")

        return obs, reward, terminated, truncated, info


class MultiEpisodeGPEnv(SingleEpisodeGPEnv):
    def __init__(self, reward, space, reactor_sim, training_reactor_sim, plot_manager=None, training_starts=0):
        super().__init__(
            reward=reward, 
            space=space, 
            reactor_sim=reactor_sim, 
            training_reactor_sim=training_reactor_sim, 
            plot_manager=plot_manager, 
            training_starts=training_starts
        )

        self.action_space = Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = Box(0.0, 10.0, shape=(6,), dtype=np.float32)
        self.current_layout = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_layout = np.random.rand(5).astype(np.float32)
        info = {}
        return np.append(self.current_layout, 0), info

    def _get_obs(self):
        progress = (self.iters % 10) / 10
        return np.append(self.current_layout, progress)
    
    def _get_info(self):
        return {}
    
    def step(self, action):
        #self.current_layout = np.clip(self.current_layout + action, 0.0, 1.0)
        #self.current_solution = self.current_layout
        
        new_layout = np.empty_like(self.current_layout)
        for i in range(len(action)):
            if action[i] >= 0:
                increment = action[i] * 0.1 * (1.0 - self.current_layout[i]) ** (1/2)
                new_layout[i] = self.current_layout[i] + increment
            else:
                increment = action[i] * 0.1 * (self.current_layout[i] - 0.0) ** (1/2)
                new_layout[i] = self.current_layout[i] + increment
        self.current_layout = np.clip(new_layout, 0.0, 1.0)
        self.current_solution = self.current_layout

        

        obs = self._get_obs()
        info = self._get_info()
        x = torch.from_numpy(self.current_layout).unsqueeze(0).double()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            keff = self.keff_gp(x).mean.item()
            pkf = self.pkf_gp(x).mean.item()
            heating_rate = self.heating_rate_gp(x).mean.item()

        reward, keff_rew, pkf_rew, heating_rate_rew = self.reward(keff=keff, peaking_factor=pkf, heating_rate=heating_rate)

        enrichment_ring1 = 17 * (self.current_layout[2]) + 2
        enrichment_ring2 = 17 * (self.current_layout[3]) + 2
        enrichment_ring3 = 17 * (self.current_layout[4]) + 2

        plot_data = {
            "enrichment_ring1": enrichment_ring1,
            "enrichment_ring2": enrichment_ring2, 
            "enrichment_ring3": enrichment_ring3, 
            "enrichment-composite": {"enrichment_ring1": enrichment_ring1, "enrichment_ring2": enrichment_ring2, "enrichment_ring3": enrichment_ring3},
            "fuel_radius": self.current_layout[0],
            "min_dist_pin2pin": self.current_layout[1],
            "keff": keff,
            "pkf": pkf,
            "reward-composite": {"Full": reward, "k_eff": keff_rew, "peaking_factor": pkf_rew, "heating_rate": heating_rate_rew},
            "reward-comparison": {"Gaussian Process": reward},
        }

        self.iters += 1

        if self.iters % 1000 == 0 and self.iters >= self.training_starts:
            obs_sim = self.reactor_sim(torch.tensor(self.current_layout, dtype=torch.float64))
            reward_sim = self.reward(keff=obs_sim["k_eff.n"], peaking_factor=obs_sim["peaking_factor"], heating_rate=obs_sim["heating_rate"])[0]
            plot_data["reward-comparison"]["Simulation"] = reward_sim

        info = self._get_info()
        if self.iters % 10 == 0:  
            reward *= 4
            terminated = True
            if self.iters > self.training_starts:
                print(f"Iteration {self.iters} Config tested: {self.current_layout[0]:.2f}, {self.current_layout[1]:.2f}, {self.current_layout[2]:.2f}, {self.current_layout[3]:.2f}, {self.current_layout[4]:.2f} --- Reward: {reward / 4}")
                self.plot_manager.add(self.iters - self.training_starts, plot_data)
            else:
                print(f"Iteration {self.iters}")
        else:                  
            terminated = False

        return obs, reward, terminated, False, info


# Optimizer Class
class OptimizerRL(ReactorSandbox):
    def __init__(self, space, plot_manager, output_dir, reward, learning_starts=15, **kwargs):
        super().__init__(space=space, **kwargs)
        self.n_actions = len(space.keys())
        
        self.output_dir = output_dir
        self.learning_starts = learning_starts
        self.space = space
        self.plot_manager = plot_manager
        self.max_sigma = 0.015
        self.reward=reward

        self.action_noise = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=self.max_sigma * np.ones(self.n_actions))

        self.env = MultiEpisodeGPEnv(
            reward=self.reward, 
            space=self.space,
            reactor_sim=self.minimal_model_run,
            training_reactor_sim=self.training_model_run,
            training_starts=self.learning_starts,
            plot_manager=self.plot_manager,
        )

        self.model = TD3(
            policy="MlpPolicy",
            env=self.env,
            gamma=0.99,
            learning_starts=learning_starts,
            learning_rate=3e-4,
            batch_size=256,
            train_freq=1,
            action_noise=self.action_noise,
            buffer_size=100000,
            verbose=1,
        )

    def train(self, n_iters):
        start_time = time.time()
        self.model.learn(
            total_timesteps=n_iters,
            callback=ActionNoiseCallback(optimizer=self, action_noise=self.action_noise, max_iters=n_iters),
            log_interval=100
        )
        print(f"Time Elapsed: {time.time() - start_time}")
        self.eval_model_run(torch.tensor(self.get_results(), dtype=torch.float64), self.reward, self.output_dir, start_time)

    def get_results(self):
        return self.env.current_solution

