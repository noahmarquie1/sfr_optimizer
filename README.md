# ⚛ SFR Nuclear Reactor Optimizer

Bayesian Optimization and Reinforcement Learning implementations
for uranium-based fast-reactor core optimization.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage (Sandbox)](#-usage-sandbox)
- [Usage (Optimizer)](#-usage-optimizer)
- [License](#-license)

---

## 🔍 Overview

This repository was inspired by my work as an AI/ML Engineering Intern at Transmutex, 
a Geneva-based company developing software solutions to pioneer safe, sustainable 
transmutation-based nuclear energy. During my internship, I developed a suite of optimization 
tools for Transmutex’s proprietary thorium-based, lead-cooled fast reactor 
design. Note on IP: the work presented here does not contain any proprietary information or inventions
owned by Transmutex SA. 

Included here are two of the most effective approaches I implemented:

* Bayesian Optimization using BoTorch, a scalable, flexible framework built on PyTorch for sample-efficient black-box 
optimization in high-dimensional parameter spaces.
* Reinforcement Learning using the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm from Stable-Baselines3, 
applied to learn optimal core configurations through interaction with a physics-informed simulation environment.

Although my original work targeted Transmutex's thorium-based reactor configuration, the methods demonstrated here are 
applied to an industry-standard uranium-based fast reactor model.

---

## ✨ Features

- ✅ [Interactive CLI for Sandbox and Optimizers]
- 📊 [Visualized Results]
- 🐳 [Containerized with Docker]

---

## 🛠️ Tech Stack

- **Language**: Python 3.12
- **Libraries**: OpenMC, BoTorch, PyTorch, StableBaselines3, Gymnasium, SciPy, NumPy, Pandas Matplotlib
- **Tools**: Docker, Git (Github)

---

### 🔧 Prerequisites

- Python 3.10+
- Docker

---

### 🔨 Installation

```bash
git clone https://github.com/project.git sfr_optimizer
cd sfr_optimizer
docker buildx build -t sfr-image .
docker run -it -v "$(pwd)":/workspace sfr-image
cd workspace
```

**Note:** Building the Docker image may take 5-10 minutes.

--- 

### ⚡️ Usage (Sandbox)

To visualize the geometry before running:

```python
python reactor_core.py --plot_geom
```

To run:
```python
python reactor_core.py --run 
```

Final output should look like:

```bash
============================>     RESULTS     <============================

 k-effective (Collision)     = 1.25286 +/- 0.00323
 k-effective (Track-length)  = 1.25246 +/- 0.00356
 k-effective (Absorption)    = 1.24991 +/- 0.00456
 Combined k-effective        = 1.25178 +/- 0.00247
 Leakage Fraction            = 0.28848 +/- 0.00160

--------------output for optimization------------------
k_eff: 1.25e+00 [n_neutrons/source]
Total fission heating rate: 8.34e+07 [eV/source]
Total capture rate of Th: 1.34e-01 [n_captures/source]
Total capture rate of Pu: 2.75e-02 [n_captures/source]
Peaking factor: 3.441
--------------output for optimization------------------
```


---

### ⚡️ Usage (Optimizer)

To run a Bayesian Optimization loop:

```bash
python main.py --train_bo {num-random-starts} {num-total-iterations}
```

To run a Reinforcement Learning Optimization loop:

```bash
python main.py --train_rl {num-random-starts} {num-total-iterations}
```

Each iteration should look like:
```bash
============ Iteration: 42 (TRAINING) ============
(PRE)  : fuel_radius: 1.5cm, min_dist_pin2pin: 0.5cm
       : enrichment_ring1: 10%, enrichment_ring2: 12%
       : enrichment_ring3: 14%, enrichment_ring4: 16%
       
(POST) : k_eff: 0.975 ± 86, peaking_factor: 1.2
       : capture_rate_U: 0.7, heating_rate: 0.2
       : loss: 0.98
```

The program will print the following plots to /metrics:
1. Fuel Radius vs. iterations
2. MinDist Pin2Pin vs. iterations
3. Enrichments 1-3 vs. iterations 
4. Reward vs. iterations
5. K_eff (criticality) vs. iterations
6. Peaking Factor vs. iterations
8. Heating Rate vs. iterations

And the following movies to respective folders within /output_BO or /output_RL:
1. Fission Heating Rate vs. iterations 
2. Plot XY vs. iterations 
3. Plot YZ vs. iterations 
4. Radial Heating Distribution vs. iterations

--- 

### 📄 License

This project is licensed under the [MIT License](LICENSE).