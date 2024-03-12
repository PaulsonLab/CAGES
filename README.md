# CAGES
Cost-Aware Gradient Entropy Search for Efficient Local
Multi-Fidelity Bayesian Optimization

Our code implementation extends the [GIBO's codebase](https://arxiv.org/abs/2106.11899) and [LVGP's codebase](https://arxiv.org/abs/1806.07504). More detail can be found in their repository ([GIBO](https://github.com/sarmueller/gibo/tree/main))([LVGP-python implementation](https://github.com/balaranjan/LVGP/tree/main))([LVGP-matlab implementation](https://doi.org/10.1080/00401706.2019.1638834)).

# Codes for the repo
[cartpole](https://github.com/PaulsonLab/CAGES/blob/1c9525c7246ac3c7511f79fa02d784f689c59aed/cartpole.py) : Please replace this file with the original cartpole.py file inside the gymnasium package (~/gymnasium/envs/classic_control/cartpole.py) to enable the changes of step time (tau).\
[src\environment_api](https://github.com/PaulsonLab/CAGES/blob/48ca4862a56500a48b9537c3e8df5c0817c4a78e/src/environment_api.py): Interface for interactions with reinforcement learning environments of OpenAI Gym.\
[RL_function](URL): Function that takes policy parameters and qualatative variable as input and return the reward for the RL problem.\
[acquisition_function](URL): Custom entropy-based acquisition function for gradient information.\
[LVGP_main](URL): Implementation of the Latent Variable Gaussian Process (LVGP) model by Zhang et al. [1]\
[lvgp_grad](URL): Calculate the gradient for the LVGP model (Jacobian LVGP).

# Usage
Vanilla BO
------------------------------
Run the following command to execute vanilla BO with expected imrpovement. User can specify which problem (Rosenbrock, OTL, or Cartpole) to be optimized in line 31-33.

Two numpy files (EI_cost.np/EI_reward.np) will be saved in user's local directory including the accumulated cost and best found value per iteration.
```sh
python Vanilla_BO.py
```

GIBO and ARS
------------------------------
Run this following command to execute GIBO or ARS algorithm. User can specify which problem (Rosenbrock, OTL, or Cartpole) to be optimized in line 37-39 and specify GIBO or ARS in line 41.

Two numpy files (GIBO_cost.np/GIBO_reward.np) will be saved in user's local directory including the accumulated cost and best found value per iteration.
```sh
cd GIBO
python local_GIBO_exe.py
```

CAGES
------------------------------
Run the following command to execute CAGES for the multi-information Rosenbrock test problem.

Two numpy files (Rosenbrock_cost_CAGES.np/Rosenbrock_reward_CAGES.np) will be saved in user's local directory including the accumulated cost and best found value per iteration.

```sh
python lvgp_optimization_new_Rosenbrock.py
```

Run the following command to execute CAGES for the multi-information OTL test problem.

Two numpy files (OTL_cost_CAGES.np/OTL_reward_CAGES.np) will be saved in user's local directory including the accumulated cost and best found value per iteration.

```sh
python lvgp_optimization_new_OTL.py
```

Run the following command to execute CAGES for the multi-information Cartpole RL control problem.

Two numpy files (Cartpole_cost_CAGES.np/Cartpole_reward_CAGES.np) will be saved in user's local directory including the accumulated cost and best found value per iteration.

```sh
python lvgp_optimization_new_RL.py
```

Plotting
------------------------------
Please run the following command to load saved numpy files and generate plots. User may need to modify the path to load the file.
```sh
cd Plotting
python Plotting.py
```
Here is an example plot for the Rosenbrock problem that user can reproduce based on the np files under ~\Results\Rosenbrock:

![image](https://github.com/PaulsonLab/CAGES/assets/101409886/aa7c160a-8474-47c5-9e2f-589516dba528)

# Reference
[1] Zhang, Y., Tao, S., Chen, W., & Apley, D. W. (2020). A latent variable approach to Gaussian process modeling with qualitative and quantitative factors. Technometrics, 62(3), 291-302.\
[2] Müller, S., von Rohr, A., & Trimpe, S. (2021). Local policy search with Bayesian optimization. Advances in Neural Information Processing Systems, 34, 20708-20720.
