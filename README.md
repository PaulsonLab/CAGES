# CAGES
Cost-Aware Gradient Entropy Search for Efficient Local
Multi-Fidelity Bayesian Optimization

Our code implementation extends the [GIBO's codebase](https://arxiv.org/abs/2106.11899) and [LVGP's codebase](https://arxiv.org/abs/1806.07504). More detail can be found in their repository ([GIBO](https://github.com/sarmueller/gibo/tree/main))([LVGP-python implementation](https://github.com/balaranjan/LVGP/tree/main))([LVGP-matlab implementation](https://doi.org/10.1080/00401706.2019.1638834).

# Codes for the repo
[cartpole](https://github.com/PaulsonLab/CAGES/blob/1c9525c7246ac3c7511f79fa02d784f689c59aed/cartpole.py) : Please replace this file with the original cartpole.py file inside the gymnasium package (~/gymnasium/envs/classic_control/cartpole.py) to enable the changes of step time (tau).\
[src2\environment_api](URL): Interface for interactions with reinforcement learning environments of OpenAI Gym.\
[RL_function](URL): Function that takes policy parameters and qualatative variable as input and return the reward for the RL problem.\
[acquisition_function](URL): Custom entropy-based acquisition function for gradient information.\
[Vanilla_BO](URL): Run this file to execute vanilla BO with expected imrpovement. User can specify which problem to be optimized in line 31-33.\
[GIBO\local_GIBO_exe](URL): Run this file to execute GIBO or ARS algorithm. User can specify which problem to be optimized in line 37-39 and specify GIBO or ARS in line 41.\
[LVGP_main](URL): Implementation of the Latent Variable Gaussian Process (LVGP) model by Zhang et al.\
[lvgp_grad](URL): Calculate the gradient for the LVGP model.\
[lvgp_optimization_new_Rosenbrock](URL): Run this file to execute CAGES for the multi-information Rosenbrock test problem.\
[lvgp_optimization_new_OTL](URL):Run this file to execute CAGES for the multi-information OTL test problem.\
[lvgp_optimization_new_RL](URL):Run this file to execute CAGES for the multi-information Cartpole RL control problem.\

# Usage
Vanilla BO
------------------------------

GIBO and ARS
------------------------------

CAGES
------------------------------

# Reference
Müller, S., von Rohr, A., & Trimpe, S. (2021). Local policy search with Bayesian optimization. Advances in Neural Information Processing Systems, 34, 20708-20720.\
Zhang, Y., Tao, S., Chen, W., & Apley, D. W. (2020). A latent variable approach to Gaussian process modeling with qualitative and quantitative factors. Technometrics, 62(3), 291-302.
