# CAGES
Cost-Aware Gradient Entropy Search for Efficient Local
Multi-Fidelity Bayesian Optimization

Our code implementation extends the [GIBO's codebase](URL) and [LVGP's codebase](URL). More detail can be found in their repository ([GIBO](URL))([LVGP](URL)).

# Codes for the repo
[cartpole](https://github.com/PaulsonLab/CAGES/blob/1c9525c7246ac3c7511f79fa02d784f689c59aed/cartpole.py) : Please replace this file with the original cartpole.py file inside the gymnasium package (gymnasium/envs/classic_control/cartpole.py) to enable the changes of step time (tau).\
[src2\environment_api](URL): Interface for interactions with reinforcement learning environments of OpenAI Gym.\
[RL_function](URL): Function that takes policy parameters and qualatative variable as input and return the reward for the RL problem.\
[acquisition_function](URL): Custom entropy-based acquisition function for gradient information.\
[Vanilla_BO](URL): Run this file to execute vanilla BO with expected imrpovement. User can specify which problem to be optimized in line 31-33.\
[GIBO\local_GIBO_exe](URL): Run this file to execute GIBO or ARS algorithm. User can specify which problem to be optimized in line 43-45 and specify GIBO or ARS in line 47.\
[LVGP_main](URL): Implementation of the Latent Variable Gaussian Process (LVGP) model by Zhang et al.\
[lvgp_grad](URL): Calculate the gradient for the LVGP model.\
[lvgp_optimization_new_Rosenbrock](URL): Run this file to execute CAGES for the multi-information Rosenbrock test function.\
[lvgp_optimization_new_OTL](URL):Run this file to execute CAGES for the multi-information OTL test function.\
[lvgp_optimization_new_RL](URL):Run this file to execute CAGES for the multi-information Cartpole RL control problem.\

