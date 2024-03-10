# CAGES
Cost-Aware Gradient Entropy Search for Efficient Local
Multi-Fidelity Bayesian Optimization

# Codes for the repo
[cartpole](https://github.com/PaulsonLab/CAGES/blob/1c9525c7246ac3c7511f79fa02d784f689c59aed/cartpole.py) : Please replace this file with the original cartpole.py file inside the gymnasium package (gymnasium/envs/classic_control/cartpole.py) to enable the changes of step time (tau) and episode length.\
[Vanilla_BO](URL): Run this file to execute vanilla BO with expected imrpovement. User can specify which problem to be optimized in line 31-33.\
[GIBO\local_GIBO_exe](URL): Run this file to execute GIBO or ARS algorithm. User can specify which problem to be optimized in line 43-45 and specify GIBO or ARS in line 47.\
[lvgp_optimization_new_Rosenbrock](URL): Run this file to execute CAGES for the multi-information Rosenbrock test function.\
[lvgp_optimization_new_OTL](URL):Run this file to execute CAGES for the multi-information OTL test function.\
[lvgp_optimization_new_RL](URL):Run this file to execute CAGES for the multi-information Cartpole RL control problem.\

