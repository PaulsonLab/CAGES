#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:07:02 2023

@author: tang.1856
"""

# All codes are built on "minimization" problem

import numpy as np
import os
import gpytorch
import botorch
import torch
from torch.quasirandom import SobolEngine
from pyDOE2 import lhs
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
from botorch.test_functions.synthetic import Ackley, Rosenbrock, Levy, StyblinskiTang, Griewank, Michalewicz, Rastrigin, DixonPrice
from src2.model import DerivativeExactGPSEModel
from src2.acquisition_function import GradientInformation, DownhillQuadratic,  GradientInformation_entropy
from src2.acquisition_function import optimize_acqf_vanilla_bo, optimize_acqf_custom_bo
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from gpytorch.utils.cholesky import psd_safe_cholesky
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from torch.distributions import Normal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel
from gpytorch.constraints import Positive
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from src2.synthetic_functions import (
    generate_objective_from_gp_post,
    generate_training_samples,
    get_maxima_objectives,
    get_lengthscales,
    factor_hennig,
    get_lengthscale_hyperprior,
)

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
    "dtype": torch.double,
    
}

# Codes are built upon  https://github.com/kayween/local-bo-mpd/blob/main/src/optimizers.py  and  https://github.com/sarmueller/gibo/blob/main/src/optimizers.py
class GIBO(): # This class contains all gradient-based algorithm
    def __init__(self, objective, dim, delta, lr, epsilon_diff_acq_value, lb,ub, Ninit, reward_list, cost_list):
        self.objective = objective
        self.dim = dim
        self.delta = delta
        self.epsilon_diff_acq_value = epsilon_diff_acq_value
        self.normalize_gradient: bool = True,
        self.standard_deviation_scaling: bool = False,
        self.lr_schedular = None
        self.lr = lr
        self.Ninit = Ninit
        self.lb = lb
        self.ub = ub
        self.reward_list = reward_list
        self.cost_list = cost_list
             
    def __call__(self, X, Y, params):
        torch.manual_seed(1)
        
        self.lb = self.lb.to("cpu")
        self.ub = self.ub.to("cpu")
        X = X.to(**tkwargs)
        if Y.size(-1) != len(Y):
            Y = Y.to(**tkwargs).squeeze(-1)
        else:
            Y = Y.to(**tkwargs)
              
        self.params = params.to(**tkwargs)
       
        self.optimizer_torch = torch.optim.SGD([self.params],lr=self.lr)
        
        self.gp = DerivativeExactGPSEModel(self.dim, ard_num_dims=self.dim)
        self.gp.append_train_data(X, Y)
        self.acquisition_fcn = GradientInformation(self.gp)
        
        self.gp.posterior(self.params) # Call this to update prediction strategy of GPyTorch (get_L_lower, get_K_XX_inv)
        self.acquisition_fcn.update_theta_i(self.params)
        
        bounds = torch.tensor([[-self.delta], [self.delta]]) + self.params
        bounds[bounds<0] = 0
        bounds[bounds>1] = 1
        bounds = bounds.to("cpu")
        
        # train GP
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp
        )
        
        botorch.fit.fit_gpytorch_mll(mll)
        
        self.gp.posterior(
            self.params
        )  # Call this to update prediction strategy of GPyTorch.
        
        lengthscale = self.gp.covar_module.base_kernel.lengthscale
        print('lengthscale:', lengthscale)
        # print('output scale:', self.gp.covar_module.outputscale)
        # print('noise:', self.gp.likelihood.noise)
        
        self.acquisition_fcn.update_theta_i(self.params) 
              
        # inner loop for GIBO (enhance gradient information)
        acq_value_old = None
        max_samples_per_iteration = int(0.5*self.dim)
        for i in range(max_samples_per_iteration):
            
            new_x, acq_value = optimize_acqf_custom_bo(self.acquisition_fcn, bounds, q=1, num_restarts = 5, raw_samples = 20)
            new_y = self.objective(self.lb + (self.ub - self.lb) *new_x)
            
            if new_y.max() > Y.max():
                ind_best = new_y.argmax()               
                print(
                    f"New best query: {new_y[ind_best].item():.3f}"
                )
                
            X = torch.cat((new_x, X))
            Y = torch.cat((new_y, Y))
            
            self.reward_list.append(float(max(new_y, self.reward_list[-1])))
            self.cost_list.append(len(X))
            
            self.gp.append_train_data(new_x, new_y)
            self.gp.posterior(self.params)
            self.acquisition_fcn.update_K_xX_dx()
            
                           
            if acq_value_old is not None:
                diff = acq_value - acq_value_old
                if diff < self.epsilon_diff_acq_value:
                    print(f"Stop sampling after {i+1} samples, since gradient certainty is {diff}.")
                    break
                
            acq_value_old = acq_value
       
        mean_d, variance_d = self.gp.posterior_derivative(self.params) # gradient predicted by GP
        print('gradient GP=', mean_d*Y.std()/(self.ub-self.lb))
        
        
        self.move_gibo(method="step")
        Y_next = torch.cat([self.objective(self.lb + (self.ub - self.lb) *self.params)])   
        
        X = torch.cat((self.params,X))
        Y = torch.cat((Y_next,Y))    
         
        self.reward_list.append(float(max(Y_next, self.reward_list[-1])))
        self.cost_list.append(len(X))
      
        if Y_next.max() > Y[1:].max():
            ind_best = Y_next.argmax()          
            print(
                f"New best moving: {Y_next[ind_best].item():.3f}"
            )
                
        return X, Y, (self.params.flatten()).unsqueeze(0), self.reward_list, self.cost_list
    
    
    def move_gibo(self, method):
        if method == "step":
            with torch.no_grad():
                self.optimizer_torch.zero_grad()
                self.params.grad = torch.zeros_like(self.params)
                mean_d, variance_d = self.gp.posterior_derivative(self.params)
                params_grad = -mean_d.view(1, self.dim)
                if self.normalize_gradient:
                    lengthscale = (
                        self.gp.covar_module.base_kernel.lengthscale.detach()
                    )
                    params_grad = (
                        torch.nn.functional.normalize(params_grad) * lengthscale
                    )
                # if self.standard_deviation_scaling:
                #     params_grad = params_grad / torch.diag(
                #         variance_d.view(self.dim, self.dim)
                #     )
                if self.lr_schedular:
                    lr = [
                        v for k, v in self.lr_schedular.items() if k <= self.iteration
                    ][-1]
                    self.params.grad[:] = lr * params_grad  # Define as gradient ascent.
                else:
                    self.params.grad[:] = params_grad  # Define as gradient ascent.
                self.optimizer_torch.step()
                self.params[self.params<0] = 0
                self.params[self.params>1] = 1

class GIBO_LF(): # This class contains all gradient-based algorithm
    def __init__(self, objective, objective_HF, dim, delta, lr, epsilon_diff_acq_value, lb,ub, Ninit, reward_list, cost_list):
        self.objective = objective
        self.objective_HF = objective_HF
        self.dim = dim
        self.delta = delta
        self.epsilon_diff_acq_value = epsilon_diff_acq_value
        self.normalize_gradient: bool = True,
        self.standard_deviation_scaling: bool = False,
        self.lr_schedular = None
        self.lr = lr
        self.Ninit = Ninit
        self.lb = lb
        self.ub = ub
        self.reward_list = reward_list
        self.cost_list = cost_list
             
    def __call__(self, X, Y, params):
        torch.manual_seed(1)
        
        self.lb = self.lb.to("cpu")
        self.ub = self.ub.to("cpu")
        X = X.to(**tkwargs)
        if Y.size(-1) != len(Y):
            Y = Y.to(**tkwargs).squeeze(-1)
        else:
            Y = Y.to(**tkwargs)
              
        self.params = params.to(**tkwargs)
       
        self.optimizer_torch = torch.optim.SGD([self.params],lr=self.lr)
        
        self.gp = DerivativeExactGPSEModel(self.dim, ard_num_dims=self.dim)
        self.gp.append_train_data(X, Y)
        self.acquisition_fcn = GradientInformation(self.gp)
        
        self.gp.posterior(self.params) # Call this to update prediction strategy of GPyTorch (get_L_lower, get_K_XX_inv)
        self.acquisition_fcn.update_theta_i(self.params)
        
        bounds = torch.tensor([[-self.delta], [self.delta]]) + self.params
        bounds[bounds<0] = 0
        bounds[bounds>1] = 1
        bounds = bounds.to("cpu")
        
        # train GP
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp
        )
        
        botorch.fit.fit_gpytorch_mll(mll)
        
        self.gp.posterior(
            self.params
        )  # Call this to update prediction strategy of GPyTorch.
        
        lengthscale = self.gp.covar_module.base_kernel.lengthscale
        print('lengthscale:', lengthscale)
        # print('output scale:', self.gp.covar_module.outputscale)
        # print('noise:', self.gp.likelihood.noise)
        
        self.acquisition_fcn.update_theta_i(self.params) 
              
        # inner loop for GIBO (enhance gradient information)
        acq_value_old = None
        max_samples_per_iteration = int(0.5*self.dim)
        for i in range(max_samples_per_iteration):
            
            new_x, acq_value = optimize_acqf_custom_bo(self.acquisition_fcn, bounds, q=1, num_restarts = 5, raw_samples = 20)
            new_y = self.objective(self.lb + (self.ub - self.lb) *new_x)
            
            if new_y.max() > Y.max():
                ind_best = new_y.argmax()               
                print(
                    f"New best query: {new_y[ind_best].item():.3f}"
                )
                
            X = torch.cat((new_x, X))
            Y = torch.cat((new_y, Y))
            new_y_HF = self.objective_HF(self.lb + (self.ub - self.lb) *new_x)
            self.reward_list.append(float(max(new_y_HF, self.reward_list[-1])))
            self.cost_list.append(len(X))
            
            self.gp.append_train_data(new_x, new_y)
            self.gp.posterior(self.params)
            self.acquisition_fcn.update_K_xX_dx()
            
                           
            if acq_value_old is not None:
                diff = acq_value - acq_value_old
                if diff < self.epsilon_diff_acq_value:
                    print(f"Stop sampling after {i+1} samples, since gradient certainty is {diff}.")
                    break
                
            acq_value_old = acq_value
       
        mean_d, variance_d = self.gp.posterior_derivative(self.params) # gradient predicted by GP
        print('gradient GP=', mean_d*Y.std()/(self.ub-self.lb))
        
        
        self.move_gibo(method="step")
        Y_next = torch.cat([self.objective(self.lb + (self.ub - self.lb) *self.params)])   
        
        X = torch.cat((self.params,X))
        Y = torch.cat((Y_next,Y))    
        Y_next_HF = self.objective_HF(self.lb + (self.ub - self.lb) *self.params)
        self.reward_list.append(float(max(Y_next_HF, self.reward_list[-1])))
        self.cost_list.append(len(X))
      
        if Y_next.max() > Y[1:].max():
            ind_best = Y_next.argmax()          
            print(
                f"New best moving: {Y_next[ind_best].item():.3f}"
            )
                
        return X, Y, (self.params.flatten()).unsqueeze(0), self.reward_list, self.cost_list
    
    
    def move_gibo(self, method):
        if method == "step":
            with torch.no_grad():
                self.optimizer_torch.zero_grad()
                self.params.grad = torch.zeros_like(self.params)
                mean_d, variance_d = self.gp.posterior_derivative(self.params)
                params_grad = -mean_d.view(1, self.dim)
                if self.normalize_gradient:
                    lengthscale = (
                        self.gp.covar_module.base_kernel.lengthscale.detach()
                    )
                    params_grad = (
                        torch.nn.functional.normalize(params_grad) * lengthscale
                    )
                # if self.standard_deviation_scaling:
                #     params_grad = params_grad / torch.diag(
                #         variance_d.view(self.dim, self.dim)
                #     )
                if self.lr_schedular:
                    lr = [
                        v for k, v in self.lr_schedular.items() if k <= self.iteration
                    ][-1]
                    self.params.grad[:] = lr * params_grad  # Define as gradient ascent.
                else:
                    self.params.grad[:] = params_grad  # Define as gradient ascent.
                self.optimizer_torch.step()
                self.params[self.params<0] = 0
                self.params[self.params>1] = 1
class ARS():
    def __init__(self, objective, lb, ub, reward_list, cost_list, samples_per_iteration=1, exploration_noise=0.02, step_size=0.025,  num_top_directions=4, standard_deviation_scaling=True):
        self.lb = lb
        self.ub = ub
        self.objective = objective
        self.samples_per_iteration = samples_per_iteration
        self.exploration_noise = exploration_noise
        self.step_size = step_size
        self.param_args_ignore = None
        self.standard_deviation_scaling = standard_deviation_scaling
        self.num_top_directions = num_top_directions
        self.reward_list = reward_list
        self.cost_list = cost_list
        
        if num_top_directions is None:
            num_top_directions = self.samples_per_iteration
        self.num_top_directions = num_top_directions
        
    def __call__(self, X, Y, params):  
        self.params = params.to(**tkwargs)
        # self.params = self.lb + (self.ub-self.lb)*params.to(**tkwargs)
        self._deltas = torch.empty(self.samples_per_iteration, self.params.shape[-1], dtype=torch.float64) 
        # 1. Sample deltas.
        torch.randn(*self._deltas.shape, out=self._deltas)
        if self.param_args_ignore is not None:
            self._deltas[:, self.param_args_ignore] = 0.0
        # 2. Scale deltas.
        perturbations = self.exploration_noise * self._deltas
        # 3. Compute rewards
        rewards_plus = torch.tensor(
            [
                self.objective(self.lb+(self.ub-self.lb)*(self.params + perturbation))
                for perturbation in perturbations
            ]
        )
        
        for q in range(len(perturbations)):
            self.reward_list.append(float(max(self.reward_list[-1], rewards_plus[q])))
            self.cost_list.append(self.cost_list[-1]+1)
            
        rewards_minus = torch.tensor(
            [
                self.objective(self.lb+(self.ub-self.lb)*(self.params - perturbation))
                for perturbation in perturbations
            ]
        )
        
        for q in range(len(perturbations)):
            self.reward_list.append(float(max(self.reward_list[-1], rewards_minus[q])))
            self.cost_list.append(self.cost_list[-1]+1)
            
        if self.num_top_directions < self.samples_per_iteration:
            # 4. Using top performing directions.
            args_sorted = torch.argsort(
                torch.max(rewards_plus, rewards_minus), descending=True
            )
            args_relevant = args_sorted[: self.num_top_directions]
        else:
            args_relevant = slice(0, self.num_top_directions)
        if self.standard_deviation_scaling is not None:
            # 5. Perform standard deviation scaling.
            std_reward = torch.cat(
                [rewards_plus[args_relevant], rewards_minus[args_relevant]]
            ).std()
        else:
            std_reward = 1.0

        # 6. Update parameters. (gradiend ascent)
        self.params.add_(
            (rewards_plus[args_relevant] - rewards_minus[args_relevant]).to(torch.float64)
            @ self._deltas[args_relevant],
            alpha=self.step_size / (self.num_top_directions * std_reward),
        )
        
        self.params[self.params>1] = 1
        self.params[self.params<0] = 0
        # self.params = torch.min(self.params, self.ub)
        # self.params = torch.max(self.params, self.lb)
        new_y = self.objective(self.lb+(self.ub-self.lb)*self.params)
        self.reward_list.append(float(max(self.reward_list[-1], new_y)))
        self.cost_list.append(self.cost_list[-1]+1)
        
        # 7. Save new parameters.
        # if (type(self.objective._func) is EnvironmentObjective) and (
        #     self.objective._func._manipulate_state is not None
        # ):
        #     self.params_history_list.append(
        #         self.objective._func._manipulate_state.unnormalize_params(self.params)
        #     )
        #     # 8. Perform state normalization update.
        #     self.objective._func._manipulate_state.apply_update()
        # else:
        #     self.params_history_list.append(self.params.clone())

        # if self.verbose:
        #     print(f"Parameter {self.params.numpy()}.")
        #     print(
        #         f"Mean of (b) perturbation rewards {torch.mean(torch.cat([rewards_plus[args_relevant], rewards_minus[args_relevant]])) :.2f}."
        #     )
        #     if self.standard_deviation_scaling:
        #         print(f"Std of perturbation rewards {std_reward:.2f}.")                
        return X, Y, (self.params.flatten()).unsqueeze(0), self.reward_list, self.cost_list