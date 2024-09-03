#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:43:17 2024

@author: tang.1856
"""

import os
import torch
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from torch import Tensor
from botorch.models.deterministic import DeterministicModel
import numpy as np
from test_function import Rosenbrock, Borehole, OTL, Piston
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from typing import Any, Callable, Dict, List, Optional
from gpytorch.constraints import GreaterThan
from pyDOE2 import lhs

dim = 12
tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

target_fidelities = {dim: 1.0}

class FlexibleFidelityCostModel(DeterministicModel):
    def __init__(
        self,
        fidelity_dims: list = [-1],
         values = {'2': 1, '1.0': 10},
         fixed_cost: float = 0,
         )->None:
        r'Gets the cost according to the fidelity level'
        super().__init__()
        self.cost_values=values
        self.fixed_cost=fixed_cost
        self.fidelity_dims=fidelity_dims
        self.register_buffer("weights", torch.tensor([1.0]))
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        
        cost = list(map(lambda x: self.cost_values[str(float(x))], X[..., self.fidelity_dims].flatten()))
        cost = torch.tensor(cost).to(X)
        cost = cost.reshape(X[..., self.fidelity_dims].shape)
        return self.fixed_cost + cost

def cont_kernel_factory(
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> RBFKernel:
    return RBFKernel(
        batch_shape=batch_shape,
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
        lengthscale_constraint=GreaterThan(1e-04),
    )

def generate_initial_data(seed = None, fun = None, dim=-1):
    # generate training data
   
    np.random.seed((seed)*1)
    N_l1 = 4 # number of training data for the first level -1 
    N_l2 = 5 # number of training data for the second level                  
    ind_qual = [dim] # column index for the qualatative variable
    
    
    # Initial location for local algorithm
    N_test = 1
    X_te_normalized = 0.2+0.6*np.random.rand(N_test,dim) # random select a point that we want to estimate gradient 
    X_te = lb+(ub-lb)*X_te_normalized  # rescale   
    qualatative_column_te = np.random.choice([1], size=N_test) # we want to estimate the gradient for the highest fidelity function
    if ind_qual is not None:
        X_te = np.column_stack((X_te, qualatative_column_te)) # concatenate the qualatative variable into testing data
        X_te_normalized  = np.column_stack((X_te_normalized , qualatative_column_te))
       
    # Generate initial training data for GP
    # np.random.seed((seed)*1)
    # X_l1_normalized = np.random.rand(N_l1,dim)
    X_l1_normalized = lhs(dim, samples = N_l1, random_state=seed)
    X_l1 = lb+(ub-lb)*X_l1_normalized # generate initial training data (at level1) for GP
    qualatative_column = np.random.choice([1.0], size=N_l1)  
    if ind_qual is not None:
        X_l1 = np.column_stack((X_l1, qualatative_column)) # concatenate the qualatative varibale into training data set
        X_l1_normalized = np.column_stack((X_l1_normalized, qualatative_column))
        
    # X_l2 = np.concatenate((X_l1[:,0:dim],X_te[:,0:dim]),axis=0) # initial data for the second level
    # X_l2_normalized = np.concatenate((X_l1_normalized[:,0:dim],X_te_normalized[:,0:dim]),axis=0) # initial data for the second level
    
    X_l2_normalized = lhs(dim, samples = N_l2, random_state=seed+10)
    X_l2 = lb+(ub-lb)*X_l2_normalized # generate initial training data (at level1) for GP
    qualatative_column = np.random.choice([2.0], size=N_l2)    
    if ind_qual is not None:
        X_l2 = np.column_stack((X_l2, qualatative_column)) # concatenate the qualatative varibale into training data set
        X_l2_normalized = np.column_stack((X_l2_normalized, qualatative_column))
      
    train_x_full= np.concatenate((X_l1, X_l2))
    train_x_full_normalized = np.concatenate((X_l1_normalized, X_l2_normalized))
    train_x_full = np.concatenate((train_x_full, X_te)) # need to include the estimated point into the training data   
    train_x_full_normalized = np.concatenate((train_x_full_normalized, X_te_normalized))
          
    train_obj = fun(torch.tensor(train_x_full)).unsqueeze(1) # calculate the true function value
    return torch.tensor(train_x_full_normalized), train_obj


def initialize_model(train_x, train_obj, dim):
    
    model = MixedSingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=1), cat_dims = [dim], cont_kernel_factory=cont_kernel_factory)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def project(X):
    
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)


def get_mfkg(model, dim):

    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=dim_all,
        columns=[dim],
        values=[1.0],
    )

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=5,
        raw_samples=20,
        options={"batch_limit": 1, "maxiter": 50},
    )

    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=16,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


def optimize_mfkg_and_get_observation(mfkg_acqf, fun, dim):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""
    
    # generate new candidates
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{dim: 2.0}, {dim: 1.0}],
        q=1,
        num_restarts=5,
        raw_samples=20,
        # batch_initial_conditions=X_init,
        options={"batch_limit": 1, "maxiter": 50},
    )

    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach().clone()
    new_x[:,0:dim] = torch.tensor(lb).to(device) + (torch.tensor(ub-lb)).to(device)*new_x[:,0:dim]
    new_obj = fun(new_x.clone()).unsqueeze(-1)
    # new_obj = problem(new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return candidates.detach(), new_obj, cost

def get_recommendation(model):
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=dim_all,
        columns=[dim],
        values=[1],
    )

    final_rec, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=5,
        raw_samples=20,
        # options={"batch_limit": 1, "maxiter": 20},
    )

    final_rec = rec_acqf._construct_X_full(final_rec)

    # objective_value = -fun(final_rec)
    # print(f"recommended point:\n{final_rec}\n\nobjective value:\n{objective_value}")
    return final_rec

if __name__ == '__main__':
    torch.set_printoptions(precision=3, sci_mode=False)
    
    
    dim_all = dim+1
    N_ITER = 500
    replicate = 10
    initial_cost = 55
    lb = np.array([0]*(dim)) # lb for Rosenbrock
    ub = np.array([2]*(dim)) # ub for Rosenbrock
    bounds = torch.tensor([[0.0] * dim_all, [1.0] * dim_all], **tkwargs) # bounds for optimizing acquisition
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], **tkwargs)
    bound_cat = torch.tensor([1.0,2.0]) # bound for categorical variable
    bounds = torch.cat((bounds, bound_cat.unsqueeze(1)), dim=1).to(device)
    # Define testing function
    fun = Rosenbrock(dim=dim, LVGP=True, negate=True)
    # cost_model = AffineFidelityCostModel(fidelity_weights={6: 1.0}, fixed_cost=5.0)
    cost_model = FlexibleFidelityCostModel(values={'1.0':10, '2.0':1}, fixed_cost=0)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    cost_list = [[] for _ in range(replicate)]
    best_Y_list = [[] for _ in range((replicate))] 
    
    for seed in range(replicate):
        train_x, train_obj = generate_initial_data(seed=seed, fun = fun, dim=dim)
        train_x = train_x.to(device)
        train_obj = train_obj.to(device)
        cost_list[seed].append(initial_cost)
        
        best_Y_list[seed].append(-float(train_obj[-1]))
        cumulative_cost = initial_cost
        
        
        for i in range(N_ITER):
            
            torch.cuda.empty_cache()
            mll, model = initialize_model(train_x, train_obj, dim)
            fit_gpytorch_mll(mll)
            mfkg_acqf = get_mfkg(model, dim)
            new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf, fun, dim)
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])
            cumulative_cost += cost
            cost_list[seed].append(int(cumulative_cost))
            
            # if new_x[0][-1] == 1: # query at the highest fidelity function
            #     # cost_list[seed].append(int(cumulative_cost))
            #     best_Y_list[seed].append(min(best_Y_list[seed][-1],-float(new_obj[0])))
            # else: # query at low fidelity function
                # 
                # best_Y_list[seed].append(best_Y_list[seed][-1])
                
            final_rec = get_recommendation(model)
            final_rec[:,0:dim] = torch.tensor(lb).to(device)+torch.tensor(ub-lb).to(device)*final_rec[:,0:dim]
            final_rec_obj = fun(final_rec.clone())
            best_Y_list[seed].append(-float(final_rec_obj))
            
            if cumulative_cost>400:
                break
            
    # xx = np.array([[tensor for tensor in sublist] for sublist in cost_list])
    # # yy = np.array([[tensor.item() for tensor in sublist] for sublist in norm_list])
    # yy = np.array([[tensor for tensor in sublist] for sublist in best_Y_list])
    
    
    max_length = max(len(row) for row in cost_list)
    padded_list = np.array([row + [row[-1]] * (max_length - len(row)) for row in cost_list])
    xx = torch.tensor(padded_list)
    
    max_length1 = max(len(row) for row in best_Y_list)
    padded_list1 = np.array([row + [row[-1]] * (max_length1 - len(row)) for row in best_Y_list])
    yy = torch.tensor(padded_list1)
    
    # Save results
    np.save('Rosenbrock_cost_MFBO1.npy',xx)
    np.save('Rosenbrock_reward_MFBO1.npy',yy)
