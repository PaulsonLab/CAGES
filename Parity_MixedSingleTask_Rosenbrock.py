#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:22:37 2024

@author: tang.1856
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from test_function import Rosenbrock, Borehole, OTL, Piston, OTL2
from pyDOE2 import lhs
import matplotlib.pyplot as plt
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from typing import Any, Callable, Dict, List, Optional
from gpytorch.constraints import GreaterThan
from botorch.acquisition import AnalyticAcquisitionFunction
import gpytorch

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

N_l1 = 80 # number of data for the first fidelity level
N_l2 = 80 # number of data for the second fidelity level  
dim = 12 # Dimension excludes qualatative variable
ind_qual = [dim] # column index for the qualatative variable

# Define testing function
fun = Rosenbrock(dim=dim, LVGP=True)

lb = 0 # lower bound
ub = 2 # upper bound

N_test = 1000 # number of test points

# Generate testing data
np.random.seed(0)
X_te_normalized = np.random.rand(N_test,dim) 
X_te = lb+(ub-lb)*X_te_normalized  # rescale   
qualatative_column_te = np.random.choice([1], size=N_test) 
if ind_qual is not None:
    X_te = np.column_stack((X_te, qualatative_column_te)) # concatenate the qualatative variable into testing data
   
Y_te = fun(torch.tensor(X_te)).numpy()
    
# Generate training data for GP
X_l1 = lb+(ub-lb)*(np.random.rand(N_l1,dim)) # generate training data (at level1) for GP
qualatative_column = np.random.choice([1], size=N_l1) 
if ind_qual is not None:
    X_l1 = np.column_stack((X_l1, qualatative_column))
   
X_l2 = lb+(ub-lb)*(lhs(dim, samples = N_l2)) # generate training data (at level2) for GP
qualatative_column = np.random.choice([2], size=N_l2) 

if ind_qual is not None:
    X_l2 = np.column_stack((X_l2, qualatative_column))
     
X = np.concatenate((X_l1, X_l2))
Y = fun(torch.tensor(X)).numpy() # calculate the true function value


# Fit GP
train_X = torch.tensor(X)
train_X[:,0:-1] = (train_X[:,0:-1]-torch.tensor(lb).unsqueeze(0))/(torch.tensor(ub-lb).unsqueeze(0))
train_Y = torch.tensor(Y).unsqueeze(1)

model = MixedSingleTaskGP(train_X, train_Y, cat_dims = [-1], outcome_transform=Standardize(1), cont_kernel_factory=cont_kernel_factory)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

test_X = torch.tensor(np.column_stack((X_te_normalized, qualatative_column_te)))

posterior = model.posterior(test_X)
mean = (posterior.mean).detach().numpy() # posterior mean for GP (predicted value)

# Plotting
plt.scatter(Y_te, mean, label='Mixed SingleTask GP')
plt.plot([min(Y_te), max(Y_te)], [min(Y_te), max(Y_te)], 'k--') 
plt.xlabel('True value')
plt.ylabel('predicted value')
plt.title('Parity Plot (Mixed SingleTask GP)')
plt.grid(True)