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
# from LVGP_main import LVGP
# from pyDOE import lhs
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

N_l1 = 5 # number of data for the first level
N_l2 = 5
N_l3 = 5
N_l4 = 5
       
dim = 5 # Dimension excludes qualatative variable
      
ind_qual = [dim] # column index for the qualatative variable

# Define testing function
fun = OTL(dim=dim, LVGP=True)
# fun = Rosenbrock(dim=dim, LVGP=True)

n_opt = 4 # The number of times the log-likelihood function is optimized
dim_z = 2 # Dimensionality of latent space, usually 1 or 2 but can be higher (Cartpole)

lb = np.array([50, 25, 0.5, 1.2, 0.25]) 
ub = np.array([150, 70, 3, 2.5, 1.2])

# lb = np.array([50,25,1.2,0.25]) 
# ub = np.array([150, 70, 2.5, 1.2])
# lb = 0
# ub = 2

cost = [1000, 100, 10, 1] 

if ind_qual is not None:
    level_set1 = [1,2,3,4] # define level set
else:
    level_set1 = [1] # define level set for no qualatative variable case

# Initial location for local algorithm
N_test = 1000

np.random.seed(0)
X_te_normalized = np.random.rand(N_test,dim) # starting point for the algorithm
X_te = lb+(ub-lb)*X_te_normalized  # rescale   
qualatative_column_te = np.random.choice([1], size=N_test) 
if ind_qual is not None:
    X_te = np.column_stack((X_te, qualatative_column_te)) # concatenate the qualatative variable into testing data
   
Y_te = fun(torch.tensor(X_te)).numpy()
    
# Generate initial training data for GP
X_l1 = lb+(ub-lb)*(np.random.rand(N_l1,dim)) # generate initial training data (at level1) for GP
# X_l1 = lb+(ub-lb)*(lhs(dim, samples = N_l1))
qualatative_column = np.random.choice([1], size=N_l1) # randomly generate qualatative variable 
if ind_qual is not None:
    X_l1 = np.column_stack((X_l1, qualatative_column))
    # X_l1 = np.column_stack((X_l1, qualatative_column))
  

X_l2 = lb+(ub-lb)*(np.random.rand(N_l2,dim)) # generate initial training data (at level2) for GP
# X_l2 = lb+(ub-lb)*(lhs(dim, samples = N_l2))
# X_l2 = X_l1[:, 0:dim]
qualatative_column = np.random.choice([2], size=N_l2) 

if ind_qual is not None:
    X_l2 = np.column_stack((X_l2, qualatative_column))
    # X_l2 = np.column_stack((X_l2, qualatative_column))

X_l3 = lb+(ub-lb)*(np.random.rand(N_l3,dim)) # generate initial training data (at level3) for GP
# X_l3 = lb+(ub-lb)*(lhs(dim, samples = N_l3))
# X_l3 = X_l1[:, 0:dim]
qualatative_column = np.random.choice([3], size=N_l3) 

if ind_qual is not None:
    X_l3 = np.column_stack((X_l3, qualatative_column)) 
    # X_l3 = np.column_stack((X_l3, qualatative_column)) 

X_l4 = lb+(ub-lb)*(np.random.rand(N_l4,dim)) # generate initial training data (at level4) for GP
# X_l4 = lb+(ub-lb)*(lhs(dim, samples = N_l4))
# X_l4 = X_l1[:, 0:dim]
qualatative_column = np.random.choice([4], size=N_l4) 

if ind_qual is not None:
    X_l4 = np.column_stack((X_l4, qualatative_column))
     
X = np.concatenate((X_l1, X_l2))
X = np.concatenate((X,X_l3))
# X = np.concatenate((X,X_l4))  
      
Y = fun(torch.tensor(X)).numpy() # calculate the true function value


train_X = torch.tensor(X)
train_X[:,0:-1] = (train_X[:,0:-1]-torch.tensor(lb).unsqueeze(0))/(torch.tensor(ub-lb).unsqueeze(0))
train_Y = torch.tensor(Y).unsqueeze(1)

model = MixedSingleTaskGP(train_X, train_Y, cat_dims = [-1], outcome_transform=Standardize(1), cont_kernel_factory=cont_kernel_factory)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

test_X = torch.tensor(np.column_stack((X_te_normalized, qualatative_column_te)))


posterior = model.posterior(test_X)
mean = (posterior.mean).detach().numpy()

plt.scatter(Y_te, mean, label='Mixed SingleTask GP')

plt.plot([min(Y_te), max(Y_te)], [min(Y_te), max(Y_te)], 'k--') 
plt.xlabel('True value')
plt.ylabel('predicted value')
plt.title('Parity Plot (Mixed SingleTask GP)')
plt.grid(True)