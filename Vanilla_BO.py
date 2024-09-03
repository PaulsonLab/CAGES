#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:20:27 2024

@author: tang.1856
"""

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.test_functions.synthetic import Branin, Rosenbrock
import numpy as np
from test_function import Borehole, OTL, Piston
from RL_function_new import RL_fun
from pyDOE2 import lhs
import matplotlib.pyplot as plt
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from botorch.models.transforms import Normalize, Standardize

replicate = 20
replicate_list = [10, 40, 50, 80, 90, 140, 170, 210, 220, 280]

cost_list = [[] for _ in range((replicate))] 
best_Y_list = [[] for _ in range((replicate))] 

RL = True
rosenbrock = False
otl = False

for seed in range(replicate):
    
    if RL:
        dim = 10
        N_init = 12 # number of initial training data -1
        BO_iter = 27
        cost = 10 # cost per querying HF model
        lb = torch.ones(dim)*-1
        ub = torch.ones(dim)*1          
        X_te_normalized = (torch.ones((1,dim))*0.75).to(torch.float64) # starting point for the algorithm (GIBO's paper start with the mid-point)
        # np.random.seed((replicate_list[seed])*1)  
        # np.random.seed((seed)*1) 
        train_X = torch.tensor(lhs(dim, samples=N_init, random_state=seed)) # initial training data for GP     
        # fun = RL_fun(dim=dim, LVGP=False, negate=False)
        fun = RL_fun(dim=dim, LVGP=False)
        
    elif rosenbrock:
        dim = 12
        N_init = 4
        BO_iter = 45 
        cost = 10
        lb = torch.ones(dim)*0
        ub = torch.ones(dim)*2
        N_test = 1
        np.random.seed((seed)*1)        
        X_te_normalized = torch.tensor(0.2+0.6*np.random.rand(N_test,dim)) 
        # np.random.seed((seed)*1)  
        train_X = torch.tensor(lhs(dim, samples=N_init, random_state=seed))
        # train_X = torch.tensor(np.random.rand(N_init,dim))
        fun = Rosenbrock(dim=dim, negate=True) # botorch does the maximization
    elif otl:
        dim = 5
        N_init = 9
        BO_iter = 20
        cost = 1000
        lb = torch.tensor([50,25,0.5,1.2,0.25]) # lb for OTL
        ub = torch.tensor([150, 70, 3, 2.5, 1.2])
        N_test = 1
        np.random.seed((seed)*1)        
        X_te_normalized = torch.tensor(0.2+0.6*np.random.rand(N_test,dim)) 
        np.random.seed((seed)*1)   
        train_X = torch.tensor(np.random.rand(N_init,dim))
        fun = OTL(dim=dim, LVGP=False, negate=True)
        
        
    train_X = torch.cat((train_X, X_te_normalized))             
    train_Y = fun(lb+(ub-lb)*train_X).reshape(-1,1).to(torch.float64) # need to rescale x, Y is original scale

    # train_Y = standardize(Y) # standardize Y
    best_Y_list[seed].append(float(train_Y[-1]))
    cost_list[seed].append(cost*(N_init+1)) # Initial cost 
    
        
    # Start BO loop
    for i in range(BO_iter):
        
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim)) # define the kernel
        # covar_module = ScaleKernel(MaternKernel(ard_num_dims=dim))
        best_f = max(train_Y)
        # covar_module = ScaleKernel(RBFKernel(ard_num_dims=dim))
        gp = SingleTaskGP(train_X, train_Y, covar_module = covar_module, outcome_transform=Standardize(1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
       
        # if (i)%(dim)==0: # we can re-fit the GP per d iteration
        fit_gpytorch_mll(mll)
        
        EI = LogExpectedImprovement(gp, best_f)
        
        
        bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        candidate, acq_value = optimize_acqf(
            EI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )
        candidate = candidate.to(torch.float64) # argmax(EI)
        
        train_X = torch.cat((train_X, candidate))
        Y_next = fun(lb+(ub-lb)*candidate).unsqueeze(1) # need to rescale x
        train_Y = torch.cat((train_Y, Y_next)) # Y in original scale
        # train_Y = standardize(Y) # standardize Y
        # gp.set_train_data(train_X, train_Y.squeeze(1), strict=False)
        
        best_Y_list[seed].append(max(float(Y_next), best_Y_list[seed][-1])) # best found value
        cost_list[seed].append(cost_list[seed][-1]+cost) # add cost for querying point
    
xx = np.array([[tensor for tensor in sublist] for sublist in cost_list]) # convert cost_list to tensor
# yy = np.array([[tensor.item() for tensor in sublist] for sublist in norm_list])
yy = np.array([[tensor for tensor in sublist] for sublist in best_Y_list]) # convert best_Y_list to tensor

# Save the results
np.save('Cartpole_cost_EI.npy',xx)
np.save('Cartpole_reward_EI.npy',yy)
