#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:58:47 2023

@author: tang.1856
"""
import numpy as np
import os
import torch
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt
from Algorithm import GIBO, ARS, GIBO_LF
from botorch.test_functions.synthetic import Rosenbrock
import sys
# sys.path.append('/home/tang.1856/Jonathan/LVGP/LVGP-main') # add the path
sys.path.append('/home/tang.1856/CAGES/GIBO')
from RL_function_new import RL_fun
from pyDOE2 import lhs
# from test_function import Borehole, OTL, Piston

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
    "dtype": torch.double,
    
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
if __name__ == "__main__": 
           
    replicate = 20   
    replicate_list = [10, 40, 50, 80, 90, 140, 170, 210, 220, 280] # random replicate list
    
    # select which problem we want to optimize
    RL = True
    rosenbrock = False
    otl = False
    
    gibo = True # if False, execute ARS
   
    reward_gibo_list, cost_list_gibo = [[] for _ in range((replicate))], [[] for _ in range((replicate))]   
        
    for seed in range(replicate):
                      
        if RL: # Cartpole Problem
            dim = 10
            # np.random.seed((replicate_list[seed])*1) 
            np.random.seed(seed)
            
            if gibo:
                N_func_evaluation = 5
                N_INIT = 64 # initial training data point -1                         
                X = torch.tensor(lhs(dim, samples=N_INIT, random_state=seed))   
            else:
                N_func_evaluation = 15
                N_INIT = 0 # ARS doesn't require any initial data to fit surrogate
                X = torch.tensor(np.random.rand(N_INIT,dim))
                
            delta = 0.05 # searching range for the GIBO's acquisition
            lr = 0.5 # learning rate for gradient ascent
            cost = 2 # cost per querying high fidelity model
            step_size = 0.025 # step size for ARSD
            init_x = (0.75*torch.ones(dim)).unsqueeze(0) # starting point for the algorithm (GIBO's paper start with the mid-point)
            lb = -1*torch.ones(dim) # lower bound for the variables
            ub = 1*torch.ones(dim) # upper bound for the variables
            objective = RL_fun(dim = dim, LVGP=False, GIBO_LF=True, HF=False, LF=True) # define objective function
            objective_HF = RL_fun(dim = dim, LVGP=False, GIBO_LF=True, HF=True, LF=False)
            
        elif rosenbrock:
            dim = 12           
            if gibo:
                N_INIT = 4 
                N_func_evaluation = 15
            else:
                N_INIT = 0
                N_func_evaluation = 15
            delta = 0.1
            lr = 0.05
            cost = 10
            step_size = 0.25
            np.random.seed((seed)*1)
            init_x = torch.tensor(0.2+0.6*np.random.rand(1,dim)) 
            # np.random.seed((seed)*1)
            X = torch.tensor(np.random.rand(N_INIT,dim))
            # X = torch.tensor(lhs(dim, samples = N_INIT, random_state=seed))
            lb = 0*torch.ones(dim)
            ub = 2*torch.ones(dim)
            objective = Rosenbrock(dim=dim, negate=True)
            
                
                                               
        X = torch.cat((init_x, X))                                                     
        Y = objective(lb + (ub - lb) *X).unsqueeze(-1)          
        reward_list = [float(objective_HF(lb + (ub - lb) *X[0].unsqueeze(0)))]
        
        if gibo:
            BO1 = GIBO_LF(objective, objective_HF, dim, delta=delta, lr = lr, epsilon_diff_acq_value=0.1, lb = lb, ub = ub,  Ninit = N_INIT, reward_list = reward_list, cost_list = [len(X)])
        else:
            BO1 = ARS(objective, lb, ub, step_size = step_size, reward_list= [float(Y[0])], cost_list = [len(X)])
            
        params = init_x # Define initial moving point
        print('starting point=', params)
            
        moving_time, fun_evaluation = 0, 0
            
        # Start the BO loop
        while fun_evaluation<N_func_evaluation: 
                
            torch.cuda.empty_cache()
                                   
            X, Y, params, reward_list, cost_list = BO1(X, Y, params)    
            reward_gibo_list[seed]=reward_list  
            cost_list_gibo[seed]=[p*cost for p in cost_list]                                           
            fun_evaluation+=1
                
    min_length = min(len(sublist) for sublist in reward_gibo_list)
    reward_gibo_list = [sublist[:min_length] for sublist in reward_gibo_list]
    min_length = min(len(sublist) for sublist in cost_list_gibo)
    cost_list_gibo = [sublist[:min_length] for sublist in cost_list_gibo]
    xx = np.array([[tensor for tensor in sublist] for sublist in cost_list_gibo])
    # yy = np.array([[tensor.item() for tensor in sublist] for sublist in norm_list])
    yy = np.array([[tensor for tensor in sublist] for sublist in reward_gibo_list])  
    
    # Save the results
    if gibo:
        np.save('GIBO_LF_Cartpole_cost.npy',xx)
        np.save('GIBO_LF_Cartpole_reward.npy',yy)
    else:
        np.save('ARS_Cartpole_cost.npy',xx)
        np.save('ARS_Cartpole_reward.npy',yy)
        
                    
           

