#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:21:55 2024

@author: tang.1856
"""


from torch.distributions.multivariate_normal import MultivariateNormal
import yaml
import argparse
from src import config

import torch
# from synthetic_function.generate_objective_synthetic_new import generate_objective_synthetic   
from src.environment_api import StateNormalizer, EnvironmentObjective, manipulate_reward
from src.policy_parameterizations import MLP, discretize
from torch.quasirandom import SobolEngine
import gymnasium as gym

class RL_fun():
    
    def __init__(self, dim, LVGP = True, negate=True):
        self.dim = dim
        self.LVGP = LVGP
        self.negate = negate
          
    def __call__(self, X):
        
        
        parser = argparse.ArgumentParser(
            description='Run optimization with given optimization method.'
        )
        parser.add_argument('-c', '--config ', type=str, help='Path to config file.')
        parser.set_defaults(config='/home/tang.1856/Jonathan/LVGP/LVGP-main/configs/rl_experiment/cartpole/gibo_10runs.yaml') # need to change to user's local path
      
        
        args = parser.parse_args()
        with open(args.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    
        # Translate config dictionary.
        cfg = config.insert(cfg, config.insertion_config)
    
    
        # Usecase 1: optimizing policy for a reinforcement learning environment.
        mlp = MLP(*cfg['mlp']['layers'], add_bias=cfg['mlp']['add_bias'])
        len_params = mlp.len_params
    
        if cfg['mlp']['discretize'] is not None:
            mlp = discretize(mlp, cfg['mlp']['discretize'])
    
        # Evaluate config dictionary (functions etc.).
        cfg = config.evaluate(cfg, len_params)
        
      
        if cfg['mlp']['state_normalization']:
            state_norm = StateNormalizer(
                normalize_params=mlp.normalize_params,
                unnormalize_params=mlp.unnormalize_params,
            )
        else:
            state_norm = None
    
                       
        ylist = []
        for element in X:
            if self.LVGP:
                
                if element[-1]==1: # HF                   
                    l = 1
                    tau = 0.02
                    level = 500
                    scale = level
                    
                elif element[-1]==2: # LF1                  
                    l = 0.4
                    tau = 0.04
                    level = 250
                    scale = level
                    
                elif element[-1]==3: # LF2                  
                    l = 0.1
                    tau = 0.02
                    level = 500
                    scale = level                
            
            else:

                level = 500
                tau = 0.02
                scale = level
                l=1
                
            
                
            reward_func = manipulate_reward(
                cfg['mlp']['manipulate_reward']['shift'],
                scale
            )
            
            objective = EnvironmentObjective(
                env=gym.make(cfg['environment_name']),
                episode_length = level,
                policy=mlp,
                manipulate_state=state_norm,
                manipulate_reward=reward_func,
            )
            
            new_y = 0
            for i in range(int(100*l)):
                new_y += objective(element[0:self.dim], i, tau)
            new_y/=100*l
            ylist.append(new_y)
        
        if self.negate:
            fun_val = -torch.tensor(ylist)
        else:
            fun_val = torch.tensor(ylist)
        return fun_val
