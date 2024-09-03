#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:19:05 2024

@author: tang.1856
"""
import gymnasium as gym
import torch

class RL_fun():
    
    def __init__(self, dim, LVGP = True, GIBO_LF = False, negate=True, HF=False, LF = False):
        self.dim = dim
        self.LVGP = LVGP
        self.HF = HF
        self.LF = LF
        self.GIBO_LF = GIBO_LF
        # self.tau = tau
        # self.max_step = max_step
        # self.initial_state = initial_state
        # self.LVGP = LVGP
        # self.negate = negate
    
    def policy(self, param, state):
    
        p1 = param[0]*state[0] + param[1]*state[1] + param[2]*state[2] + param[3]*state[3] 
        p2 = param[4]*state[0] + param[5]*state[1] + param[6]*state[2] + param[7]*state[3]
        
        
        p1 = torch.relu(p1)
        p2 = torch.relu(p2)
        
        p3 = param[8]*p1 + param[9]*p2
        p3 = torch.sigmoid(p3)

        if p3>0.5:
            p3 = 1 
        else:
            p3 = 0
        
        
        return int(p3)
    
    def environment(self, param, seed):    
    
        env = gym.make("CartPole-v1")    
        env.unwrapped.update_tau(self.tau)
        # options = {'goal_cell':np.array([5,2]), 'reset_cell':np.array([7,4])}
        observation, info = env.reset(seed=seed)
       
        # initial_dist = np.linalg.norm(observation['achieved_goal']-observation['desired_goal'])
        Reward = 0
        for i in range(self.max_step):
           
            action = self.policy(param, observation)       
            observation, reward, terminated, truncated, info = env.step(action)
            Reward+=reward
            if terminated or truncated:
              # observation['achieved_goal'] = observation['desired_goal']
              break
               
        # final_dist = np.linalg.norm(observation['achieved_goal']-observation['desired_goal'])
        # Reward = (initial_dist-final_dist)/initial_dist
      
        env.close()
        return Reward

    def __call__(self,x):
        
        Reward_list = []
        for element in x:
            if self.LVGP:
                if element[-1]==1: # HF                   
                    initial_state = 100
                    self.tau = 0.02
                    self.max_step = 500
                      
                elif element[-1]==2: # LF1                  
                    initial_state = 40
                    self.tau = 0.04
                    self.max_step = 250
                           
                elif element[-1]==3: # LF2                  
                    initial_state = 10
                    self.tau = 0.02
                    self.max_step = 500
                    
            elif self.GIBO_LF:
                if self.HF: # HF                   
                    initial_state = 100
                    self.tau = 0.02
                    self.max_step = 500
                           
                elif self.LF: # LF2                  
                    initial_state = 40
                    self.tau = 0.04
                    self.max_step = 250
                
            else:
                initial_state = 100
                self.tau = 0.02
                self.max_step = 500
             
                
            Reward_total = 0
            for seed in range(initial_state):
                Reward_total += self.environment(element,seed)
                
            Reward_total/=initial_state
            Reward_total/=self.max_step
            Reward_list.append(Reward_total) 
        
        return torch.tensor(Reward_list).to(torch.float64)
    
# import gymnasium as gym
# env = gym.make("CartPole-v1", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(500):
#     action = env.action_space.sample()  # this is where you would insert your policy
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#       observation, info = env.reset()

# env.close()

# fun = RL_fun(4, 0.02, 500, 10)
# # X = torch.rand(10,4)
# X = torch.tensor([[0.5,0.5,0.5,0.5,1],[0.4,0.2,0.3,0.5,3]])
# fun(X)