#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 09:32:38 2024

@author: tang.1856
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

marker_interval = 1
text_size = 24
marker_size = 18
linewidth=4
weight='bold'
# Plot for Cartpole problem
indice = [[0,1,2,3,4,5,6,7,8,9]]

plt.figure(figsize=(14,12))
#CAGES
xx0 = np.load('/home/tang.1856/CAGES/Results/Cartpole/Cartpole_cost_CAGES_MFBO.npy')[0:10]
yy0 = (np.load('/home/tang.1856/CAGES/Results/Cartpole/Cartpole_reward_CAGES_MFBO.npy')*500)[0:10]

xx_mean0= np.mean(xx0,axis=0)
yy_mean0 = np.mean(yy0,axis=0)
yy_std0 = 0.5*np.std(yy0, axis=0)
plt.plot(xx_mean0, yy_mean0,label='CAGES', marker='o', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean0, yy_mean0 - yy_std0, yy_mean0 + yy_std0,  alpha=0.5)

# MFBO
xx = np.load('/home/tang.1856/CAGES/Results/Cartpole/cartpole_cost_MFBO.npy')[0:10]# need to add the initial cost
yy = (np.load('/home/tang.1856/CAGES/Results/Cartpole/cartpole_reward_MFBO.npy')[0:10])*500

xx_mean= np.mean(xx,axis=0) 
yy_mean = (np.mean(yy,axis=0))
yy_std = 0.5*np.std(yy, axis=0)
plt.plot(xx_mean[::marker_interval+1], (yy_mean)[::marker_interval+1],label='MFBO',marker='^', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)


# GIBO
xx = np.load('/home/tang.1856/CAGES/Results/Cartpole/GIBO_HF_Cartpole_cost.npy')[0:10]# need to add the initial cost
yy = (np.load('/home/tang.1856/CAGES/Results/Cartpole/GIBO_HF_Cartpole_reward.npy')[0:10])*500

xx_mean= np.mean(xx,axis=0) 
yy_mean = (np.mean(yy,axis=0))
yy_std = 0.5*np.std(yy, axis=0)
plt.plot(xx_mean[::marker_interval], (yy_mean)[::marker_interval],label='GIBO',marker='s', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)

# log EI
xx = np.load('/home/tang.1856/CAGES/Results/Cartpole/Cartpole_cost_EI.npy')[0:10]# need to add the initial cost
yy = (np.load('/home/tang.1856/CAGES/Results/Cartpole/Cartpole_reward_EI.npy')[0:10])*500

xx_mean= np.mean(xx,axis=0) 
yy_mean = (np.mean(yy,axis=0))
yy_std = 0.5*np.std(yy, axis=0)
plt.plot(xx_mean[::marker_interval], (yy_mean)[::marker_interval],label='Log EI',marker='*', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)

# ARS
xx = np.load('/home/tang.1856/CAGES/Results/Cartpole/ARS_Cartpole_cost.npy')[0:10]# need to add the initial cost
yy = (np.load('/home/tang.1856/CAGES/Results/Cartpole/ARS_Cartpole_reward.npy')[0:10])*500

xx_mean= np.mean(xx,axis=0) 
yy_mean = (np.mean(yy,axis=0))
yy_std = 0.5*np.std(yy, axis=0)
plt.plot(xx_mean[::marker_interval], (yy_mean)[::marker_interval],label='ARS', marker='>', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)

plt.legend(prop={'weight':'bold','size':text_size})
plt.xlabel('Total Budget Spent', fontsize=text_size, fontweight=weight)
plt.ylabel('Recommended Reward Value', fontsize=text_size, fontweight=weight)

plt.title('Cartpole-V1 (d = 10)', fontsize=text_size, fontweight=weight)
plt.tick_params(axis='both',
                which='both',
                width=2)
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontsize(text_size)
    label.set_fontweight('bold')
    
for label in ax.get_yticklabels():
    label.set_fontsize(text_size)
    label.set_fontweight('bold')
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

plt.xlim(125,300)
plt.ylim(20,520)
plt.grid(alpha=0.5, linewidth=2.0)

# Plot for Rosenbrock
marker_interval = 2
plt.figure(figsize=(14,12))
#CAGES
xx0 = np.load('/home/tang.1856/CAGES/Results/Rosenbrock/Rosenbrock_cost_CAGES_MFBO.npy')
yy0 = (np.load('/home/tang.1856/CAGES/Results/Rosenbrock/Rosenbrock_reward_CAGES_MFBO.npy'))

xx_mean0= np.mean(xx0,axis=0)
yy_mean0 = np.mean(yy0,axis=0)
yy_std0 = 0.5*np.std(yy0, axis=0)
plt.semilogy(xx_mean0, (yy_mean0),label='CAGES',marker='o', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean0, yy_mean0 - yy_std0, yy_mean0 + yy_std0,  alpha=0.5)

# MFBO
xx = np.load('/home/tang.1856/CAGES/Results/Rosenbrock/Rosenbrock_cost_MFBO1.npy')# need to add the initial cost
yy = (np.load('/home/tang.1856/CAGES/Results/Rosenbrock/Rosenbrock_reward_MFBO1.npy'))

xx_mean= np.mean(xx,axis=0) 
yy_mean = (np.mean(yy,axis=0))
yy_std = 0.5*np.std(yy, axis=0)
plt.semilogy(xx_mean[::marker_interval+3], (yy_mean)[::marker_interval+3],label='MFBO', marker='^', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)


# GIBO
xx = np.load('/home/tang.1856/CAGES/Results/Rosenbrock/GIBO_HF_Rosenbrock_cost.npy')# need to add the initial cost
yy = -(np.load('/home/tang.1856/CAGES/Results/Rosenbrock/GIBO_HF_Rosenbrock_reward.npy'))

xx_mean= np.mean(xx,axis=0) 
yy_mean = (np.mean(yy,axis=0))
yy_std = 0.5*np.std(yy, axis=0)
plt.semilogy(xx_mean[::marker_interval], (yy_mean)[::marker_interval],label='GIBO', marker='s', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)


# log EI
xx = np.load('/home/tang.1856/CAGES/Results/Rosenbrock/Rosenbrock_cost_EI.npy')# need to add the initial cost
yy = (np.load('/home/tang.1856/CAGES/Results/Rosenbrock/Rosenbrock_reward_EI.npy'))

xx_mean= np.mean(xx,axis=0) 
yy_mean = (np.mean(yy,axis=0))
yy_std = 0.5*np.std(yy, axis=0)
plt.semilogy(xx_mean[::marker_interval], (yy_mean)[::marker_interval],label='Log EI', marker='*', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)


# ARS
xx = np.load('/home/tang.1856/CAGES/Results/Rosenbrock/ARS_Rosenbrock_cost.npy')# need to add the initial cost
yy = -(np.load('/home/tang.1856/CAGES/Results/Rosenbrock/ARS_Rosenbrock_reward.npy'))

xx_mean= np.mean(xx,axis=0) 
yy_mean = (np.mean(yy,axis=0))
yy_std = 0.5*np.std(yy, axis=0)
plt.semilogy(xx_mean[::marker_interval], (yy_mean)[::marker_interval],label='ARS', marker='>', markersize=marker_size, linewidth=linewidth)
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)

plt.legend(prop={'weight':'bold','size':text_size})
plt.xlabel('Total Budget Spent', fontsize=text_size, fontweight=weight)
plt.ylabel('Recommended Objective Value', fontsize=text_size, fontweight=weight)
# plt.ylim(100,530)

plt.title('Rosenbrock (d = 12)', fontsize=text_size, fontweight=weight)

plt.tick_params(axis='both',
                which='both',
                width=2)
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontsize(text_size)
    label.set_fontweight('bold')
    
for label in ax.get_yticklabels():
    label.set_fontsize(text_size)
    label.set_fontweight('bold')
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

plt.grid(alpha=0.5, linewidth=2.0)
plt.xlim(40,400)
# plt.ylim(50,520)