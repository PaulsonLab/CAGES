#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 10:17:00 2024

@author: tang.1856
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
# Plot for Cartpole problem

#MFBO
xx0 = np.load('/home/tang.1856/CAGES/Plotting/Results/Cartpole/cartpole_cost_MFBO.npy')[:,0:50]
yy0 = (np.load('/home/tang.1856/CAGES/Plotting/Results/Cartpole/cartpole_reward_MFBO.npy')[:,0:50]*-500)

xx_mean0= np.mean(xx0,axis=0)
yy_mean0 = np.mean(yy0,axis=0)
yy_std0 = np.std(yy0, axis=0)
plt.plot(xx_mean0, (yy_mean0),label='MFBO',linewidth=2, marker='o')
plt.fill_between(xx_mean0, yy_mean0 - yy_std0, yy_mean0 + yy_std0,  alpha=0.5)

# CAGES
xx = np.load('Results/Cartpole/cartpole_cost_NewAcq.npy')+65 # need to add the initial cost
yy = (np.load('Results/Cartpole/cartpole_reward_NewAcq.npy')*-500)

# xx = np.concatenate((xx[0:6],xx[7:]))
# yy = np.concatenate((yy[0:6],yy[7:]))
xx_mean= np.mean(xx,axis=0) 
yy_mean = (np.mean(yy,axis=0))
yy_std = np.std(yy, axis=0)
plt.plot(xx_mean, (yy_mean),label='CAGES',linewidth=2, marker='o')
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)

# Vanilla BO
xx1 = np.load('Results/Cartpole/cartpole_cost_EI_new.npy')
yy1 = np.load('Results/Cartpole/cartpole_reward_EI_new.npy')*-500

xx_mean1 = np.mean(xx1,axis=0)
yy_mean1 = (np.mean(yy1,axis=0))
yy_std1 = np.std(yy1, axis=0)
plt.plot(xx_mean1, (yy_mean1),label='EI',linewidth=2, marker='D')
plt.fill_between(xx_mean1, yy_mean1 - yy_std1, yy_mean1 + yy_std1,  alpha=0.5)

# GIBO
xx2 = np.load('Results/Cartpole/cartpole_cost_GIBO.npy')
yy2 = (np.load('Results/Cartpole/cartpole_reward_GIBO.npy')*500)

xx_mean2 = np.mean(xx2,axis=0)
yy_mean2 = np.mean(yy2,axis=0)
yy_std2 = np.std(yy2, axis=0)
plt.plot(xx_mean2, yy_mean2,label='GIBO',linewidth=2, marker='^')
plt.fill_between(xx_mean2, yy_mean2 - yy_std2, yy_mean2 + yy_std2,  alpha=0.5)

# ARS
xx3 = np.load('Results/Cartpole/cartpole_cost_ARS.npy')
yy3 = (np.load('Results/Cartpole/cartpole_reward_ARS.npy')*500)

xx_mean3 = np.mean(xx3,axis=0)
yy_mean3 = np.mean(yy3,axis=0)
yy_std3 = np.std(yy3, axis=0)
plt.plot(xx_mean3, yy_mean3,label='ARS',linewidth=2, marker='x')
plt.fill_between(xx_mean3, yy_mean3 - yy_std3, yy_mean3 + yy_std3,  alpha=0.5)
plt.xlim(55,400)
custom_ticks = list(range(55, 410, 50))
plt.xticks(custom_ticks)
plt.xlabel('Total Cost')
plt.ylabel('Best Reward Found')
plt.ylim(100,530)
plt.legend()
plt.title('Cartpole-V1 (d = 4)')
#################################################################################################################################################

# Plot for OTL function
plt.figure()

# CAGES
xx = np.load('Results/OTL/OTL_cost_NewAcq.npy')+10110
yy = (np.load('Results/OTL/OTL_reward_NewAcq.npy'))*-1

xx_mean= np.mean(xx,axis=0)
yy_mean = (np.mean(yy,axis=0))
yy_std = np.std(yy, axis=0)
plt.plot(xx_mean, (yy_mean),label='CAGES',linewidth=2, marker='o')
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)

# Vanilla BO
xx1 = np.load('Results/OTL/OTL_cost_EI_new.npy')
yy1 = (np.load('Results/OTL/OTL_reward_EI_new.npy'))*-1

xx_mean1 = np.mean(xx1,axis=0)
yy_mean1 = (np.mean(yy1,axis=0))
yy_std1 = np.std(yy1, axis=0)
plt.plot(xx_mean1, (yy_mean1),label='EI',linewidth=2, marker='D')
plt.fill_between(xx_mean1, yy_mean1 - yy_std1, yy_mean1 + yy_std1,  alpha=0.5)

# GIBO
xx2 = np.load('Results/OTL/OTL_cost_GIBO.npy')
yy2 = (np.load('Results/OTL/OTL_reward_GIBO.npy'))

xx_mean2 = np.mean(xx2,axis=0)
yy_mean2 = np.mean(yy2,axis=0)
yy_std2 = np.std(yy2, axis=0)
plt.plot(xx_mean2, yy_mean2,label='GIBO',linewidth=2, marker='^')
plt.fill_between(xx_mean2, yy_mean2 - yy_std2, yy_mean2 + yy_std2,  alpha=0.5)

# ARS
xx3 = np.load('Results/OTL/OTL_cost_ARS.npy')
yy3 = (np.load('Results/OTL/OTL_reward_ARS.npy'))

xx_mean3 = np.mean(xx3,axis=0)
yy_mean3 = np.mean(yy3,axis=0)
yy_std3 = np.std(yy3, axis=0)
plt.plot(xx_mean3, yy_mean3,label='ARS',linewidth=2, marker='x')
plt.fill_between(xx_mean3, yy_mean3 - yy_std3, yy_mean3 + yy_std3,  alpha=0.5)

plt.xlim(9900,20000)
plt.ylim(-6,-2.5)
custom_ticks = list(range(10000, 22000, 2000))
plt.xticks(custom_ticks)
plt.xlabel('Total Cost')
plt.ylabel('- Best Value Found')
plt.legend()
plt.title('OTL (d = 5)')
######################################################################################################################################
# Rosenbrock function
plt.figure()

# MFBO
xx0 = torch.load('/home/tang.1856/CAGES/Plotting/Results/Rosenbrock/Rosenbrock_cost_MFBO.pt').to(torch.float64)
yy0 = torch.log(torch.load('/home/tang.1856/CAGES/Plotting/Results/Rosenbrock/Rosenbrock_reward_MFBO.pt')).to(torch.float64)*-1

xx_mean0= torch.mean(xx0,dim=0)[0:-22].numpy()
yy_mean0 = (torch.mean(yy0,dim=0))[0:-22].numpy()
yy_std0 = torch.std(yy0, axis=0)[0:-22].numpy()
plt.plot(xx_mean0, (yy_mean0),label='MFBO',linewidth=2, marker='o')
plt.fill_between(xx_mean0, yy_mean0 - yy_std0, yy_mean0 + yy_std0,  alpha=0.5)

# CAGES
xx = np.load('Results/Rosenbrock/Rosenbrock_cost_NewAcq.npy')+55
yy = np.log((np.load('Results/Rosenbrock/Rosenbrock_reward_NewAcq.npy')))*-1

xx_mean= np.mean(xx,axis=0)
yy_mean = (np.mean(yy,axis=0))
yy_std = np.std(yy, axis=0)
plt.plot(xx_mean, (yy_mean),label='CAGES',linewidth=2, marker='o')
plt.fill_between(xx_mean, yy_mean - yy_std, yy_mean + yy_std,  alpha=0.5)

# Vanilla BO
xx1 = np.load('Results/Rosenbrock/Rosenbrock_cost_EI_new.npy')
yy1 = np.log(np.load('Results/Rosenbrock/Rosenbrock_reward_EI_new.npy'))*-1

xx_mean1 = np.mean(xx1,axis=0)
yy_mean1 = (np.mean(yy1,axis=0))
yy_std1 = np.std(yy1, axis=0)
plt.plot(xx_mean1, (yy_mean1),label='EI',linewidth=2, marker='D')
plt.fill_between(xx_mean1, yy_mean1 - yy_std1, yy_mean1 + yy_std1,  alpha=0.5)

# GIBO
xx2 = np.load('Results/Rosenbrock/Rosenbrock_cost_GIBO.npy')
yy2 = -np.log((np.load('Results/Rosenbrock/Rosenbrock_reward_GIBO.npy'))*-1)

xx_mean2 = np.mean(xx2,axis=0)
yy_mean2 = np.mean(yy2,axis=0)
yy_std2 = np.std(yy2, axis=0)
plt.plot(xx_mean2, yy_mean2,label='GIBO',linewidth=2, marker='^')
plt.fill_between(xx_mean2, yy_mean2 - yy_std2, yy_mean2 + yy_std2,  alpha=0.5)

# ARS
xx3 = np.load('Results/Rosenbrock/Rosenbrock_cost_ARS.npy')
yy3 = -np.log((np.load('Results/Rosenbrock/Rosenbrock_reward_ARS.npy'))*-1)

xx_mean3 = np.mean(xx3,axis=0)
yy_mean3 = np.mean(yy3,axis=0)
yy_std3 = np.std(yy3, axis=0)
plt.plot(xx_mean3, yy_mean3,label='ARS',linewidth=2, marker='x')
plt.fill_between(xx_mean3, yy_mean3 - yy_std3, yy_mean3 + yy_std3,  alpha=0.5)

plt.xlim(50,500)
# plt.ylim(-2.5,0)
# plt.ylim(-300,5)
plt.xlabel('Total Cost')
# plt.ylabel('-log(Best Value Found)')
plt.ylabel('Best Value Found')
plt.legend()
plt.title('Rosenbrock (d = 6)')