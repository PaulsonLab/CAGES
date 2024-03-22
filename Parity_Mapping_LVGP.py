#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:13:00 2024

@author: tang.1856
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from test_function import Rosenbrock, Borehole, OTL, Piston
from LVGP_main import LVGP
from pyDOE import lhs
import matplotlib.pyplot as plt

np.random.seed(0)
N_l1 = 5 # number of data for the first level
N_l2 = 5
N_l3 = 5
N_l4 = 5
       
dim = 5 # Dimension excludes qualatative variable
      
ind_qual = [dim] # column index for the qualatative variable

# Define testing function
fun = OTL(dim=dim, LVGP=True)

n_opt = 4 # The number of times the log-likelihood function is optimized
dim_z = 2 # Dimensionality of latent space, usually 1 or 2 but can be higher (Cartpole)

lb = np.array([50,25,0.5,1.2,0.25]) 
ub = np.array([150, 70, 3, 2.5, 1.2])
      
cost = [1000, 100, 10, 1] 

if ind_qual is not None:
    level_set1 = [1,2,3,4] # define level set
else:
    level_set1 = [1] # define level set for no qualatative variable case

# Initial location for local algorithm
N_test = 100

X_te_normalized = 0.2+0.6*np.random.rand(N_test,dim) # starting point for the algorithm
X_te = lb+(ub-lb)*X_te_normalized  # rescale   
qualatative_column_te = np.random.choice([1], size=N_test) 
if ind_qual is not None:
    X_te = np.column_stack((X_te, qualatative_column_te)) # concatenate the qualatative variable into testing data
    
Y_te = fun(torch.tensor(X_te)).numpy()
    
# Generate initial training data for GP
X_l1 = lb+(ub-lb)*(np.random.rand(N_l1,dim)) # generate initial training data (at level1) for GP
qualatative_column = np.random.choice([1], size=N_l1) # randomly generate qualatative variable 
if ind_qual is not None:
    X_l1 = np.column_stack((X_l1, qualatative_column))
  

X_l2 = lb+(ub-lb)*(np.random.rand(N_l2,dim)) # generate initial training data (at level2) for GP
qualatative_column = np.random.choice([2], size=N_l2) 

if ind_qual is not None:
    X_l2 = np.column_stack((X_l2, qualatative_column))

X_l3 = lb+(ub-lb)*(np.random.rand(N_l3,dim)) # generate initial training data (at level3) for GP
qualatative_column = np.random.choice([3], size=N_l3) 

if ind_qual is not None:
    X_l3 = np.column_stack((X_l3, qualatative_column)) 

X_l4 = lb+(ub-lb)*(np.random.rand(N_l4,dim)) # generate initial training data (at level4) for GP
qualatative_column = np.random.choice([4], size=N_l4) 

if ind_qual is not None:
    X_l4 = np.column_stack((X_l4, qualatative_column))
     
X = np.concatenate((X_l1, X_l2))
X = np.concatenate((X,X_l3))
X = np.concatenate((X,X_l4))  
      
Y = fun(torch.tensor(X)).numpy() # calculate the true function value

LVGP_class = LVGP(X, Y, ind_qual=ind_qual, dim_z=dim_z, n_opt=n_opt, progress=False, noise=False, lb=lb, ub=ub) # define LVGP class
model = LVGP_class.lvgp_fit(X,Y) # fit LVGP

latent_variable = model['qualitative_params']['z_vec'] # each qualatativa variable corresponds to 2 latent variable (z1, z2)
latent_variable = np.concatenate((np.zeros(dim_z), latent_variable)) # noted that the latent variable for the first level is always set to be (0,0)

y_pred = LVGP_class.lvgp_predict(X_te, model)['Y_hat'] # predicted value based on LVGP


##############################################################################################################################################################
# Plotting (Parity)
plt.scatter(Y_te, y_pred)
plt.plot([min(Y_te), max(Y_te)], [min(Y_te), max(Y_te)], 'k--', label='Ideal Fit') 
plt.xlabel('True value')
plt.ylabel('predicted value')
plt.title('Parity Plot')
plt.grid(True)

# Splitting the array into x and y coordinates
xx = latent_variable[::2]  
yy = latent_variable[1::2]  

# Plotting (latent variable mapping)
plt.figure(figsize=(8, 5))

# Use a colormap to generate different colors for each point
colors = plt.cm.viridis(np.linspace(0, 1, len(xx)))

for i in range(len(xx)):
    plt.scatter(xx[i], yy[i], color=colors[i])
    # Adding text labels for each point
    plt.text(xx[i]-0.003, yy[i]+0.02, r'$\ell_{'+str(i)+'}$', color='black', fontsize=9)
       
plt.xlim(-0.02,0.22)
plt.ylim(-0.25,0.25)
plt.grid(True)
# Adding title and labels for clarity
plt.title('Latent Variable Mapping')
plt.xlabel('z1')
plt.ylabel('z2')
