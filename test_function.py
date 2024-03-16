#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:52:07 2024

@author: tang.1856
"""
import torch

class Rosenbrock():
    def __init__(self,dim, LVGP=True):
        self.dim=dim
        self.LVGP = LVGP
    def __call__(self,X):
        X[X[:, -1] == 2, -1] = 0.9
        fun_val = 0
        for d in range(1,self.dim):
            X_curr = X[..., d-1:d]
            X_next = X[..., d:d+1]
            
            if self.LVGP:
                t1 = 100 * (X_next - X_curr**2) ** 2 + (1-X[..., -1:])*torch.sin(10*X_curr+5*X_next) # refer to MISO paper
                # t1 = 100 * (X_next - X_curr**2 + 0.1 * (1 - X[..., -1:])) ** 2 # refer to BOtorch
            else:
                t1 = 100 * (X_next - X_curr**2) ** 2
                
           
            t2 = (X_curr - 1) ** 2
            fun_val+=((t1 + t2).sum(dim=-1))
        return fun_val
    
    
class Borehole():
    def __init__(self,dim, negate=False, LVGP=True):
        self.dim = dim
        self.negate = negate
        self.LVGP = LVGP
    def __call__(self,X):
        Tu = X[..., 0:1]
        r = X[..., 1:2]
        Hu = X[..., 2:3]
        Tl = X[..., 3:4]
        Hl = X[..., 4:5]
        L = X[..., 5:6]
        Kw = X[..., 6:7]
        if self.LVGP:
            rw = X[..., 7:8]*0.05 # qualatative variable
        else:
            rw = 0.05
        
        if self.negate:
            fun_val = 2*torch.pi*Tu*(Hu - Hl)*(torch.log(r/rw) * (Tu/Tl + 1 + 2*(L*Tu/(torch.log(r/rw) * rw**2 * Kw))))**-1
            fun_val*=-1
        else:           
            fun_val = 2*torch.pi*Tu*(Hu - Hl)*(torch.log(r/rw) * (Tu/Tl + 1 + 2*(L*Tu/(torch.log(r/rw) * rw**2 * Kw))))**-1
        
        return fun_val.squeeze(1)
        
class OTL():
    def __init__(self,dim, negate=False, LVGP=True):
        self.dim = dim
        self.negate = negate
        self.LVGP = LVGP
        
    def __call__(self,X):
        Rb1 = X[..., 0:1]
        Rb2 = X[..., 1:2]
        Rf = X[..., 2:3]
        Rc1 = X[..., 3:4]
        Rc2 = X[..., 4:5]
        if self.LVGP:
            B = X[..., 5:6]*50
        else:
            B = 50
        
        Vb1 = 12*Rb2/(Rb1+Rb2)
        
        if self.negate:
            fun_val = B * (Vb1+0.74)*(Rc2+9)/(B*(Rc2+9)+Rf) + 11.35*Rf/(B*(Rc2+9) + Rf) + 0.74*B*Rf*(Rc2+9)/(Rc1*(B*(Rc2+9)+Rf))
            fun_val*=-1
        else:
            fun_val = B * (Vb1+0.74)*(Rc2+9)/(B*(Rc2+9)+Rf) + 11.35*Rf/(B*(Rc2+9) + Rf) + 0.74*B*Rf*(Rc2+9)/(Rc1*(B*(Rc2+9)+Rf))
        
        return fun_val.squeeze(1)

class Piston():
    def __init__(self, dim, negate=False, LVGP = True):
        self.dim = dim
        self.negate = negate
        self.LVGP = LVGP
    def __call__(self, X):
        M = X[..., 0:1]
        S = X[..., 1:2]
        V0 = X[..., 2:3]
        P0 = X[..., 3:4]
        Ta = X[..., 4:5]
        T0 = X[..., 5:6] 
        if self.LVGP:
            k = X[..., 6:7] *1000
        else:
            k = 1000
        
        A = P0*S+19.62*M-k*V0/S
        V = (S/(2*k)) * ((A**2+4*k*P0*V0*Ta/T0)**0.5-A)
        C = 2*torch.pi*(M/(k+S**2*P0*V0*Ta/(T0*V**2)))**0.5
        
        if self.negate:
            C*=-1
            
        return C.squeeze(1)
        
# import numpy as np  
# from scipy.optimize import minimize      
# x0 = np.array([70,30,1,2,1])
# lb = np.array([50,25,0.5,1.2,0.25]) # lb for OTL
# ub = np.array([150, 70, 3, 2.5, 1.2])

# lb = np.array([63070, 100, 990, 63.1, 700, 1120, 9855]) # lb for borhole
# ub = np.array([115600, 50000, 1110, 116, 820, 1680, 12045])

# # lb = np.array([30,0.005,0.002,90000,290,340]) # bound for Piston
# # ub = np.array([60,0.02,0.01,110000,296,360])

# x0 = (lb+ub)*0.5
# bounds = [(lb[i], ub[i]) for i in range(7)]
# function = Borehole(dim=7,LVGP=False)
# result =  minimize(function,x0,bounds=bounds, method='L-BFGS-B')
# result.fun
     
