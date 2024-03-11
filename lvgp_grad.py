#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:47:05 2024

@author: tang.1856
"""
import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky
import time
from skopt.sampler import Lhs
from skopt.space import Space
import warnings
from multiprocessing import Pool
import torch


class lvgp_gradient():
    """
        This class is used to calculate the gradient of a LVGP 
    """
    def __init__(self, model, lvgp_to_latent, lvgp_kernel):
        self.lvgp_to_latent = lvgp_to_latent
        self.lvgp_kernel = lvgp_kernel
        self.model = model
        # get params
        self.X_old_full = model['data']['X_full']   
        self.X_quant = model['data']['X_quant']
        
        self.N = len(self.X_old_full)
        self.Y = model['data']['Y']
        self.p_qual = model['data']['p_qual']
        if self.p_qual==0:
            self.D = len(self.X_old_full[0])
        else:
            self.D = len(self.X_old_full[0]) - self.p_qual*model['qualitative_params']['dim_z']
        self.X_quant_min = model['data']['X_quant_min']
        self.X_quant_max = model['data']['X_quant_max']
        self.Y_min = model['data']['Y_min']
        self.Y_max = model['data']['Y_max']
        self.Y_std = model['data']['Y_std']

        self.lvs_qual = model['data']['lvs_qual']
        self.n_lvs_qual = model['data']['n_lvs_qual']
        self.ind_qual = model['data']['ind_qual']
    
        self.phi = model['quantitative_params']['phi']
        self.dim_z = model['qualitative_params']['dim_z']
        self.phi_full = np.array([*self.phi, *[0. for _ in range(self.p_qual * self.dim_z)]])
        
        self.z_vec = model['qualitative_params']['z_vec']
    
        self.beta_hat = model['fit_details']['beta_hat']
        self.RinvPYminusMbetaP = model['fit_details']['RinvPYminusMbetaP']
        self.p_all = model['data']['p_all']
        self.sigma2 = model['fit_details']['sigma2']
                
    def _get_KxX_dx_SE(self, x):
        """Computes the analytic derivative of the SE kernel K(x,X) w.r.t. x.
    
        Args:
            x: (n x D) Test points.
    
        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        
        X = torch.tensor(self.X_old_full)       
        X = X.to(torch.float64)
               
        n = x.shape[0]
        # K_xX = torch.tensor(self.sigma2*self.lvgp_kernel(x, self.X_old_full, self.phi_full))
        K_xX = torch.tensor(self.lvgp_kernel(x, self.X_old_full, self.phi_full))
        lengthscale = torch.tensor((1/10**self.phi_full)**0.5)[0:self.D]
        x = torch.tensor(x)
        x = x.to(torch.float64)
        
        return (
            -torch.eye(self.D, device=x.device)
            / lengthscale ** 2
            @ (
                (x[:,0:self.D].view(n, 1, self.D) - X[:,0:self.D].view(1, self.N, self.D))
                * K_xX.view(n, self.N, 1)
            ).transpose(1, 2)
        )

       
    def lvgp_posterior_mean_gradient(self, X_new):
        if not ('model' in self.model and self.model['model'] == 'LVGP model'):
            print('The 2nd input should be a model of class "LVGP model".')
            exit(0)
    
        if not isinstance(X_new, np.ndarray):
            print('X_new should be a numpy array.')
            exit(0)
      
        if X_new.shape[1] != self.p_all:
            print('The dimensionality of X_new is not correct!')
            exit(0)
       
        _NUMERIC_KINDS = {'b', 'f', 'i', 'u'}
    
        # process X_new
        m = X_new.shape[0]
    
        if self.p_qual == 0:
            X_new_qual = None
            X_new_quant = X_new
    
            if not (np.asarray(X_new_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(X_new_quant))):
                print('All the elements of X_new must be finite numbers.')
                exit(0)
    
            X_new_quant = ((X_new_quant.T - self.X_quant_min.reshape(-1, 1)) / (self.X_quant_max - self.X_quant_min).reshape(-1, 1)).T
            X_new_full = X_new_quant
            # phi_full = self.phi
        else:
            X_new_qual = X_new[:, self.ind_qual]
    
            if self.p_qual == self.p_all:
                X_new_quant = None
            else:
                X_new_quant = X_new[:, [ii for ii in range(self.p_all) if ii not in self.ind_qual]]
                if not (np.asarray(X_new_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(X_new_quant))):
                    print('All the elements of X_new must be finite numbers.')
                    exit(0)
    
                X_new_quant = ((X_new_quant.T - self.X_quant_min.reshape(-1, 1)) / (self.X_quant_max - self.X_quant_min).reshape(-1, 1)).T
    
            X_new_qual_la = self.lvgp_to_latent(X_new_qual, self.lvs_qual, self.n_lvs_qual, self.p_qual, self.z_vec, self.dim_z, m)
    
            if X_new_quant is not None:
                X_new_full = np.hstack([X_new_quant, X_new_qual_la])
                # phi_full = np.array([*self.phi, *[0. for _ in range(self.p_qual * self.dim_z)]])
            else:
                X_new_full = X_new_qual_la
                # phi_full = np.array([0. for _ in range(self.p_qual * self.dim_z)])
    
        
        # KXX = self.sigma2 * self.lvgp_kernel(self.X_old_full,self.X_old_full, self.phi_full)
        # KXX = self.lvgp_kernel(self.X_old_full, self.X_old_full, phi_full)
        # KXX_inv = torch.tensor(np.linalg.inv(KXX))
        K_xX_dx = self._get_KxX_dx_SE(X_new_full)
        mean_d = (K_xX_dx.to(torch.float64) @ torch.tensor(self.RinvPYminusMbetaP).T[0])
        # mean_d = K_xX_dx.to(torch.float64) @ KXX_inv.to(torch.float64) @ torch.tensor(self.Y.squeeze(1) - self.beta_hat)
       
        # mean_d = mean_d*(self.Y_max-self.Y_min)/(self.X_quant_max - self.X_quant_min) # rescale
        # mean_d = mean_d * self.Y_std/(self.X_quant_max - self.X_quant_min)
        # print('gradient GP=',mean_d*(self.Y_max-self.Y_min)/(self.X_quant_max - self.X_quant_min))
        print('gradient GP=',mean_d*self.Y_std/(self.X_quant_max - self.X_quant_min))
        return mean_d