#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:55:30 2024

@author: tang.1856
"""
import numpy as np
import torch
from scipy.optimize import minimize
from src.cholesky import one_step_cholesky
import torch.optim as optim

class GradientInformation_entropy():
    """Acquisition function to sample points for gradient information.

    Attributes:
        model: Gaussian process model that supplies the Jacobian (e.g. DerivativeExactGPSEModel).
    """

    def __init__(self, model, lvgp_to_latent, lvgp_kernel, cost, params):
        # super(GradientInformation_entropy, self).__init__()
        """Inits acquisition function with model."""
        self.cost = cost
        self.theta_i = params
        self.lvgp_to_latent = lvgp_to_latent
        self.lvgp_kernel = lvgp_kernel
        self.model = model
        # get params
        self.p_qual = model['data']['p_qual']
        self.X_old_full = model['data']['X_full']   
        self.X_quant = model['data']['X_quant']
        self.D = len(self.X_old_full[0]) - self.p_qual*model['qualitative_params']['dim_z'] # dimension excludes qualatative variable
        self.D_full = len(self.X_old_full[0])
        self.N = len(self.X_old_full)
        self.Y = model['data']['Y']
        
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
        

    def update(self, level):
        self.level = level # update the qualatative level (we optimize each level separately)
        
    def _get_KxX_dx_SE(self, x, X):
        """Computes the analytic derivative of the SE kernel K(x,X) w.r.t. x.
    
        Args:
            x: (n x D) Test points.
    
        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
                 
        n = x.shape[0]
        K_xX = (self.sigma2*self.lvgp_kernel(x, X, self.phi_full))
        lengthscale = ((1/10**self.phi_full)**0.5)[0:self.D]
        
        return (
            -np.eye(self.D)
            / lengthscale ** 2
            @ (
                (x[:,0:self.D].reshape(n, 1, self.D) - X[:,0:self.D].reshape(1, len(X), self.D))
                * K_xX.reshape(n, len(X), 1)
            ).transpose(0,2,1)
        )
    
    def _get_Kxx_dx2(self):
        """Computes the analytic second derivative of the kernel K(x,x) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x D) The second derivative of K(x,x) w.r.t. x.
        """
        
        lengthscale = ((1/10**self.phi_full)**0.5)[0:self.D]
        
        return (
            np.eye(self.D) / lengthscale ** 2
        ) * self.sigma2
    
    def __call__(self, theta):
        """Evaluate the acquisition function on the candidate set thetas.

        Args:
            thetas: A (b) x D-dim Tensor of (b) batches with a d-dim theta points each.

        Returns:
            A (b)-dim Tensor of acquisition function values at the given theta points.
        """
        # define cost function
        cost = self.cost[self.level-1]
        
        sigma_n = 0
        D = self.p_all

        x = self.theta_i # point that we want to estimate the gradient
        
        _NUMERIC_KINDS = {'b', 'f', 'i', 'u'}
        
        ###################################################################################################################################################
        # process X_new
        m = x.shape[0]
        
        if self.p_qual == 0:# no qualatative variable
            X_new_qual = None
            X_new_quant = x
    
            if not (np.asarray(X_new_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(X_new_quant))):
                print('All the elements of X_new must be finite numbers.')
                exit(0)
    
            X_new_quant = ((X_new_quant.T - self.X_quant_min.reshape(-1, 1)) / (self.X_quant_max - self.X_quant_min).reshape(-1, 1)).T
            X_new_full = X_new_quant
          
        else:
            X_new_qual = x[:, self.ind_qual]
            
            if self.p_qual == self.p_all:
                X_new_quant = None
            else:
                X_new_quant = x[:, [ii for ii in range(self.p_all) if ii not in self.ind_qual]]
                if not (np.asarray(X_new_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(X_new_quant))):
                    print('All the elements of X_new must be finite numbers.')
                    exit(0)
    
                X_new_quant = ((X_new_quant.T - self.X_quant_min.reshape(-1, 1)) / (self.X_quant_max - self.X_quant_min).reshape(-1, 1)).T

            X_new_qual_la = self.lvgp_to_latent(X_new_qual, self.lvs_qual, self.n_lvs_qual, self.p_qual, self.z_vec, self.dim_z, m)
    
            if X_new_quant is not None:
                X_new_full = np.hstack([X_new_quant, X_new_qual_la])             
            else:
                X_new_full = X_new_qual_la
               
        ###################################################################################################################################################
        
        # process the theta (point that we want to add into training data: D_{t+1})
        
        if self.p_qual == 0:# no qualatative variable
            theta_new_qual = None
            theta_new_quant = theta
    
            if not (np.asarray(theta_new_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(theta_new_quant))):
                print('All the elements of X_new must be finite numbers.')
                exit(0)
    
            # theta_new_quant = ((theta_new_quant.T - self.X_quant_min.reshape(-1, 1)) / (self.X_quant_max - self.X_quant_min).reshape(-1, 1)).T
            theta_new_full = theta_new_quant.reshape(1,-1)

        else:
            theta = (np.append(theta, self.level)).reshape(1,-1)# add the qualatative variable to the last column
            theta_new_qual = theta[:, self.ind_qual]
            
            if self.p_qual == self.p_all:
                theta_new_quant = None
            else:
                theta_new_quant = theta[:, [ii for ii in range(self.p_all) if ii not in self.ind_qual]]
                if not (np.asarray(theta_new_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(theta_new_quant))):
                    print('All the elements of X_new must be finite numbers.')
                    exit(0)
    
                # theta_new_quant = ((theta_new_quant.T - self.X_quant_min.reshape(-1, 1)) / (self.X_quant_max - self.X_quant_min).reshape(-1, 1)).T
    
            theta_new_qual_la = self.lvgp_to_latent(theta_new_qual, self.lvs_qual, self.n_lvs_qual, self.p_qual, self.z_vec, self.dim_z, m)
    
            if theta_new_quant is not None:
                theta_new_full = np.hstack([theta_new_quant, theta_new_qual_la])              
            else:
                theta_new_full = theta_new_qual_la
               
        ##################################################################################################################################################
        
        # Compute original variance (before adding the "imaginary sampled" theta).
        KXX = self.sigma2 * self.lvgp_kernel(self.X_old_full, self.X_old_full, self.phi_full)
        KXX_inv = (np.linalg.inv(KXX + 1e-6*np.eye(len(KXX))))
        
        K_xX_dx = self._get_KxX_dx_SE(X_new_full, self.X_old_full)
        variance_d_old = self._get_Kxx_dx2()-K_xX_dx[0] @ KXX_inv @ K_xX_dx[0].T
        # log_det_old = 2 * torch.linalg.cholesky(variance_d_old).diagonal(dim1=-2, dim2=-1).log().sum(-1) 
        log_det_old = np.linalg.slogdet(variance_d_old)[1] # Calculate the log determinant
        
        # Compute variance after adding new data theta
        self.X_train_new = np.concatenate((self.X_old_full, theta_new_full), axis=0) # add the "imaginary sampled" theta into current data set 
        KXX = self.sigma2 * self.lvgp_kernel(self.X_train_new, self.X_train_new, self.phi_full) 
        KXX_inv = (np.linalg.inv(KXX+ 1e-6*np.eye(len(KXX))))
        
        K_xX_dx = self._get_KxX_dx_SE(X_new_full, self.X_train_new)
        variance_d_new = self._get_Kxx_dx2()-K_xX_dx[0] @ KXX_inv @ K_xX_dx[0].T
        # log_det_new = 2 * torch.linalg.cholesky(variance_d_new).diagonal(dim1=-2, dim2=-1).log().sum(-1) 
        log_det_new = np.linalg.slogdet(variance_d_new)[1] # Calculate the log determinant
      
        acq_val = (log_det_old - log_det_new)/cost # Calculate acquisition function per cost: (entropy_old - entropy_new)/cost        
        return -acq_val # add minus sign because scipy does minimization
