#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:43:17 2024

@author: tang.1856
"""

import os
import torch
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from torch import Tensor
from botorch.models.deterministic import DeterministicModel
import numpy as np
from test_function import Rosenbrock, Borehole, OTL, Piston
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from typing import Any, Callable, Dict, List, Optional
from gpytorch.constraints import GreaterThan
from botorch.acquisition import AnalyticAcquisitionFunction
import gpytorch
from pyDOE2 import lhs
tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

target_fidelities = {6: 1.0}

class FlexibleFidelityCostModel(DeterministicModel):
    def __init__(
        self,
        fidelity_dims: list = [-1],
         values = {'2': 1, '1.0': 10},
         fixed_cost: float = 0,
         )->None:
        r'Gets the cost according to the fidelity level'
        super().__init__()
        self.cost_values=values
        self.fixed_cost=fixed_cost
        self.fidelity_dims=fidelity_dims
        self.register_buffer("weights", torch.tensor([1.0]))
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        
        cost = list(map(lambda x: self.cost_values[str(float(x))], X[..., self.fidelity_dims].flatten()))
        cost = torch.tensor(cost).to(X)
        cost = cost.reshape(X[..., self.fidelity_dims].shape)
        return self.fixed_cost + cost

class CustomAcquisitionFunction(AnalyticAcquisitionFunction):
    def __init__(self, model, current_theta):
        '''Inits acquisition function with model.'''
        super().__init__(model=model)
        self.current_theta = current_theta # point that we want to evaluate gradient        
    
    def _get_KxX_dx(self, x, X) -> torch.Tensor:
        '''Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        '''
        x_cont = x[:,0:-1]
        X_cont = X[:,0:-1]
        N = X.shape[0]
        n = x.shape[0]
        self.model.D = X.shape[1] - 1 # dimension for continuous variable

        outputscale1 = self.model.covar_module.kernels[0].outputscale
        outputscale2 = self.model.covar_module.kernels[1].outputscale
        K_xX_SE1 = outputscale1*self.model.covar_module.kernels[0].base_kernel.kernels[0](x_cont, X_cont).evaluate() # covariance vector for SE kernel
        K_xX_SE2 = outputscale2*self.model.covar_module.kernels[1].base_kernel.kernels[0](x_cont, X_cont).evaluate()
 
        lengthscale1 = self.model.covar_module.kernels[0].base_kernel.kernels[0].lengthscale.detach() # lengthscale for the RBF kernel
        lengthscale2 = self.model.covar_module.kernels[1].base_kernel.kernels[0].lengthscale.detach() # lengthscale for the RBF kernel
        K_cat = self.model.covar_module.kernels[1].base_kernel.kernels[1](x, X).evaluate() # covariance vector for categorical kernel 
        
        dk_SE1_dx = -torch.eye(self.model.D, device=X.device)/ lengthscale1 ** 2 @ ((x_cont.view(n, 1, self.model.D) - X_cont.view(1, N, self.model.D))* K_xX_SE1.view(n, N, 1)).transpose(1, 2) # gradient of SE kernel
        dk_SE2_dx = -torch.eye(self.model.D, device=X.device)/ lengthscale2 ** 2 @ ((x_cont.view(n, 1, self.model.D) - X_cont.view(1, N, self.model.D))* K_xX_SE2.view(n, N, 1)).transpose(1, 2)
        return dk_SE1_dx + dk_SE2_dx * K_cat
    
    def _get_KxX_dx2(self, x, X):
        outputscale1 = self.model.covar_module.kernels[0].outputscale
        outputscale2 = self.model.covar_module.kernels[1].outputscale
        # self.model.D = X.shape[1] - 1 # dimension for continuous variable
        lengthscale1 = self.model.covar_module.kernels[0].base_kernel.kernels[0].lengthscale.detach() # lengthscale for the RBF kernel
        lengthscale2 = self.model.covar_module.kernels[1].base_kernel.kernels[0].lengthscale.detach() # lengthscale for the RBF kernel
        K_cat = self.model.covar_module.kernels[1].base_kernel.kernels[1](x, X).evaluate() # covariance vector for categorical kernel 
        
        dk_SE1_dx2 = outputscale1*torch.eye(self.model.D, device=X.device)/ lengthscale1 ** 2
        dk_SE2_dx2 = outputscale2*torch.eye(self.model.D, device=X.device)/ lengthscale2 ** 2
        
        return dk_SE1_dx2 + dk_SE2_dx2*K_cat
        
    def calculate_gradient(self):
        """Compute the gradient for posterior mean function"""
        X = self.model.train_inputs[0] # training data for GP
        Y = self.model.train_targets
        noise = self.model.likelihood.noise
        KxX_dx = self._get_KxX_dx(self.current_theta, X)
        K_XX = self.model.covar_module(X, X).evaluate()
        K_XX_inv = torch.inverse(K_XX + noise*torch.eye(len(X), device=X.device))
        prior_mean = self.model.mean_module(X)[0]
        mean = KxX_dx @ K_XX_inv @ (Y - prior_mean) # graident of posterior mean
        return mean
    
    # @t_batch_mode_transform(expected_q=1)
    def forward(self,thetas):
        """Compute the acquisition function value at thetas."""

        acquisition_values = []
        for theta in thetas:
            noise = self.model.likelihood.noise
            
            # variance of gradient before augmenting imaginary data point
            X_old = self.model.train_inputs[0]
            KxX_dx_old = self._get_KxX_dx(self.current_theta, X_old)
            K_XX_old = self.model.covar_module(X_old, X_old).evaluate()
            K_XX_inv_old = torch.inverse(K_XX_old + noise*torch.eye(len(X_old), device=X_old.device))
            K_xX_dx2_old = self._get_KxX_dx2(self.current_theta, self.current_theta)
            variance_old = K_xX_dx2_old - KxX_dx_old @ K_XX_inv_old @ KxX_dx_old.transpose(1,2) # variance of gradient
            log_det_old = torch.logdet(variance_old)
            
            # variance of gradient after augmenting imaginary data point
            X = torch.cat((self.model.train_inputs[0], theta))
            KxX_dx = self._get_KxX_dx(self.current_theta, X)
            K_XX = self.model.covar_module(X, X).evaluate()
            K_XX_inv = torch.inverse(K_XX + noise*torch.eye(len(X), device=X.device))
            K_xX_dx2 = self._get_KxX_dx2(self.current_theta, self.current_theta)
            variance_new = K_xX_dx2 - KxX_dx @ K_XX_inv @ KxX_dx.transpose(1,2) # variance of gradient
            log_det_new  = torch.logdet(variance_new)
            
            acq_val = 0.5*(log_det_old - log_det_new)/float(cost_model(theta)[0][0])
            acquisition_values.append(acq_val)
          
        
        return torch.cat(acquisition_values,dim=0).flatten()

    
def cont_kernel_factory(
    batch_shape: torch.Size,
    ard_num_dims: int,
    active_dims: List[int],
) -> RBFKernel:
    return RBFKernel(
        batch_shape=batch_shape,
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
        lengthscale_constraint=GreaterThan(1e-04),
        
    )

def generate_initial_data(seed = None, fun = None, dim=-1):
    # generate training data
   
    np.random.seed((seed)*1)
    N_l1 = 4 # number of training data for the first level -1 
    N_l2 = 5 # number of training data for the second level                  
    ind_qual = [dim] # column index for the qualatative variable
    
    
    # Initial location for local algorithm
    N_test = 1
    X_te_normalized = 0.2+0.6*np.random.rand(N_test,dim) # random select a point that we want to estimate gradient 
    X_te = lb+(ub-lb)*X_te_normalized  # rescale   
    qualatative_column_te = np.random.choice([1], size=N_test) # we want to estimate the gradient for the highest fidelity function
    if ind_qual is not None:
        X_te = np.column_stack((X_te, qualatative_column_te)) # concatenate the qualatative variable into testing data
        X_te_normalized  = np.column_stack((X_te_normalized , qualatative_column_te))
       
    # Generate initial training data for GP
    # np.random.seed((seed)*1)
    # X_l1_normalized = np.random.rand(N_l1,dim)
    X_l1_normalized = lhs(dim, samples = N_l1, random_state=seed)
    X_l1 = lb+(ub-lb)*X_l1_normalized # generate initial training data (at level1) for GP
    qualatative_column = np.random.choice([1.0], size=N_l1)  
    if ind_qual is not None:
        X_l1 = np.column_stack((X_l1, qualatative_column)) # concatenate the qualatative varibale into training data set
        X_l1_normalized = np.column_stack((X_l1_normalized, qualatative_column))
        
    # X_l2 = np.concatenate((X_l1[:,0:dim],X_te[:,0:dim]),axis=0) # initial data for the second level
    # X_l2_normalized = np.concatenate((X_l1_normalized[:,0:dim],X_te_normalized[:,0:dim]),axis=0) # initial data for the second level
    
    X_l2_normalized = lhs(dim, samples = N_l2, random_state=seed+10)
    X_l2 = lb+(ub-lb)*X_l2_normalized 
    qualatative_column = np.random.choice([2.0], size=N_l2) 
    
    if ind_qual is not None:
        X_l2 = np.column_stack((X_l2, qualatative_column)) # concatenate the qualatative varibale into training data set
        X_l2_normalized = np.column_stack((X_l2_normalized, qualatative_column))
      
    train_x_full= np.concatenate((X_l1, X_l2))
    train_x_full_normalized = np.concatenate((X_l1_normalized, X_l2_normalized))
    train_x_full = np.concatenate((train_x_full, X_te)) # need to include the estimated point into the training data   
    train_x_full_normalized = np.concatenate((train_x_full_normalized, X_te_normalized))
          
    train_obj = fun(torch.tensor(train_x_full)).unsqueeze(1) # calculate the true function value
    return torch.tensor(train_x_full_normalized), train_obj


def initialize_model(train_x, train_obj, dim):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = GreaterThan(1e-04))
    model = MixedSingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=1), likelihood = likelihood, cat_dims = [dim], cont_kernel_factory=cont_kernel_factory)
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # model.covar_module(train_x, train_x).evaluate()
    # model.covar_module.kernels[0].base_kernel.kernels[1](train_x, train_x).evaluate()
    return mll, model



if __name__ == '__main__':
    # torch.set_printoptions(precision=6, sci_mode=False)
    
    dim = 12
    dim_all = dim+1
    lr = 0.05 # learning rate for gradient ascent
    N_ITER = 300
    N_iner = int(0.5*dim)
    replicate = 10
    initial_cost = 55
    lb = np.array([0]*(dim)) # lb for Rosenbrock
    ub = np.array([2]*(dim)) # ub for Rosenbrock
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], **tkwargs) # bounds for optimizing acquisition    
    bound_cat = torch.tensor([1.0,2.0]) # bound for categorical variable
    bounds = torch.cat((bounds, bound_cat.unsqueeze(1)), dim=1)
    # Define testing function
    fun = Rosenbrock(dim=dim, LVGP=True, negate=True)
    # cost_model = AffineFidelityCostModel(fidelity_weights={6: 1.0}, fixed_cost=5.0)
    cost_model = FlexibleFidelityCostModel(values={'1.0':10, '2.0':1}, fixed_cost=0)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    cost_list = [[] for _ in range(replicate)]
    best_Y_list = [[] for _ in range((replicate))] 
    
    for seed in range(replicate):
        train_x, train_obj = generate_initial_data(seed=seed, fun = fun, dim=dim)
        train_x = train_x.to(device)
        train_obj = train_obj.to(device)
        cost_list[seed].append(initial_cost)
        best_Y_list[seed].append(-float(train_obj[-1]))
        cumulative_cost = initial_cost
       
        theta = train_x[-1].unsqueeze(0).clone()
        
        for i in range(N_ITER):
            mll, model = initialize_model(train_x, train_obj, dim)
            try:
                fit_gpytorch_mll(mll)
            except:
                print('cant fit GP')         
            CAGES_acq = CustomAcquisitionFunction(model, theta)
            # inner loop for querying point to decrease gradient uncertainty
            for I in range(N_iner):
                # mfkg_acqf = get_mfkg(model, dim)
                # new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf, fun, dim)
                lb_acq = theta.clone()
                ub_acq = theta.clone()
                lb_acq = lb_acq - 0.1
                ub_acq = ub_acq + 0.1
                bounds_acq = torch.cat([lb_acq, ub_acq])
                bounds_acq[bounds_acq>1] = 1
                bounds_acq[bounds_acq<0] = 0
                candidate, _ = optimize_acqf_mixed(
                    acq_function=CAGES_acq, 
                    bounds = bounds_acq,
                    fixed_features_list=[{dim: 2.0}, {dim: 1.0}],
                    q=1,
                    num_restarts=5,
                    raw_samples=20
                )
                
                new_x = candidate.clone()
                new_x[:,0:dim] = torch.tensor(lb).to(device) + (torch.tensor(ub-lb)).to(device)*new_x[:,0:dim]
                new_obj = fun(new_x).unsqueeze(1)
                train_x = torch.cat([train_x, candidate]).detach()
                train_obj = torch.cat([train_obj, new_obj]).detach()  
                
                mll, model = initialize_model(train_x, train_obj, dim)
                try:
                    fit_gpytorch_mll(mll)
                except:
                    print('cant fit GP')
                CAGES_acq = CustomAcquisitionFunction(model, theta)
                
                cost = float(cost_model(new_x)[0][0])
                cumulative_cost += cost
                # cost_list[seed].append(int(cumulative_cost))
                # best_Y_list[seed].append(min(best_Y_list[seed][-1],-float(new_obj[0])))
                
            # moving via gradient ascent
            
            gradient = CAGES_acq.calculate_gradient()
            theta[:,0:dim] = (theta[:,0:dim] + lr*gradient/torch.norm(gradient)).clone()
            # theta[:,0:dim] = theta[:,0:dim] + lr*gradient
            theta_rescale = theta.clone()
            theta_rescale[:,0:dim] = torch.tensor(lb) + (torch.tensor(ub-lb))*theta_rescale[:,0:dim]
            new_obj = fun(theta_rescale.clone()).unsqueeze(1)
            train_x = torch.cat([train_x, theta]).detach()
            train_obj = torch.cat([train_obj, new_obj]).detach()  
            cost = float(cost_model(theta)[0][0])
            cumulative_cost += cost
            cost_list[seed].append(int(cumulative_cost))
            best_Y_list[seed].append(min(best_Y_list[seed][-1],-float(new_obj[0])))
            print(new_obj)
            if cumulative_cost>400:
                break
            
    max_length = max(len(row) for row in cost_list)
    padded_list = np.array([row + [row[-1]] * (max_length - len(row)) for row in cost_list])
    xx = torch.tensor(padded_list)
    
    max_length1 = max(len(row) for row in best_Y_list)
    padded_list1 = np.array([row + [row[-1]] * (max_length1 - len(row)) for row in best_Y_list])
    yy = torch.tensor(padded_list1)
    
    # Save results
    np.save('Rosenbrock_cost_CAGES_MFBO.npy',xx)
    np.save('Rosenbrock_reward_CAGES_MFBO.npy',yy)
