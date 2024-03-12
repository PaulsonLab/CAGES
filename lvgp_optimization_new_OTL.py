import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
import torch
from lvgp_grad import lvgp_gradient
import matplotlib.pyplot as plt
from acquisition_function import GradientInformation_entropy
import torch.optim as optim
import random
from scipy.stats import norm
from test_function import Rosenbrock, Borehole, OTL, Piston
from LVGP_main import LVGP
from RL_function import RL_fun
from pyDOE import lhs

if __name__ == '__main__':
     
    replicate = 10
    replicate_list = [10, 40, 50, 80, 90, 140, 170, 210, 220, 280]
    norm_list = [[] for _ in range((replicate))]
    cost_list = [[] for _ in range((replicate))] 
    best_Y_list = [[] for _ in range((replicate))] 
    
    random_search = False
    new_acq = True
    
    for seed in range(replicate):
    
        np.random.seed((seed)*1)
        N_l1 = 8 
        N_l2 = 10
        N_l3 = 10
        N_l4 = 10
               
        dim = 5 # Dimension excludes qualatative variable
              
        ind_qual = [dim] # column index for the qualatative variable
        # ind_qual = None
        if ind_qual is not None:
            LVGP_ = True
        else:
            LVGP_ = False 
        
        # Define testing function
        fun = OTL(dim=dim, LVGP=LVGP_)
        
        n_opt = 4 # The number of times the log-likelihood function is optimized
        dim_z = 2 # Dimensionality of latent space, usually 1 or 2 but can be higher (Cartpole)
        
        lb = np.array([50,25,0.5,1.2,0.25]) 
        ub = np.array([150, 70, 3, 2.5, 1.2])
              
        cost = [1000, 100, 10, 1] 
        s = [2, 3, 4] 
        
        if ind_qual is not None:
            level_set1 = [1,2,3,4] # define level set
        else:
            level_set1 = [1] # define level set for no qualatative variable case
            
        BO_iter = 4
        Inner_iter = dim 
        lr = 1 
        
        # Initial location for local algorithm
        N_test = 1
        np.random.seed((seed)*1)
        X_te_normalized = 0.2+0.6*np.random.rand(N_test,dim) # starting point for the algorithm
        X_te = lb+(ub-lb)*X_te_normalized  # rescale   
        qualatative_column_te = np.random.choice([1], size=N_test) 
        if ind_qual is not None:
            X_te = np.column_stack((X_te, qualatative_column_te)) # concatenate the qualatative variable into testing data
           
        # Generate initial training data for GP
        np.random.seed((seed)*1)
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
        X = np.concatenate((X, X_te)) # need to include the estimated point into the training data   
         
        X1 = X.copy()
    
        # Convert the level to s val as to calculate the function value
        X1[X1[:, -1] == 2, -1] = s[0]
        X1[X1[:, -1] == 3, -1] = s[1]
        X1[X1[:, -1] == 4, -1] = s[2]
              
        Y = fun(torch.tensor(X1)).numpy() # calculate the true function value
        best_Y_list[seed].append(float(Y[-1]))
         
        X1_te = X_te.copy()
        XX = torch.tensor(X1_te).requires_grad_()
    
        X1_te[X1_te[:, -1] == 2, -1] = s[0]
        X1_te[X1_te[:, -1] == 3, -1] = s[1]
        X1_te[X1_te[:, -1] == 4, -1] = s[2]
        
        Y_te = fun(torch.tensor(X1_te)).numpy()
                   
        cost_list[seed].append(0)
        accu_cost = 0 # accumulated cost
        # outer loop for local BO
        for bo_iter in range(BO_iter):
            
            LVGP_class = LVGP(X, Y, ind_qual=ind_qual, dim_z=dim_z, n_opt=n_opt, progress=False, noise=False, lb=lb, ub=ub) # define LVGP class
            model = LVGP_class.lvgp_fit(X,Y) # fit LVGP
            grad_class = lvgp_gradient(model, LVGP_class.lvgp_to_latent, LVGP_class.lvgp_kernel) # Initialize the class for computing gradient
            gradient_GP = grad_class.lvgp_posterior_mean_gradient(X_te) # computing gradient for point that we want to estimate
           
            # inner loop for querying point to decrease gradient uncertainty
            for iteration in range(Inner_iter):
                                           
                opt_acq_list, argopt_acq_list = np.zeros((len(level_set1),1)), np.zeros((len(level_set1),dim)) # create for each level
                multi_starts = 5     
  
                lb_acq = X_te_normalized[0] - 0.1
                lb_acq[lb_acq<0] = 0
                ub_acq = X_te_normalized[0] + 0.1
                ub_acq[ub_acq>1]=1
                
                if new_acq:
                    bounds = [(lb_acq[i], ub_acq[i]) for i in range(dim)] # define the bound for optimizing acquisition function
                    function = GradientInformation_entropy(model, LVGP_class.lvgp_to_latent, LVGP_class.lvgp_kernel, cost, X_te)
                           
                    for level in level_set1[0:]: # optimize each level separately
                        for j in range(multi_starts):
                            
                            function.update(level) # update the level (cost function) for acquisition function
                            x_starts = np.random.uniform(lb_acq, ub_acq) # in normalized scale
                            res = minimize(function, x_starts, method='L-BFGS-B' ,bounds = bounds)               
                            
                            if -res.fun>opt_acq_list[level-1]:                     
                                opt_acq_list[level-1]=(-res.fun)
                                argopt_acq_list[level-1] = lb+(ub-lb)*(res.x) # need to rescale  
                                
                    query_level = np.array(np.argmax(opt_acq_list)+1) # next query level(the best one among level_set)                   
                    query_x = argopt_acq_list[np.argmax(opt_acq_list)] # next query point    
                    
                                       
                elif random_search:              
                    query_level = random.choice(level_set1)
                    query_x = lb+(ub-lb)*np.random.uniform(lb_acq, ub_acq)
                
                if ind_qual is not None:
                    query_x = np.append(query_x, query_level)
                    
                X = np.concatenate((X,np.array([query_x])))
                
                query_x[query_x[-1] == 2, -1] = s[0] # convert the qualatative level to s value
                query_x[query_x[-1] == 3, -1] = s[1] # convert the qualatative level to s value 
                query_x[query_x[-1] == 4, -1] = s[2]
                query_y = fun(torch.tensor(query_x.reshape(1,-1))).numpy()
                
                accu_cost+=cost[int(query_level-1)] 
                
                Y = np.append(Y,query_y)
                LVGP_class.update_LVGP(X, Y) # update the training data (X,Y) for LVGP without refitting the model
                # model = LVGP_class.lvgp_fit(X, Y) # we can also re-fit the LVGP per iteration
                              
                grad_class = lvgp_gradient(model, LVGP_class.lvgp_to_latent, LVGP_class.lvgp_kernel)
                gradient_GP = grad_class.lvgp_posterior_mean_gradient(X_te)

            X_te_normalized = (X_te_normalized[:,:dim] - lr*(gradient_GP/torch.norm(gradient_GP)).numpy()) # gradient descent
            X_te_normalized[X_te_normalized>1] = 1
            X_te_normalized[X_te_normalized<0] = 0
            X_te = lb + (ub-lb) * X_te_normalized # rescale
            if ind_qual is not None:
                X_te = np.column_stack((X_te, qualatative_column_te)) # add back the qualatative variable (level 1)
            Y_next = fun(torch.tensor(X_te)).numpy()
            
            best_Y_list[seed].append(min(best_Y_list[seed][-1],float(Y_next[0])))
            accu_cost+=cost[0]
            cost_list[seed].append(accu_cost) 
            
            X = np.concatenate((X,X_te))
            Y = np.append(Y,Y_next)
                       
            model = LVGP_class.lvgp_fit(X, Y) # fit the LVGP model
            
            XX = torch.tensor(X_te).requires_grad_()
            grad_true = torch.autograd.grad(fun(XX),XX)[0][0:dim]
            print('grad true=',grad_true)
            
    
    xx = np.array([[tensor for tensor in sublist] for sublist in cost_list])
    yy = np.array([[tensor for tensor in sublist] for sublist in best_Y_list])
    
    # Save Results
    np.save('Piston_cost_CAGES.npy',xx)
    np.save('Piston_reward_CAGES.npy',yy)
   
