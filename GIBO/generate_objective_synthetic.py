#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:13:30 2023

@author: tang.1856
"""
import torch
from src.synthetic_functions import (
    generate_objective_from_gp_post,
    generate_training_samples,
    get_maxima_objectives,
    get_lengthscales,
    factor_hennig,
    get_lengthscale_hyperprior,
    generate_objective_gradient_from_gp_post,
)

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
    "dtype": torch.double,
    
}

def generate_objective_synthetic(dim,seed):
    torch.manual_seed(seed)
    
    train_x_dict = {}
    train_y_dict = {}
    lengthscales_dict = {}
    l = get_lengthscales(dim, factor_hennig)
    m = torch.distributions.Uniform(
        2 * l * (1 - 0.3),
        2 * l * (1 + 0.3),
    )
    lengthscale = m.sample((1, dim))
    train_x, train_y = generate_training_samples(
        num_objectives=1,
        dim=dim,
        num_samples=1000,
        gp_hypers={
            "covar_module.base_kernel.lengthscale": lengthscale,
            "covar_module.outputscale": torch.tensor(
                1
            ),
        },
        seed = seed,
    )
    
    train_x_dict[dim] = train_x
    train_y_dict[dim] = train_y
    lengthscales_dict[dim] = lengthscale
    
    f_max_dict, argmax_dict = get_maxima_objectives(
        lengthscales=lengthscales_dict,
        noise_variance=0.01,
        train_x=train_x_dict,
        train_y=train_y_dict,
        n_max=15,
    )
    
    objective = generate_objective_from_gp_post(
        train_x_dict[dim][0].to(**tkwargs),
        train_y_dict[dim][0].to(**tkwargs),
        noise_variance=0.01,
        gp_hypers={
            "covar_module.base_kernel.lengthscale": lengthscales_dict[dim].to(**tkwargs),
            "covar_module.outputscale": torch.tensor(
                1
            ).to(**tkwargs),
        },
    )
    
    objective_gradient = generate_objective_gradient_from_gp_post(
        train_x_dict[dim][0].to(**tkwargs),
        train_y_dict[dim][0].to(**tkwargs),
        noise_variance=0.01,
        gp_hypers={
            "covar_module.base_kernel.lengthscale": lengthscales_dict[dim].to(**tkwargs),
            "covar_module.outputscale": torch.tensor(
                1
            ).to(**tkwargs),
        },
    )
    print(f"Max of objective: {f_max_dict[dim]}.")
    
    return objective, objective_gradient, f_max_dict, [2 * l * (1 - 0.3), 2 * l * (1 + 0.3)]