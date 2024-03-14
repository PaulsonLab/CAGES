from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List

import numpy as np
import torch
from torch import Tensor
import botorch
import gpytorch
from src.model import DerivativeExactGPSEModel
# from .exact_prediction_strategies import prediction_strategy
delta_n = lambda n: ((1 / 6) * n) ** (1 / 2) * (
    1 / 3 * (1 + 2 * (1 - 3 / (5 * n)) ** (1 / 2))
) ** (1 / 2)
delta_n_lower_bound = lambda n: 1 / 3 * n ** (1 / 2)
get_lengthscales = lambda n, factor: delta_n(n) * factor
factor_hennig = 0.1 / delta_n(2)


def irregular_grid(dim: int, seed = None) -> Tensor:
    """Generate irragular grid with a quasirandom sobol sampler.

    Args:
        dim: Dimensions of grid.

    Return:
        Grid sample.
    """
    soboleng = torch.quasirandom.SobolEngine(dimension=dim, seed = seed)

    def sample(number: int):
        return soboleng.draw(number)

    return sample


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model with constant mean and SE-kernel.

    Attributes:
        train_x: The training features X.
        train_y: The training targets y.
        likelihood: The model's likelihood.
    """

    def __init__(self, train_x, train_y, likelihood, ard_num_dims):
        """Inits the model."""
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )

    def forward(self, x):
        """Compute the prior latent distribution on a given input.

        Args:
            x: The test points.

        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def sample_from_gp_prior(
    dim: int,
    num_samples: int = 1000,
    gp_hypers: Dict[str, Tensor] = {
        "covar_module.base_kernel.lengthscale": torch.tensor(0.1),
        "covar_module.outputscale": torch.tensor(1.0),
    },
    seed: int = None
) -> Tuple[Tensor, Tensor]:
    """Sample random points from gp prior.

    Args:
        dim: Dimension of sample grid for train_x data.
        num_samples: Number of train_x samples.
        gp_hypers: GP model hyperparameters.

    Returns:
        Trainings features and targets.
    """

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(None, None, likelihood, ard_num_dims=dim)
    model.initialize(**gp_hypers)
    train_x = irregular_grid(dim, seed)(num_samples)
    train_y = sample_from_gp_prior_helper(model)(train_x)
    return train_x, train_y


def sample_from_gp_prior_helper(model) -> Tensor:
    """Helper function to sample from GP prior model.

    Args:
        model: GP prior model.

    Returns:
        GP prior sample.
    """

    def sample(x):
        # torch.manual_seed(0)
        model.train(False)
        with gpytorch.settings.prior_mode(True):
            mvn = model(x)
        return mvn.sample().flatten()

    return sample


def generate_objective_from_gp_post(
    train_x: Tensor,
    train_y: Tensor,
    noise_variance: float = 1e-6,
    gp_hypers: Dict[str, Tensor] = {
        "covar_module.base_kernel.lengthscale": torch.tensor(0.1),
        "covar_module.outputscale": torch.tensor(1.0),
    },
) -> Callable[[Tensor], float]:
    """Generate objective function with given train_x, train_y and hyperparameters.

    Args:
        train_x: The training features X.
        train_y: The training targets y.
        noise_variance: Observation noise.
        gp_hypers: GP model hyperparameters.

    Returns:
        Objective function.
    """
    train_x = train_x.to(dtype=torch.float64)
    train_y = train_y.to(dtype=torch.float64)
    dim = train_x.shape[-1]
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.Interval(0,1e-4))
    model = ExactGPModel(None, None, likelihood, ard_num_dims=dim)
    model.initialize(**gp_hypers)
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)

    def objective(x, observation_noise=False, requires_grad=False):
        model.eval()
        with gpytorch.settings.fast_pred_var():
            posterior = model(x)
            m = posterior.mean.flatten()
            if observation_noise:
                m += torch.randn_like(m) * np.sqrt(noise_variance)
        if requires_grad:
            return m
        else:
            return m.detach()

    return objective

def get_KXX_inv(model,X):
    """Get the inverse matrix of K(X,X).

    Returns:
        The inverse of K(X,X).
    """
    # L_inv_upper = gpytorch.models.exact_prediction_strategies.DefaultPredictionStrategy.covar_cache.detach()
    # return L_inv_upper @ L_inv_upper.transpose(0, 1)
    # print('noise = ', model.likelihood.noise)
    return torch.inverse(model.covar_module(X, X).evaluate() + torch.eye(X.shape[0]) * model.likelihood.noise)

def _get_KxX_dx(x, model, dim):
    """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

    Args:
        x: (n x D) Test points.

    Returns:
        (n x D) The derivative of K(x,X) w.r.t. x.
    """
    
    X = model.train_inputs[0]
    
    X = X.to(torch.float64)
    x = x.to(torch.float64)
    
    n = x.shape[0]
    K_xX = model.covar_module(x, X).evaluate()
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().to(torch.float64)
    # lengthscale = self.covar_module.base_kernel.lengthscale.detach()
    return (
        -torch.eye(dim, device=x.device)
        / lengthscale ** 2
        @ (
            (x.view(n, 1, dim) - X.view(1, X.shape[0], dim))
            * K_xX.view(n, X.shape[0], 1)
        ).transpose(1, 2)
    )

def generate_objective_gradient_from_gp_post(
    train_x: Tensor,
    train_y: Tensor,
    noise_variance: float = 1e-6,
    gp_hypers: Dict[str, Tensor] = {
        "covar_module.base_kernel.lengthscale": torch.tensor(0.1),
        "covar_module.outputscale": torch.tensor(1.0),
    },
) -> Callable[[Tensor], float]:
    """Generate objective function with given train_x, train_y and hyperparameters.

    Args:
        train_x: The training features X.
        train_y: The training targets y.
        noise_variance: Observation noise.
        gp_hypers: GP model hyperparameters.

    Returns:
        Objective function.
    """
    
    train_x = train_x.to(dtype=torch.float64)
    train_y = train_y.to(dtype=torch.float64)
    dim = train_x.shape[-1]
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.Interval(0,1e-4))
    model = ExactGPModel(None, None, likelihood, ard_num_dims=dim)
    model.initialize(**gp_hypers)
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)
    
    
    
    def objective_gradient(x):
        """Computes the true derivative and Hessian objective function w.r.t. the given test
        points x.

        Args:
            x: (n x D) Test points.

        Returns:
            A GPyTorchPosterior.
        """
        
        lengthscale = model.covar_module.base_kernel.lengthscale.detach().to(torch.float64)
        X = model.train_inputs[0]
        alpha = get_KXX_inv(model, X) @ model.train_targets
        # print('true alpha = ', alpha)
        gradient_true = []
        for i in range(x.shape[1]):
            
            dfdx = model.covar_module(x, X).evaluate() * -(x[0][i]-X[:,i])/lengthscale[0][i]**2 # partial derivative of kernel w.r.t. the ith dimension
            gradient_true.append(torch.sum(alpha * dfdx))
        
        hessian = torch.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                if i==j:
                    gradient = model.covar_module(x, X).evaluate() * -(x[0][i]-X[:,i])/lengthscale[0][i]**2 # gradient of each kernel         
                    d2k_dx2 = gradient * -(x[0][i]-X[:,i])/lengthscale[0][i]**2 - model.covar_module(x, X).evaluate()/lengthscale[0][i]**2 # second derivative of each kernel
                    d2k_dx2 = torch.sum(d2k_dx2 * alpha)
                    hessian[i,j] = d2k_dx2
                else:
                    gradient = model.covar_module(x, X).evaluate() * -(x[0][i]-X[:,i])/lengthscale[0][i]**2 # gradient of each kernel   
                    d2k_dx2 = gradient * -(x[0][j]-X[:,j])/lengthscale[0][j]**2 # second derivative of each kernel
                    d2k_dx2 = torch.sum(d2k_dx2 * alpha)
                    hessian[i,j] = d2k_dx2
                    
                    
                    
        return torch.tensor(gradient_true), hessian
    
    return objective_gradient

def generate_training_samples(
    num_objectives: int, dim: int, num_samples: int, gp_hypers: Dict[str, Tensor], seed: int
) -> Tuple[List[Tensor], List[Tensor]]:
    """Generate training samples for `num_objectives` objectives.

    Args:
        num_objectives: Number of objectives.
        dim: Dimension of parameter space/sample grid.
        num_samples: Number of grid samples.
        gp_hypers: GP model hyperparameters.

    Returns:
        List of trainings features and targets.
    """

    train_x = []
    train_y = []
    for _ in range(num_objectives):
        x, y = sample_from_gp_prior(dim, num_samples, gp_hypers, seed)
        train_x.append(x)
        train_y.append(y)
    return train_x, train_y


# Compute the reward (output value) for the current smapling point
def compute_rewards(
    params: Tensor, objective: Callable[[Tensor], float], verbose: bool = False
) -> List[float]:
    """Compute rewards as return of objective function with given parameters.

    Args:
        params: Parameters as input for objective function.
        objective: Objective function.
        verbose: If True an output is logged.

    Returns:
        Rewards for parameters.
    """

    rewards = []
    for i, param in enumerate(params):
        # reward = objective(param, observation_noise=False).item() # uncomment this if using synthetic function
        reward = -objective(param).item() # uncomment this if using benchmark function
        rewards.append(reward)
        if verbose:
            print(f"Iteration {i+1}, reward {reward :.2f}.")
    return rewards


def get_maxima_objectives(
    lengthscales: Dict[int, Tensor],
    noise_variance: float,
    train_x: Dict[int, Tensor],
    train_y: Dict[int, Tensor],
    n_max: Optional[int],
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    """Compute maxima of synthetic objective functions.

    Args:
        lengthscales: Hyperparameter for GP model.
        noise_variance: Observation noise.
        train_x: The training features X.
        train_y: The training targets y.
        n_max: Number of train_x samples for optimization's starting points.

    Returns:
        Maxima values (max) and positions (argmax).
    """
    f_max_dict = {}
    argmax_dict = {}
    dimensions = list(lengthscales.keys())
    number_objectives = len(train_y[dimensions[0]])
    for dim in dimensions:
        f_max_dim = []
        argmax_dim = []
        for index_objective in range(number_objectives):
            objective = generate_objective_from_gp_post(
                train_x[dim][index_objective],
                train_y[dim][index_objective],
                noise_variance=noise_variance,
                gp_hypers={
                    "covar_module.base_kernel.lengthscale": lengthscales[dim],
                    "covar_module.outputscale": torch.tensor(1.0),
                },
            )
            f = lambda x: objective(x, observation_noise=False, requires_grad=True)
            if n_max:
                _, indices_sort = torch.sort(train_y[dim][index_objective])
                init_cond = train_x[dim][index_objective][indices_sort[:n_max]]
            else:
                init_cond = train_x[dim][index_objective]
            clamped_candidates, batch_acquisition = botorch.gen_candidates_scipy(
                initial_conditions=init_cond,
                acquisition_function=f,
                lower_bounds=torch.tensor([[0.0] * dim]),
                upper_bounds=torch.tensor([[1.0] * dim]),
            )
            max_optimizer, index_optimizer = torch.max(batch_acquisition, dim=0)
            max_train_samples, index_max_train_samples = torch.max(
                train_y[dim][index_objective], dim=0
            )
            if max_optimizer > max_train_samples:
                f_max_dim.append(max_optimizer.clone().item())
                argmax_dim.append(clamped_candidates[index_optimizer])
            else:
                f_max_dim.append(max_train_samples.clone().item())
                argmax_dim.append(
                    train_x[dim][index_objective][index_max_train_samples]
                )
        f_max_dict[dim] = f_max_dim
        argmax_dict[dim] = argmax_dim
    return f_max_dict, argmax_dict


def get_lengthscale_hyperprior(dim: int, factor_lengthscale: int, gamma: float):
    """Compute hyperprior for lengthscale.

    Args:
        dim: Dimension of search space.
        factor_lengthscale: Scale for upper bound of lengthscales' sample
            distribution.
        gamma: Noise parameter for uniform sample distribution for lengthscales.

    Returns:
        Gpytorch hyperprior for lengthscales.
    """
    l = get_lengthscales(dim, factor_hennig)
    a = factor_lengthscale * l * (1 - gamma)
    b = factor_lengthscale * l * (1 + gamma)
    return gpytorch.priors.UniformPrior(a, b)
