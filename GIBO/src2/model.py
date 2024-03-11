import torch
import gpytorch
import botorch


class ExactGPSEModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    """An exact Gaussian process (GP) model with a squared exponential (SE) kernel.

    ExactGP: The base class of gpytorch for any Gaussian process latent function to be
        used in conjunction with exact inference.
    GPyTorchModel: The easiest way to use a GPyTorch model in BoTorch.
        This adds all the api calls that botorch expects in its various modules.

    Attributes:
        train_x: (N x D) The training features X.
        train_y: (N x 1) The training targets y.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    _num_outputs = 1  # To inform GPyTorchModel API.

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint=None,
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint = noise_constraint, noise_prior=None
        )
        if noise_hyperprior is not None:
            likelihood.noise = noise_hyperprior[0][0]
        if train_y is not None:
            train_y = train_y.squeeze(-1)
        super(ExactGPSEModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        if prior_mean != 0:
            self.mean_module.initialize(constant=prior_mean)
            self.mean_module.constant.requires_grad = False

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dims,
                #lengthscale_prior=lengthscale_hyperprior,
                # lengthscale_constraint=gpytorch.constraints.Interval(0,10),
                lengthscale_constraint=lengthscale_constraint,
            ),
            # outputscale_constraint = gpytorch.constraints.Interval(0,100)
            outputscale_constraint = outputscale_constraint
          
        )
        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale_hyperprior
        if outputscale_hyperprior is not None:
            self.covar_module.outputscale = outputscale_hyperprior

    def forward(self, x):
        """Compute the prior latent distribution on a given input.

        Typically, this will involve a mean and kernel function. The result must be a
        MultivariateNormal. Calling this model will return the posterior of the latent
        Gaussian process when conditioned on the training data. The output will be a
        MultivariateNormal.

        Args:
            x: (n x D) The test points.

        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DerivativeExactGPSEModel(ExactGPSEModel):
    """Derivative of the ExactGPSEModel w.r.t. the test points x.

    Since differentiation is a linear operator this is again a Gaussian process.

    Attributes:
        D: Dimension of train_x-/input-data.
        normalize: Optional normalization function for policy parameterization.
        unnormalize: Optional unnormalization function for policy
            parameterization.
        N_max: Maximum number of training samples (train_x, N) for model inference.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    def __init__(
        self,
        D: int,
        normalize=None,
        unnormalize=None,
        N_max=None,
        lengthscale_constraint=None,
        lengthscale_hyperprior=None,
        outputscale_constraint=None,
        outputscale_hyperprior=None,
        noise_constraint=None,
        noise_hyperprior=None,
        ard_num_dims=None,
        prior_mean=0.0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        train_x_init, train_y_init = (
            torch.empty(0, D),
            torch.empty(0),
        )
        super(DerivativeExactGPSEModel, self).__init__(
            train_x_init,
            train_y_init,
            lengthscale_constraint,
            lengthscale_hyperprior,
            outputscale_constraint,
            outputscale_hyperprior,
            noise_constraint,
            noise_hyperprior,
            ard_num_dims,
            prior_mean,
        )

        self.N_max = N_max
        self.D = D
        self.N = 0
        self.train_xs = train_x_init
        self.train_ys = train_y_init
        if normalize is None:
            normalize = lambda params: params
        self.normalize = normalize
        if unnormalize is None:
            unnormalize = lambda params: params
        self.unnormalize = unnormalize

    def append_train_data(self, train_x, train_y):
        """Adaptively append training data.

        Optionally translates train_x data for the state normalization of the
            MLP.

        Args:
            train_x: (1 x D) New training features.
            train_y: (1 x 1) New training target.
        """
        self.train_xs = torch.cat([self.unnormalize(train_x), self.train_xs])
        self.train_ys = torch.cat([train_y, self.train_ys])
        
        # self.N_max = 5*self.D
        if (self.N_max is not None) or (self.N_max != -1):
            # args = torch.argsort(
            #    self.covar_module(self.train_xs, self.unnormalize(train_x))
            #    .evaluate()
            #    .view(-1),
            #    descending=False,
            # )
            # self.train_xs = self.train_xs[args][: self.N_max]
            # self.train_ys = self.train_ys[args][: self.N_max]
            self.train_xs = self.train_xs[: self.N_max]
            self.train_ys = self.train_ys[: self.N_max]
            
        if self.train_ys.size(-1) == 1:
            targets = self.train_ys
        else:
            targets = (self.train_ys - self.train_ys.mean()) / (
                self.train_ys.std(unbiased=False) + 1e-10
            )
       

        self.set_train_data(
            inputs=self.normalize(self.train_xs),
            targets=targets,
            strict=False,
        )

        self.N = self.train_xs.shape[0]

    def get_L_lower(self):
        """Get Cholesky decomposition L, where L is a lower triangular matrix.

        Returns:
            Cholesky decomposition L.
        """
        return (
            self.prediction_strategy.lik_train_train_covar.root_decomposition()
            .root.evaluate()
            .detach()
        )

    def get_KXX_inv(self):
        """Get the inverse matrix of K(X,X).

        Returns:
            The inverse of K(X,X).
        """
        L_inv_upper = self.prediction_strategy.covar_cache.detach()
        return L_inv_upper @ L_inv_upper.transpose(0, 1)

    def get_KXX_inv_old(self):
        """Get the inverse matrix of K(X,X).

        Not as efficient as get_KXX_inv.

        Returns:
            The inverse of K(X,X).
        """
        X = self.train_inputs[0]
        sigma_n = self.likelihood.noise_covar.noise.detach()
        return torch.inverse(
            self.covar_module(X).evaluate() + torch.eye(X.shape[0]) * sigma_n
        )

    def _get_KxX_dx_SE(self, x):
        """Computes the analytic derivative of the SE kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        
        X = self.train_inputs[0]
        
        X = X.to(torch.float64)
        x = x.to(torch.float64)
        
        n = x.shape[0]
        K_xX = self.covar_module(x, X).evaluate()
        lengthscale = self.covar_module.base_kernel.lengthscale.detach().to(torch.float64)
        # lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.D, device=x.device)
            / lengthscale ** 2
            @ (
                (x.view(n, 1, self.D) - X.view(1, self.N, self.D))
                * K_xX.view(n, self.N, 1)
            ).transpose(1, 2)
        )
    
   
        
        
    def _get_Kxx_dx2(self):
        """Computes the analytic second derivative of the kernel K(x,x) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x D) The second derivative of K(x,x) w.r.t. x.
        """
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        sigma_f = self.covar_module.outputscale.detach()
        return (
            torch.eye(self.D, device=lengthscale.device) / lengthscale ** 2
        ) * sigma_f


    def _get_KxX_dx2(self, x):
        """Computes the analytic second derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x D) The second derivative of K(x,X) w.r.t. x.
        """
        
        X = self.train_inputs[0]
        
        X = X.to(torch.float64)
        x = x.to(torch.float64)
        n = x.shape[0]
        K_xX = self.covar_module(x, X).evaluate()
        
        lengthscale = self.covar_module.base_kernel.lengthscale.detach().to(torch.float64)
        sigma_f = self.covar_module.outputscale.detach().to(torch.float64)
        
        L = torch.eye(self.D, device=lengthscale.device) / lengthscale ** 2
        L = L.to(torch.float64)
        I = torch.eye(self.D, device=lengthscale.device).to(torch.float64)
        
        x1_x2 = (x.view(n, 1, self.D) - X.view(1, self.N, self.D)).transpose(1,2)
        
        K_xX_dx2 = []
        for i in range(self.N):
            x1_x2_current = x1_x2[0][:,i].unsqueeze(0)
            K_xX_dx2.append(-((L @ (I - x1_x2_current.transpose(-2,1) @ x1_x2_current @ L)) * K_xX[0][i]))
            
        return K_xX_dx2
        
        
        
        
    def posterior_derivative(self, x):
        """Computes the posterior of the derivative of the GP w.r.t. the given test
        points x.

        Args:
            x: (n x D) Test points.

        Returns:
            A GPyTorchPosterior.
        """
        
        if self.prediction_strategy is None:
            self.posterior(x)  # Call this to update prediction strategy of GPyTorch.
            
        K_xX_dx = self._get_KxX_dx_SE(x)
        mean_d = K_xX_dx @ self.get_KXX_inv() @ self.train_targets
        variance_d = (
            self._get_Kxx_dx2() - K_xX_dx @ self.get_KXX_inv() @ K_xX_dx.transpose(1, 2)
        )
        variance_d = variance_d.clamp_min(1e-9)

        return mean_d, variance_d
    
    
    def posterior_Hessian(self, x):
        
        """Computes the posterior of the Hessian of the GP w.r.t. the given test points x.
        
        Args:
            x: (n x D) Test points.
        
        Returns:
            A GPyTorchPosterior
        """
        lengthscale = self.covar_module.base_kernel.lengthscale.detach().to(torch.float64)
        X = self.train_inputs[0]
        alpha = self.get_KXX_inv() @ self.train_targets
        # print('alpha GP:', alpha)
        hessian = torch.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                if i==j:
                    gradient = self.covar_module(x, X).evaluate() * -(x[0][i]-X[:,i])/lengthscale[0][i]**2 # gradient of each kernel         
                    d2k_dx2 = gradient * -(x[0][i]-X[:,i])/lengthscale[0][i]**2 - self.covar_module(x, X).evaluate()/lengthscale[0][i]**2 # second derivative of each kernel
                    d2k_dx2 = torch.sum(d2k_dx2 * alpha)
                    hessian[i,j] = d2k_dx2
                else:
                    gradient = self.covar_module(x, X).evaluate() * -(x[0][i]-X[:,i])/lengthscale[0][i]**2 # gradient of each kernel   
                    d2k_dx2 = gradient * -(x[0][j]-X[:,j])/lengthscale[0][j]**2 # second derivative of each kernel
                    d2k_dx2 = torch.sum(d2k_dx2 * alpha)
                    hessian[i,j] = d2k_dx2
                    
        # print('Hessian GP:', hessian)            
        second_derivative = self._get_KxX_dx2(x)
        K_XX_inverse = self.get_KXX_inv()
        alpha = K_XX_inverse @ self.train_targets
        
        mean_hessian = 0
        for i in range(self.N):
            mean_hessian+=second_derivative[i]*alpha[i]
            
        return mean_hessian
    
    def Calculate_true_gp(self, x):
        """Computes the true gradient for the GP at test points x
        """
        
        lengthscale = self.covar_module.base_kernel.lengthscale.detach().to(torch.float64)
        X = self.train_inputs[0]
        alpha = self.get_KXX_inv() @ self.train_targets
        
        gradient = []
        for i in range(x.shape[1]):
            
            dfdx = self.covar_module(x, X).evaluate() * -(x[0][i]-X[:,i])/lengthscale[0][i]**2
            gradient.append(torch.sum(alpha * dfdx))
        
        return torch.tensor(gradient)
        
        
        
        