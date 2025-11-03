#!/usr/bin/env python3
"""
Template for implementing custom generative models
Students should create their own implementation by inheriting from the base classes.

This file provides skeleton code for implementing generative models.
Students need to implement the TODO sections in their own files.
"""

import math
import torch
import copy
from src.base_model import BaseScheduler, BaseGenerativeModel
from src.network import UNet
from src.utils import pseudo_huber, expand_t

class CMScheduler(BaseScheduler):
    """
    Custom Scheduler Skeleton
    
    TODO: Students need to implement this class in their own file.
    Required methods:
    1. sample_timesteps: Sample random timesteps for training
    2. forward_process: Apply forward process to transform data
    3. reverse_process_step: Perform one step of the reverse process
    4. get_target: Get target for model prediction
    """
    
    def __init__(self, 
                num_train_timesteps: int,
                # Continuous-time domain
                t_min: float, 
                t_max: float, 

                # Discretization
                rho: float, 

                # Discretization scheduling
                s0: int,
                s1: int,
                num_iterations: int,

                # Timestep sampling
                p_mean: float,
                p_std: float,
                **kwargs
        ):
        super().__init__(num_train_timesteps, **kwargs)

        self.t_min = t_min
        self.t_max = t_max
        self.rho = rho
        
        # Discretization scheduling
        self.s0 = s0
        self.s1 = s1
        self.k_prime = int(num_iterations / (math.log2(s1/s0) + 1))

        # Timestep sampling
        self.p_mean = p_mean
        self.p_std = p_std

        # Caching
        self.n_k_prev = None
        self.p_cached = None
        self.timesteps_cached = None

    def compute_num_timesteps(self, k: int)-> int: # N(k) from the paper
        return min(self.s0 * (2 ** int(k/self.k_prime)), self.s1) + 1
    
    def compute_timesteps(self, n: int, device: torch.device):
        i = torch.arange(1, n + 1, dtype=torch.float32, device= device)
        t_min_rho = self.t_min ** (1 / self.rho)
        t_max_rho = self.t_max ** (1 / self.rho)
        t_i_rho = t_min_rho + ((i - 1) / (n - 1)) * (t_max_rho - t_min_rho)
        t_i = t_i_rho ** self.rho
        return t_i
    
    def sample_timesteps(self, k: int, batch_size: int, device: torch.device):
        """
        Sample random timesteps for training.
        
        Returns:
            Tensor of shape (batch_size,) with timestep values
        """
        n_k = self.compute_num_timesteps(k)
        if n_k != self.n_k_prev:
            timesteps = self.compute_timesteps(n_k, device= device)
            sigmas = timesteps
            log_sigmas = torch.log(sigmas)
            sqrt_2 = torch.sqrt(torch.tensor(2.0, device= device))

            erf_upper = torch.erf((log_sigmas[1:] - self.p_mean) / (sqrt_2 * self.p_std))
            erf_lower = torch.erf((log_sigmas[:-1] - self.p_mean) / (sqrt_2 * self.p_std))

            p = erf_upper - erf_lower
            p = p / p.sum() 

            self.n_k_prev = n_k
            self.p_cached = p.clone()
            self.timesteps_cached = sigmas.clone()

        else:
            p = self.p_cached
            sigmas = self.timesteps_cached
        
        indices = torch.multinomial(p, batch_size, replacement=True).to(device)

        t_i = sigmas[indices]
        t_i_plus_1 = sigmas[indices + 1]

        return torch.stack([t_i, t_i_plus_1], dim=1) # (batch_size, 2)
    
    def lambda_weight(self, timesteps_pair: torch.tensor) -> torch.tensor:
        """
        Calculate the weighting function lambda(sigma_i) = 1 / (sigma_{i+1} - sigma_i)
        
        Args:
            timesteps_pair: Tensor of shape (batch_size, 2) with [t_i, t_{i+1}]
            
        Returns:
            Tensor of shape (batch_size, ) with the calculated weight
        """
        t_i = timesteps_pair[:, 0]        # (B,)
        t_i_p1 = timesteps_pair[:, 1] # (B,)
        step_size = t_i_p1 - t_i
        weight = 1.0 / step_size # (B,)

        return weight

    def forward_process(self, data, noise, t):
        """
        Apply forward process to add noise to clean data.
        
        Args:
            data: Clean data tensor
            noise: Noise tensor
            t: Timestep tensor
            
        Returns:
            Noisy data at timestep t
        """
        from src.utils import expand_t
        t = expand_t(t, data)
        return data + t * noise
    
    def reverse_process_step(self, xt, pred, t, t_next):
        """
        Perform one step of the reverse (denoising) process.
        
        Args:
            xt: Current noisy data
            pred: Model prediction (e.g., predicted noise, velocity, or x0)
            t: Current timestep
            t_next: Next timestep
            
        Returns:
            Updated data at timestep t_next
        """
        # Nothing to do with consistency model
        return 
    
    def get_target(self, data, noise, t):
        """
        Get the target for model prediction (what the network should learn to predict).
        
        Args:
            data: Clean data
            noise: Noise
            t: Timestep
            
        Returns:
            Target tensor (e.g., noise for DDPM, velocity for Flow Matching)
        """

        # Nothing to do with consistency model
        return


class CMGenerativeModel(BaseGenerativeModel):
    """
    Custom Generative Model Skeleton
    
    Students need to implement this class by inheriting from BaseGenerativeModel.
    This class wraps the network and scheduler to provide training and sampling interfaces.
    """
    
    def __init__(self, network, scheduler, d: int, sigma_data:float, **kwargs):
        super().__init__(network, scheduler, **kwargs)

        self.d = d
        self.c = 0.00054 * math.sqrt(d)
        self.sigma_data = sigma_data

        self.teacher = copy.deepcopy(self.network)
        self.teacher.eval() 
    
    
    def compute_loss(self, data, current_iteration, **kwargs):
        """
        Compute the training loss.
        
        Args:
            data: Clean data batch
            current_iteration : current iteration number
            
        Returns:
            Loss tensor
        """
        B = data.shape[0]
        timesteps_pair = self.scheduler.sample_timesteps(
            k=current_iteration, 
            batch_size=B, 
            device=data.device
        )

        t_i = timesteps_pair[:, 0]        
        t_i_p1 = timesteps_pair[:, 1] 

        noise = torch.randn_like(data)

        x_t_i = self.scheduler.forward_process(data, noise, t_i)
        x_t_i_p1 = self.scheduler.forward_process(data, noise, t_i_p1)

        pred_t_i = self.predict(x_t_i, t_i)
        with torch.no_grad():
            pred_t_i_p1 = self.predict_with_teacher(x_t_i_p1, t_i_p1)

        weight = self.scheduler.lambda_weight(timesteps_pair) # (B,)
        dis = pseudo_huber(pred_t_i, pred_t_i_p1, self.c, dim=(1,2,3)) # (B,)

        return (weight * dis).mean()
    
    def update_teacher(self, decay = 1.0):
        with torch.no_grad():
            student_params = self.network.parameters()
            teacher_params = self.teacher.parameters()
            
            for teacher_p, student_p in zip(teacher_params, student_params):
                teacher_p.data.mul_(decay).add_(student_p.data, alpha=1. - decay)
        return 

    def predict_with_teacher(self, xt, t, **kwargs):
        eps = self.scheduler.t_min # float
        t_eps_squared = (t - eps) ** 2 # (B,)
        t_squared = t ** 2 # (B,)
        sigma_data_squared = self.sigma_data * self.sigma_data # float
    
        c_skip_t = sigma_data_squared / (t_eps_squared + sigma_data_squared) # (B, )
        c_out_t = ((t - eps) * self.sigma_data) / (t_squared + sigma_data_squared).sqrt() # (B,)
        c_skip_t = expand_t(c_skip_t, xt)
        c_out_t = expand_t(c_out_t, xt)
        output = self.teacher(xt, t, **kwargs)
        prediction = c_skip_t * xt + c_out_t * output
        return prediction
    
    def predict(self, xt, t, **kwargs):
        """
        Make prediction given noisy data and timestep.
        
        Args:
            xt: Noisy data
            t: Timestep
            **kwargs: Additional arguments (e.g., condition for additional timestep)
            
        Returns:
            Model prediction
        """
        eps = self.scheduler.t_min # float
        t_eps_squared = (t - eps) ** 2 # (B,)
        t_squared = t ** 2 # (B,)
        sigma_data_squared = self.sigma_data * self.sigma_data # float
    
        c_skip_t = sigma_data_squared / (t_eps_squared + sigma_data_squared) # (B, )
        c_out_t = ((t - eps) * self.sigma_data) / (t_squared + sigma_data_squared).sqrt() # (B,)
        c_skip_t = expand_t(c_skip_t, xt)
        c_out_t = expand_t(c_out_t, xt)
        output = self.network(xt, t, **kwargs)
        prediction = c_skip_t * xt + c_out_t * output
        return prediction
    
    def sample(self, shape, num_inference_timesteps=20, return_traj=False, verbose=False, **kwargs):
        """
        Generate samples from noise using the reverse process.
        
        Args:
            shape: Shape of samples to generate (batch_size, channels, height, width)
            num_inference_timesteps: Number of denoising steps (NFE)
            return_traj: Whether to return the full trajectory
            verbose: Whether to show progress
            **kwargs: Additional arguments
            
        Returns:
            Generated samples (or trajectory if return_traj=True)
        """

        traj = []
        device = next(self.network.parameters()).device
        B, C, H, W = shape
        noise = torch.randn(shape, device= device)
        timesteps = self.scheduler.compute_timesteps(num_inference_timesteps, device= device)
        eps = timesteps[0]
        timesteps = timesteps.flip(dims=[0])

        x = self.predict(noise * timesteps[0], timesteps[0])
        traj.append(x.detach().clone())

        for i in range(1, num_inference_timesteps):
            noise = torch.randn(shape, device= device)
            x = x + torch.sqrt(timesteps[i] ** 2 - eps **2) * noise
            x = self.predict(x, timesteps[i])

            traj.append(x.detach().clone())

        if return_traj:
            return traj
        else:
            return x

def create_custom_model(device="cpu", **kwargs):
    """
    Example function to create a custom generative model.
    
    Students should modify this function to create their specific model.
    
    Args:
        device: Device to place model on
        **kwargs: Additional arguments that can be passed to network or scheduler
                  (e.g., num_train_timesteps, use_additional_condition for scalar conditions
                   like step size in Shortcut Models or end timestep in Consistency Trajectory Models, etc.)
    """
    
    # Create U-Net backbone with FIXED hyperparameters
    # DO NOT MODIFY THESE HYPERPARAMETERS
    network = UNet(
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_additional_condition=kwargs.get('use_additional_condition', False)
    )

    
    # Create your scheduler
    scheduler = CMScheduler(
        num_train_timesteps=kwargs.get('num_train_timesteps', 1000), 
        t_min= kwargs.get('t_min', 0.002), 
        t_max= kwargs.get('t_max', 80), 
        rho= kwargs.get('rho', 7.0), 
        s0= kwargs.get('s0', 10),
        s1= kwargs.get('s1', 1280),
        num_iterations= kwargs.get('num_iterations', 100000),
        p_mean= kwargs.get('p_mean', -1.1),
        p_std= kwargs.get('p_std', 2.0)
    )
    
    # Create your model
    model = CMGenerativeModel(
        network,
        scheduler,
        d= kwargs.get('d', 64*64*3),
        sigma_data= kwargs.get('sigma_data', 0.5)
    )
    
    return model.to(device)