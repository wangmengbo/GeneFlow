import math
import torch
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class GaussianDiffusion:
    """
    Implements Gaussian diffusion process for generative modeling.
    This follows the DDPM (Denoising Diffusion Probabilistic Models) approach
    with improvements from more recent papers.
    """
    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2,
        clip_denoised=True,
        predict_noise=True,  # True: predict noise, False: predict x_0
        device="cpu"
    ):
        """
        Initialize the Gaussian diffusion model.
        
        Args:
            timesteps: Number of diffusion steps
            beta_schedule: Schedule for noise variance (linear or cosine)
            beta_start: Starting value for beta schedule
            beta_end: Ending value for beta schedule
            clip_denoised: Whether to clip denoised values to valid range
            predict_noise: Whether the model predicts noise (True) or x_0 (False)
            device: Device to place the diffusion parameters on
        """
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clip_denoised = clip_denoised
        self.predict_noise = predict_noise
        self.device = device
        
        # Set up beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, timesteps, dtype=torch.float32, device=device
            )
        elif beta_schedule == "cosine":
            # Cosine schedule from improved DDPM paper
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32, device=device)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Calculate diffusion process parameters
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior variance
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Adds noise to the initial sample according to the diffusion schedule.
        
        Args:
            x_0: Initial clean sample [B, C, H, W]
            t: Timestep(s) to diffuse to [B]
            noise: Optional noise to add [B, C, H, W]
            
        Returns:
            x_t: Noisy sample at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Get diffusion coefficients for this timestep
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        # Forward process: q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * ε
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Calculate the mean and variance of the posterior distribution q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0: Predicted clean sample [B, C, H, W]
            x_t: Noisy sample at timestep t [B, C, H, W]
            t: Timesteps [B]
            
        Returns:
            Dictionary containing:
                mean: Posterior mean
                variance: Posterior variance
                log_variance: Log posterior variance
        """
        posterior_mean_coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
        }
    
    def p_mean_variance(self, model, x_t, t, rna_expr, gene_mask=None, num_cells=None, is_multi_cell=False, clip_denoised=None):
        """
        Calculate the mean and variance of p(x_{t-1} | x_t) using the model prediction.
        
        Args:
            model: The neural network model
            x_t: Noisy sample at timestep t [B, C, H, W]
            t: Timesteps [B]
            rna_expr: RNA expression data
            gene_mask: Optional gene mask
            num_cells: Optional number of cells per patch
            is_multi_cell: Whether using multi-cell model
            clip_denoised: Whether to clip denoised values
            
        Returns:
            Dictionary containing:
                mean: Model predicted mean
                variance: Model predicted variance
                log_variance: Log of model predicted variance
                pred_x_0: Predicted clean sample
        """
        if clip_denoised is None:
            clip_denoised = self.clip_denoised
        
        # Get model prediction
        if is_multi_cell:
            model_output = model(x_t, t, rna_expr, num_cells, gene_mask)
        else:
            model_output = model(x_t, t, rna_expr, gene_mask)
        
        if self.predict_noise:
            # Model predicts noise ε
            pred_noise = model_output
            # Calculate x_0 from the noise prediction
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
            alpha_cumprod_t_sqrt = torch.sqrt(alpha_cumprod_t)
            beta_cumprod_t_sqrt = torch.sqrt(1 - alpha_cumprod_t)
            pred_x_0 = (x_t - beta_cumprod_t_sqrt * pred_noise) / alpha_cumprod_t_sqrt
        else:
            # Model directly predicts x_0
            pred_x_0 = model_output
            
        if clip_denoised:
            # pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
            pred_x_0 = torch.clamp(pred_x_0, 0.0, 1.0)
            
        # Calculate model mean (posterior mean) and variance
        posterior = self.q_posterior_mean_variance(x_0=pred_x_0, x_t=x_t, t=t)
        model_mean = posterior["mean"]
        posterior_variance = posterior["variance"]
        posterior_log_variance = posterior["log_variance"]
        
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_x_0": pred_x_0,
            "pred_noise": pred_noise if self.predict_noise else None,
        }
        
    def p_sample(self, model, x_t, t, rna_expr, gene_mask=None, num_cells=None, is_multi_cell=False):
        """
        Sample x_{t-1} from the model.
        
        Args:
            model: The neural network model
            x_t: Noisy sample at timestep t [B, C, H, W]
            t: Timesteps [B]
            rna_expr: RNA expression data
            gene_mask: Optional gene mask
            num_cells: Optional number of cells per patch
            is_multi_cell: Whether using multi-cell model
            
        Returns:
            Denoised sample x_{t-1} and predicted clean sample x_0
        """
        out = self.p_mean_variance(
            model=model,
            x_t=x_t,
            t=t,
            rna_expr=rna_expr,
            gene_mask=gene_mask,
            num_cells=num_cells,
            is_multi_cell=is_multi_cell,
        )
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        # Add noise scaled by the posterior variance
        noise = torch.randn_like(x_t)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        
        return sample, out["pred_x_0"]
    
    def p_sample_loop(self, model, rna_expr, img_shape, gene_mask=None, num_cells=None, is_multi_cell=False, progress=True):
        """
        Generate samples from the model using the reverse diffusion process.
        
        Args:
            model: The neural network model
            rna_expr: RNA expression data [B, gene_dim] or [B, num_cells, gene_dim]
            img_shape: Shape of the image to generate (C, H, W)
            gene_mask: Optional gene mask
            num_cells: Optional number of cells per patch
            is_multi_cell: Whether using multi-cell model
            progress: Whether to show progress bar
            
        Returns:
            Generated samples [B, C, H, W]
        """
        device = next(model.parameters()).device
        batch_size = rna_expr.shape[0]
        
        # Start from pure noise (x_T)
        img = torch.randn(batch_size, *img_shape, device=device)
        
        # Show progress bar if requested
        iterator = tqdm(
            reversed(range(0, self.timesteps)), 
            desc="Diffusion sampling",
            total=self.timesteps,
            disable=not progress,
        )
        
        for i in iterator:
            # Create batch of same timestep
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # Sample from p(x_{t-1} | x_t)
            with torch.no_grad():
                img, _ = self.p_sample(
                    model=model,
                    x_t=img,
                    t=t,
                    rna_expr=rna_expr,
                    gene_mask=gene_mask,
                    num_cells=num_cells,
                    is_multi_cell=is_multi_cell,
                )
                
        return img
    
    def ddim_sample(
        self, 
        model, 
        rna_expr, 
        img_shape, 
        timesteps_subset=None, 
        eta=0.0, 
        gene_mask=None, 
        num_cells=None, 
        is_multi_cell=False, 
        progress=True
    ):
        """
        Generate samples using DDIM (Denoising Diffusion Implicit Models) sampling.
        
        Args:
            model: The neural network model
            rna_expr: RNA expression data
            img_shape: Shape of the image to generate (C, H, W)
            timesteps_subset: Subset of timesteps to use (for faster sampling)
            eta: Controls the stochasticity (0 = deterministic, 1 = DDPM)
            gene_mask: Optional gene mask
            num_cells: Optional number of cells per patch
            is_multi_cell: Whether using multi-cell model
            progress: Whether to show progress bar
            
        Returns:
            Generated samples [B, C, H, W]
        """
        device = next(model.parameters()).device
        batch_size = rna_expr.shape[0]
        
        # Set up timesteps
        if timesteps_subset is None:
            # Default: use all timesteps
            timesteps_subset = list(range(0, self.timesteps))
        
        # Total number of steps to take
        total_steps = len(timesteps_subset)
        
        # Get subset of alphas for DDIM
        alphas = self.alphas_cumprod.to(device)
        alphas_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        
        # Start from pure noise (x_T)
        img = torch.randn(batch_size, *img_shape, device=device)
        
        # Show progress bar if requested
        iterator = tqdm(
            reversed(range(0, total_steps)), 
            desc="DDIM sampling",
            total=total_steps,
            disable=not progress,
        )
        
        for i in iterator:
            # Get current timestep
            t = timesteps_subset[i]
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get alpha and alpha_prev
            a_t = alphas[t]
            a_prev = alphas_prev[t]
            
            # Predict x_0 or noise
            if is_multi_cell:
                model_output = model(img, t_tensor, rna_expr, num_cells, gene_mask)
            else:
                model_output = model(img, t_tensor, rna_expr, gene_mask)
            
            if self.predict_noise:
                # Model predicts noise
                pred_noise = model_output
                # Calculate x_0 from the noise prediction
                pred_x_0 = (img - torch.sqrt(1 - a_t) * pred_noise) / torch.sqrt(a_t)
                if self.clip_denoised:
                    # pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
                    pred_x_0 = torch.clamp(pred_x_0, 0.0, 1.0)
            else:
                # Model directly predicts x_0
                pred_x_0 = model_output
                if self.clip_denoised:
                    # pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
                    pred_x_0 = torch.clamp(pred_x_0, 0.0, 1.0)
                # Compute implied noise
                pred_noise = (img - torch.sqrt(a_t) * pred_x_0) / torch.sqrt(1 - a_t)
            
            # DDIM deterministic or stochastic update
            sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
            pred_dir = torch.sqrt(1 - a_prev - sigma_t**2) * pred_noise
            
            # Compute x_{t-1} using the DDIM formula
            x_prev = torch.sqrt(a_prev) * pred_x_0 + pred_dir
            
            # Add noise if eta > 0 (stochastic sampling)
            if eta > 0:
                noise = torch.randn_like(img)
                x_prev = x_prev + sigma_t * noise
            
            # Update img for next iteration
            img = x_prev
        
        return img
    
    def loss_fn(self, model, x_0, t, rna_expr, gene_mask=None, num_cells=None, is_multi_cell=False, noise=None):
        """
        Compute the diffusion loss between predicted and target noise.
        
        Args:
            model: The neural network model
            x_0: Clean image samples [B, C, H, W]
            t: Timesteps [B]
            rna_expr: RNA expression data
            gene_mask: Optional gene mask
            num_cells: Optional number of cells per patch
            is_multi_cell: Whether using multi-cell model
            noise: Optional pre-generated noise
            
        Returns:
            Loss value
        """
        # Add noise to x_0 according to timestep t
        x_t, target = self.q_sample(x_0, t, noise=noise)
        
        # Get model prediction
        if is_multi_cell:
            pred = model(x_t, t, rna_expr, num_cells, gene_mask)
        else:
            pred = model(x_t, t, rna_expr, gene_mask)
        
        if self.predict_noise:
            # Simple MSE loss between predicted and target noise
            loss = torch.mean((pred - target) ** 2)
        else:
            # Model predicts x_0, so we compare with the original clean image
            loss = torch.mean((pred - x_0) ** 2)
            
        return loss
    
    def sample_path(self, x_1, t, noise=None):
        """
        Sample from the path at time t.
        Compatible API with RectifiedFlow for easy integration.
        
        Args:
            x_1: Target data sample (B, C, H, W)
            t: Time variable in [0, 1] (B,)
            noise: Optional noise to use (B, C, H, W)
            
        Returns:
            Dictionary containing:
                x_t: Sample at time t
                velocity: Target noise
        """
        # Convert continuous t in [0, 1] to discrete timesteps in [0, timesteps-1]
        # Map t=0 (pure noise) to timestep=timesteps-1, and t=1 (clean) to timestep=0
        timestep = ((1 - t) * self.timesteps).long().clamp(0, self.timesteps - 1)
        
        # Get noisy sample and target noise
        x_t, target_noise = self.q_sample(x_1, timestep, noise=noise)
        
        # For compatibility with rectified flow, we return both the noisy sample and target
        return {
            "x_t": x_t,
            "velocity": target_noise,  # In diffusion, we predict the noise
            "noise": noise if noise is not None else torch.randn_like(x_1),
            "sigma_t": self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, x_1.shape)
        }
    
    def _extract(self, a, t, x_shape):
        """
        Extract the appropriate t index for a batch of indices.
        
        Args:
            a: Tensor to extract from
            t: Indices to extract
            x_shape: Shape of the target tensor
            
        Returns:
            Tensor with appropriate coefficients
        """
        batch_size = t.shape[0]
        # Move tensor 'a' to the same device as tensor 't'
        a = a.to(t.device)
        out = a.gather(-1, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out


class DiffusionSampler:
    """
    Sampler for generating images from the diffusion model.
    """
    def __init__(self, model, diffusion):
        """
        Initialize the diffusion sampler.
        
        Args:
            model: The neural network model that predicts denoised samples
            diffusion: The diffusion process object
        """
        self.model = model
        self.diffusion = diffusion
        self.img_channels = model.img_channels
        self.img_size = model.img_size
    
    def generate_sample(self, rna_expr, num_steps=100, device="cuda", method="ddpm", gene_mask=None, num_cells=None, is_multi_cell=False):
        """
        Generate a sample by running the reverse diffusion process.
        
        Args:
            rna_expr: RNA expression data (B, gene_dim) or (B, num_cells, gene_dim)
            num_steps: Number of steps for the sampling process 
                       (if method="ddim", this sets how many steps to use out of the full diffusion steps)
            device: The device to run the computation on
            method: Sampling method ('ddpm' for standard sampling or 'ddim' for accelerated sampling)
            gene_mask: Optional gene mask
            num_cells: Optional number of cells per patch
            is_multi_cell: Whether using multi-cell model
            
        Returns:
            Generated sample
        """
        # Get shapes for image generation
        img_shape = (self.img_channels, self.img_size, self.img_size)
        
        # Choose sampling method
        if method == "ddpm":
            # Standard DDPM sampling
            x = self.diffusion.p_sample_loop(
                model=self.model,
                rna_expr=rna_expr,
                img_shape=img_shape,
                gene_mask=gene_mask,
                num_cells=num_cells,
                is_multi_cell=is_multi_cell,
                progress=True
            )
        elif method == "ddim":
            # Accelerated DDIM sampling
            # Select subset of timesteps for faster sampling
            timestep_seq = torch.linspace(0, self.diffusion.timesteps-1, num_steps).long()
            
            x = self.diffusion.ddim_sample(
                model=self.model,
                rna_expr=rna_expr,
                img_shape=img_shape,
                timesteps_subset=timestep_seq,
                eta=0.0,  # 0.0 for deterministic (faster), ~0.5 for better quality
                gene_mask=gene_mask,
                num_cells=num_cells,
                is_multi_cell=is_multi_cell,
                progress=True
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Scale to [0, 1] range for output
        # x = (x + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        x = torch.clamp(x, 0, 1)
        
        return x