import torch
import torch.nn as nn
import logging
import math
from rectified.rectified_flow import RectifiedFlow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class WaddingtonRectifiedFlow(RectifiedFlow):
    """
    Extends RectifiedFlow with Waddington landscape energy guidance
    to make the generative process biologically plausible.
    """
    def __init__(self, sigma_min=0.002, sigma_max=80.0, energy_weight=1.0):
        """
        Initialize the Waddington-guided rectified flow model.
        
        Args:
            sigma_min: Minimum noise level (same as original RectifiedFlow)
            sigma_max: Maximum noise level (same as original RectifiedFlow)
            energy_weight: Weight of the energy term in the loss function
        """
        super().__init__(sigma_min, sigma_max)
        self.energy_weight = energy_weight
        self.waddington_energy = None  # Will be set externally
    
    def loss_fn(self, model_output, target_velocity, gene_expr=None, x_t=None, t=None):
        """
        Compute the loss between predicted and target velocities with Waddington energy guidance.
        
        Args:
            model_output: Predicted velocity from the model
            target_velocity: Target velocity from the path
            gene_expr: Gene expression data for energy calculation
            x_t: Current state for energy calculation
            t: Time variable
            
        Returns:
            Tuple of (combined_loss, flow_loss, energy_term)
        """
        # Base rectified flow loss (same as original)
        flow_loss = torch.mean((model_output - target_velocity) ** 2)
        
        # Add Waddington energy guidance if available
        energy_term = torch.tensor(0.0, device=flow_loss.device)
        if self.waddington_energy is not None and gene_expr is not None and x_t is not None:
            # Calculate energy at current state
            energy = self.waddington_energy(gene_expr, x_t, t)
            energy_term = torch.mean(energy)
            
            # Combined loss with energy guidance
            combined_loss = flow_loss + self.energy_weight * energy_term
        else:
            combined_loss = flow_loss
        
        return combined_loss, flow_loss, energy_term
    
    def sample_path_with_guidance(self, x_1, t, gene_expr, noise=None):
        """
        Sample from the path at time t using non-linear interpolation with energy guidance.
        
        Args:
            x_1: Target data sample (B, C, H, W)
            t: Time variable in [0, 1] (B,)
            gene_expr: Gene expression data for energy guidance
            noise: Optional noise to use (B, C, H, W)
            
        Returns:
            Dictionary containing samples and velocities
        """
        # Get base path sample (same as original RectifiedFlow)
        path_sample = self.sample_path(x_1, t, noise)
        
        # Modify path with energy guidance if energy model is available
        if self.waddington_energy is not None:
            try:
                x_t = path_sample["x_t"]
                
                # Calculate energy gradient for guidance
                x_t_requires_grad = x_t.detach().clone().requires_grad_(True)
                
                # Make sure t is also a tensor that can be used in the computation
                if isinstance(t, torch.Tensor):
                    t_for_energy = t.detach().clone()
                else:
                    t_for_energy = torch.tensor(t, device=x_t.device)
                
                # Calculate energy with the gradient-tracking tensor
                energy = self.waddington_energy(gene_expr, x_t_requires_grad, t_for_energy)
                
                # Check if energy actually depends on x_t_requires_grad
                if energy.sum().requires_grad:
                    energy_sum = torch.sum(energy)
                    
                    # Calculate gradient of energy with respect to x_t
                    grad_energy = torch.autograd.grad(
                        energy_sum, 
                        x_t_requires_grad, 
                        create_graph=False, 
                        allow_unused=True,
                        retain_graph=False
                    )[0]
                    
                    if grad_energy is not None:
                        # Modify velocity to follow energy gradient
                        t_expanded = t.view(-1, *([1] * (len(x_1.shape) - 1)))
                        guidance_strength = (1 - t_expanded) * self.energy_weight
                        modified_velocity = path_sample["velocity"] - guidance_strength * grad_energy
                        
                        # Update the path sample with modified velocity
                        path_sample["velocity"] = modified_velocity
                        path_sample["energy"] = energy.detach()
                        
                        # Fix the norm calculation - use Frobenius norm instead of trying multi-dim
                        # This flattens everything except the batch dimension
                        path_sample["energy_gradient_norm"] = torch.norm(
                            grad_energy.view(grad_energy.shape[0], -1), 
                            dim=1  # Only keep batch dimension
                        ).detach()
                    else:
                        # No usable gradient, log warning
                        print("Warning: Energy gradient is None, using original velocity")
                else:
                    # Energy doesn't depend on x_t_requires_grad, log warning
                    print("Warning: Energy doesn't depend on image tensor, using original velocity")
                    
            except Exception as e:
                # Fallback to original velocity on any error
                print(f"Error in energy guidance: {str(e)}, using original velocity")
        
        return path_sample