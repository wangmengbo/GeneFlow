import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class WaddingtonEnergy(nn.Module):
    """
    Implements Waddington landscape energy guidance for cellular differentiation paths.
    Models the energy landscape that guides cells toward stable attractors during development.
    """
    def __init__(self, gene_dim, hidden_dim=128, energy_scale=1.0, num_attractor_states=5, img_size=64, img_channels=3):
        super().__init__()
        self.gene_dim = gene_dim
        self.energy_scale = energy_scale
        self.num_attractor_states = num_attractor_states
        self.img_size = img_size
        self.img_channels = img_channels
        
        # Network to estimate energy potential from gene expression (keep this as is)
        self.energy_net = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learn attractor states (stable cell states in gene expression space)
        self.attractor_states = nn.Parameter(torch.randn(num_attractor_states, gene_dim))
        
        # Attention mechanism for attractors
        self.attractor_attention = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_attractor_states)
        )
        
        # Network to learn barrier heights between attractors
        self.barrier_net = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Instead of calculating exact dimensions, we'll use a simpler pooling+CNN approach
        # that can handle variable input sizes
        self.image_net = nn.Sequential(
            nn.AvgPool2d(kernel_size=8),  # More aggressive pooling to reduce size
            nn.Conv2d(img_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(kernel_size=2),  # Further reduce size
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling to fixed size
            nn.Flatten(),
            nn.Linear(8, 1)  # Output single energy value
        )
    
    def attractor_energy(self, gene_expr):
        """Calculate energy from distance to attractor states"""
        batch_size = gene_expr.shape[0]
        
        # Calculate attention weights for each attractor
        # Shape: [batch_size, num_attractor_states]
        attention_logits = self.attractor_attention(gene_expr)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # Calculate squared distance to each attractor
        # Reshape for broadcasting: [batch_size, 1, gene_dim] and [1, num_attractors, gene_dim]
        gene_expr_expanded = gene_expr.unsqueeze(1)
        attractors_expanded = self.attractor_states.unsqueeze(0)
        
        # Calculate squared distances: [batch_size, num_attractors]
        squared_distances = torch.sum((gene_expr_expanded - attractors_expanded)**2, dim=2)
        
        # Get barrier heights for this expression state
        barriers = self.barrier_net(gene_expr)
        
        # Apply barrier height to increase energy away from attractors
        # Shape: [batch_size, num_attractors]
        modulated_distances = squared_distances * barriers
        
        # Weight distances by attention and sum
        # Shape: [batch_size]
        weighted_distances = torch.sum(attention_weights * modulated_distances, dim=1)
        
        return weighted_distances
    
    def forward(self, gene_expr, images=None, t=None):
        """
        Calculate Waddington energy with proper image dependency.
        
        Args:
            gene_expr: Gene expression data [B, gene_dim] or [B, C, gene_dim] for multi-cell
            images: Generated images [B, C, H, W] that should influence the energy
            t: Optional time parameter for time-dependent energy
            
        Returns:
            Energy value (lower is better)
        """
        # Handle multi-cell case by averaging gene expression per patch first
        if len(gene_expr.shape) == 3:
            gene_expr = gene_expr.mean(dim=1)
        
        # Calculate direct energy estimate from gene expression
        direct_energy = self.energy_net(gene_expr).squeeze()
        
        # Calculate attractor-based energy
        attractor_energy = self.attractor_energy(gene_expr)
        
        # Combined energy from gene expression
        gene_energy = direct_energy + attractor_energy
        
        # Image energy component - IMPORTANT for gradient guidance
        image_energy = 0.0
        if images is not None:
            try:
                # Handle channel dimension mismatches
                if images.shape[1] != self.img_channels:
                    # logger.info(f"Adapting image channels from {images.shape[1]} to {self.img_channels}")
                    
                    if images.shape[1] > self.img_channels:
                        # Use first n channels
                        images_processed = images[:, :self.img_channels]
                    else:
                        # Pad with zeros to match expected channels
                        pad_size = self.img_channels - images.shape[1]
                        pad = torch.zeros(images.shape[0], pad_size, images.shape[2], 
                                        images.shape[3], device=images.device)
                        images_processed = torch.cat([images, pad], dim=1)
                else:
                    images_processed = images
                    
                # Get image energy contribution
                image_energy = self.image_net(images_processed).squeeze()
                
                # Scale image energy to be smaller than gene energy initially
                image_energy = 0.01 * image_energy
                
            except Exception as e:
                print(f"Error processing image in energy calculation: {str(e)}")
                # Fallback to zero image energy
                image_energy = torch.zeros_like(gene_energy)
        
        # Combine gene expression energy and image energy
        total_energy = gene_energy + image_energy
        
        # Apply time-dependent scaling if t is provided
        if t is not None and isinstance(t, torch.Tensor):
            # Make time factor tensor-based for proper gradient flow
            t_factor = torch.sin(t * torch.pi) * 0.5 + 0.5  # Peaks at t=0.5
            t_factor = t_factor.view(-1) if len(t_factor.shape) > 0 else t_factor
            if len(total_energy.shape) > 0:
                total_energy = total_energy * t_factor
        
        # Scale energy and return
        return self.energy_scale * total_energy
    
    def get_attractor_states(self):
        """Return the learned attractor states for visualization"""
        return self.attractor_states.detach().cpu()
    
    def analyze_landscape(self, gene_expr_dataset, device):
        """
        Analyze the Waddington landscape for a dataset of gene expressions
        
        Args:
            gene_expr_dataset: Dataset with gene expression data
            device: Computation device
            
        Returns:
            Dictionary with analysis results
        """
        self.eval()
        with torch.no_grad():
            all_energies = []
            all_attractor_weights = []
            
            # Process in batches
            batch_size = 64
            num_samples = len(gene_expr_dataset)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = slice(i, min(i + batch_size, num_samples))
                batch_data = [gene_expr_dataset[j] for j in range(i, min(i + batch_size, num_samples))]
                
                # Extract gene expressions (might need to adapt based on dataset structure)
                if isinstance(batch_data[0], dict):
                    gene_exprs = torch.stack([item['gene_expr'] for item in batch_data]).to(device)
                else:
                    gene_exprs = torch.stack(batch_data).to(device)
                
                # Handle multi-cell case
                if len(gene_exprs.shape) == 3:
                    gene_exprs = gene_exprs.mean(dim=1)
                
                # Calculate energies
                energies = self.forward(gene_exprs)
                all_energies.append(energies.cpu())
                
                # Calculate attractor weights
                attention_logits = self.attractor_attention(gene_exprs)
                attractor_weights = F.softmax(attention_logits, dim=1)
                all_attractor_weights.append(attractor_weights.cpu())
            
            # Combine results
            all_energies = torch.cat(all_energies, dim=0).numpy()
            all_attractor_weights = torch.cat(all_attractor_weights, dim=0).numpy()
            
            # Analyze results
            mean_energy = all_energies.mean()
            std_energy = all_energies.std()
            min_energy = all_energies.min()
            max_energy = all_energies.max()
            
            # Find dominant attractors
            dominant_attractors = all_attractor_weights.argmax(axis=1)
            attractor_counts = {i: (dominant_attractors == i).sum() for i in range(self.num_attractor_states)}
            
            logger.info(f"Waddington landscape analysis:")
            logger.info(f"Mean energy: {mean_energy:.4f}, Std: {std_energy:.4f}")
            logger.info(f"Energy range: [{min_energy:.4f}, {max_energy:.4f}]")
            logger.info(f"Attractor distribution: {attractor_counts}")
            
            return {
                "mean_energy": mean_energy,
                "std_energy": std_energy,
                "min_energy": min_energy,
                "max_energy": max_energy,
                "attractor_counts": attractor_counts,
                "all_energies": all_energies,
                "all_attractor_weights": all_attractor_weights,
                "attractor_states": self.get_attractor_states().numpy()
            }