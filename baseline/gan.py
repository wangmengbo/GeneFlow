import math
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class Discriminator(nn.Module):
    """
    Discriminator network for the GAN.
    Classifies images as real or generated.
    """
    def __init__(
        self,
        img_channels=3,
        img_size=64,
        ndf=64,  # Number of discriminator filters
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False
    ):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        
        # No normalization for the first layer
        model = [
            nn.Conv2d(img_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Add downsampling layers
        current_size = img_size // 2
        current_channels = ndf
        for i in range(1, n_layers):
            next_channels = current_channels * 2
            model += [
                nn.Conv2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(next_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            current_channels = next_channels
            current_size = current_size // 2
        
        # Add final layers
        model += [
            nn.Conv2d(current_channels, current_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(current_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final convolutional layer to produce one-channel output
            nn.Conv2d(current_channels * 2, 1, kernel_size=4, stride=1, padding=0, bias=False),
        ]
        
        # Add sigmoid for binary classification if requested
        if use_sigmoid:
            model += [nn.Sigmoid()]
            
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        """
        Forward pass through the discriminator.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Classification score
        """
        return self.model(x)

class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator network for the GAN.
    Classifies image-RNA pairs as real or generated.
    """
    def __init__(
        self,
        img_channels=3,
        img_size=64,
        rna_dim=1000,
        rna_embed_dim=128,
        ndf=64,  # Number of discriminator filters
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False
    ):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.rna_dim = rna_dim
        
        # RNA embedding
        self.rna_embedding = nn.Sequential(
            nn.Linear(rna_dim, rna_embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(rna_embed_dim, rna_embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Image processing layers
        self.image_layers = nn.Sequential(
            # First layer without normalization
            nn.Conv2d(img_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Additional downsampling layers
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate the size of the flattened feature map
        # After 4 stride-2 convolutions, spatial dimensions are reduced by factor of 16
        feature_size = (img_size // 16) ** 2 * ndf * 8
        
        # Combined processing (image features + RNA embedding)
        self.combined_layers = nn.Sequential(
            nn.Linear(feature_size + rna_embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )
        
        # Add sigmoid for binary classification if requested
        if use_sigmoid:
            self.combined_layers.add_module("sigmoid", nn.Sigmoid())
    
    def forward(self, x, rna_expr):
        """
        Forward pass through the conditional discriminator.
        
        Args:
            x: Input image tensor [B, C, H, W]
            rna_expr: RNA expression tensor [B, rna_dim]
            
        Returns:
            Classification score
        """
        # Process image
        img_features = self.image_layers(x)
        img_features = img_features.view(x.size(0), -1)  # Flatten
        
        # Process RNA expression
        rna_features = self.rna_embedding(rna_expr)
        
        # Combine features
        combined = torch.cat([img_features, rna_features], dim=1)
        
        # Final classification
        return self.combined_layers(combined)

class GANGenerator(nn.Module):
    """
    Generator for Cell GAN.
    Adapts existing UNet to work with noise vectors instead of time steps.
    """
    def __init__(
        self,
        rna_to_hne_model,  # Existing model to adapt
        z_dim=100
    ):
        super().__init__()
        self.z_dim = z_dim
        self.base_model = rna_to_hne_model
        self.img_channels = rna_to_hne_model.img_channels
        self.img_size = rna_to_hne_model.img_size
        self.rna_dim = rna_to_hne_model.rna_dim
        
        # Replace time embedding with noise embedding
        if hasattr(self.base_model, 'unet'):
            # For multi-cell model
            self.base_model.unet.time_embed = nn.Sequential(
                nn.Linear(z_dim, self.base_model.unet.model_channels * 4),
                nn.SiLU(),
                nn.Linear(self.base_model.unet.model_channels * 4, self.base_model.unet.model_channels * 4),
            )
        else:
            # For single-cell model
            self.base_model.time_embed = nn.Sequential(
                nn.Linear(z_dim, self.base_model.model_channels * 4),
                nn.SiLU(),
                nn.Linear(self.base_model.model_channels * 4, self.base_model.model_channels * 4),
            )
    
    def forward(self, z, rna_expr, num_cells=None, gene_mask=None):
        """
        Forward pass for the GAN generator.
        
        Args:
            z: Noise vector [B, z_dim]
            rna_expr: RNA expression tensor [B, rna_dim] or [B, C, rna_dim] for multi-cell
            num_cells: Optional number of cells per patch for multi-cell model
            gene_mask: Optional gene mask tensor
            
        Returns:
            Generated images [B, C, H, W]
        """
        batch_size = z.shape[0]
        
        # Create initial image tensor filled with zeros
        x = torch.zeros(
            batch_size, 
            self.img_channels, 
            self.img_size, 
            self.img_size, 
            device=z.device
        )
        
        # Use the existing model's forward method but pass noise vector instead of timesteps
        if hasattr(self.base_model, 'unet'):
            # For multi-cell model
            rna_embedding = self.base_model.rna_encoder(rna_expr, mask=gene_mask, num_cells=num_cells)
            output = self.base_model.unet(x, z, extra={"rna_embedding": rna_embedding})
        else:
            # For single-cell model
            if gene_mask is not None:
                rna_embedding = self.base_model.rna_encoder(rna_expr, gene_mask)
            else:
                rna_embedding = self.base_model.rna_encoder(rna_expr)
            output = self.base_model.unet(x, z, extra={"rna_embedding": rna_embedding})
        
        # Scale to [0, 1] range for images
        output = torch.sigmoid(output)
        
        return output

class CellGAN:
    """
    GAN implementation for generating cell images from RNA expression data.
    """
    def __init__(
        self,
        generator,
        is_multi_cell=False,
        adv_weight=1.0,
        l1_weight=10.0,
        gp_weight=10.0,
        use_wgan=True,
        z_dim=100
    ):
        """
        Initialize the CellGAN model.
        
        Args:
            generator: The generator model (RNAtoHnEModel or MultiCellRNAtoHnEModel)
            is_multi_cell: Flag indicating if the model is for multi-cell or single-cell
            adv_weight: Weight for the adversarial loss
            l1_weight: Weight for the L1 regression loss (optional)
            gp_weight: Weight for gradient penalty (if using WGAN-GP)
            use_wgan: Whether to use Wasserstein GAN with gradient penalty
            z_dim: Dimension of the noise vector
        """
        self.is_multi_cell = is_multi_cell
        self.adv_weight = adv_weight
        self.l1_weight = l1_weight
        self.gp_weight = gp_weight
        self.use_wgan = use_wgan
        self.z_dim = z_dim
        
        # Wrap the generator with the GAN adapter
        self.generator = GANGenerator(generator, z_dim=z_dim)
        
        # Create discriminator based on generator properties
        img_channels = generator.img_channels
        img_size = generator.img_size
        rna_dim = generator.rna_dim
        
        # Create conditional discriminator
        self.discriminator = ConditionalDiscriminator(
            img_channels=img_channels,
            img_size=img_size,
            rna_dim=rna_dim,
            use_sigmoid=not use_wgan  # No sigmoid for WGAN
        )
        
        logger.info(f"Initialized {'Multi-Cell' if is_multi_cell else 'Single-Cell'} CellGAN")
        logger.info(f"Using {'WGAN-GP' if use_wgan else 'Standard GAN'} approach")
    
    def generator_loss(self, fake_pred, fake_img, real_img):
        """
        Compute the generator loss.
        
        Args:
            fake_pred: Discriminator output for fake images
            fake_img: Generated images
            real_img: Real images
            
        Returns:
            Dictionary of loss components
        """
        # Adversarial loss
        if self.use_wgan:
            # WGAN loss: -E[D(G(z))]
            adv_loss = -fake_pred.mean()
        else:
            # Standard GAN loss with binary cross entropy
            targets = torch.ones_like(fake_pred)
            adv_loss = F.binary_cross_entropy_with_logits(fake_pred, targets)
        
        # L1 loss for image reconstruction (optional)
        l1_loss = torch.mean(torch.abs(fake_img - real_img)) if self.l1_weight > 0 else 0
        
        # Total generator loss
        total_loss = self.adv_weight * adv_loss + self.l1_weight * l1_loss
        
        return {
            "gen_total": total_loss,
            "gen_adv": adv_loss,
            "gen_l1": l1_loss
        }
    
    def discriminator_loss(self, real_pred, fake_pred, real_img, fake_img, rna_expr):
        """
        Compute the discriminator loss.
        
        Args:
            real_pred: Discriminator output for real images
            fake_pred: Discriminator output for fake images
            real_img: Real images
            fake_img: Generated images
            rna_expr: RNA expression data
            
        Returns:
            Dictionary of loss components
        """
        if self.use_wgan:
            # WGAN loss: E[D(x)] - E[D(G(z))]
            wgan_loss = fake_pred.mean() - real_pred.mean()
            
            # Gradient penalty for WGAN-GP
            if self.gp_weight > 0:
                gp = self.gradient_penalty(real_img, fake_img, rna_expr)
                disc_loss = wgan_loss + self.gp_weight * gp
            else:
                gp = torch.tensor(0.0, device=real_pred.device)
                disc_loss = wgan_loss
            
            return {
                "disc_total": disc_loss,
                "disc_wgan": wgan_loss,
                "disc_gp": gp
            }
        else:
            # Standard GAN with binary cross entropy
            real_targets = torch.ones_like(real_pred)
            fake_targets = torch.zeros_like(fake_pred)
            
            real_loss = F.binary_cross_entropy_with_logits(real_pred, real_targets)
            fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_targets)
            
            disc_loss = real_loss + fake_loss
            
            return {
                "disc_total": disc_loss,
                "disc_real": real_loss,
                "disc_fake": fake_loss
            }
    
    def gradient_penalty(self, real_img, fake_img, rna_expr):
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            real_img: Real images
            fake_img: Fake images
            rna_expr: RNA expression data
            
        Returns:
            Gradient penalty value
        """
        batch_size = real_img.size(0)
        
        # Create random interpolation between real and fake
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_img.device)
        interpolated = alpha * real_img + (1 - alpha) * fake_img
        interpolated.requires_grad_(True)
        
        # Calculate discriminator output for interpolated images
        disc_interpolates = self.discriminator(interpolated, rna_expr)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Calculate gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def sample_noise(self, batch_size, device):
        """
        Sample noise for the generator.
        
        Args:
            batch_size: Number of noise samples
            device: Computation device
            
        Returns:
            Noise tensor
        """
        # Create random noise
        z = torch.randn(batch_size, self.z_dim, device=device)
        return z
    
    def generate_images(self, rna_expr, gene_mask=None, num_cells=None, z=None):
        """
        Generate images from RNA expression data.
        
        Args:
            rna_expr: RNA expression tensor
            gene_mask: Optional gene mask tensor
            num_cells: Optional number of cells per patch for multi-cell model
            z: Optional noise tensor
            
        Returns:
            Generated images tensor
        """
        batch_size = rna_expr.shape[0]
        device = rna_expr.device
        
        # Sample random noise if not provided
        if z is None:
            z = self.sample_noise(batch_size, device)
        
        # Generate images based on model type
        with torch.no_grad():
            if self.is_multi_cell:
                generated_images = self.generator(z, rna_expr, num_cells, gene_mask)
            else:
                generated_images = self.generator(z, rna_expr, gene_mask)
                
        # Ensure valid image range
        generated_images = torch.clamp(generated_images, 0, 1)
        
        return generated_images