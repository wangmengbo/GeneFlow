import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from rectified.rectified_flow import DOPRI5Solver


def generate_images_with_rectified_flow(
    model,
    rectified_flow, 
    gene_expr, 
    device, 
    num_steps=100,
    gene_mask=None,
    num_cells=None,
    is_multi_cell=False
):
    """
    Generate cell images from gene expression profiles using rectified flow and DOPRI5 solver
    
    Args:
        model: The RNA to H&E model (can be DDP wrapped)
        rectified_flow: The rectified flow module
        gene_expr: RNA expression tensor
        device: Computation device
        num_steps: Number of steps for the solver
        gene_mask: Optional gene mask tensor
        num_cells: Optional number of cells per patch for multi-cell model
        is_multi_cell: Whether using multi-cell model
        
    Returns:
        Generated images tensor
    """
    # Handle DDP model wrapping for inference
    actual_model = model.module if isinstance(model, DDP) else model
    
    # Create the solver with modified forward method for multi-cell model
    if is_multi_cell:
        class MultiCellModelWrapper:
            def __init__(self, model):
                self.model = model
                self.img_channels = model.img_channels
                self.img_size = model.img_size
                
            def __call__(self, x, t, rna_expr):
                # Forward gene_mask as None if not provided
                return self.model(x, t, rna_expr, num_cells, gene_mask)
                
        model_wrapper = MultiCellModelWrapper(actual_model)
        solver = DOPRI5Solver(model_wrapper, rectified_flow)
    else:
        # For single-cell model, use standard wrapper
        class SingleCellModelWrapper:
            def __init__(self, model):
                self.model = model
                self.img_channels = model.img_channels
                self.img_size = model.img_size
                
            def __call__(self, x, t, rna_expr):
                return self.model(x, t, rna_expr, gene_mask)
                
        model_wrapper = SingleCellModelWrapper(actual_model)
        solver = DOPRI5Solver(model_wrapper, rectified_flow)
    
    # Generate images
    generated_images = solver.generate_sample(
        rna_expr=gene_expr,
        num_steps=num_steps,
        device=device
    )
    
    # Denormalize images
    generated_images = torch.clamp(generated_images, 0, 1)
    
    return generated_images

