import os, sys, argparse, logging
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="A tool for generating and managing prompts.")
    parser.add_argument("--adata", type=str, default="cell_256_aux/input/adata.h5ad", help="Path to the AnnData object.")
    parser.add_argument("--layer", type=str, default=None, help="Layer to use for the AnnData object.")
    parser.add_argument("--cell_type", type=str, default=None)
    parser.add_argument("--cell_type_label", type=str, default="cell_type")
    parser.add_argument("--min_total_counts", type=int, default=0)
    parser.add_argument("--max_total_counts", type=int, default=np.inf)
    parser.add_argument("--min_total_pct", type=float, default=0.0)
    parser.add_argument("--max_total_pct", type=float, default=1.0)
    parser.add_argument("--use_full_gene_list", action="store_true", default=False, help="Use the full gene list instead of the top 1000 highly variable genes.")
    parser.add_argument("--gene_symbols", type=str, default="gene_symbols.txt", help="Path to the gene symbol list.")
    parser.add_argument("--only_inference", action="store_true", default=False, help="Only run inference.")
    parser.add_argument('--only_importance_anlaysis', action='store_true', default=False, help='Only run gene importance analysis.')
    parser.add_argument('--analysis_timesteps', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9], help='List of timesteps (0 to 1) to use for importance analysis.')
    parser.add_argument('--analysis_batches', type=int, default=10, help='Number of validation batches to use for importance analysis.')
    parser.add_argument('--missing_gene_symbols', type=str, default=None, help='Path to a file containing missing gene symbols, one per line.')
    parser.add_argument('--concat_mask', action='store_true', default=False, help='Concatenate mask to the input of the RNA encoder.')
    return parser

def parse_adata(args=None, 
                adata=None,
                layer=None,
                cell_type=None, 
                cell_type_label=None, 
                min_total_counts=None, 
                max_total_counts=None, 
                min_total_pct=None, 
                max_total_pct=None,
                use_full_gene_list=None,
                gene_symbols=None,
                missing_gene_symbols=None,
                concat_mask=None,
                ):
    # clean arguments
    if args is not None:
        if adata is None and args.adata is not None:
            adata = args.adata
        if layer is None and args.layer is not None:
            layer = args.layer
        if cell_type is None and args.cell_type is not None:
            cell_type = args.cell_type
        if cell_type_label is None:
            cell_type_label = args.cell_type_label
        if min_total_counts is None and args.min_total_counts is not None:
            min_total_counts = args.min_total_counts
        if max_total_counts is None and args.max_total_counts is not None:
            max_total_counts = args.max_total_counts
        if min_total_pct is None and args.min_total_pct is not None:
            min_total_pct = args.min_total_pct
        if max_total_pct is None and args.max_total_pct is not None:
            max_total_pct = args.max_total_pct
        if use_full_gene_list is None:
            use_full_gene_list = args.use_full_gene_list
        if gene_symbols is None:
            gene_symbols = args.gene_symbols
        if missing_gene_symbols is None:
            missing_gene_symbols = args.missing_gene_symbols
    
    # parse adata
    if type(adata) is str:
        adata = sc.read_h5ad(adata)
    
    if layer is not None:
        adata.X = adata.layers[layer]

    if cell_type is not None:
        adata = adata[adata.obs[cell_type_label] == cell_type]

    if min_total_counts is not None and min_total_counts > 0:
        logger.info(f"Filtering cells with total counts < {min_total_counts}")
        adata = adata[adata.obs["total_counts"] >= min_total_counts]
    
    if max_total_counts is not None and max_total_counts < np.inf:
        logger.info(f"Filtering cells with total counts > {max_total_counts}")
        adata = adata[adata.obs["total_counts"] <= max_total_counts]
    
    if min_total_pct is not None and min_total_pct > 0.0:
        logger.info(f"Filtering cells with total pct < {min_total_pct * 100}%")
        threshold = np.percentile(adata.obs["total_counts"], min_total_pct * 100)
        adata = adata[adata.obs["total_counts"] >= threshold]
    
    if max_total_pct is not None and max_total_pct < 1.0:
        logger.info(f"Filtering cells with total pct > {max_total_pct * 100}%")
        threshold = np.percentile(adata.obs["total_counts"], max_total_pct * 100)
        adata = adata[adata.obs["total_counts"] <= threshold]

    if missing_gene_symbols is not None:
        if type(missing_gene_symbols) is str:
            missing_gene_symbols = pd.read_csv(missing_gene_symbols, header=None)[0].tolist()
            logger.info(f"Loaded {len(missing_gene_symbols)} missing gene symbols from {args.missing_gene_symbols}")

    ngenes = adata.n_vars
    genes = adata.var_names.tolist()
    expr = adata.to_df()
    # mask = None
    if use_full_gene_list:
        if type(gene_symbols) is str:
            gene_symbols = pd.read_csv(gene_symbols, header=None)[0].tolist()
        ngenes = len(gene_symbols)
        genes = gene_symbols
        expr = pd.DataFrame(np.zeros((adata.n_obs, ngenes)), index=adata.obs_names, columns=gene_symbols)
        expr.update(adata.to_df())
        if missing_gene_symbols is None:
            missing_gene_symbols = list(set(gene_symbols) - set(adata.var_names))
        else:
            missing_gene_symbols = list(set(missing_gene_symbols) + set(gene_symbols) - set(adata.var_names))

    # mask = pd.DataFrame(np.ones((adata.n_obs, ngenes)).astype(int), index=adata.obs_names, columns=genes)
    # mask.loc[:, missing_gene_symbols] = 0
    
    return expr, missing_gene_symbols
    

def analyze_gene_importance(
    model,
    data_loader,
    rectified_flow, # Pass RectifiedFlow object if needed for sampling x_t
    device,
    gene_names,
    output_path,
    timesteps_to_analyze=[0.1, 0.5, 0.9],
    num_batches_to_analyze=np.inf, # Limit number of batches for efficiency
):
    """
    Performs gradient-based gene importance analysis.

    Args:
        model: The trained RNAtoHnEModel.
        data_loader: DataLoader (e.g., validation loader) to get RNA samples.
        rectified_flow: RectifiedFlow instance (optional, for sampling x_t).
        device: Computation device.
        gene_names: List of gene names corresponding to rna_expr dimensions.
        output_path: Path to save the CSV results.
        timesteps_to_analyze: List of timesteps (0 to 1) to analyze.
        num_batches_to_analyze: Max number of batches to process from data_loader.
    """
    logger.info("Starting gradient-based gene importance analysis...")
    model.eval() # Ensure model is in eval mode

    # Initialize tensor to store cumulative absolute gradients for each gene
    # Use float64 for accumulation to prevent potential overflow/precision issues
    gene_gradients_sum = torch.zeros(model.rna_dim, dtype=torch.float64, device=device)
    num_samples_processed = 0
    batches_processed = 0

    timesteps_tensor = torch.tensor(timesteps_to_analyze, device=device)

    # Get expected image shape once
    try:
        # Attempt to get shape from a sample batch image if possible
        sample_batch = next(iter(data_loader))
        _, C, H, W = sample_batch['image'].shape
        logger.info(f"Inferred image shape: C={C}, H={H}, W={W}")
    except Exception:
        # Fallback to model config
        C, H, W = model.img_channels, model.img_size, model.img_size
        logger.warning(f"Could not infer image shape from data, using model config: C={C}, H={H}, W={W}")

    # Iterate through data loader batches
    pbar_batches = tqdm(data_loader, total=min(num_batches_to_analyze, len(data_loader)), desc="Analyzing Batches")
    for batch in pbar_batches:
        if batches_processed >= num_batches_to_analyze:
            break

        rna_expr_batch = batch['gene_expr'].to(device)
        current_batch_size = rna_expr_batch.shape[0]

        # Enable gradient calculation for this specific RNA input batch
        rna_expr_batch.requires_grad_(True)

        # Generate noise once per batch (or sample x_t if preferred)
        # Using same noise across timesteps for this batch for simplicity
        x_t_noise = torch.randn(current_batch_size, C, H, W, device=device)

        # Iterate through specified timesteps
        for t_val in timesteps_tensor:
            t_batch = torch.full((current_batch_size,), t_val.item(), device=device)

            # --- Gradient Calculation ---
            with torch.set_grad_enabled(True):
                # Zero gradients before calculation
                model.zero_grad()
                if rna_expr_batch.grad is not None:
                    rna_expr_batch.grad.zero_()

                # Forward pass
                v_pred = model(x_t_noise, t_batch, rna_expr_batch)

                # Scalar output: L2 norm squared of velocity, summed over batch
                scalar_output = torch.sum(v_pred**2)

                # Backward pass
                scalar_output.backward()

            # --- Accumulate Gradients ---
            if rna_expr_batch.grad is not None:
                # Sum absolute gradients across the batch dimension for this timestep
                # Move to float64 before summing potentially large numbers
                batch_gene_grads = rna_expr_batch.grad.abs().sum(dim=0).to(torch.float64)
                gene_gradients_sum += batch_gene_grads
                num_samples_processed += current_batch_size # Increment by samples in this timestep analysis
            else:
                logger.warning(f"No gradient computed for rna_expr_batch at t={t_val.item()} in batch {batches_processed}")

            # Detach noise and predicted velocity to free memory if not needed further
            # v_pred = v_pred.detach() # Uncomment if memory becomes an issue

        # Detach the input batch after processing all timesteps for it
        rna_expr_batch = rna_expr_batch.detach()
        batches_processed += 1
        pbar_batches.set_postfix({"Samples processed": num_samples_processed})

    pbar_batches.close()

    if num_samples_processed > 0:
        # Average the summed absolute gradients over all samples processed (batches * timesteps)
        avg_gene_importance = (gene_gradients_sum / num_samples_processed).cpu().numpy()

        # Create DataFrame and save results
        importance_df = pd.DataFrame({
            'gene_name': gene_names,
            'importance_score': avg_gene_importance
        })
        importance_df = importance_df.sort_values(by='importance_score', ascending=False)

        importance_df.to_csv(output_path, index=False)
        logger.info(f"Gene importance scores saved to {output_path}")
        logger.info("Top 5 important genes:")
        logger.info(importance_df.head(5))
    else:
        logger.warning("No samples were processed for gradient analysis. Importance scores not calculated.")
    
    return importance_df
