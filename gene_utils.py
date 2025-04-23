# gene_utils.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def visualize_gene_importance(model, gene_names, output_dir, expr_df=None):
    """Create visualizations of gene importance after model training"""
    # Get gene attention weights
    if hasattr(model.rna_encoder, 'gene_attention'):
        importance = model.rna_encoder.get_gene_importance().cpu().detach().numpy()
    else:
        # Alternatively, use first layer weights
        importance = np.abs(model.rna_encoder.encoder[0].weight.mean(dim=0).cpu().detach().numpy())
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Gene': gene_names,
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot top 30 genes
    plt.figure(figsize=(12, 8))
    top_genes = importance_df.head(30)
    plt.barh(top_genes['Gene'], top_genes['Importance'])
    plt.xlabel('Importance Score')
    plt.ylabel('Gene')
    plt.title('Top 30 Most Important Genes')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gene_importance_bar.png")
    
    # Create heatmap if expression data is provided
    if expr_df is not None:
        # Get top 20 genes
        top_genes = importance_df.head(20)['Gene'].tolist()
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            expr_df[top_genes].T,
            cmap='viridis',
            yticklabels=top_genes,
            xticklabels=expr_df.index,
            cbar_kws={'label': 'Expression Level'}
        )
        plt.title('Expression of Top 20 Important Genes Across Cells')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gene_expression_heatmap.png")
    
    # Return the ranked list of genes by importance
    return importance_df