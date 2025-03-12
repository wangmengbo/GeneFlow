import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_enhanced_evaluation_plots(
    output_dir,
    all_ssim_scores,
    all_psnr_scores,
    valid_fids,
    valid_uni2h_fids,
    all_biological_results,
    mean_ssim,
    mean_psnr,
    mean_batch_fid,
    mean_uni2h_fid
):
    """Create comprehensive evaluation plots"""
    logger.info("Creating evaluation plots...")
    
    plt.figure(figsize=(20, 10))
    
    # Row 1: Traditional metrics
    plt.subplot(2, 4, 1)
    plt.hist(all_ssim_scores, bins=50, alpha=0.7, color='blue')
    plt.axvline(mean_ssim, color='red', linestyle='--', label=f'Mean: {mean_ssim:.4f}')
    plt.xlabel('SSIM')
    plt.ylabel('Frequency')
    plt.title('SSIM Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 4, 2)
    plt.hist(all_psnr_scores, bins=50, alpha=0.7, color='green')
    plt.axvline(mean_psnr, color='red', linestyle='--', label=f'Mean: {mean_psnr:.4f}')
    plt.xlabel('PSNR')
    plt.ylabel('Frequency')
    plt.title('PSNR Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 4, 3)
    if valid_fids:
        plt.hist(valid_fids, bins=50, alpha=0.7, color='orange')
        plt.axvline(mean_batch_fid, color='red', linestyle='--', label=f'Mean: {mean_batch_fid:.4f}')
    plt.xlabel('Inception FID')
    plt.ylabel('Frequency')
    plt.title('Inception FID Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 4, 4)
    if valid_uni2h_fids:
        plt.hist(valid_uni2h_fids, bins=50, alpha=0.7, color='purple')
        plt.axvline(mean_uni2h_fid, color='red', linestyle='--', label=f'Mean: {mean_uni2h_fid:.4f}')
    plt.xlabel('UNI2-H FID')
    plt.ylabel('Frequency')
    plt.title('UNI2-H FID Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Row 2: Biological validation metrics
    if all_biological_results:
        cell_type_accuracies = [r['cell_type_accuracy'] for r in all_biological_results 
                              if 'cell_type_accuracy' in r and not np.isnan(r['cell_type_accuracy'])]
        plt.subplot(2, 4, 5)
        if cell_type_accuracies:
            plt.hist(cell_type_accuracies, bins=30, alpha=0.7, color='cyan')
            plt.axvline(np.mean(cell_type_accuracies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(cell_type_accuracies):.4f}')
        plt.xlabel('Cell Type Accuracy')
        plt.ylabel('Frequency')
        plt.title('Cell Type Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)

        uni2h_similarities = [r['uni2h_embedding_similarity'] for r in all_biological_results 
                            if 'uni2h_embedding_similarity' in r and not np.isnan(r['uni2h_embedding_similarity'])]
        plt.subplot(2, 4, 6)
        if uni2h_similarities:
            plt.hist(uni2h_similarities, bins=30, alpha=0.7, color='magenta')
            plt.axvline(np.mean(uni2h_similarities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(uni2h_similarities):.4f}')
        plt.xlabel('UNI2-H Embedding Similarity')
        plt.ylabel('Frequency')
        plt.title('UNI2-H Feature Similarity')
        plt.legend()
        plt.grid(True, alpha=0.3)

        nuclear_areas = [r['nuclear_area_similarity'] for r in all_biological_results 
                        if 'nuclear_area_similarity' in r and not np.isnan(r['nuclear_area_similarity'])]
        plt.subplot(2, 4, 7)
        if nuclear_areas:
            plt.hist(nuclear_areas, bins=30, alpha=0.7, color='brown')
            plt.axvline(np.mean(nuclear_areas), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(nuclear_areas):.4f}')
        plt.xlabel('Nuclear Area Similarity')
        plt.ylabel('Frequency')
        plt.title('Nuclear Morphometry')
        plt.legend()
        plt.grid(True, alpha=0.3)

        bio_plausibilities = [r['overall_biological_plausibility'] for r in all_biological_results 
                            if 'overall_biological_plausibility' in r and not np.isnan(r['overall_biological_plausibility'])]
        plt.subplot(2, 4, 8)
        if bio_plausibilities:
            plt.hist(bio_plausibilities, bins=30, alpha=0.7, color='gold')
            plt.axvline(np.mean(bio_plausibilities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(bio_plausibilities):.4f}')
        plt.xlabel('Overall Biological Plausibility')
        plt.ylabel('Frequency')
        plt.title('Biological Validation Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_metrics_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"evaluation plots saved to {plot_path}")
    return plot_path


def create_rna_correlation_plots(output_dir, all_rna_prediction_metrics, gene_names=None):
    """Create RNA prediction correlation plots"""
    if not all_rna_prediction_metrics:
        logger.info("No RNA prediction metrics available for plotting")
        return None
    
    logger.info("Creating RNA correlation plots...")
    
    plt.figure(figsize=(15, 5))
    
    # Sample correlations
    sample_corrs = [m.get('rna_sample_correlation_mean', 0) for m in all_rna_prediction_metrics 
                   if 'rna_sample_correlation_mean' in m and not np.isnan(m['rna_sample_correlation_mean'])]
    
    plt.subplot(1, 3, 1)
    if sample_corrs:
        plt.hist(sample_corrs, bins=30, alpha=0.7, color='skyblue')
        plt.axvline(np.mean(sample_corrs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sample_corrs):.4f}')
    plt.xlabel('Sample-wise RNA Correlation')
    plt.ylabel('Frequency')
    plt.title('RNA Sample Correlation Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gene correlations
    gene_corrs = [m.get('rna_gene_correlation_mean', 0) for m in all_rna_prediction_metrics 
                 if 'rna_gene_correlation_mean' in m and not np.isnan(m['rna_gene_correlation_mean'])]
    
    plt.subplot(1, 3, 2)
    if gene_corrs:
        plt.hist(gene_corrs, bins=30, alpha=0.7, color='lightgreen')
        plt.axvline(np.mean(gene_corrs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(gene_corrs):.4f}')
    plt.xlabel('Gene-wise RNA Correlation')
    plt.ylabel('Frequency')
    plt.title('RNA Gene Correlation Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MSE values
    mse_values = [m.get('rna_mse_mean', 0) for m in all_rna_prediction_metrics 
                 if 'rna_mse_mean' in m and not np.isnan(m['rna_mse_mean'])]
    
    plt.subplot(1, 3, 3)
    if mse_values:
        plt.hist(mse_values, bins=30, alpha=0.7, color='salmon')
        plt.axvline(np.mean(mse_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(mse_values):.4f}')
    plt.xlabel('RNA MSE')
    plt.ylabel('Frequency')
    plt.title('RNA MSE Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rna_prediction_correlation_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"RNA correlation plots saved to {plot_path}")
    return plot_path


def create_summary_comparison_plot(output_dir, results_summary):
    """Create a summary comparison plot of all key metrics"""
    logger.info("Creating summary comparison plot...")
    
    # Extract key metrics for comparison
    metrics = {
        'SSIM': results_summary.get('mean_ssim', 0),
        'PSNR': results_summary.get('mean_psnr', 0) / 40,  # Normalize to 0-1 range
        'Inception FID': 1 / (1 + results_summary.get('overall_fid', float('inf'))),  # Convert to similarity
        'UNI2-H FID': 1 / (1 + results_summary.get('overall_uni2h_fid', float('inf'))),  # Convert to similarity
        'Biological Score': results_summary.get('mean_overall_biological_plausibility', 0),
        'RNA Correlation': results_summary.get('mean_rna_gene_correlation_mean', 0)
    }
    
    # Remove metrics with invalid values
    valid_metrics = {k: v for k, v in metrics.items() if not np.isnan(v) and np.isfinite(v)}
    
    if not valid_metrics:
        logger.warning("No valid metrics for summary plot")
        return None
    
    plt.figure(figsize=(12, 8))
    
    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(valid_metrics), endpoint=False).tolist()
    values = list(valid_metrics.values())
    labels = list(valid_metrics.keys())
    
    # Close the plot
    angles += angles[:1]
    values += values[:1]
    
    plt.subplot(111, projection='polar')
    plt.plot(angles, values, 'o-', linewidth=2, label='Model Performance')
    plt.fill(angles, values, alpha=0.25)
    plt.xticks(angles[:-1], labels)
    plt.ylim(0, 1)
    plt.title('Overall Model Performance Summary', size=16, y=1.1)
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, 'model_performance_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Summary comparison plot saved to {plot_path}")
    return plot_path


def save_all_evaluation_plots(
    output_dir,
    all_ssim_scores,
    all_psnr_scores,
    valid_fids,
    valid_uni2h_fids,
    all_biological_results,
    all_rna_prediction_metrics,
    results_summary,
    mean_ssim,
    mean_psnr,
    mean_batch_fid,
    mean_uni2h_fid,
    gene_names=None
):
    """Create and save all evaluation plots"""
    plot_paths = {}
    
    # evaluation plots
    plot_paths['enhanced_metrics'] = create_enhanced_evaluation_plots(
        output_dir, all_ssim_scores, all_psnr_scores, valid_fids, valid_uni2h_fids,
        all_biological_results, mean_ssim, mean_psnr, mean_batch_fid, mean_uni2h_fid
    )
    
    # RNA correlation plots
    if all_rna_prediction_metrics:
        plot_paths['rna_correlations'] = create_rna_correlation_plots(
            output_dir, all_rna_prediction_metrics, gene_names
        )
    
    # Summary comparison plot
    plot_paths['summary_comparison'] = create_summary_comparison_plot(
        output_dir, results_summary
    )
    
    logger.info(f"All evaluation plots saved to {output_dir}")
    return plot_paths