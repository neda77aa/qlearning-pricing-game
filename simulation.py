import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_comparative_heatmaps(results_dir, experiment_dirs, metric_name="Price", figsize=(12, 8)):
    """
    Creates a 2-row heatmap layout:
    - First row: Two heatmaps (No Reference & Reference) with a shared colorbar.
    - Second row: Difference heatmap with its own colorbar.

    Parameters:
    -----------
    results_dir : str
        Directory containing experiment results.
    experiment_dirs : dict
        Dictionary with keys 'reference' and 'noreference' pointing to their respective directories.
    metric_name : str
        The metric to visualize (e.g., "Price", "Profit", "Cycle Length").
    figsize : tuple
        Figure size for plotting.
    """

    # Extract data for both cases
    alpha_noref, beta_noref, metric_noref = extract_metric_data(experiment_dirs['noreference'], metric_name)
    alpha_ref, beta_ref, metric_ref = extract_metric_data(experiment_dirs['reference'], metric_name)

    # Convert to numpy arrays
    alpha_noref, beta_noref, metric_noref = np.array(alpha_noref), np.array(beta_noref), np.array(metric_noref)
    alpha_ref, beta_ref, metric_ref = np.array(alpha_ref), np.array(beta_ref), np.array(metric_ref)

    # Get global min and max for a shared color scale
    vmin = min(np.nanmin(metric_noref), np.nanmin(metric_ref))
    vmax = max(np.nanmax(metric_noref), np.nanmax(metric_ref))

    # Compute the difference between Reference and No Reference
    metric_diff = metric_ref - metric_noref
    vmin_diff = min(np.nanmin(metric_diff), 0)
    vmax_diff = max(np.nanmax(metric_diff), 0)

    # Create figure with two rows (first row: 2 plots, second row: 1 plot)
    fig, axes = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'height_ratios': [1, 0.8]})  
    ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0]  # Assign subplots

    # Create heatmaps
    heatmaps = []  # Store heatmap objects for colorbars

    for i, (alpha_vals, beta_vals, metric_vals, title, ax, cmap, norm) in enumerate([
        (alpha_noref, beta_noref, metric_noref, f"{metric_name} (No Reference)", ax1, "Blues", mcolors.Normalize(vmin=vmin, vmax=vmax)),
        (alpha_ref, beta_ref, metric_ref, f"{metric_name} (Reference)", ax2, "Blues", mcolors.Normalize(vmin=vmin, vmax=vmax)),
        (alpha_ref, beta_ref, metric_diff, f"{metric_name} Difference (Ref - No Ref)", ax3, "coolwarm", mcolors.TwoSlopeNorm(vmin=vmin_diff, vcenter=0, vmax=np.abs(vmin_diff)))
    ]):
        unique_alphas = np.sort(np.unique(alpha_vals))
        unique_betas = np.sort(np.unique(beta_vals))
        data_grid = np.full((len(unique_alphas), len(unique_betas)), np.nan)  # Initialize grid with NaNs

        # Fill data grid
        for a, b, value in zip(alpha_vals, beta_vals, metric_vals):
            i = np.where(unique_alphas == a)[0][0]
            j = np.where(unique_betas == b)[0][0]
            data_grid[i, j] = value

        # Replace NaNs with smallest non-NaN value
        valid_values = metric_vals[~np.isnan(metric_vals)]
        min_valid = np.nanmin(valid_values) if valid_values.size > 0 else 0
        data_grid = np.nan_to_num(data_grid, nan=min_valid)

        # Create heatmap
        im = ax.imshow(data_grid, aspect='auto', origin='lower', 
                       extent=[min(unique_betas), max(unique_betas), min(unique_alphas), max(unique_alphas)],
                       cmap=cmap, norm=norm)
        
        ax.set_xlabel(r'$\beta \times 10^5$')
        ax.set_ylabel(r'$\alpha$')
        ax.set_title(title)

        # Add contour lines for better readability
        ax.contour(data_grid, levels=6, colors="black", linewidths=0.5, extent=[min(unique_betas), max(unique_betas), min(unique_alphas), max(unique_alphas)])

        # Store heatmap reference for colorbar
        heatmaps.append(im)

        # Hide the empty second subplot in the second row
        axes[1, 1].axis("off")

    # Add a shared colorbar for the first two heatmaps (No Reference & Reference)
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])  # Positioning for shared colorbar
    cbar = fig.colorbar(heatmaps[0], cax=cbar_ax, orientation="vertical")
    cbar.set_label(metric_name)

    # Add a separate colorbar for the difference heatmap
    cbar_diff_ax = fig.add_axes([0.52, 0.1, 0.02, 0.3])  # Adjust positioning for difference colorbar
    cbar_diff = fig.colorbar(heatmaps[2], cax=cbar_diff_ax, orientation="vertical")
    cbar_diff.set_label(f"{metric_name} Difference")

    # Adjust spacing instead of using tight_layout
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    return plt.gcf()



def extract_metric_data(experiment_dir, metric_name):
    """Extracts alpha, beta, and a given metric (price, profit, or cycle length) from experiment results."""
    pattern = "alpha_*_beta_*"
    run_dirs = glob(os.path.join(experiment_dir, pattern))

    alpha_values, beta_values, metric_values = [], [], []

    for run_dir in run_dirs:
        try:
            dir_name = os.path.basename(run_dir)
            alpha = float(dir_name.split('alpha_')[1].split('_beta_')[0])
            beta = float(dir_name.split('beta_')[1].split('_')[0])
            
            stats_file = os.path.join(run_dir, "cycle_statistics.csv")
            if os.path.exists(stats_file):
                df = pd.read_csv(stats_file)

                # Extract metric value
                if metric_name == "Cycle Length":
                    metric_value = float(df["mean_cycle_length"].iloc[0])
                elif metric_name == "Surplus":
                    metric_value = float(df['mean_consumer_surplus'].iloc[0])
                else:
                    metric_columns = [col for col in df.columns if col.startswith(f'mean_{metric_name.lower().replace(" ", "_")}_p')]
                    metric_value = df[metric_columns].mean(axis=1).iloc[0] if metric_columns else np.nan

                alpha_values.append(alpha)
                beta_values.append(beta)
                metric_values.append(metric_value)

        except (IndexError, ValueError) as e:
            print(f"Skipping {dir_name} due to parsing error: {e}")
            continue

    return alpha_values, beta_values, metric_values




experiment_base_name = "reference_impact_loss_aversion/test_loss_aversion"
# Store experiment directories for later comparison
experiment_dirs = {}


for demand_type in ['noreference','reference']:
    experiment_name = experiment_base_name + "_" + demand_type
    experiment_dirs[demand_type] = os.path.join("../Results/experiments", experiment_name)
    # Create "Figures" directory

# Generate heatmaps
figures_dir = os.path.join("../Results/experiments", experiment_base_name, "Figures_comparison")
os.makedirs(figures_dir, exist_ok=True)


# Run side-by-side heatmaps for price, profit, and cycle length
fig1 = create_comparative_heatmaps("../Results/experiments", experiment_dirs, metric_name="Price")
fig2 = create_comparative_heatmaps("../Results/experiments", experiment_dirs, metric_name="Profit")
fig3 = create_comparative_heatmaps("../Results/experiments", experiment_dirs, metric_name="Surplus")
fig4 = create_comparative_heatmaps("../Results/experiments", experiment_dirs, metric_name="Cycle Length")

fig1.savefig(os.path.join(figures_dir, "price_dual_heatmap.png"))

fig2.savefig(os.path.join(figures_dir, "profit_dual_heatmap.png"))

fig3.savefig(os.path.join(figures_dir, "consumer_surplus_dual_heatmap.png"))

fig4.savefig(os.path.join(figures_dir, "cyclelength_dual_heatmap.png"))