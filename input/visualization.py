import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd
from glob import glob
import random
import matplotlib.cm as cm

################################
# Single figures 

def create_single_heatmap(results_dir, experiment_name="*", metric_name="Profit Gain",figsize=(10, 8)):
    """
    Creates a heatmap of mean profit gains for a specific player across different alpha and beta values.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing experiment results (can be relative or absolute)
    player_num : int
        Player number to plot (1, 2, etc.)
    experiment_name : str
        Pattern to match specific experiments (default "*" matches all)
    figsize : tuple
        Figure size in inches
    """

    # Resolve the full path if a relative path is provided
    results_dir = os.path.abspath(results_dir)
    
    # First get the experiment directory
    exp_dir = os.path.join(results_dir, experiment_name)
    if not os.path.exists(exp_dir):
        raise ValueError(f"No experiment directory found: {exp_dir}")
    
    
    # Get all alpha-beta directories (including timestamps)
    pattern = "alpha_*_beta_*" #_*"  # Matches "alpha_X_beta_Y_timestamp"
    run_dirs = glob(os.path.join(exp_dir, pattern))
    
    if not run_dirs:
        raise ValueError(f"No run directories found matching pattern '{pattern}' in {exp_dir}")
    

    # Extract alpha and beta values from directory names
    alpha_values = []
    beta_values = []
    metric_values = []
    
    for run_dir in run_dirs:
        try:
            # Extract alpha and beta from directory name
            dir_name = os.path.basename(run_dir)
            # The format should be "alpha_X_beta_Y_timestamp"
            alpha_str = dir_name.split('alpha_')[1].split('_beta_')[0]
            beta_str = dir_name.split('beta_')[1].split('_')[0]
            
            alpha = float(alpha_str)
            beta = float(beta_str)
            
            # Read cycle statistics
            stats_file = os.path.join(run_dir, "cycle_statistics.csv")
            if os.path.exists(stats_file):
                df = pd.read_csv(stats_file)

                # Find all columns that match the pattern for the metric (mean_xxx_p1, mean_xxx_p2, ...)
                metric_columns = [col for col in df.columns if col.startswith(f'mean_{metric_name.lower().replace(" ", "_")}_p')]

                if metric_columns:
                    # Compute the average across all player columns
                    metric_value = df[metric_columns].mean(axis=1).iloc[0]  # Mean across players for this row

                alpha_values.append(alpha)
                beta_values.append(beta)
                metric_values.append(metric_value)
                
        except (IndexError, ValueError) as e:
            print(f"Skipping directory {dir_name} due to parsing error: {e}")
            continue
    
    if not alpha_values:
        raise ValueError("No valid data found to create heatmap")
    
    # Create and return the heatmap
    fig = _create_heatmap(alpha_values, beta_values, metric_values, metric_name, figsize)
    return fig


# Replace NaNs and zeros in the data grid
def replace_nans_and_zeros(data_grid, data_values):
    """Replaces NaN and zero values with the smallest nonzero non-NaN value in the dataset."""

    # Find the minimum nonzero, non-NaN value
    valid_values = data_values[~np.isnan(data_values) & (data_values != 0)]
    
    if valid_values.size > 0:
        min_valid_value = np.min(valid_values)  # Smallest positive nonzero value
    else:
        min_valid_value = 1e-6  # Default small value if all values are NaN or zero
    
    # Replace NaNs
    nan_mask = np.isnan(data_grid)
    if np.any(nan_mask):
        print(f"Replacing NaNs with: {min_valid_value}")
        data_grid[nan_mask] = min_valid_value  

    # Replace zeros
    zero_mask = (data_grid == 0)
    if np.any(zero_mask):
        print(f"Replacing zeros with: {min_valid_value}")
        data_grid[zero_mask] = min_valid_value

    return data_grid



def _create_heatmap(alpha_values, beta_values, data_values, metric_name, figsize):
    """
    Helper function to create heatmaps for different metrics (Profit Gain or Mean Price).

    Parameters:
    -----------
    alpha_values : list
        List of alpha values
    beta_values : list
        List of beta values
    data_values : list
        Corresponding values for the heatmap
    player_num : int
        Player number being plotted
    metric_name : str
        Title of the heatmap (e.g., "Profit Gain", "Mean Price")
    figsize : tuple
        Size of the figure
    """

    # Convert inputs to numpy arrays
    alpha_values = np.array(alpha_values)
    beta_values = np.array(beta_values)
    data_values = np.array(data_values)

    # Get unique values for the grid
    unique_alphas = np.sort(np.unique(alpha_values))
    unique_betas = np.sort(np.unique(beta_values))

    # Create grid for the heatmap
    data_grid = np.zeros((len(unique_alphas), len(unique_betas)))

    # Fill grid with values
    for alpha, beta, value in zip(alpha_values, beta_values, data_values):
        i = np.where(unique_alphas == alpha)[0][0]  # Find row index
        j = np.where(unique_betas == beta)[0][0]  # Find column index
        data_grid[i, j] = value

    data_grid = replace_nans_and_zeros(data_grid, data_values)

    if figsize is not None:
        plt.figure(figsize=figsize)

    # Set colormap based on metric
    cmap_choice = 'Reds' if "Gain" in metric_name else ('Blues' if "Profit" in metric_name else 'Greens')

    # Define colormap limits
    # vmin_value = 0.25 if "Profit Gain" in metric_name else 1.3
    # vmax_value = 0.7 if "Profit Gain" in metric_name else 1.5

    # Create the heatmap with defined bounds
    im = plt.imshow(
        data_grid, 
        aspect='auto',
        origin='lower',
        extent=[min(unique_betas), max(unique_betas), min(unique_alphas), max(unique_alphas)],
        cmap=cmap_choice,
        # vmin=vmin_value,  # Set minimum colormap value
        # vmax=vmax_value   # Set maximum colormap value
    )
    # Add color bar
    plt.colorbar(im, label=metric_name)

    # Set labels and title
    plt.xlabel(r'$\beta \times 10^5$')
    plt.ylabel(r'$\alpha$')
    plt.title(f'{metric_name}')

    return plt.gcf()

###############################


################################
# Comparison Figures

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
    colormap_options = [
        "Blues", "Greens", "Purples", "Oranges", "Reds", "Greys", "PuRd", "YlRd"
    ]

    # Seed random choice to make it consistent for a given metric
    random.seed(hash(metric_name) % 100)

    # Pick different random colormaps for the first two heatmaps
    cmap = random.choice(colormap_options)

    for i, (alpha_vals, beta_vals, metric_vals, title, ax, cmap, norm) in enumerate([
        (alpha_noref, beta_noref, metric_noref, f"{metric_name} (No Reference)", ax1, cmap, mcolors.Normalize(vmin=vmin, vmax=vmax)),
        (alpha_ref, beta_ref, metric_ref, f"{metric_name} (Reference)", ax2, cmap, mcolors.Normalize(vmin=vmin, vmax=vmax)),
        (alpha_ref, beta_ref, metric_diff, f"{metric_name} Difference (Ref - No Ref)", ax3, "coolwarm", mcolors.TwoSlopeNorm(vmin=vmin_diff - 0.01, vcenter=0, vmax=vmax_diff + 0.01))
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


###########################