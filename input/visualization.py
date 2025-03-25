import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import pandas as pd
from glob import glob
import random
import matplotlib.cm as cm


###############################
# visualization alpha beta

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
    # colormap_options = [
    #     "Blues", "Greens", "Purples", "Oranges", "Reds", "Greys"
    # ]

    # # Seed random choice to make it consistent for a given metric
    # random.seed(hash(metric_name) % 100)

    # # Pick different random colormaps for the first two heatmaps
    # cmap = random.choice(colormap_options)
    cmap = 'Reds' if "Gain" in metric_name else ('Blues' if "Profit" in metric_name else ( 'Purples' if "Cycle Length" in metric_name else ( 'YlOrRd' if "Surplus" in metric_name else 'Greens')))


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



def create_comparative_heatmaps_miss(results_dir, experiment_dirs, metric_name="Price", figsize=(14, 10)):
    """
    Creates a 2-row heatmap layout:
    - First row: Three heatmaps (No Reference, Reference, Misspecification) with a shared colorbar.
    - Second row: Two difference heatmaps (Ref - No Ref, Misspecification - Ref) with individual colorbars.

    Parameters:
    -----------
    results_dir : str
        Directory containing experiment results.
    experiment_dirs : dict
        Dictionary with keys 'reference', 'noreference', and 'misspecification' pointing to their respective directories.
    metric_name : str
        The metric to visualize (e.g., "Price", "Profit", "Cycle Length").
    figsize : tuple
        Figure size for plotting.
    """

    # Extract data for all three cases
    alpha_noref, beta_noref, metric_noref = extract_metric_data(experiment_dirs['noreference'], metric_name)
    alpha_ref, beta_ref, metric_ref = extract_metric_data(experiment_dirs['reference'], metric_name)
    alpha_mis, beta_mis, metric_mis = extract_metric_data(experiment_dirs['misspecification'], metric_name)

    # Convert to numpy arrays
    alpha_noref, beta_noref, metric_noref = np.array(alpha_noref), np.array(beta_noref), np.array(metric_noref)
    alpha_ref, beta_ref, metric_ref = np.array(alpha_ref), np.array(beta_ref), np.array(metric_ref)
    alpha_mis, beta_mis, metric_mis = np.array(alpha_mis), np.array(beta_mis), np.array(metric_mis)

    # Get global min and max for a shared color scale
    vmin = min(np.nanmin(metric_noref), np.nanmin(metric_ref), np.nanmin(metric_mis))
    vmax = max(np.nanmax(metric_noref), np.nanmax(metric_ref), np.nanmax(metric_mis))

    # Compute differences for comparative analysis
    metric_diff_ref_noref = metric_ref - metric_noref  # (Reference - No Reference)
    metric_diff_mis_ref = metric_mis - metric_ref      # (Misspecification - Reference)

    # Get color scale limits for difference heatmaps
    vmin_diff = min(np.nanmin(metric_diff_ref_noref), np.nanmin(metric_diff_mis_ref), 0)
    vmax_diff = max(np.nanmax(metric_diff_ref_noref), np.nanmax(metric_diff_mis_ref), 0)

    # Create figure with three heatmaps in first row, two in second row
    fig, axes = plt.subplots(2, 3, figsize=figsize, gridspec_kw={'height_ratios': [1, 0.8]})  
    ax1, ax2, ax3 = axes[0]  # First row: No Ref, Ref, Misspecification
    ax4, ax5, ax6 = axes[1]  # Second row: Difference (Ref - NoRef), Difference (Mis - Ref), Empty

    # Define colormap options
    # colormap_options = ["Blues", "Greens", "Purples", "Oranges", "Reds", "Greys"]

    # # Pick consistent colormaps based on metric name hash
    # random.seed(hash(metric_name) % 100)
    # cmap = random.choice(colormap_options)  # Random but fixed colormap
    cmap = 'Reds' if "Gain" in metric_name else ('Blues' if "Profit" in metric_name else ( 'Purples' if "Cycle Length" in metric_name else ( 'YlOrRd' if "Surplus" in metric_name else 'Greens')))


    # List of heatmap configurations
    heatmap_data = [
        (alpha_noref, beta_noref, metric_noref, f"{metric_name} (No Reference)", ax1, cmap, mcolors.Normalize(vmin=vmin, vmax=vmax)),
        (alpha_ref, beta_ref, metric_ref, f"{metric_name} (Reference)", ax2, cmap, mcolors.Normalize(vmin=vmin, vmax=vmax)),
        (alpha_mis, beta_mis, metric_mis, f"{metric_name} (Misspecification)", ax3, cmap, mcolors.Normalize(vmin=vmin, vmax=vmax)),
        (alpha_ref, beta_ref, metric_diff_ref_noref, f"{metric_name} Difference (Ref - No Ref)", ax4, "coolwarm", mcolors.TwoSlopeNorm(vmin=vmin_diff, vcenter=0, vmax=vmax_diff)),
        (alpha_mis, beta_mis, metric_diff_mis_ref, f"{metric_name} Difference (Mis - Ref)", ax5, "coolwarm", mcolors.TwoSlopeNorm(vmin=vmin_diff, vcenter=0, vmax=vmax_diff))
    ]

    # Store heatmap references for colorbars
    heatmaps = []

    for alpha_vals, beta_vals, metric_vals, title, ax, cmap, norm in heatmap_data:
        unique_alphas = np.sort(np.unique(alpha_vals))
        unique_betas = np.sort(np.unique(beta_vals))
        data_grid = np.full((len(unique_alphas), len(unique_betas)), np.nan)

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

    # Hide the empty third subplot in the second row
    ax6.axis("off")

    # Add a shared colorbar for the first three heatmaps (No Reference, Reference, Misspecification)
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])  
    cbar = fig.colorbar(heatmaps[0], cax=cbar_ax, orientation="vertical")
    cbar.set_label(metric_name)

    # Add separate colorbars for the difference heatmaps
    #cbar_diff_ax1 = fig.add_axes([0.42, 0.1, 0.02, 0.3])  
    #cbar_diff1 = fig.colorbar(heatmaps[3], cax=cbar_diff_ax1, orientation="vertical")
    #cbar_diff1.set_label(f"{metric_name} Diff (Ref - No Ref)")

    cbar_diff_ax2 = fig.add_axes([0.67, 0.1, 0.02, 0.3])  
    cbar_diff2 = fig.colorbar(heatmaps[4], cax=cbar_diff_ax2, orientation="vertical")
    cbar_diff2.set_label(f"{metric_name} Diff (Mis - Ref)")

    # Adjust spacing
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


#############################################################################################








###############################
# visualization gamma lambda

################################
# Single figures 

def create_single_heatmap_gl(results_dir, experiment_name="*", metric_name="Profit Gain", desired_experiment = 'alpha_beta' ,figsize=(10, 8)):
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
    pattern = "gamma_*_lambda_*"

    run_dirs = glob(os.path.join(exp_dir, pattern))
    
    if not run_dirs:
        raise ValueError(f"No run directories found matching pattern '{pattern}' in {exp_dir}")
    

    # Extract alpha and beta values from directory names
    gamma_values, lambda_values, metric_values = [], [], []
    
    for run_dir in run_dirs:
        try:
            # Extract alpha and beta from directory name
            dir_name = os.path.basename(run_dir)
            # The format should be "alpha_X_beta_Y_timestamp"
            gamma_str = dir_name.split('gamma_')[1].split('_lambda_')[0]
            lambda_str = dir_name.split('lambda_')[1].split('_')[0]
            
            gamma = float(gamma_str)
            lambda_ = float(lambda_str)
            
            # Read cycle statistics
            stats_file = os.path.join(run_dir, "cycle_statistics.csv")
            if os.path.exists(stats_file):
                df = pd.read_csv(stats_file)
                if metric_name == 'mean_cycle_length':
                    # Find all columns that match the pattern for the metric (mean_xxx_p1, mean_xxx_p2, ...)
                    metric_columns = [col for col in df.columns if col.startswith(f'mean_cycle_length')]
                
                else:
                    # Find all columns that match the pattern for the metric (mean_xxx_p1, mean_xxx_p2, ...)
                    metric_columns = [col for col in df.columns if col.startswith(f'mean_{metric_name.lower().replace(" ", "_")}_p')]

                if metric_columns:
                    # Compute the average across all player columns
                    metric_value = df[metric_columns].mean(axis=1).iloc[0]  # Mean across players for this row

                gamma_values.append(gamma)
                lambda_values.append(lambda_)
                metric_values.append(metric_value)
                    
        except (IndexError, ValueError) as e:
            print(f"Skipping directory {dir_name} due to parsing error: {e}")
            continue
    
    if not gamma_values:
        raise ValueError("No valid data found to create heatmap")
    
    # Create and return the heatmap
    fig = _create_heatmap_gl(gamma_values, lambda_values, metric_values, metric_name, figsize)
    return fig





def _create_heatmap_gl(gamma_values, lambda_values, data_values, metric_name, figsize):
    """
    Helper function to create heatmaps for different metrics (Profit Gain or Mean Price).

    Parameters:
    -----------
    gamma_values : list
        List of alpha values
    lambda_values : list
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
    gamma_values = np.array(gamma_values)
    lambda_values = np.array(lambda_values)
    data_values = np.array(data_values)

    # Get unique values for the grid
    unique_gammas = np.sort(np.unique(gamma_values))
    unique_lambdas = np.sort(np.unique(lambda_values))

    # Create grid for the heatmap
    data_grid = np.zeros((len(unique_gammas), len(unique_lambdas)))

    # Fill grid with values
    for gamma, lambda_, value in zip(gamma_values, lambda_values, data_values):
        i = np.where(unique_gammas == gamma)[0][0]  # Find row index
        j = np.where(unique_lambdas == lambda_)[0][0]  # Find column index
        data_grid[i, j] = value
    
    # remove last lambda 
    # data_grid = data_grid[:,0:20-1]
    data_grid = replace_nans_and_zeros(data_grid, data_values)

    if figsize is not None:
        plt.figure(figsize=figsize)

    # Set colormap based on metric
    cmap_choice = 'Reds' if "Gain" in metric_name else ('Blues' if "Profit" in metric_name else ( 'Purples' if "mean_cycle_length" in metric_name else 'Greens'))

    # Define colormap limits
    # vmin_value = 0.25 if "Profit Gain" in metric_name else 1.3
    # vmax_value = 0.7 if "Profit Gain" in metric_name else 1.5

    # Create the heatmap with defined bounds
    im = plt.imshow(
        data_grid, 
        aspect='auto',
        origin='lower',
        extent=[min(unique_lambdas), max(unique_lambdas), min(unique_gammas), max(unique_gammas)],
        cmap=cmap_choice,
        # vmin=vmin_value,  # Set minimum colormap value
        # vmax=vmax_value   # Set maximum colormap value
    )
    # Add color bar
    plt.colorbar(im, label=metric_name)

    # Set labels and title
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\gamma$')
    plt.title(f'{metric_name}')

    return plt.gcf()

###############################


################################
# Comparison Figures

def create_comparative_heatmaps_gl(results_dir, experiment_dirs, metric_name="Price", figsize=(12, 8)):
    """
    Creates a 2-row heatmap layout:
    - First row: Two heatmaps (Reference & Misspecification) with a shared colorbar.
    - Second row: Difference heatmap (Misspecification - Reference) with its own colorbar.

    Parameters:
    -----------
    results_dir : str
        Directory containing experiment results.
    experiment_dirs : dict
        Dictionary with keys 'reference' and 'misspecification' pointing to their respective directories.
    metric_name : str
        The metric to visualize (e.g., "Price", "Profit", "Cycle Length").
    figsize : tuple
        Figure size for plotting.
    """

    # Extract data for both cases
    gamma_ref, lambda_ref, metric_ref = extract_metric_data_gl(experiment_dirs['reference'], metric_name)
    gamma_mis, lambda_mis, metric_mis = extract_metric_data_gl(experiment_dirs['misspecification'], metric_name)

    # Convert to numpy arrays
    gamma_ref, lambda_ref, metric_ref = np.array(gamma_ref), np.array(lambda_ref), np.array(metric_ref)
    gamma_mis, lambda_mis, metric_mis = np.array(gamma_mis), np.array(lambda_mis), np.array(metric_mis)

    # Get global min and max for a shared color scale
    vmin = min(np.nanmin(metric_ref), np.nanmin(metric_mis))
    vmax = max(np.nanmax(metric_ref), np.nanmax(metric_mis))

    # Compute the difference between Misspecification and Reference
    metric_diff = metric_mis - metric_ref
    vmin_diff = min(np.nanmin(metric_diff), 0)
    vmax_diff = max(np.nanmax(metric_diff), 0)

    # Create figure with two rows (first row: 2 plots, second row: 1 plot)
    fig, axes = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'height_ratios': [1, 0.8]})  
    ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0]  # Assign subplots

    # Create heatmaps
    heatmaps = []  # Store heatmap objects for colorbars
    # colormap_options = ["Blues", "Greens", "Purples", "Oranges", "Reds", "Greys"]

    # # Seed random choice to make it consistent for a given metric
    # random.seed(hash(metric_name) % 100)

    # # Pick different random colormaps for the first two heatmaps
    # cmap = random.choice(colormap_options)
    # Set colormap based on metric
    cmap = 'Reds' if "Gain" in metric_name else ('Blues' if "Profit" in metric_name else ( 'Purples' if "Cycle Length" in metric_name else ( 'YlOrRd' if "Surplus" in metric_name else 'Greens')))


    for i, (gamma_vals, lambda_vals, metric_vals, title, ax, cmap, norm) in enumerate([
        (gamma_ref, lambda_ref, metric_ref, f"{metric_name} (Reference)", ax1, cmap, mcolors.Normalize(vmin=vmin, vmax=vmax)),
        (gamma_mis, lambda_mis, metric_mis, f"{metric_name} (Misspecification)", ax2, cmap, mcolors.Normalize(vmin=vmin, vmax=vmax)),
        (gamma_mis, lambda_mis, metric_diff, f"{metric_name} Difference (Miss - Ref)", ax3, "coolwarm", mcolors.TwoSlopeNorm(vmin=vmin_diff - 0.01, vcenter=0, vmax=vmax_diff + 0.01))
    ]):
        unique_gammas = np.sort(np.unique(gamma_vals))
        unique_lambdas = np.sort(np.unique(lambda_vals))
        data_grid = np.full((len(unique_gammas), len(unique_lambdas)), np.nan)  # Initialize grid with NaNs

        # Fill data grid
        for g, l, value in zip(gamma_vals, lambda_vals, metric_vals):
            i = np.where(unique_gammas == g)[0][0]
            j = np.where(unique_lambdas == l)[0][0]
            data_grid[i, j] = value

        # Replace NaNs with smallest non-NaN value
        valid_values = metric_vals[~np.isnan(metric_vals)]
        min_valid = np.nanmin(valid_values) if valid_values.size > 0 else 0
        data_grid = np.nan_to_num(data_grid, nan=min_valid)

        # Create heatmap
        im = ax.imshow(data_grid, aspect='auto', origin='lower', 
                       extent=[min(unique_lambdas), max(unique_lambdas), min(unique_gammas), max(unique_gammas)],
                       cmap=cmap, norm=norm)
        
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$\gamma$')
        ax.set_title(title)

        # Add contour lines for better readability
        ax.contour(data_grid, levels=6, colors="black", linewidths=0.5, 
                   extent=[min(unique_lambdas), max(unique_lambdas), min(unique_gammas), max(unique_gammas)])

        # Store heatmap reference for colorbar
        heatmaps.append(im)

    # Hide the empty second subplot in the second row
    axes[1, 1].axis("off")

    # Add a shared colorbar for the first two heatmaps (Reference & Misspecification)
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


def extract_metric_data_gl(experiment_dir, metric_name):
    """Extracts alpha, beta, and a given metric (price, profit, or cycle length) from experiment results."""
    pattern =  "gamma_*_lambda_*"
    run_dirs = glob(os.path.join(experiment_dir, pattern))

    gamma_values, lambda_values, metric_values = [], [], []

    for run_dir in run_dirs:
        try:
            dir_name = os.path.basename(run_dir)
            gamma = float(dir_name.split('gamma_')[1].split('_lambda_')[0])
            lambda_ = float(dir_name.split('lambda_')[1].split('_')[0])
            
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

                gamma_values.append(gamma)
                lambda_values.append(lambda_)
                metric_values.append(metric_value)

        except (IndexError, ValueError) as e:
            print(f"Skipping {dir_name} due to parsing error: {e}")
            continue

    return gamma_values, lambda_values, metric_values




###############################
# visualization lossaversion

################################
# Single figures 

def create_single_heatmap_lossaversion(results_dir, experiment_name="*", metric_name="Profit Gain", figsize=(10, 8)):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import pandas as pd
    from glob import glob

    results_dir = os.path.abspath(results_dir)
    exp_dir = os.path.join(results_dir, experiment_name)

    if not os.path.exists(exp_dir):
        raise ValueError(f"No experiment directory found: {exp_dir}")

    pattern = "lossaversion_*"
    run_dirs = glob(os.path.join(exp_dir, pattern))

    if not run_dirs:
        raise ValueError(f"No run directories found matching pattern '{pattern}' in {exp_dir}")

    lossaversion_values = []
    metric_means = []
    metric_stds = []

    coop_vals = []
    nash_vals = []
    diffs = []

    for run_dir in run_dirs:
        try:
            dir_name = os.path.basename(run_dir)
            lossaversion_str = dir_name.split('lossaversion_')[1]
            lossaversion = float(lossaversion_str)

            stats_file = os.path.join(run_dir, "cycle_statistics.csv")
            if os.path.exists(stats_file):
                df = pd.read_csv(stats_file)

                # Extract both mean and std values
                if metric_name == 'Cycle Length':
                    mean_col = ['mean_cycle_length']
                    std_col = ['std_cycle_length']
                    if mean_col and std_col:
                        mean_val = df[mean_col].mean(axis=1).iloc[0]
                        std_val = df[std_col].mean(axis=1).iloc[0]

                        lossaversion_values.append(lossaversion)
                        metric_means.append(mean_val)
                        metric_stds.append(std_val)

                if metric_name == 'FOC':
                    # Coop and Nash columns for player 1
                    coop_col = [col for col in df.columns if col.startswith('p_coop_p1')]
                    nash_col = [col for col in df.columns if col.startswith('p_nash_p1')]
                    mean_col = [col for col in df.columns if col.startswith(f'mean_price_p')]
                    std_col = [col for col in df.columns if col.startswith(f'std_price_p')]
                    if mean_col and std_col:
                        mean_val = df[mean_col].mean(axis=1).iloc[0]
                        std_val = df[std_col].mean(axis=1).iloc[0]

                        metric_means.append(mean_val)
                        metric_stds.append(std_val)

                    if coop_col and nash_col:
                        coop_val = df[coop_col].iloc[0].values[0]
                        nash_val = df[nash_col].iloc[0].values[0]

                        lossaversion_values.append(lossaversion)
                        coop_vals.append(coop_val)
                        nash_vals.append(nash_val)
                        diffs.append(coop_val - nash_val)

                else:
                    mean_col = [col for col in df.columns if col.startswith(f'mean_{metric_name.lower().replace(" ", "_")}_p')]
                    std_col = [col for col in df.columns if col.startswith(f'std_{metric_name.lower().replace(" ", "_")}_p')]
                    if mean_col and std_col:
                        mean_val = df[mean_col].mean(axis=1).iloc[0]
                        std_val = df[std_col].mean(axis=1).iloc[0]

                        lossaversion_values.append(lossaversion)
                        metric_means.append(mean_val)
                        metric_stds.append(std_val)
        except Exception as e:
            print(f"Skipping {run_dir}: {e}")
            continue

    if not lossaversion_values:
        raise ValueError("No valid data found to plot")

    # Convert and sort
    lossaversion_values = np.array(lossaversion_values)
    sort_idx = np.argsort(lossaversion_values)
    lossaversion_values = lossaversion_values[sort_idx]

    if metric_name == 'FOC':
        coop_vals = np.array(coop_vals)[sort_idx]
        nash_vals = np.array(nash_vals)[sort_idx]
        metric_means = np.array(metric_means)[sort_idx]
        metric_stds = np.array(metric_stds)[sort_idx]
        diffs = np.array(diffs)[sort_idx]

        # Plot Nash, Coop, and their difference
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(lossaversion_values, nash_vals, label='Nash Price', color='blue', marker='o')
        ax.plot(lossaversion_values, coop_vals, label='Coop Price', color='green', marker='o')
        #ax.plot(lossaversion_values, diffs, label='Difference (Coop - Nash)', color='red', linestyle='--', marker='x')
        ax.plot(lossaversion_values, metric_means, marker='o', linestyle='-', color='black', label='Price')
        # Add standard deviation shading if available
        if metric_stds is not None:
            ax.fill_between(lossaversion_values, metric_means - metric_stds, metric_means + metric_stds, 
                            color='black', alpha=0.2, label=f"Price ± std")

        ax.set_xlabel(r'Loss Aversion')
        ax.set_ylabel("Price Level")
        ax.set_title("Nash vs Coop Prices vs Loss Aversion")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        return fig

    else:
        # Normal metric
        metric_means = np.array(metric_means)[sort_idx]
        metric_stds = np.array(metric_stds)[sort_idx] if metric_stds else None
        return _create_heatmap_lossaversion(lossaversion_values, metric_means, metric_stds, metric_name, figsize)







def _create_heatmap_lossaversion(lossaversion_values, data_means, data_stds=None, metric_name="Metric", figsize=(8,6)):
    """
    Helper function to create heatmaps for different metrics (Profit Gain or Mean Price).

    Parameters:
    -----------
    gamma_values : list
        List of alpha values
    lambda_values : list
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
    lossaversion_values = np.array(lossaversion_values)
    data_means = np.array(data_means)
    if data_stds is not None:
        data_stds = np.array(data_stds)

    # Sort values based on loss aversion parameters
    sorted_indices = np.argsort(lossaversion_values)
    lossaversion_values = lossaversion_values[sorted_indices]
    data_means = data_means[sorted_indices]
    if data_stds is not None:
        data_stds = data_stds[sorted_indices]

    # Choose colors dynamically based on metric type
    if "Gain" in metric_name:
        color = 'red'
    elif "Profit" in metric_name:
        color = 'blue'
    elif "Cycle" in metric_name:
        color = 'purple'
    elif "Consumer Surplus" in metric_name:
        color = 'green'
    else:
        color = 'black'  # Default case

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(lossaversion_values, data_means, marker='o', linestyle='-', color=color, label=metric_name)

    # Add standard deviation shading if available
    if data_stds is not None:
        ax.fill_between(lossaversion_values, data_means - data_stds, data_means + data_stds, 
                        color=color, alpha=0.2, label=f"{metric_name} ± std")

    # Set labels and title
    ax.set_xlabel(r'Loss Aversion')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs Loss Aversion')

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)

    # Show legend
    ax.legend()

    return fig

###############################
