import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import argparse
import glob

from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter

# Set up matplotlib to use LaTeX for rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{relsize}",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
})

# Define results directory
RESULTS_DIR = "results"

def list_available_files():
    """List all JSON files with yield data in the results directory."""
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' not found. Run run_simulation.py first.")
        return
        
    files = glob.glob(os.path.join(RESULTS_DIR, "Runs_*_Length_*_*.json"))
    if not files:
        print(f"No yield data files found in '{RESULTS_DIR}'. Run run_simulation.py first.")
        return
    
    print(f"Available yield data files in '{RESULTS_DIR}':")
    for i, file in enumerate(files, 1):
        print(f"{i}. {os.path.basename(file)}")
    return files

def generate_yields_plot(filename, output_filename=None, selected_n=None):
    """
    Generate a boxplot from data saved in a JSON file.
    
    Args:
        filename: Name of the file with the data
        output_filename: Name of the file to save the plot
        selected_n: List of n values to include in the plot (if None, include all)
    """
    # Check if filename contains path, if not, prepend results directory
    if not os.path.dirname(filename):
        filepath = os.path.join(RESULTS_DIR, filename)
    else:
        filepath = filename
    
    # Check if data file exists
    if not os.path.exists(filepath):
        print(f"Error: Data file '{filepath}' not found.")
        print("Run run_simulation.py first to generate the data.")
        return
        
    # Load data
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    n_values = data['n_values']
    y_expected = data['y_expected']
    y_signal = data['y_signal']
    y_decoy = data['y_decoy']    
    
    # Filter data by selected n values if provided
    if selected_n is not None:
        # Convert selected_n to list of integers
        selected_indices = []
        filtered_n_values = []
        
        for n in selected_n:
            if n in n_values:
                idx = n_values.index(n)
                selected_indices.append(idx)
                filtered_n_values.append(n)
        
        # Filter data
        n_values = filtered_n_values
        y_expected = [y_expected[i] for i in selected_indices]
        y_signal = [y_signal[i] for i in selected_indices]
        y_decoy = [y_decoy[i] for i in selected_indices]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Set background color
    ax.set_facecolor('#f8f8f8')
    
    # Prepare data for boxplot
    boxplot_data = []
    labels = []
    
    for i, n in enumerate(n_values):
        boxplot_data.append(y_expected[i])
        boxplot_data.append(y_signal[i])
        boxplot_data.append(y_decoy[i])
        
        labels.append(r"$Y_{" + str(n) + r"}$")
        labels.append(r"$Y_{" + str(n) + r"}^{\mu}$")
        labels.append(r"$Y_{" + str(n) + r"}^{\nu}$")
    
    # If no data to plot
    if not boxplot_data:
        print("No data to plot. Check if the selected n values exist in the data file.")
        return
    
    # Create boxplot
    box = ax.boxplot(
        boxplot_data, 
        patch_artist=True,
        widths=0.4,
        medianprops={'color': 'darkred', 'linewidth': 2.5},
        boxprops={'facecolor': 'lightblue', 'alpha': 0.8, 'edgecolor': 'navy', 'linewidth': 1},
        whiskerprops={'color': 'navy', 'linewidth': 1.2},
        capprops={'color': 'navy', 'linewidth': 1.2},
        flierprops={'marker': '+', 'markeredgecolor': 'navy', 'markersize': 8}
    )

    # Calculate medians, standard deviations and Q3 with NumPy
    medians = [np.median(data) for data in boxplot_data]
    stdevs = [np.std(data) for data in boxplot_data]
    q4s = [np.percentile(data, 100) for data in boxplot_data]

    # Get limits to calculate dynamic offset
    y_min, y_max = ax.get_ylim()
    offset_value = (y_max - y_min) * 0.02  # 2% of the range

    # Add annotation always above Q4
    for i, (median, std, q4) in enumerate(zip(medians, stdevs, q4s)):
        y_pos = q4 + offset_value
        text = r'$\tilde{{x}} = {:.4f}$'.format(median)
        if std > 1e-8:
            text += '\n' + r'$\sigma = {:.4f}$'.format(std)
        ax.text(
            i+1, 
            y_pos, 
            text,
            ha='center', 
            va='bottom', 
            color='black'
        )

    ax.margins(y=0.2)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(r"$\text{Yield}$")
    
    # Add grid with specified parameters
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    fig.tight_layout()
    
    # Make sure results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Generate output filename if not provided
    if output_filename is None:
        n_str = "_".join(str(n) for n in n_values) if selected_n else "all"
        base_filename = os.path.basename(filename).replace('.json', '')
        output_filename = f"plot_{base_filename}_n{n_str}.svg"  # Changed extension to .svg
    
    # Prepare full path for output file
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    # Save as SVG format
    fig.savefig(output_path, format='svg')
    plt.show()
    
    print(f"Plot saved as '{output_path}'")
    
    # Print the average yields for each n
    print("\nAverage Yields for each n:")
    for i, n in enumerate(n_values):
        print(f"n={n}:")
        print(f"  Expected yield: {np.mean(y_expected[i]):.6f}")
        print(f"  Signal yield:   {np.mean(y_signal[i]):.6f}")
        print(f"  Decoy yield:    {np.mean(y_decoy[i]):.6f}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate yield plots from simulation data')
    parser.add_argument('--file', type=str, help='Input data file (in results directory)')
    parser.add_argument('--n', type=int, nargs='+', help='Specific n values to plot')
    parser.add_argument('--list', action='store_true', help='List available data files')
    parser.add_argument('--output', type=str, help='Output image filename')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_files()
    elif args.file:
        generate_yields_plot(filename=args.file, output_filename=args.output, selected_n=args.n)
    else:
        # If no specific file, check for any available files
        files = list_available_files()
        if files:
            print("\nUse --file FILENAME to generate a plot for a specific file.")
            print("Use --n N1 N2 ... to select specific n values (e.g., --n 1 3 5).")
        else:
            print("Run run_simulation.py first to generate simulation data.")