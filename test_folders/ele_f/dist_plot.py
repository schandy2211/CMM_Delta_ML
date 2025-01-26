import re
import numpy as np
import os
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from utils import *
from model_v1 import feed_forward

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_r2(y_true, y_pred):
    """
    Calculate R² using the Excel RSQ method.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation = correlation_matrix[0, 1]
    return correlation ** 2

# Debugging helpers
def debug_print(*args):
    print("[DEBUG]:", *args)

# Load configuration
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'args.txt')
ele, model_struc, epochs = load_config_values(file_path)
debug_print("Loaded configuration:", ele, model_struc, epochs)

# Create directories for saving errors and plots
error_file = 'errors_nd_plots'
path = os.path.join(current_dir, error_file)
os.makedirs(path, exist_ok=True)
save_path1 = path + '/'
plot_path = path + '/plots/'
os.makedirs(plot_path, exist_ok=True)

label_type = 'eda'
save_path = label_type + '/'

# Load XYZ data
filepath_sobol = f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/xyz/h2o_{ele}_sobol.xyz'
filepath_aimd = f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/xyz/h2o_{ele}_aimd.xyz'

xyz_sobol = load_xyz(filepath_sobol)
xyz_aimd = load_xyz(filepath_aimd)
xyz = np.concatenate((xyz_sobol, xyz_aimd), axis=0)

debug_print("Loaded XYZ data: sobol:", xyz_sobol.shape, "aimd:", xyz_aimd.shape, "combined:", xyz.shape)

# Load EDA and FF data
original_labels = ['cls_elec', 'ct', 'mod_pauli', 'pol', 'disp']
new_labels = ['Electrostatics', 'Charge Transfer', 'Pauli Repulsion', 'Polarization', 'Dispersion']

E_eda_sobol = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/eda_labels/h2o_{ele}_sobol.csv')[original_labels]
E_eda_aimd = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/eda_labels/h2o_{ele}_aimd.csv')[original_labels]
E_ff_sobol = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/cmm_labels/h2o_{ele}_sobol.csv')[original_labels]
E_ff_aimd = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/cmm_labels/h2o_{ele}_aimd.csv')[original_labels]

E_eda_sobol.columns = new_labels
E_eda_aimd.columns = new_labels
E_ff_sobol.columns = new_labels
E_ff_aimd.columns = new_labels

E_eda_tot = np.concatenate((E_eda_sobol, E_eda_aimd), axis=0)
E_ff_tot = np.concatenate((E_ff_sobol, E_ff_aimd), axis=0)

debug_print("Loaded EDA and FF data:", "E_eda_tot:", E_eda_tot.shape, "E_ff_tot:", E_ff_tot.shape)

# Compute distances
def compute_distances(xyz):
    distances = []
    for frame in xyz:
        oxygen = frame[0]
        ion = frame[3]
        dist = np.linalg.norm(oxygen[1:] - ion[1:])
        distances.append(dist)
    return np.array(distances)

distances = compute_distances(xyz)

# Split data into train, test, valid
size = len(xyz)
indices = list(range(size))
random.shuffle(indices)

train_size = int(0.8 * size)
valid_size = int(0.1 * size)
test_size = int(0.1 * size)
train_indices = indices[:train_size]
test_indices = indices[train_size:(train_size+test_size)]
valid_indices = indices[(train_size+test_size):(train_size+2*test_size)]

xyz_train, E_train = xyz[train_indices], E_eda_tot[train_indices]
xyz_test, E_test = xyz[test_indices], E_eda_tot[test_indices]
xyz_valid, E_valid = xyz[valid_indices], E_eda_tot[valid_indices]
E_ff_valid = E_ff_tot[valid_indices]

hyb_train = hyb_n(xyz_train)
hyb_test = hyb_n(xyz_test)
hyb_valid = hyb_n(xyz_valid)

x_train = torch.tensor(hyb_train).view(train_size, 16).to(torch.float32).to(device)
x_test = torch.tensor(hyb_test).view(test_size, 16).to(torch.float32).to(device)
x_valid = torch.tensor(hyb_valid).view(valid_size, 16).to(torch.float32).to(device)

# Load model
model = torch.load(save_path + "best_model.pt")
model.to(device)
model.eval()

# Predict delta corrections
E_valid_pred = model(x_valid).detach().cpu().numpy()
E_ff_delta_valid = E_ff_valid + E_valid_pred

# Distance cutoff analysis
def distance_cutoff_analysis(E_eda, E_ff, distances, cutoff):
    mask_less = distances < cutoff
    mask_more = distances >= cutoff
    results = {}
    for mask, range_label in zip([mask_less, mask_more], ['<cutoff', '>=cutoff']):
        E_eda_cutoff = E_eda[mask]
        E_ff_cutoff = E_ff[mask]
        mae = mean_absolute_error(E_eda_cutoff, E_ff_cutoff)
        mse = mean_squared_error(E_eda_cutoff, E_ff_cutoff)
        r2 = calculate_r2(E_eda_cutoff, E_ff_cutoff)
        results[range_label] = {'MAE': mae, 'MSE': mse, 'R2': r2}
    return results

cutoff = 3.0
results_all = {}

for lbl_idx, lbl in enumerate(new_labels):
    E_eda = E_eda_tot[:, lbl_idx]
    E_ff = E_ff_tot[:, lbl_idx]
    results_all[f'{lbl}_original'] = distance_cutoff_analysis(E_eda, E_ff, distances, cutoff)

results_all['Total Energy_original'] = distance_cutoff_analysis(E_eda_tot.sum(axis=1), E_ff_tot.sum(axis=1), distances, cutoff)

for lbl_idx, lbl in enumerate(new_labels):
    E_eda = E_eda_tot[valid_indices][:, lbl_idx]
    E_ff_delta = E_ff_delta_valid[:, lbl_idx]
    results_all[f'{lbl}_delta'] = distance_cutoff_analysis(E_eda, E_ff_delta, distances[valid_indices], cutoff)

results_all['Total Energy_delta'] = distance_cutoff_analysis(E_eda_tot[valid_indices].sum(axis=1), E_ff_delta_valid.sum(axis=1), distances[valid_indices], cutoff)


def save_cutoff_results(results, labels, save_path):
    """
    Save the distance cutoff analysis results to separate tables for less than and greater than the cutoff.
    """
    less_cutoff_data = []
    more_cutoff_data = []

    # Loop through labels and include both original and delta-corrected versions
    for lbl in labels + ["Total Energy"]:
        for metric in ["MAE", "MSE", "R2"]:
            # Original results
            if f"{lbl}_original" in results:
                less_cutoff_data.append({
                    "Label": f"{lbl} (Original)",
                    "Metric": metric,
                    "Value": results[f"{lbl}_original"]["<cutoff"][metric]
                })
                more_cutoff_data.append({
                    "Label": f"{lbl} (Original)",
                    "Metric": metric,
                    "Value": results[f"{lbl}_original"][">=cutoff"][metric]
                })

            # Delta-corrected results
            if f"{lbl}_delta" in results:
                less_cutoff_data.append({
                    "Label": f"{lbl} (Delta)",
                    "Metric": metric,
                    "Value": results[f"{lbl}_delta"]["<cutoff"][metric]
                })
                more_cutoff_data.append({
                    "Label": f"{lbl} (Delta)",
                    "Metric": metric,
                    "Value": results[f"{lbl}_delta"][">=cutoff"][metric]
                })

    # Convert to DataFrames and save
    less_df = pd.DataFrame(less_cutoff_data)
    more_df = pd.DataFrame(more_cutoff_data)

    less_df.to_csv(os.path.join(save_path, "less_than_cutoff_analysis.csv"), index=False)
    more_df.to_csv(os.path.join(save_path, "greater_than_cutoff_analysis.csv"), index=False)

    print(f"Saved less_than_cutoff_analysis.csv and greater_than_cutoff_analysis.csv to {save_path}")


def plot_cutoff_results(results, labels, cutoff, save_path):
    """
    Create bar plots for MAE less than and greater than the cutoff for all labels.
    """
    mae_less = []
    mae_more = []
    label_names = []

    # Loop through labels and collect MAE data
    for lbl in labels + ["Total Energy"]:
        mae_less.append(results[f"{lbl}_original"]["<cutoff"]["MAE"])
        mae_more.append(results[f"{lbl}_original"][">=cutoff"]["MAE"])
        label_names.append(f"{lbl} (Original)")

        if f"{lbl}_delta" in results:
            mae_less.append(results[f"{lbl}_delta"]["<cutoff"]["MAE"])
            mae_more.append(results[f"{lbl}_delta"][">=cutoff"]["MAE"])
            label_names.append(f"{lbl} (Delta)")

    # Create bar plot
    x = range(len(label_names))
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, mae_less, width, label=f"MAE < {cutoff} Å")
    ax.bar(x, mae_more, width, bottom=mae_less, label=f"MAE >= {cutoff} Å")

    # Add labels and title
    ax.set_xlabel("Labels", fontsize=12)
    ax.set_ylabel("MAE (kJ/mol)", fontsize=12)
    ax.set_title(f"Distance Cutoff MAE Analysis (< {cutoff} Å and >= {cutoff} Å)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"cutoff_mae_analysis_{cutoff}A.png"), dpi=300)
    plt.show()


# Save results
save_cutoff_results(results_all, new_labels, save_path1)
plot_cutoff_results(results_all, new_labels, cutoff=3.0, save_path=save_path1)

debug_print("Distance cutoff analysis completed and saved.")


# Compute metrics
def compute_metrics(E_eda, E_ff):
    mae = mean_absolute_error(E_eda, E_ff)
    mse = mean_squared_error(E_eda, E_ff)
    r2 = calculate_r2(E_eda, E_ff)
    return mae, mse, r2

# Plotting function for scatter
def plot_scatter(E_eda, E_ff, distances, labels, title, filename):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=18)

    for idx, lbl in enumerate(labels + ['Interaction Energy']):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        if lbl == 'Interaction Energy':
            E_eda_comp = E_eda.sum(axis=1)
            E_ff_comp = E_ff.sum(axis=1)
        else:
            lbl_idx = labels.index(lbl)
            E_eda_comp = E_eda[:, lbl_idx]
            E_ff_comp = E_ff[:, lbl_idx]

        mae, mse, r2 = compute_metrics(E_eda_comp, E_ff_comp)

        ax.scatter(distances, E_eda_comp, label=f"DFT {lbl}", alpha=0.5)
        ax.scatter(distances, E_ff_comp, label=f"FF {lbl}", alpha=0.5, color="orange")
        ax.set_title(lbl, fontsize=14)
        ax.set_xlabel("Distance (Å)", fontsize=12)
        ax.set_ylabel("Energy (kJ/mol)", fontsize=12)
        ax.legend()
        ax.grid()

        # Add metrics to the plot
        textstr = f"MAE = {mae:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    # Hide any unused subplots
    if len(labels) + 1 < 6:
        axes[-1, -1].axis("off")

    plt.tight_layout()
    plt.savefig(plot_path + filename, dpi=300)
    plt.show()

# Plotting function for linear regression
def plot_linear(E_eda, E_ff, labels, title, filename):
    """
    Plot linear regression with EDA vs. FF, showing MAE, MSE, and R² for actual data.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=18)

    for idx, lbl in enumerate(labels + ['Interaction Energy']):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        if lbl == 'Interaction Energy':
            E_eda_comp = E_eda.sum(axis=1).reshape(-1, 1)
            E_ff_comp = E_ff.sum(axis=1)
        else:
            lbl_idx = labels.index(lbl)
            E_eda_comp = E_eda[:, lbl_idx].reshape(-1, 1)
            E_ff_comp = E_ff[:, lbl_idx]

        # Compute the regression line
        reg = LinearRegression().fit(E_eda_comp, E_ff_comp)
        pred = reg.predict(E_eda_comp)

        # Compute metrics (MAE, MSE, R²) for actual data
        mae, mse, r2 = compute_metrics(E_eda_comp.ravel(), E_ff_comp)

        # Plot scatter points and regression line
        ax.scatter(E_eda_comp, E_ff_comp, alpha=0.5)#, label="DFT vs FF")
        ax.plot(E_eda_comp, pred, color="red", linewidth=2)#, label="Best Fit Line")
        ax.set_title(lbl, fontsize=14)
        ax.set_xlabel("DFT Energy (kJ/mol)", fontsize=12)
        ax.set_ylabel("FF Energy (kJ/mol)", fontsize=12)
        ax.legend()
        ax.grid()

        # Add metrics to the plot
        textstr = f"MAE = {mae:.4f}\nR² = {r2:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.75,0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)

    # Hide any unused subplots
    if len(labels) + 1 < 6:
        axes[-1, -1].axis("off")

    plt.tight_layout()
    plt.savefig(plot_path + filename, dpi=300)
    plt.show()
# Save results
with open(save_path1 + "distance_cutoff_analysis.txt", "w") as f:
    for lbl, results in results_all.items():
        f.write(f"Label: {lbl}\n")
        for range_type, metrics in results.items():
            f.write(f"  Range {range_type}: MAE = {metrics['MAE']:.4f}, MSE = {metrics['MSE']:.4f}, R2 = {metrics['R2']:.4f}\n")

print("Distance cutoff analysis completed, tables and bar plots saved.")

# Generate plots
debug_print("Generating scatter plots...")
plot_scatter(E_eda_tot[valid_indices], E_ff_valid, distances[valid_indices], new_labels,
             "Original Energies vs. Distances (Validation Set)", "scatter_original.png")
plot_scatter(E_eda_tot[valid_indices], E_ff_delta_valid, distances[valid_indices], new_labels,
             "Delta-Corrected Energies vs. Distances (Validation Set)", "scatter_delta.png")

debug_print("Generating linear regression plots...")
plot_linear(E_eda_tot[valid_indices], E_ff_valid, new_labels,
            "Original Energies Linear Fit (Validation Set)", "linear_original.png")
plot_linear(E_eda_tot[valid_indices], E_ff_delta_valid, new_labels,
            "Delta-Corrected Energies Linear Fit (Validation Set)", "linear_delta.png")

debug_print("All plots generated and saved.")


# Define the plotting function 
def plot_energy_vs_distance(E_valid_pred, distances, labels, title, save_path, filename):
    """
    delta vs dist
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=18)

    for idx, lbl in enumerate(labels + ["Interaction Energy"]):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        if lbl == "Interaction Energy":
            E_pred_comp = E_valid_pred.sum(axis=1)
        else:
            lbl_idx = labels.index(lbl)
            E_pred_comp = E_valid_pred[:, lbl_idx]

        ax.scatter(distances, E_pred_comp, alpha=0.5, label=f"Predicted {lbl}")
        ax.set_title(lbl, fontsize=14)
        ax.set_xlabel("Distance (Å)", fontsize=12)
        ax.set_ylabel("Energy (kcal/mol)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid()

    # Hide unused subplot
    if len(labels) + 1 < 6:
        axes[-1, -1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.show()

# Example usage
plot_path = "./errors_nd_plots/plots"  # Define the directory for saving plots
os.makedirs(plot_path, exist_ok=True)

# Assuming `E_valid_pred` is a numpy array with predicted energies for validation set
# Assuming `distances_valid` contains distances for the validation set
plot_energy_vs_distance(
    E_valid_pred, 
    distances[valid_indices], 
    labels=["Electrostatics", "Charge Transfer", "Pauli Repulsion", "Polarization", "Dispersion"],
    title="Predicted Energies vs. Distances (Validation Set)", 
    save_path=plot_path, 
    filename="energy_vs_distance.png"
)
