import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import random
import matplotlib.pyplot as plt
from utils import *
from model import feed_forward

# Load configuration values
file_path = '/pscratch/sd/s/schandy/delta_learning/new_data/ele_f/args.txt'
ele, model_struc, epochs = load_config_values(file_path)
label_type = 'eda'
save_path = label_type + '/aimd_sobol/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load XYZ data
xyz_sobol = load_xyz(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/xyz/h2o_{ele}_sobol.xyz')
xyz_aimd = load_xyz(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/xyz/h2o_{ele}_aimd.xyz')
xyz_train = np.concatenate((xyz_sobol, xyz_aimd), axis=0)
xyz_valid = xyz_aimd

# Load EDA and force field data
labels = ['cls_elec', 'ct', 'mod_pauli', 'pol', 'disp']
E_eda_sobol = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/eda_labels/h2o_{ele}_sobol.csv')[labels]
E_eda_aimd = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/eda_labels/h2o_{ele}_aimd.csv')[labels]
E_ff_sobol = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/cmm_labels/h2o_{ele}_sobol.csv')[labels]
E_ff_aimd = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/cmm_labels/h2o_{ele}_aimd.csv')[labels]
E_train = np.concatenate((E_eda_sobol - E_ff_sobol, E_eda_aimd - E_ff_aimd), axis=0)
E_valid = E_eda_aimd - E_ff_aimd

# Convert XYZ to hybrid matrices
hyb_train = hyb_n(xyz_train)
hyb_valid = hyb_n(xyz_valid)

# Prepare tensors and dataloaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def prepare_tensors(hyb_data, E_data):
    x = torch.tensor(hyb_data).view(len(hyb_data), 16).float().to(device)
    y = torch.tensor(E_data).float().to(device)
    return TensorDataset(x, y)

train_set, valid_set = prepare_tensors(hyb_train, E_train), prepare_tensors(hyb_valid, E_valid)
train_loader, valid_loader = DataLoader(train_set, batch_size=512, shuffle=True), DataLoader(valid_set, batch_size=512, shuffle=False)

# Model training
model = feed_forward(model_struc).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_mae_label, valid_mae_label = [], []
for epoch in range(epochs):
    model.train()
    epoch_mae_label = torch.zeros(len(labels)).to(device)
    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        mae_loss = torch.mean(torch.abs(outputs - y), dim=0)
        mae_loss.mean().backward()
        optimizer.step()
        epoch_mae_label += mae_loss
    train_mae_label.append(epoch_mae_label.detach().cpu().numpy() / len(train_loader))

    model.eval()
    with torch.no_grad():
        valid_mae_label_epoch = torch.zeros(len(labels)).to(device)
        for x, y in valid_loader:
            outputs = model(x)
            valid_mae_label_epoch += torch.mean(torch.abs(outputs - y), dim=0)
        valid_mae_label.append(valid_mae_label_epoch.cpu().numpy() / len(valid_loader))
    
    torch.save(model.state_dict(), save_path + 'best_model.pt')

# Save results
np.savetxt(save_path + 'Valid_MAE_Labels.txt', valid_mae_label, fmt='%.8f')

# Evaluate performance after delta learning
model.load_state_dict(torch.load(save_path + 'best_model.pt'))
model.eval()
output = model(torch.tensor(hyb_valid).view(len(hyb_valid), 16).float().to(device)).cpu().detach().numpy()
E_pred = E_ff_aimd + output
mae_after = np.mean(np.abs(E_pred - E_eda_aimd), axis=0)
mae_before = np.mean(np.abs(E_ff_aimd - E_eda_aimd), axis=0)

# Save MAE before and after delta learning in a single file
mae_results = np.column_stack((mae_before, mae_after))
np.savetxt(save_path + 'MAE_Comparison.txt', mae_results, header='Before After', fmt='%.8f', delimiter=' ', comments='')

print("AIMD and Sobol data training complete, AIMD validation complete.")

