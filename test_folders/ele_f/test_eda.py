import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from model import feed_forward

# Load configuration values
file_path = '/pscratch/sd/s/schandy/delta_learning/new_data/ele_f/args.txt'
ele, model_struc, epochs = load_config_values(file_path)
label_type = 'eda'
save_path = label_type + '/'

# Load XYZ data
xyz_sobol = load_xyz(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/xyz/h2o_{ele}_sobol.xyz')
xyz_aimd = load_xyz(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/xyz/h2o_{ele}_aimd.xyz')
xyz = np.concatenate((xyz_sobol, xyz_aimd), axis=0)

# Load EDA and force field data
labels = ['cls_elec', 'ct', 'mod_pauli', 'pol', 'disp']
E_eda_sobol = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/eda_labels/h2o_{ele}_sobol.csv')[labels]
E_eda_aimd = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/eda_labels/h2o_{ele}_aimd.csv')[labels]
E_ff_sobol = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/cmm_labels/h2o_{ele}_sobol.csv')[labels]
E_ff_aimd = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/cmm_labels/h2o_{ele}_aimd.csv')[labels]
E = np.concatenate((E_eda_sobol - E_ff_sobol, E_eda_aimd - E_ff_aimd), axis=0)

# Split dataset into train, validation, and test sets
size = len(xyz)
indices = list(range(size))
random.shuffle(indices)
train_size, valid_size, test_size = int(0.8 * size), int(0.1 * size), int(0.1 * size)
train_indices, valid_indices, test_indices = indices[:train_size], indices[-valid_size:], indices[train_size:-valid_size]
xyz_train, xyz_valid, xyz_test = xyz[train_indices], xyz[valid_indices], xyz[test_indices]
E_train, E_valid, E_test = E[train_indices], E[valid_indices], E[test_indices]

# Convert XYZ to hybrid matrices
hyb_train, hyb_valid, hyb_test = hyb_n(xyz_train), hyb_n(xyz_valid), hyb_n(xyz_test)

# Prepare tensors and dataloaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def prepare_tensors(hyb_data, E_data, size):
    x = torch.tensor(hyb_data).view(size, 16).float().to(device)
    y = torch.tensor(E_data).float().to(device)
    return TensorDataset(x, y)

train_set, valid_set, test_set = prepare_tensors(hyb_train, E_train, train_size), prepare_tensors(hyb_valid, E_valid, valid_size), prepare_tensors(hyb_test, E_test, test_size)
train_loader, valid_loader, test_loader = DataLoader(train_set, batch_size=512, shuffle=True), DataLoader(valid_set, batch_size=512, shuffle=False), DataLoader(test_set, batch_size=512, shuffle=False)

# Model training
model = feed_forward(model_struc).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae = [], [], [], [], [], []
train_mae_label, valid_mae_label, test_mae_label = [], [], []
for epoch in range(epochs):
    model.train()
    epoch_mse_loss, epoch_mae_loss = 0, 0
    epoch_mae_label = torch.zeros(len(labels)).to(device)
    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        mse_loss = criterion(outputs, y)
        mse_loss.backward()
        optimizer.step()
        epoch_mse_loss += mse_loss.item()
        epoch_mae_loss += torch.mean(torch.abs(outputs - y)).item()
        epoch_mae_label += torch.mean(torch.abs(outputs - y), dim=0)
    train_mse.append(epoch_mse_loss / len(train_loader))
    train_mae.append(epoch_mae_loss / len(train_loader))
    train_mae_label.append(epoch_mae_label.cpu().numpy() / len(train_loader))

    model.eval()
    with torch.no_grad():
        valid_mse_loss, valid_mae_loss = 0, 0
        valid_mae_label = torch.zeros(len(labels)).to(device)
        for x, y in valid_loader:
            outputs = model(x)
            valid_mse_loss += criterion(outputs, y).item()
            valid_mae_loss += torch.mean(torch.abs(outputs - y)).item()
            valid_mae_label += torch.mean(torch.abs(outputs - y), dim=0)
        valid_mse.append(valid_mse_loss / len(valid_loader))
        valid_mae.append(valid_mae_loss / len(valid_loader))
        valid_mae_label.append(valid_mae_label.cpu().numpy() / len(valid_loader))
    
    torch.save(model.state_dict(), save_path + 'best_model.pt')

# Save results
np.savetxt(save_path + 'Valid_MSE.txt', valid_mse, fmt='%.8f')
np.savetxt(save_path + 'Valid_MAE.txt', valid_mae, fmt='%.8f')
np.savetxt(save_path + 'Valid_MAE_Labels.txt', valid_mae_label, fmt='%.8f')

