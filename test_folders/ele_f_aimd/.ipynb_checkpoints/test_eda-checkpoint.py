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
import os

# Load configuration values
file_path = '/pscratch/sd/s/schandy/delta_learning/new_data/ele_f/args.txt'
ele, model_struc, epochs = load_config_values(file_path)
label_type = 'eda'
save_path = label_type + '/aimd/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load AIMD data
xyz_aimd = load_xyz(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/xyz/h2o_{ele}_aimd.xyz')
labels = ['cls_elec', 'ct', 'mod_pauli', 'pol', 'disp']
E_eda_aimd = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/eda_labels/h2o_{ele}_aimd.csv')[labels]
E_ff_aimd = pd.read_csv(f'/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/cmm_labels/h2o_{ele}_aimd.csv')[labels]
E = E_eda_aimd - E_ff_aimd

# Split dataset into train, validation, and test sets
size = len(xyz_aimd)
indices = list(range(size))
random.shuffle(indices)
train_size, valid_size, test_size = int(0.8 * size), int(0.1 * size), int(0.1 * size)
train_indices, valid_indices, test_indices = indices[:train_size], indices[-valid_size:], indices[train_size:-valid_size]
xyz_train, xyz_valid, xyz_test = xyz_aimd[train_indices], xyz_aimd[valid_indices], xyz_aimd[test_indices]
E_train, E_valid, E_test = E.iloc[train_indices], E.iloc[valid_indices], E.iloc[test_indices]

# Convert XYZ to hybrid matrices
hyb_train, hyb_valid, hyb_test = hyb_n(xyz_train), hyb_n(xyz_valid), hyb_n(xyz_test)

# Prepare tensors and dataloaders
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def prepare_tensors(hyb_data, E_data, size):
    x = torch.tensor(hyb_data).view(size, 16).float().to(device)
    y = torch.tensor(E_data.values).float().to(device)
    return TensorDataset(x, y)

train_set, valid_set, test_set = prepare_tensors(hyb_train, E_train, train_size), prepare_tensors(hyb_valid, E_valid, valid_size), prepare_tensors(hyb_test, E_test, test_size)
train_loader, valid_loader, test_loader = DataLoader(train_set, batch_size=512, shuffle=True), DataLoader(valid_set, batch_size=512, shuffle=False), DataLoader(test_set, batch_size=512, shuffle=False)

# Model training
model = feed_forward(model_struc).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_mse, valid_mse, test_mse, train_mae, valid_mae, test_mae = [], [], [], [], [], []
for epoch in range(epochs):
    model.train()
    epoch_mse_loss, epoch_mae_loss = 0, 0
    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        mse_loss = criterion(outputs, y)
        mse_loss.backward()
        optimizer.step()
        epoch_mse_loss += mse_loss.item()
        epoch_mae_loss += torch.mean(torch.abs(outputs - y)).item()
    train_mse.append(epoch_mse_loss / len(train_loader))
    train_mae.append(epoch_mae_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        valid_mse_loss, valid_mae_loss = 0, 0
        for x, y in valid_loader:
            outputs = model(x)
            valid_mse_loss += criterion(outputs, y).item()
            valid_mae_loss += torch.mean(torch.abs(outputs - y)).item()
        valid_mse.append(valid_mse_loss / len(valid_loader))
        valid_mae.append(valid_mae_loss / len(valid_loader))
    
    torch.save(model.state_dict(), save_path + 'best_model.pt')

# Save results
np.savetxt(save_path + 'Valid_MSE.txt', valid_mse, fmt='%.8f')
np.savetxt(save_path + 'Valid_MAE.txt', valid_mae, fmt='%.8f')

# Scatter plots
model.load_state_dict(torch.load(save_path + 'best_model.pt'))
model.eval()
output = model(torch.tensor(hyb_valid).view(valid_size, 16).float().to(device)).cpu().detach().numpy()
for i, lbl in enumerate(labels):
    plot_scatter(E_valid[labels[i]], output[:, i], save_path, f'{ele}_{label_type}_{lbl}')
plot_box(E_valid.values - output, save_path, ele, label_type)

print("AIMD data training and evaluation complete.")

