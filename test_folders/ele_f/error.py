import re
import numpy as np
import itertools
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.data import random_split, RandomSampler
import random 
import matplotlib.pyplot as plt
from utils import *
from model_v1 import feed_forward
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, mean_squared_error,  mean_absolute_error

current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'args.txt')
ele, model_struc, epochs = load_config_values(file_path)
print(ele)


error_file = 'errors_nd_plots'
path  = os.path.join(current_dir, error_file)
os.makedirs(path, exist_ok=True)
save_path1 = path + '/'

label_type = 'eda'

save_path = label_type +'/'

filepath_sobol = '/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/xyz/h2o_'+ele+'_sobol.xyz'
filepath_aimd = '/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/xyz/h2o_'+ele+'_aimd.xyz'
xyz_sobol = load_xyz(filepath_sobol)
xyz_aimd = load_xyz(filepath_aimd)
xyz = np.concatenate((xyz_sobol,xyz_aimd),axis=0)


##   SOBOL + AIMD

label = ['cls_elec','ct','mod_pauli','pol','disp']

E_eda_sobol = pd.read_csv('/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/eda_labels/h2o_'+ele+'_sobol.csv')[label]
E_eda_aimd = pd.read_csv('/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/eda_labels/h2o_'+ele+'_aimd.csv')[label]

#if label_type == 'eda_aniso':
E_ff_sobol = pd.read_csv('/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/cmm_labels/h2o_'+ele+'_sobol.csv')[label]
E_ff_aimd = pd.read_csv('/pscratch/sd/s/schandy/delta_learning/new_data/dataset_v1.1/cmm_labels/h2o_'+ele+'_aimd.csv')[label]
E = np.concatenate((E_eda_sobol-E_ff_sobol,E_eda_aimd-E_ff_aimd),axis=0)

E_eda_tot = np.concatenate((E_eda_sobol, E_eda_aimd), axis=0)
E_ff_tot =  np.concatenate((E_ff_sobol, E_ff_aimd), axis=0)
column_indices = [0, 1, 2, 3, 4]  # Adjust these indices to match your data

# Sum the specified columns row-wise
E_eda_total = E_eda_tot[:, column_indices].sum(axis=1)
E_ff_total = E_ff_tot[:, column_indices].sum(axis=1)
E_eda_ele = E_eda_tot[:,0]
E_ff_ele = E_ff_tot[:,0]
E_ele_mae = mean_absolute_error(E_eda_ele, E_ff_ele)
E_ct_mae = mean_absolute_error(E_eda_tot[:,1],E_ff_tot[:,1])
E_pol_mae = mean_absolute_error(E_eda_tot[:,3],E_ff_tot[:,3])
E_pauli_mae = mean_absolute_error(E_eda_tot[:,2],E_ff_tot[:,2])
E_disp_mae = mean_absolute_error(E_eda_tot[:,4],E_ff_tot[:,4])

E_ele_r2 = r2_score(E_eda_ele, E_ff_ele)
E_ct_r2 = r2_score(E_eda_tot[:,1],E_ff_tot[:,1])
E_pol_r2 = r2_score(E_eda_tot[:,3],E_ff_tot[:,3])
E_pauli_r2 = r2_score(E_eda_tot[:,2],E_ff_tot[:,2])
E_disp_r2 = r2_score(E_eda_tot[:,4],E_ff_tot[:,4])
#print (E)
E_tot_mae = mean_absolute_error(E_eda_total, E_ff_total)
E_tot_r2 = r2_score(E_eda_total, E_ff_total)

print (f"E_ele_mae:{E_ele_mae}, E_ct_mae:{E_ct_mae}, E_pauli_mae:{E_pauli_mae}, E_pol_mae:{E_pol_mae}, E_disp_mae:{E_disp_mae}")
print (f"E_ele_r2:{E_ele_r2}, E_ct_r2:{E_ct_r2}, E_pauli_r2:{E_pauli_r2}, E_pol_r2:{E_pol_r2}, E_disp_r2:{E_disp_r2}")
print (f"E_tot_mae: {E_tot_mae}, E_tot_r2: {E_tot_r2}")


size = len(xyz)
#print(size)
indices = list(range(size))
random.shuffle(indices)
               
train_size = int(0.8*size)
valid_size  = int(0.1*size)
test_size  = int(0.1*size)
print('train_size={}, test_size={}, valid_size={}'.format(train_size,test_size,valid_size))

train_indices = indices[:train_size]
test_indices = indices[train_size:(train_size+test_size)]
valid_indices = indices[(train_size+test_size):(train_size+2*test_size)]

np.savetxt(save_path1+'train_indices.txt', train_indices, fmt='%d')
np.savetxt(save_path1+'test_indices.txt', test_indices, fmt='%d')
np.savetxt(save_path1+'valid_indices.txt', valid_indices, fmt='%d')

xyz_train, E_train = xyz[train_indices], E[train_indices]
xyz_test, E_test = xyz[test_indices], E[test_indices]
xyz_valid, E_valid = xyz[valid_indices], E[valid_indices]

hyb_train = hyb_n(xyz_train)
hyb_test = hyb_n(xyz_test)
hyb_valid = hyb_n(xyz_valid)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = torch.tensor(hyb_train)
print(x_train.shape)
x_train = x_train.view(train_size,16)
x_train = x_train.to(torch.float32)
E_train = torch.tensor(E_train)
E_train = E_train.to(torch.float32)

x_test = torch.tensor(hyb_test)
x_test = x_test.view(test_size,16)
x_test = x_test.to(torch.float32)
E_test = torch.tensor(E_test)
E_test = E_test.to(torch.float32)

x_valid = torch.tensor(hyb_valid)
x_valid = x_valid.view(valid_size,16)
x_valid = x_valid.to(torch.float32)
E_valid = torch.tensor(E_valid)
E_valid = E_valid.to(torch.float32)

train_set = TensorDataset(x_train, E_train)
test_set = TensorDataset(x_test,E_test)
valid_set = TensorDataset(x_valid,E_valid)

train_loader = DataLoader(train_set,batch_size=512, shuffle=True)
test_loader = DataLoader(test_set ,batch_size=512,shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=512,shuffle=True)
#model.to(device)

criterion = nn.MSELoss()  
#optimizer = optim.Adam(model.parameters(), lr=0.001)  


model = torch.load(save_path+"best_model.pt")
model.to(device)
model.eval()

#valid_set
x_valid = x_valid.to(device)
E_valid = E_valid.to(device)
output = model(x_valid)
E_valid_pred = output.detach().cpu().numpy()
E_valid_cpu = E_valid.detach().cpu().numpy() 
mae_valid_elec = mean_absolute_error(E_valid_cpu[:,0], E_valid_pred[:,0])
mae_valid_ct = mean_absolute_error(E_valid_cpu[:,1], E_valid_pred[:,1])
mae_valid_pauli = mean_absolute_error(E_valid_cpu[:,2], E_valid_pred[:,2])
mae_valid_pol = mean_absolute_error(E_valid_cpu[:,3], E_valid_pred[:,3])
mae_valid_disp = mean_absolute_error(E_valid_cpu[:,4], E_valid_pred[:,4])
mae_valid_all = mae_valid_elec + mae_valid_ct + mae_valid_pauli + mae_valid_pol + mae_valid_disp 

r2_valid_elec = r2_score(E_valid_cpu[:,0], E_valid_pred[:,0])
r2_valid_ct = r2_score(E_valid_cpu[:,1], E_valid_pred[:,1])
r2_valid_pauli = r2_score(E_valid_cpu[:,2], E_valid_pred[:,2])
r2_valid_pol = r2_score(E_valid_cpu[:,3], E_valid_pred[:,3])
r2_valid_disp = r2_score(E_valid_cpu[:,4], E_valid_pred[:,4])
#r2_valid_all = r2_valid_elec + r2_valid_ct + r2_valid_pauli + r2_valid_pol + r2_valid_disp
E_valid_cpu_sum = np.sum(E_valid_cpu, axis=1)
E_valid_pred_sum = np.sum(E_valid_pred, axis=1)

# Compute the R² score for the summed vectors
r2_valid_sum = r2_score(E_valid_cpu_sum, E_valid_pred_sum)
mae_valid_sum = mean_absolute_error(E_valid_cpu_sum, E_valid_pred_sum)

#print(f"Esum: {mae_valid_all}, E_valid_sum: {E_valid_sum}")

#print(f"R² score for the summed vectors: {r2_valid_sum}")

# Write each value to a new line in the file
with open(save_path1 + 'valid.txt', 'w') as file:
    file.write(f"MAE_valid_elec: {mae_valid_elec}, R2_Score_elec: {r2_valid_elec}\n")
    file.write(f"MAE_valid_ct: {mae_valid_ct}, R2_Score_ct: {r2_valid_ct}\n")
    file.write(f"MAE_valid_pauli: {mae_valid_pauli}, R2_Score_pauli: {r2_valid_pauli}\n")
    file.write(f"MAE_valid_pol: {mae_valid_pol}, R2_Score_pol: {r2_valid_pol}\n")
    file.write(f"MAE_valid_disp: {mae_valid_disp}, R2_Score_disp: {r2_valid_disp}\n")
    file.write(f"MAE_valid_all: {mae_valid_sum}, R2_Score_all: {r2_valid_sum}\n")

# Test Set
x_test = x_test.to(device)
E_test = E_test.to(device)
output = model(x_test)
E_test_pred = output.detach().cpu().numpy()
E_test_cpu = E_test.detach().cpu().numpy()

mae_test_elec = mean_absolute_error(E_test_cpu[:,0], E_test_pred[:,0])
mae_test_ct = mean_absolute_error(E_test_cpu[:,1], E_test_pred[:,1])
mae_test_pauli = mean_absolute_error(E_test_cpu[:,2], E_test_pred[:,2])
mae_test_pol = mean_absolute_error(E_test_cpu[:,3], E_test_pred[:,3])
mae_test_disp = mean_absolute_error(E_test_cpu[:,4], E_test_pred[:,4])
mae_test_all = mae_test_elec + mae_test_ct + mae_test_pauli + mae_test_pol + mae_test_disp

r2_test_elec = r2_score(E_test_cpu[:,0], E_test_pred[:,0])
r2_test_ct = r2_score(E_test_cpu[:,1], E_test_pred[:,1])
r2_test_pauli = r2_score(E_test_cpu[:,2], E_test_pred[:,2])
r2_test_pol = r2_score(E_test_cpu[:,3], E_test_pred[:,3])
r2_test_disp = r2_score(E_test_cpu[:,4], E_test_pred[:,4])

E_test_cpu_sum = np.sum(E_test_cpu, axis=1)
E_test_pred_sum = np.sum(E_test_pred, axis=1)

# Compute the R² score for the summed vectors
r2_test_sum = r2_score(E_test_cpu_sum, E_test_pred_sum)
mae_test_sum = mean_absolute_error(E_test_cpu_sum, E_test_pred_sum)

# Write each value to a new line in the file
with open(save_path1 + 'test.txt', 'w') as file:
    file.write(f"MAE_test_elec: {mae_test_elec}, R2_Score_elec: {r2_test_elec}\n")
    file.write(f"MAE_test_ct: {mae_test_ct}, R2_Score_ct: {r2_test_ct}\n")
    file.write(f"MAE_test_pauli: {mae_test_pauli}, R2_Score_pauli: {r2_test_pauli}\n")
    file.write(f"MAE_test_pol: {mae_test_pol}, R2_Score_pol: {r2_test_pol}\n")
    file.write(f"MAE_test_disp: {mae_test_disp}, R2_Score_disp: {r2_test_disp}\n")
    file.write(f"MAE_test_all: {mae_test_sum}, R2_Score_all: {r2_test_sum}\n")


# Train Set
x_train = x_train.to(device)
E_train = E_train.to(device)
output = model(x_train)
E_train_pred = output.detach().cpu().numpy()
E_train_cpu = E_train.detach().cpu().numpy()

mae_train_elec = mean_absolute_error(E_train_cpu[:,0], E_train_pred[:,0])
mae_train_ct = mean_absolute_error(E_train_cpu[:,1], E_train_pred[:,1])
mae_train_pauli = mean_absolute_error(E_train_cpu[:,2], E_train_pred[:,2])
mae_train_pol = mean_absolute_error(E_train_cpu[:,3], E_train_pred[:,3])
mae_train_disp = mean_absolute_error(E_train_cpu[:,4], E_train_pred[:,4])
mae_train_all = mae_train_elec + mae_train_ct + mae_train_pauli + mae_train_pol + mae_train_disp

r2_train_elec = r2_score(E_train_cpu[:,0], E_train_pred[:,0])
r2_train_ct = r2_score(E_train_cpu[:,1], E_train_pred[:,1])
r2_train_pauli = r2_score(E_train_cpu[:,2], E_train_pred[:,2])
r2_train_pol = r2_score(E_train_cpu[:,3], E_train_pred[:,3])
r2_train_disp = r2_score(E_train_cpu[:,4], E_train_pred[:,4])

E_train_cpu_sum = np.sum(E_train_cpu, axis=1)
E_train_pred_sum = np.sum(E_train_pred, axis=1)

# Compute the R² score for the summed vectors
r2_train_sum = r2_score(E_train_cpu_sum, E_train_pred_sum)
mae_train_sum = mean_absolute_error(E_train_cpu_sum, E_train_pred_sum)

# Write each value to a new line in the file
with open(save_path1 + 'train.txt', 'w') as file:
    file.write(f"MAE_train_elec: {mae_train_elec}, R2_Score_elec: {r2_train_elec}\n")
    file.write(f"MAE_train_ct: {mae_train_ct}, R2_Score_ct: {r2_train_ct}\n")
    file.write(f"MAE_train_pauli: {mae_train_pauli}, R2_Score_pauli: {r2_train_pauli}\n")
    file.write(f"MAE_train_pol: {mae_train_pol}, R2_Score_pol: {r2_train_pol}\n")
    file.write(f"MAE_train_disp: {mae_train_disp}, R2_Score_disp: {r2_train_disp}\n")
    file.write(f"MAE_train_all: {mae_train_sum}, R2_Score_all: {r2_train_sum}\n")


