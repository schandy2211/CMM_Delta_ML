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

file_path = '/pscratch/sd/s/schandy/delta_learning/new_data/ele_cl/args.txt'  # Replace with the path to text file
ele, model_struc, epochs = load_config_values(file_path)
print(ele)

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


#if label_type  == 'eda_dipole':
 #   E_dipole_sobol = pd.read_csv('/pscratch/sd/s/schandy/delta_learning/dataset_v1.0/ff_dipole_labels/h2o_'+ele+'_sobol_ff_dipole.csv')[label]
  #  E_dipole_aimd = pd.read_csv('/pscratch/sd/s/schandy/delta_learning/dataset_v1.0/ff_dipole_labels/h2o_'+ele+'_aimd_ff_dipole.csv')[label]
   # E = np.concatenate((E_eda_sobol-E_dipole_sobol,E_eda_aimd-E_dipole_aimd),axis=0)
    
#if label_type  == 'eda':
 #   E = np.concatenate((E_eda_sobol,E_eda_aimd),axis=0)
    

size = len(xyz)
indices = list(range(size))
random.shuffle(indices)
               

train_size = int(0.5*size)
valid_size = int(0.01*size)
test_size = int(0.01*size)
print('train_size={}, test_size={}, valid_size={}'.format(train_size,test_size,valid_size))

train_indices = indices[:train_size]
test_indices = indices[train_size:(train_size+test_size)]
valid_indices = indices[(train_size+test_size):(train_size+2*test_size)]

np.savetxt(save_path+'train_indices.txt', train_indices, fmt='%d')
np.savetxt(save_path+'test_indices.txt', test_indices, fmt='%d')
np.savetxt(save_path+'valid_indices.txt', test_indices, fmt='%d')



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



model = feed_forward(model_struc)





train_set = TensorDataset(x_train, E_train)
test_set = TensorDataset(x_test,E_test)


train_loader = DataLoader(train_set,batch_size=512, shuffle=True)
test_loader = DataLoader(test_set ,batch_size=512,shuffle=True)
model.to(device)

criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  




best_loss = float('inf')

Train_mseloss = []
Test_mseloss = []
Train_maeloss = []
Test_maeloss = []
Test_relative_e = []
for epoch in range(epochs):
    
    best_test_mseloss = float('inf')
    #model.train()
    epoch_mseloss = 0
    epoch_maeloss = 0
    
    for i, (x,y) in enumerate(train_loader):
        x,y = x.to(device), y.to(device)
        outputs = model(x)
        
        mseloss = criterion(outputs, y)
            
        optimizer.zero_grad()
        mseloss.backward()
        optimizer.step()
        
        epoch_mseloss+= mseloss
        
        maeloss =  torch.sum(abs(outputs-y),dim=0)/len(y)
        epoch_maeloss+= maeloss

    epoch_mseloss = epoch_mseloss/len(train_loader)
    epoch_maeloss = epoch_maeloss/len(train_loader)
    
    print(epoch, epoch_mseloss)
    
    epoch_mseloss = epoch_mseloss.to('cpu')
    epoch_maeloss = epoch_maeloss.to('cpu')
    Train_mseloss.append(epoch_mseloss)
    Train_maeloss.append(epoch_maeloss)
    
    
    with torch.no_grad():
        test_epoch_mseloss = 0
        test_epoch_maeloss = 0
        test_epoch_relative_e = 0
        for i, (x,y) in enumerate(test_loader):
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            test_mseloss = criterion(outputs ,y)
            test_maeloss = torch.sum(abs(outputs-y),dim=0)/len(y)
            #test_relative_error = abs(torch.sum((outputs-y)/y,dim=0))/len(y)
            
            test_epoch_mseloss += test_mseloss
            test_epoch_maeloss += test_maeloss
            #test_epoch_relative_e += test_relative_error
            
            
    test_epoch_mseloss = test_epoch_mseloss/len(test_loader)
    test_epoch_maeloss = test_epoch_maeloss/len(test_loader)
    
    #test_epoch_relative_e = test_epoch_relative_e/len(test_loader)
    
    test_epoch_mseloss = test_epoch_mseloss.to('cpu')
    test_epoch_maeloss = test_epoch_maeloss.to('cpu')
    #test_epoch_relative_e = test_epoch_relative_e.to('cpu')
    
    Test_mseloss.append(test_epoch_mseloss)
    Test_maeloss.append(test_epoch_maeloss)
    #Test_relative_e.append(test_epoch_relative_e)
    
    print(test_epoch_mseloss)
    
    if test_epoch_mseloss < best_test_mseloss:
        best_test_mseloss = test_mseloss
        torch.save(model, save_path+'best_model.pt')
    
    
Train_mseloss = np.array([x.detach().numpy() for x in Train_mseloss])
Test_mseloss = np.array([x.detach().numpy() for x in Test_mseloss])
Train_maeloss = np.array([x.detach().numpy() for x in Train_maeloss])
Test_maeloss = np.array([x.detach().numpy() for x in Test_maeloss])
#Test_relative_e = np.array([x.detach().numpy() for x in Test_relative_e])



np.savetxt(save_path + 'Train_mseloss_1.txt', Train_mseloss, fmt='%.8f', delimiter=' ')
np.savetxt(save_path + 'Train_maeloss_1.txt', Train_maeloss, fmt='%.8f', delimiter=' ')
np.savetxt(save_path + 'Test_mseloss_1.txt', Test_mseloss, fmt='%.8f', delimiter=' ')
np.savetxt(save_path + 'Test_maeloss_1.txt', Test_maeloss, fmt='%.8f', delimiter=' ')
#np.savetxt('eda_dipole_hyb/Test_relative_e_1.txt', Test_relative_e, fmt='%.8f', delimiter=' ')



model = torch.load(save_path+"best_model.pt")
model.to(device)
model.eval()
x_valid = x_valid.to(device)
output = model(x_valid)
output_numpy = output.detach().cpu().numpy()
np.savetxt(save_path+'valid_output.txt', output_numpy, fmt='%.6f', delimiter=' ')


E_deltaml = np.loadtxt(save_path+'valid_output.txt')
E_eda_ff = E_valid.numpy()
plot_scatter(E_eda_ff[:,0],E_deltaml[:,0],save_path,ele+'_'+label_type+'_cls_elec')
plot_scatter(E_eda_ff[:,1],E_deltaml[:,1],save_path,ele+'_'+label_type+'_ct')
plot_scatter(E_eda_ff[:,2],E_deltaml[:,2],save_path,ele+'_'+label_type+'_mod_pauli')
plot_scatter(E_eda_ff[:,3],E_deltaml[:,3],save_path,ele+'_'+label_type+'_pol')
plot_scatter(E_eda_ff[:,4],E_deltaml[:,4],save_path,ele+'_'+label_type+'_disp')
plot_box((E_eda_ff-E_deltaml),save_path)
