import re
import numpy as np
import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt




def dist(origin_data):
    xyz = origin_data[:,1:]
    dist = ((xyz[:, np.newaxis, :]-xyz)**2).sum(axis=2)**(1/2)
    return dist

def radii_vdw(origin_data,radii):
    r_vdw_list = [np.array(radii[radii['atom']== atom ]['r_vdw']) for atom in origin_data[:,0]]
    r_vdw = np.array(r_vdw_list).reshape((4,))
    dist_vdw = r_vdw[:,np.newaxis]+r_vdw
    return dist_vdw

def Matrix_hyb_n(dist, R_e, R_vdw):
    M_hyb = np.array(dist-R_vdw-R_e,dtype=np.float16)

    A = np.exp(-M_hyb[3,:3])
    M_hyb[3,:3] = A
    B = np.exp(-M_hyb[:3,3])
    M_hyb[:3,3] = B 
    np.fill_diagonal(dist, 2)
    M_hyb = M_hyb/dist
    np.fill_diagonal(M_hyb, 0)

    return M_hyb


def hyb_n(xyz):
    # Get the current working directory
    direct = os.getcwd()

    # Construct the file path for 'vdw_radii.txt' located in the current directory
    file_paath = os.path.join(direct, 'vdw_radii.txt')

    # Read the CSV file from the constructed path
    radii = pd.read_csv(file_paath)

    #radii = pd.read_csv('/pscratch/sd/s/schandy/delta_learning/new_data/ele_ca/vdw_radii.txt')
    dist_all = np.array([dist(x) for x in xyz],dtype=np.float16)
    R_vdw = radii_vdw(xyz[0],radii)
    R_vdw = np.array(R_vdw)
    R_vdw[0:3,0:3]=0

    R_e = np.array([[0,0.959273,0.959273,0],[0.959273,0,1.522472,0],[0.959273,1.522472,0,0],[0,0,0,0]])

    M_hyb_all = np.array([Matrix_hyb(d, R_e, R_vdw) for d in dist_all],dtype=np.float64)
    return M_hyb_all

def Matrix_hyb(dist, R_e, R_vdw):
    M_hyb = np.array(dist-R_e,dtype=np.float16)
    

    A = np.exp(-M_hyb[3,:3]/R_vdw[3,:3])
    M_hyb[3,:3] = A
    B = np.exp(-M_hyb[:3,3]/R_vdw[:3,3])
    M_hyb[:3,3] = B 
    np.fill_diagonal(dist, 1)
    M_hyb = M_hyb/dist
    np.fill_diagonal(M_hyb, 0)

    return M_hyb


def load_xyz(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    data = file_content
    
    data_blocks = data.strip().split("4\n\n")
    data_blocks = data_blocks[1:]
    
    all_coordinates = []
    
    # Parse and store coordinates for each block
    for block in data_blocks:
        lines = block.strip().split('\n')
        #print(lines)
        coordinates = []
        for line in lines:
            elements = line.split()
            #print(elements)
            element_symbol = elements[0]
            x, y, z = map(float, elements[1:])
            coordinates.append([element_symbol, x, y, z])
        all_coordinates.append(coordinates)
    
    all_coordinates_array = np.array(all_coordinates, dtype=object)
    return all_coordinates_array


def load_xyz_ele(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    data = file_content
    
    data_blocks = data.strip().split("4\n\n")
    data_blocks = data_blocks[1:]
    
    all_coordinates = []
    
    # Parse and store coordinates for each block
    for block in data_blocks:
        lines = block.strip().split('\n')
        #print(lines)
        coordinates = []
        for line in lines:
            elements = line.split()
            #print(elements)
            element_symbol = elements[0]
            x, y, z = map(float, elements[1:])
            coordinates.append([element_symbol, x, y, z])
        all_coordinates.append(coordinates)
    
    all_coordinates_array = np.array(all_coordinates, dtype=object)
    return all_coordinates_array, element_symbol


def load_config_values(file_path):
    # Initialize variables
    ele = None
    moddel_struc = None
    epoch = None

    # Open and read the text file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterate through the lines and extract the values
    for line in lines:
        if line.startswith('ele'):
            ele = str(line.split('=')[1].strip().strip("'"))
        elif line.startswith('struc'):
            moddel_struc = [int(x) for x in line.split('=')[1].strip()[1:-1].split(',')]
        elif line.startswith('epoch'):
            epoch = int(line.split('=')[1])

    return ele, moddel_struc, epoch



def plot_scatter(x, y, plot_save_path, title):
    # x : E_eda - E_ff
    # y : E_ml
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

    ax.plot(x,y,'o',zorder=10)

    # Set the same range for both x and y axes
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])

    # Make sure the aspect ratio is equal
    ax.axis('equal')
    
    # Set ticks for both axes to be the same
    x_ticks = ax.get_yticks()
    #ax.set_xticks(x_ticks)
    #ax.set_xticklabels(x_ticks)
    #ax.set_yticks(x_ticks)
    
    # Plot a diagonal dotted line for reference
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='dotted', zorder=0)
    ax.set_xlabel(r'$E_{eda}-E_{ff}$' + '(kJ/mol)')
    ax.set_ylabel(r'$E_{delta-ml}$'+'(kJ/mol)' )

    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_save_path + title + '.jpg')
    plt.show()



def plot_bar(x,plot_save_path,title=r'$E_{eda}-E_{ff} \quad vs \quad E_{delta-ml}$'):
    plt.figure(figsize=(4,3),dpi=600)
    #average_maeloss = np.sum(x,axis=0)/len(x)

    average_maeloss = np.mean(x, axis=0)
    std_maeloss = np.std(x, axis=0)

    label = ['elec','ct','pauli','pol','disp']
    plt.bar(label, average_maeloss,yerr=std_maeloss, capsize=5, align='center', alpha=0.7)
    plt.ylabel('Mae')
    plt.title(title)

    plt.tight_layout()
    plt.savefig(plot_save_path+ele+'_maebar.jpg')
    plt.show()


# box plot of the errors
def plot_box(x, plot_save_path,ele,labeltype):
    plt.figure(figsize=(4, 3), dpi=300)

    label = ['elec', 'ct', 'pauli', 'pol', 'disp']
    title=ele+'_'+r'$E_{DFT}-E_{ff} \quad vs \quad E_{delta-ml}$'
    plt.boxplot(x, labels=label,showfliers=False)
    plt.ylabel('error(kJ/mol)')
    plt.title(title)

    plt.tight_layout()
    plt.savefig(plot_save_path + ele+'_'+labeltype+'_errorbox.jpg')
    plt.show()

