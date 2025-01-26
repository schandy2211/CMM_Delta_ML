import torch
import numpy as np
from utils import *
import os
from model import feed_forward


def remove_title_card_and_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        
        i = 0
        while i < len(lines):
            atom_count = lines[i].strip()
            outfile.write(f"{atom_count}\n\n")  
            
            i += 1  
            i += 1  
            
            for j in range(int(atom_count)):  
                line = lines[i].strip().split()
                atom, x, y, z = line[0], float(line[1]), float(line[2]), float(line[3])
                formatted_line = f"{atom} {x:.16f} {y:.16f} {z:.16f}\n"
                outfile.write(formatted_line)
                i += 1  
            
           # outfile.write("\n")  

    print(f"Title cards removed and formatted output saved to '{output_file}'.")
input_file = 'f_water_f_dimers.xyz'   
output_file = 'f_water_f_dimers1.xyz' 
remove_title_card_and_format(input_file, output_file)

current_dir = os.getcwd()
filepath_cluster = os.path.join(current_dir, 'f_water_f_dimers1.xyz')
xyz = load_xyz(filepath_cluster)


filepath ='args.txt'
ele, model_struc, epochs = load_config_values(filepath)

#Calculte the input hybrid matrices
hyb_cluster = hyb_n(xyz)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_cluster = torch.tensor(hyb_cluster, dtype=torch.float32).to(device)

# Ensure the shape matches the model's expectation
cluster_size = len(xyz)
x_cluster = x_cluster.view(cluster_size, 16)
model = feed_forward(model_struc)  # Ensure model structure is defined correctly
model.load_state_dict(torch.load(os.path.join(current_dir, "best_model.pt"), map_location=device))
model = model.to(device)
model.eval()
with torch.no_grad():
    output = model(x_cluster)
output_numpy = output.cpu().numpy()
np.savetxt(os.path.join(current_dir, 'delta_pred.txt'), output_numpy, fmt='%.6f', delimiter=' ')


# labelling #
labels = ['cls_elec', 'ct', 'mod_pauli', 'pol', 'disp']

def label_data_with_titles(valid_output_file, xyz_file, output_file):
    energies = np.loadtxt(valid_output_file)

    titles = []
    with open(xyz_file, 'r') as xyz_f:
        lines = xyz_f.readlines()
        for i, line in enumerate(lines):
            if "Cluster" in line:
                titles.append(line.strip())  

    if len(energies) != len(titles):
        raise ValueError("Mismatch between number of energies and number of XYZ titles")

    with open(output_file, 'w') as f_out:
        f_out.write(",".join(labels) + ",title\n")  
        for i, energy in enumerate(energies):
            labeled_line = ", ".join([f"{value:.6f}" for value in energy])  
            f_out.write(f"{labeled_line},{titles[i]}\n")  

    #print(f"Labeled data saved to {output_file}")



def aggregate_cluster_contributions(labeled_file, cluster_output_file):
    cluster_sums = {}

    with open(labeled_file, 'r') as f_in:
        lines = f_in.readlines()[1:]  
        for line in lines:
            values, title = line.rsplit(',', 1)
            values = list(map(float, values.split(',')))

            cluster_info = title.strip().split("Cluster")[-1]
            cluster_number = int(cluster_info.strip())

            # Sum contributions for each cluster
            if cluster_number not in cluster_sums:
                cluster_sums[cluster_number] = np.zeros(len(labels))
            cluster_sums[cluster_number] += values  


    with open(cluster_output_file, 'w') as f_out:
        f_out.write("Cluster," + ",".join(labels) + "\n")  
        for cluster, sums in sorted(cluster_sums.items()):
            summed_line = ", ".join([f"{value:.6f}" for value in sums])
            f_out.write(f"{cluster},{summed_line}\n")



def main():
    current_dir = os.getcwd()

    valid_output_file = os.path.join(current_dir, 'delta_pred.txt')
    xyz_file = os.path.join(current_dir, 'f_water_f_dimers.xyz')
    labeled_output_file = os.path.join(current_dir, 'delta_pred_label.txt')
    #cluster_output_file = os.path.join(current_dir, 'cluster_contributions_ion.txt')

    label_data_with_titles(valid_output_file, xyz_file, labeled_output_file)

    #aggregate_cluster_contributions(labeled_output_file, cluster_output_file)


if __name__ == "__main__":
    main()


