import os
import re


#########################################################################################################
#This script extract the CCSD(T) energies for dimer water molecules and calculate the interaction energy#
#########################################################################################################


def extract_info_from_logs(folder_path, keyword):
    extracted_info = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if keyword in line:
                                energy_value = float(line.split('=')[1].strip())
                                extracted_info[file] = energy_value
                                break
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return extracted_info

def parse_filenames(file_names):
    pattern = re.compile(r'molecule_(\d+)([ab]?)\.log')
    parsed_files = []

    for file in file_names:
        match = pattern.match(file)
        if match:
            number = int(match.group(1))
            suffix = match.group(2)
            parsed_files.append((number, suffix, file))

    return sorted(parsed_files, key=lambda x: (x[0], x[1]))

def calculate_energy_differences(extracted_info):
    parsed_files = parse_filenames(extracted_info.keys())
    results = []
    final_numbers = []

    expected_number = None
    for i in range(0, len(parsed_files), 3):
        try:
            num, _, file_main = parsed_files[i]
            _, _, file_a = parsed_files[i + 1]
            _, _, file_b = parsed_files[i + 2]

            if expected_number is not None and num != expected_number + 1:
                print(f"Missing files: Expecting molecule_{expected_number + 1}.log")
            
            energy_main = extracted_info[file_main]
            energy_a = extracted_info[file_a]
            energy_b = extracted_info[file_b]

            diff = energy_main - (energy_a + energy_b)
            results.append((num, file_main, file_a, file_b, diff))
            final_numbers.append((num, diff))

            expected_number = num

        except IndexError:
            print("Incomplete set found for:", parsed_files[i:])
            break

    return sorted(results), sorted(final_numbers)

def write_final_numbers_to_file(final_numbers, output_file):
    with open(output_file, 'w') as f:
        for num, diff in final_numbers:
            f.write(f"molecule_{num}: {diff:.6f}\n")
    print(f"Final results written to {output_file}")

# Define folder path and keyword
#subfolder_path = '/pscratch/sd/s/schandy/meili'

subfolder_path = os.getcwd()
keyword_to_search = 'CCSD(T) total energy       ='
output_file = 'energy_differences.txt'

# Extract energies
info = extract_info_from_logs(subfolder_path, keyword_to_search)

# Process the files and calculate differences
energy_differences, final_numbers = calculate_energy_differences(info)

# Display sorted results
for num, main, a, b, diff in energy_differences:
    print(f"{main} - ({a} + {b}) = {diff:.6f}")

# Write the sorted final numbers to a file
write_final_numbers_to_file(final_numbers, output_file)

