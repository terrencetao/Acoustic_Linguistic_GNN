import pandas as pd
import os
import argparse 



parser = argparse.ArgumentParser()
parser.add_argument('--csv', help='csv file where get the param', required=True)
parser.add_argument('--param', help='column in the condition', required =True)
parser.add_argument('--val', help='value of the column', required =True)
args = parser.parse_args()

# Load the CSV file
file_path = args.csv
unit = file_path.split('_')[2]
df = pd.read_csv(file_path)

# Filter data where 'mhg' equals 'ML'
filtered_data = df[df[args.param] == args.val]

# Find best value in 'Heterogeneous Model' column
best_index = filtered_data['Heterogeneous Model'].idxmax()

# Get the row corresponding to the best value
best_row = filtered_data.loc[best_index]

# Extract the parameters from the best row
unit = unit
alpha = best_row['alpha']
num = best_row['num_n_a']
msa = best_row['msa']
msw = best_row['msw']
mgw = best_row['mgw']
mhg = best_row['mhg']

# Construct the command to run the bash script
command = f"./exec_best.sh --sub_unit {unit} --alpha {alpha} --num {num} --msa {msa} --msw {msw} --mgw {mgw} --mhg {mhg}"

# Print the constructed command (for verification)
print(f"Constructed command: {command}")

# Execute the constructed command
os.system(command)
