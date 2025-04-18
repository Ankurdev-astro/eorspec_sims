import argparse
import os
import re
from glob import glob

# Predefined step names.
step_names = ["step210", "step216", "step222", "step228", "step235"]

def process_file(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Determine header: capture all lines until the first data row (line starts with YYYY-MM-DD)
    header_lines = []
    data_start = None
    date_pattern = re.compile(r'\s*\d{4}-\d{2}-\d{2}')
    for i, line in enumerate(lines):
        if date_pattern.match(line):
            data_start = i
            break
        else:
            header_lines.append(line)
    
    if data_start is None:
        print(f"Error: No data row found based on date pattern in file '{file_path}'.")
        return

    data_lines = lines[data_start:]
    total_data = len(data_lines)
    n_steps = len(step_names)

    if total_data < n_steps:
        print(f"Not enough data lines ({total_data}) in file '{file_path}' to split into {n_steps} chunks.")
        return

    # Compute chunk size and discard extra lines.
    chunk_size = total_data // n_steps
    total_used = chunk_size * n_steps
    print(f"Processing '{file_path}': {total_data} data lines found, discarding {total_data - total_used}).")

    original_filename = os.path.basename(file_path)
    outdir = 'gen_step_schedules'

    # For round-robin distribution, consider only the first total_used data lines.
    data_lines = data_lines[:total_used]

    for i, chunk_name in enumerate(step_names):
        # Select every n_steps-th line starting from offset i.
        chunk_data = data_lines[i::n_steps]
        step_dir = os.path.join(outdir, chunk_name)
        os.makedirs(step_dir, exist_ok=True)

        # output_filename = f"{chunk_name}_{original_filename}"
        output_filename = f"{original_filename}"
        output_path = os.path.join(step_dir, output_filename)
        with open(output_path, 'w') as f_out:
            f_out.writelines(header_lines + chunk_data)
        # print(f"Created '{output_path}'")

def main():
    parser = argparse.ArgumentParser(
        description="Split all .txt schedule files in a directory into equal chunks as n_steps"
    )
    parser.add_argument("dir_path", help="Directory containing schedule .txt files")
    args = parser.parse_args()

    if not os.path.isdir(args.dir_path):
        print(f"Error: '{args.dir_path}' is not a valid directory.")
        return

    file_pattern = os.path.join(args.dir_path, '*.txt')
    files = glob(file_pattern)
    if not files:
        print(f"No .txt files found in '{args.dir_path}'.")
        return

    for file in files:
        process_file(file)

if __name__ == "__main__":
    main()

