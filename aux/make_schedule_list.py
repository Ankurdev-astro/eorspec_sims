import glob

# Define the pattern to match files
pattern = '../input_files/schedules/sch*'

# Use glob to find all files matching the pattern
file_paths = glob.glob(pattern)

print(f"Found {len(file_paths)} schedule files")

#Extract filenames from paths
filenames = [path.split('/')[-1] for path in file_paths]

# Join filenames into a single comma-separated string
filenames_str = ','.join(filenames)

# Write the filenames to schedule_list.txt
with open('schedule_list.txt', 'w') as f:
    f.write(filenames_str)

print('Filenames written to schedule_list.txt')

