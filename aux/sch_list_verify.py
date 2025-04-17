import os
sch_list_path = '../input_files/schedule_list.txt'
sch_rel_path = "../input_files/schedules/"
filesnames = []

# Open the file and read the contents
with open(sch_list_path, 'r') as file:
    content = file.read()
    filenames = content.split(',')

# Trim any whitespace around the filenames and filter out any empty strings
filenames = [name.strip() for name in filenames if name.strip()]

print(f"Indexed {len(filenames)} schedule files")
#print(filenames)
for file_count, schedule_file in enumerate(filenames):
    sch_file_path = os.path.join(sch_rel_path,schedule_file)
    print(sch_file_path, file_count+1)
    with open(sch_file_path) as sch_file:
        head = [next(sch_file) for _ in range(11)]
        print(head)

