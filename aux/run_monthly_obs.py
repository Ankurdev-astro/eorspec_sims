import os

# Define the base date and the command to run
base_date = str(input("Enter month: YYYY-MM:"))
start_time = "00:00:00"
end_time = "23:59:59"
command = "python params_gen_yaml_smallfield.py"

# Loop through all odd/even days of the month
# for day in range(1, 31, 4):
for day in range(2, 31, 4):
    day_str = f"{day:02d}"
    start_datetime = f"{base_date}-{day_str} {start_time}"
    end_datetime = f"{base_date}-{day_str} {end_time}"
    
    # Construct and execute the command
    sch_script = f'{command} "{start_datetime}" "{end_datetime}" cosmos'
    print("-----------------------------------------","\n")
    print(f"Date: {base_date}-{day_str}")
    print(f"Executing: {sch_script}")
    os.system(sch_script)

