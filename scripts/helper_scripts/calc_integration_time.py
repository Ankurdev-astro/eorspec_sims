import os
import argparse
from datetime import datetime

def parse_time(time_str):
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

def calculate_integration_time(schedule_dir):
    # Get all .txt files in the provided directory
    schedule_files = [os.path.join(schedule_dir, f) for f in os.listdir(schedule_dir) if f.endswith('.txt')]
    
    # Initialize a variable for overall total integration time
    overall_total_integration_time = 0

    # Process each schedule file
    for file_path in schedule_files:
        total_integration_time = 0  # Initialize total time for this file

        with open(file_path, 'r') as file:
            # Skip initial non-data lines (Header lines)
            for _ in range(2):  # Adjusted to match new file format
                next(file)

            for line in file:
                # Skip lines that are not data lines
                if line.startswith('#') or not line.strip():
                    continue

                # Extract start and stop times from columns
                columns = line.split()
                start_time = parse_time(columns[0] + ' ' + columns[1])
                stop_time = parse_time(columns[2] + ' ' + columns[3])

                # Calculate the integration time for this observation in seconds and add to file total
                integration_time = (stop_time - start_time).total_seconds()
                total_integration_time += integration_time

        # Convert total integration time from seconds to hours and add to overall total
        total_integration_time_hours = total_integration_time / 3600
        overall_total_integration_time += total_integration_time_hours

        # Print total integration time for this file
        if args.debug:
            print(f"Total Integration Time for {file_path}: {total_integration_time_hours:.2f} hours")

    # Print overall total integration time
    print(f"Overall Total Integration Time: {overall_total_integration_time:.2f} hours")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate integration times from schedule files.")
    parser.add_argument('schedule_dir', type=str, help="Path to the directory containing schedule files.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function to calculate integration time
    calculate_integration_time(args.schedule_dir)

