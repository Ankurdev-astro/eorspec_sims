import argparse
import subprocess as subproc

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run a sequence of processing scripts with MPI.")
    parser.add_argument('--config-file', type=str, 
            default="config.yaml", help='Path to the configuration file.')
    parser.add_argument('h5_dirs', nargs='+', 
            type=str, help='Directory containing observation files.')

    # Parse arguments
    args = parser.parse_args()

    # Run the context script with the observation director*ies
    print("\n","    Running context script...", "\n")
    subproc.run(["python", "write_context_primecam.py", *args.h5_dirs])

    # Run geometry footprint script
    print("\n","    Running footprint script...", "\n")
    subproc.run(["python", "write_footprint_primecam.py"])

    # Run the ML map-making script
    print("\n","    Running ML map-making script...", "\n")
    #subproc.run(["python", "make_ml_map_primecam.py", "--config-file", args.config_file])
    subproc.run(["mpirun", "-np", "8", "python", "make_ml_map_primecam.py", "--config-file", args.config_file])

if __name__ == "__main__":
    main()


