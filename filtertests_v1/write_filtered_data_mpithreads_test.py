import toast
import toast.io as io

import numpy as np
import os

import astropy.units as u
import matplotlib.pyplot as plt
from scipy.signal import welch

from scripts.filters import filter_chain

from concurrent.futures import ThreadPoolExecutor

comm, procs, rank = toast.get_world()

toast_comm = toast.Comm(world=comm, groupsize=1)

filt_suffix = "filt_all_comap"
obs_dir_h5 = "ccat_datacenter_mock/data_COSMOSf351_d96_3deg15/"

def process_directory(h5dir_name):
    # Create the (empty) data
    data = toast.Data(comm=toast_comm)
    h5dir_path = os.path.join(obs_dir_h5, h5dir_name)

    if os.path.isdir(h5dir_path):
        print("\n")
        print(f"Loading h5 dir: {h5dir_path} ...")

    for h5file in os.listdir(h5dir_path):
        h5file_path = os.path.join(h5dir_path, h5file)
        #print("\t",f"Loading h5 file: {h5file} ...")

        if os.path.isfile(h5file_path) and h5file.endswith(".h5"):
            # Load the observation from the HDF5 file
            obs = io.load_hdf5(path=h5file_path, comm=toast_comm)
            # Append the observation
            data.obs.append(obs)
    print(f"Number of Observations Loaded: {len(data.obs)}")
    exit(0)
    filter_chain(data)
    
    # Write to h5
    filt_outdir = os.path.relpath(obs_dir_h5) + f"_{filt_suffix}"
    save_dir = os.path.join(filt_outdir, h5dir_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Writing filtered h5 files to: {save_dir}")

    detdata_tosave = ["signal", "flags"]

    for obs in data.obs:
        io.save_hdf5(
            obs=obs,
            dir=save_dir,
            detdata=detdata_tosave
        )
    
    del data
    print("="*40,"\n")

# Get a list of directories to process
h5dir_names = [d for d in os.listdir(obs_dir_h5) if os.path.isdir(os.path.join(obs_dir_h5, d))]

# Use ThreadPoolExecutor to parallelize the processing
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(process_directory, h5dir_names)

