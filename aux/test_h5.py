'''
We test the h5py parallel and MPI set-up in this script.
'''
from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

try:
    with h5py.File('test_parallel.h5', 'w', driver='mpio', comm=comm) as f:
        dset = f.create_dataset('data', (size,), dtype='i')
        dset[rank] = rank
    print(f"Rank {rank}: Successfully wrote to 'test_parallel.h5' using MPI.")
except ValueError as e:
    print(f"Rank {rank}: Failed to use MPI with h5py. Error: {e}")

