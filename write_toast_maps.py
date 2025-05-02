import toast
import toast.io as io
import toast.ops
from toast.mpi import MPI

import numpy as np
import os

import astropy.units as u

from scripts import ccat_operators as ccat_ops

import h5py
import re


def main():
    #=============================#
    ### Setup
    #=============================#

    comm, procs, rank = toast.get_world() 
    toast_comm = toast.Comm(world=comm, groupsize=4)
    # performance improves with groupsize increasing
    # max we can do is like 8 groupsize
    # set process_rows=None
    
    # Set up logger and timer
    log_global = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer_global = toast.timing.Timer()
    timer_global.start()
    
    log_global.info_rank(f"Parallel HDF5 enabled: {h5py.get_config().mpi}", comm)
    log_global.info_rank(f"Procs, Rank: {procs, rank}", comm)
    log_global.info_rank(f"Group info: GSize GRank: {toast_comm.group_size, toast_comm.group_rank}", comm)
    log_global.info_rank(f"Number of process groups: {toast_comm.ngroups}", comm)
    
    if "OMP_NUM_THREADS" in os.environ:
        nthread = os.environ["OMP_NUM_THREADS"]
    else:
        nthread = "unknown number of"
    log_global.info_rank(
        f"Executing Filter and Bin workflow with {procs} MPI tasks, each with "
        f"{nthread} OpenMP threads",
        comm,
    )
    log_global.info_rank(f"Job group size: {toast_comm.group_size}", comm)
    log_global.info_rank(f"Group rank: {toast_comm.group_rank}", comm)
    
    #--------------------------------------------------#
    # EoR-Spec Channel IDs
    # 350 GHz Band
    # 333, 337, 340, 343, 347, 350,
    # 354, 357, 361, 365, 368

    # Input Data Dir
    parent_dir = "ccat_datacenter_mock/data_CII_tomo_ATM"
    channel_id = 333
    step = "step210"
    pattern = re.compile(rf".*_f{channel_id}$")
    
    ## Maps Output Directory
    maps_outdir = "./ccat_datacenter_mock/outmaps_fb_v1"
    savemaps_dir = os.path.join(maps_outdir, f"{channel_id}") 
    
    mapname_prefix = f"cosmos_f{channel_id}_{step}"
    os.makedirs(savemaps_dir, exist_ok=True)    
    if rank == 0:
        notes_file = os.path.join(savemaps_dir, 'notes.txt')
        # Write to the file inside savemaps_dir (this will overwrite each time)
        with open(notes_file, 'w') as f:
            # f.write(f"Poly Detrend; No CM; PCA: 5 leading components removed along axis 1; Turnarounds included \n")
            f.write(f"Testing 2.0 Deg center-width field \n")

    try:
        match_pattern = next(
                            (d for d in os.listdir(parent_dir) if pattern.match(d)),
                        None)
        data_input_path = os.path.join(parent_dir, match_pattern)
        log_global.info_rank(f"Matched Path: {data_input_path}", comm)
    except:
        raise RuntimeError(f"No match found for '*_f{channel_id}_*' in '{parent_dir}'")
    #--------------------------------------------------#    
    
    #=============================#
    ### Data Loading
    #=============================#
    
    # Create the (empty) data
    data = toast.Data(comm=toast_comm)
    sim_ground = toast.ops.SimGround()              
    
    log_global.info_rank(f"Load data...", comm)
    
    # Metadata fields to load
    meta_list = ["name", "uid", "telescope", "session", 
                "el_noise_model", "noise_model", "scan_el", 
                "scan_max_az", "scan_max_el", "scan_min_az", "scan_min_el"]

    # Shared data fields to load
    shared_list = ["azimuth", "elevation", "times",
                "flags","boresight_azel","boresight_radec",
                "position","velocity"]

    # Detector data fields to load
    detdata_list = ["signal", "flags"]

    # Interval types to load
    intervals_list = ['elnod', 'scan_leftright', 'scan_rightleft', 
                    'scanning', 'sun_close', 'sun_up', 'throw', 
                    'throw_leftright', 'throw_rightleft', 'turn_leftright', 
                    'turn_rightleft', 'turnaround']



    # Instantiate the LoadHDF5 operator
    loader = toast.ops.LoadHDF5(
        volume=data_input_path,  # Directory with observation files
        pattern="obs_.*_.*\.h5",                           # Match files like '"obs_.*_.*\.h5"'
        # files=[],                               # Use volume + pattern to find files
        meta=meta_list,     
        shared=shared_list, 
        detdata=detdata_list,                       
        intervals=intervals_list,  
        sort_by_size=False,                       # Sort observations by size
        process_rows=4,                        # Default "None" detector-major layout process_rows=1  #4
        #process_rows=1 Ensures all detectors are available on all ranks
        force_serial=False                        # Use parallel I/O if available
    )

    # Execute the operator
    loader.apply(data)
    
    log_global.info_rank(f"Number of Observations Loaded: {len(data.obs)} in", comm, timer = timer_global)
    for i,obs in enumerate(data.obs):
        log_global.info_rank(f"Shape of Signal in Obs{i}: {np.asarray(obs.detdata['signal']).shape}", comm) 
    #-------------------------------------#
    exit(1)
    #=============================#
    ### Filtering
    #=============================#
    timer.start()

    # ### Data Level 1: Deslope
    # log_global.info_rank(f"Deslope data...", comm)
    # deslope = ccat_ops.Deslope(name="deslope")
    # deslope.enabled = True  # Toggle to False to disable
    # deslope.apply(data)
    # log_global.info_rank(f"Deslope done in", comm, timer = timer) 
    
    ### Data Level 1: Deslope
    poly_detrend = ccat_ops.PolyDetrend(name="poly_detrend")
    poly_detrend.enabled = False  # Toggle to False to disable
    poly_detrend.apply(data)
    log_global.info_rank(f"Poly Detrend done in", comm, timer = timer)
    
    ### Data Level 2: Az-El Template Correction
    log_global.info_rank(f"Az-El Template Correction...", comm)
    template_azel = ccat_ops.Template_azel(name="template_azel")
    template_azel.enabled = False  # Toggle to False to disable
    template_azel.apply(data)
    log_global.info_rank(f"Az-El Template done in", comm, timer = timer)    
    # exit(1)

    ### Data Level 3: Common Mode Removal
    log_global.info_rank(f"Common Mode Removal...", comm)
    commonmode_filter = toast.ops.CommonModeFilter()
    commonmode_filter.enabled = False  # Toggle to False to disable
    commonmode_filter.apply(data)
    log_global.info_rank(f"Common Mode done in", comm, timer = timer)   
    # exit(1)

    ### Data Level 4: PCA Component Removal
    log_global.info_rank(f"PCA Component Removal...", comm)
    pca_clean = ccat_ops.PCAComp_removal(name="pca_clean")
    pca_clean.n_components = 5 #4
    pca_clean.enabled = False  # Toggle to False to disable
    pca_clean.apply(data)
    log_global.info_rank(f"PCA done in", comm, timer = timer)
    
    log_global.info_rank(f"Filtering done in", comm, timer = timer_global)  
    
    #=============================#
    ### Binning
    #=============================#
    #CAR
    # 0.6 arcmin pixes
    res = (37/3 * u.arcsec).to(u.deg)     # Resolution per Pix: 1/3rd of Beam FWHM
    mode = "I"

    log_global.info_rank(f"Generate Pointing...", comm)
    #center = RA, DEC in DEG
    #bounds = RA_MAX, RA_MIN, DEC_MIN, DEC_MAX
    pixels_wcs_radec = toast.ops.PixelsWCS(
                                    name="pixels_wcs_radec",
                                    projection="CAR",
                                    resolution=(res, res),
                                    center=(150.0*u.degree, 2.0*u.degree),
                                    # bounds=(153*u.degree,147*u.degree,-2*u.degree,6*u.degree),
                                    dimensions = (np.int32(8*u.deg/res), np.int32(8*u.deg/res)),
                                    auto_bounds=False,
                                    )
    pixels_wcs_radec.enabled = True

    # Configure Az/El and RA/DEC boresight and detector pointing and weights
    # det_pointing_azel = toast.ops.PointingDetectorSimple(name="det_pointing_azel", 
    #                                                      quats="quats_azel")
    # det_pointing_azel.enabled = True
    # det_pointing_azel.boresight = sim_ground.boresight_azel

    det_pointing_radec = toast.ops.PointingDetectorSimple(name="det_pointing_radec", 
                                                          quats="quats_radec")
    det_pointing_radec.enabled = True
    det_pointing_radec.boresight = sim_ground.boresight_radec

    #===================#
    #Set det pointing
    pixels_wcs_radec.detector_pointing = det_pointing_radec
    #=============================#
    ### Pointing Weights
    # weights_azel = toast.ops.StokesWeights(name="weights_azel", 
    #                                        weights="weights_azel",  mode=mode)
    weights_radec = toast.ops.StokesWeights(name="weights_radec", mode=mode)
    # weights_azel.enabled = True
    weights_radec.enabled = True

    # weights_azel.detector_pointing = det_pointing_azel
    weights_radec.detector_pointing = det_pointing_radec


    # #=============================#

    # # Construct a "perfect" noise model just from the focalplane parameters # after det pointing
    # default_model = toast.ops.DefaultNoiseModel(name="default_model", noise_model="noise_model")
    # default_model.apply(data)

    # # Create the Elevation modulated noise model
    # elevation_model = toast.ops.ElevationNoise(name="elevation_model",
    #                                            out_model="el_noise_model")
    # elevation_model.noise_model = default_model.noise_model
    # elevation_model.detector_pointing = det_pointing_azel
    # elevation_model.apply(data)
    # #==============================#

    log_global.info_rank(f"Binning into Maps...", comm)

    #Set up the pointing used in the binning operator
    binner_final = toast.ops.BinMap(name="binner_final", pixel_dist="pix_dist_final")
    binner_final.enabled = True
    binner_final.shared_flag_mask = 0 #No flags masked; include all data and turnarounds
    binner_final.pixel_pointing = pixels_wcs_radec
    binner_final.stokes_weights = weights_radec
    # binner_final.noise_model = elevation_model.out_model

    mapmaker = toast.ops.MapMaker(name=mapname_prefix)
    mapmaker.weather = "vacuum"
    mapmaker.write_hdf5 = False
    mapmaker.binning = binner_final
    mapmaker.map_binning = binner_final
    # mapmaker.iter_max = 10
    mapmaker.report_memory = False

    # map product options
    mapmaker.write_hits = True
    mapmaker.write_binmap = True
    mapmaker.write_cov = False
    mapmaker.write_invcov = True
    mapmaker.write_map = True
    mapmaker.write_noiseweighted_map = True
    mapmaker.write_rcond = False
    mapmaker.write_solver_products = False


    # No templates for now (will just do the binning)
    mapmaker.template_matrix = toast.ops.TemplateMatrix(templates=[])

    #if solving templates:
    # mapmaker.template_matrix = toast.ops.TemplateMatrix(templates=[toast.templates.Offset(step_time=0.5*u.second)])
    ##0.5sec is too slow

    mapmaker.output_dir = savemaps_dir
    log_global.info_rank(f"Writing maps in {savemaps_dir}", comm)
    mapmaker.enabled = True
    mapmaker.apply(data)
    log_global.info_rank(f"Binning done in", comm, timer = timer_global)  

if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()