###
#Timestream Simulation Script for EoR-Spec
###
###Last updated: April 07, 2025
###
#Author: Ankur Dev, adev@astro.uni-bonn-de
###
###Logbook###
###
###Personal Notes and Updates:

#ToDo: Check max PWV
#ToDo: Assign ATM realization
###
###[Self]Updates Log:
#20-02-2024: Scraping TOAST2, Begin migration
#14-03-2024: Upgraded to TOAST3
#19-03-2024: Checked write and read to h5
#21-03-2024: Implement dual-regime Atm sim
#22-03-2024: Begin integrated script implemention
#23-03-2024: Implemented multiple schedules
#23-03-2024: Modified schedule field name; must be %field_%dd_%mm for uid
#02-04-2024: Updated Atmosphere implementation
#10-05-2024: Implemented CAR and Healpy capabilities
#10-05-2024: Major changes to primecam_mockdata_pipeline()
#10-05-2024: Few changes to args Class, toast.mpi.get_world() call
#13-05-2024: Changes to schedule files, input CAR files
#13-05-2024: Implemented PWV limit handling
#14-05-2024: Added changes to SimGround, with median and max_pwv truncate
#17-05-2024: Implemented El Nods in SimGround
#19-05-2024: Updated reformat_dets() to be compatible with EoRSpec
#24-05-2024: sim_atm Boolean Flag added to Class args
#01-03-2025: Implemented MPI for schedules, Implemented CAR on new Tomography
#01-03-2025: Implemented h5 FP instead of pkl
#07-04-2025: Allow automatic choosing max ndets and FPI step for and FPI channel
###

"""
Description:
Timestream Simulation Script for Prime-cam.
This script is modified from TOAST Ground Simulation to perform timestream simulation
for Prime-Cam with FYST. Note: this is a work in progress.

The script demonstrates an example workflow, from generating loading detectors, scanning
an input map, making detailed atmospheric simulation and generating mock detector timestreams. 
for more complex simulations tailored to specific experimental needs.

Usage:
Ref: https://github.com/hpc4cmb/toast/blob/toast3/workflows/toast_sim_ground.py
Ref: TOAST3 Documentation: https://toast-cmb.readthedocs.io/en/toast3/intro.html

"""


import toast
import toast.io as io
import toast.ops
from toast.mpi import MPI

from scripts.helper_scripts.calc_groupsize import job_group_size, estimate_group_size

import astropy.units as u
from astropy.table import QTable

import numpy as np
import h5py
from datetime import datetime
import os, re
import argparse
import random
import time as t


# Define the global args class
class Args:
    def __init__(self, parsed_args):
        self.parsed_args = parsed_args
        self.weather = 'atacama'
        self.sim_atm = False #True Default
        self.pwv_limit = 1.27 #1.27 #mm
        self.sample_rate = 244 * u.Hz
        self.scan_rate_az = 0.5  * (u.deg / u.s) #on sky rate
        #fix_rate_on_sky (bool):  If True, `scan_rate_az` is given in sky coordinates and azimuthal
        #rate on mount will be adjusted to meet it.
        #If False, `scan_rate_az` is used as the mount azimuthal rate. (default = True)

        self.scan_accel_az = 1  * (u.deg / u.s**2)
        self.fov = 1.3 * u.deg # Field-of-view in degrees
    
        self.h5_outdir = os.path.join(
            ".", "ccat_datacenter_mock", 
            "new_CII_tomo_dump", 
            f"deg2-0_data_COSMOS_f{parsed_args.chnl}"
        )

        # CAR set-up for WCS Pixel Operator
        self.mode = "I"
        self.input_map =  f"CII_CAR_TNGval350_f{parsed_args.chnl}.fits"
        self.map_center = (150.0 *u.degree, 2.0 *u.degree)               #from CRVAL
        map_side_deg = 4 * u.degree
        self.map_shape = (1107, 1107)                                      #from NAXIS        
        self.map_resolution = (map_side_deg/self.map_shape[0], 
                               map_side_deg/self.map_shape[0])              #from CDELT
        
def get_max_ndets_step(fchannel_dir, chnl):
    """
    Scans the directory for a given channel (chnl_{chnl}) and returns a tuple:
    (max_step_value, max_ndets, total_steps)
    """
    channel_dir = os.path.join(fchannel_dir, f"chnl_{chnl}")
    if not os.path.isdir(channel_dir):
        raise ValueError(f"Channel directory '{channel_dir}' not found.")
    
    pattern = re.compile(r'step(\d+)_d(\d+)_dettable\.h5')
    max_ndets = -1
    max_step = None
    count = 0

    for root, _, files in os.walk(channel_dir):
        for file in files:
            m = pattern.search(file)
            if m:
                count += 1
                step_val = int(m.group(1))
                ndets = int(m.group(2))
                if ndets > max_ndets:
                    max_ndets = ndets
                    max_step = step_val

    return max_step, max_ndets, count

def eorspec_mockdata_pipeline(args, comm, focalplane, schedule, group_size):
    #Set up logger and timer
    log = toast.utils.Logger.get()
    timer = toast.timing.Timer()
    timer.start()
    
    if group_size is not None:
        # Create the toast communicator with specified group size
        toast_comm = toast.Comm(world=comm, groupsize=group_size)
        log.info_rank(f"Job group size: {toast_comm.group_size}", comm)
        log.info_rank(f"Number of process groups: {toast_comm.ngroups}", comm)
    else:
        log.info_rank(f"Begin job planning ...", comm)
        # grp_size_calc = job_group_size(world_comm=comm, schedule=schedule,
        #                                focalplane=focalplane)
        grp_size_calc = estimate_group_size(world_comm=comm, schedule=schedule)
        # Create the toast communicator
        toast_comm = toast.Comm(world=comm, groupsize=grp_size_calc)
        log.info_rank(f"Job group size: {toast_comm.group_size}", comm)
        log.info_rank(f"Number of process groups: {toast_comm.ngroups}", comm)
    
    # Shortcut for the world communicator
    world_comm = toast_comm.comm_world
    
    # Begin Pipeline
    #=============================# 

    site = toast.instrument.GroundSite(
        schedule.site_name,
        schedule.site_lat,
        schedule.site_lon,
        schedule.site_alt,
        weather=args.weather,
    )
    telescope = toast.instrument.Telescope(
        schedule.telescope_name, focalplane=focalplane, site=site
    )

    log.info_rank(f"Telescope metadata: \n {telescope}", world_comm)
    log.info_rank(f"FocalPlane: {focalplane}", world_comm)

    # Create the (initially empty) data
    data = toast.Data(comm=toast_comm)

    #Load SimGround

    sim_ground = toast.ops.SimGround(weather=args.weather)
    sim_ground.telescope = telescope
    sim_ground.schedule = schedule
    sim_ground.scan_rate_az =  args.scan_rate_az
    sim_ground.scan_accel_az = args.scan_accel_az
    sim_ground.max_pwv = 1.41 * u.mm #Truncate PWV
    sim_ground.median_weather = False
    
    # exit(1)
    #=============================#
    ### El Nod Tests ###

    sim_ground.scan_rate_el = 1.5 * (u.deg / u.s) #Allowed by mount #1.6
    sim_ground.el_mod_amplitude = 1.0 * u.deg #1.2
    sim_ground.el_mod_rate = 0.1 * u.Hz
    sim_ground.el_mod_sine = True
    sim_ground.elnod_every_scan = False
    
    #=============================#
    sim_ground.apply(data)
    
    log.info_rank(f"Number of Observations loaded: {len(data.obs)}", world_comm)

    #=============================#
    # Detector Pointing
    pixels_wcs_radec = toast.ops.PixelsWCS(
                                    name="pixels_wcs_radec",
                                    projection="CAR",
                                    center=args.map_center, 
                                    resolution=args.map_resolution,
                                    dimensions=args.map_shape,
                                    auto_bounds=False,
                                    )

    ### Pointing Matrix
    pixels_wcs_azel = toast.ops.PixelsWCS(
                                    name="pixels_wcs_azel",
                                    projection="CAR",
                                    resolution=(0.0 * u.degree, 0.0 * u.degree),
                                    auto_bounds=False,
                                    )


    pixels_healpix_radec = toast.ops.PixelsHealpix(name="pixels_healpix_radec")

    #Toggel for CAR / Healpy formats
    pixels_wcs_radec.enabled = True
    pixels_wcs_azel.enabled = False
    pixels_healpix_radec.enabled = False

    scan_wcs_map = toast.ops.ScanWCSMap(name="scan_wcs_map")
    scan_wcs_map.enabled = True

    scan_healpix_map = toast.ops.ScanHealpixMap(name="scan_healpix_map")
    scan_healpix_map.enabled = False

    n_enabled_solve = np.sum(
        [
            pixels_wcs_radec.enabled,
            pixels_wcs_azel.enabled,
            pixels_healpix_radec.enabled,
        ]
    )
    if n_enabled_solve != 1:
        raise RuntimeError(
            "Only one pixelization operator should be enabled for the solver."
        )

    #=============================#
    
    # Configure Az/El and RA/DEC boresight and detector pointing and weights
    det_pointing_azel = toast.ops.PointingDetectorSimple(name="det_pointing_azel",
                                                         quats="quats_azel")
    det_pointing_azel.enabled = True
    det_pointing_azel.boresight = sim_ground.boresight_azel

    det_pointing_radec = toast.ops.PointingDetectorSimple(name="det_pointing_radec",
                                                          quats="quats_radec")
    det_pointing_radec.enabled = True
    det_pointing_radec.boresight = sim_ground.boresight_radec

    #===================#

    pixels_wcs_azel.detector_pointing = det_pointing_azel
    pixels_wcs_radec.detector_pointing = det_pointing_radec
    pixels_healpix_radec.detector_pointing = det_pointing_radec

    #===================#
    ### Pointing Weights
    weights_azel = toast.ops.StokesWeights(name="weights_azel",
                                           weights="weights_azel",  mode=args.mode)
    weights_radec = toast.ops.StokesWeights(name="weights_radec",
                                            weights="weights_radec",  mode=args.mode)
    weights_azel.enabled = True
    weights_radec.enabled = True

    weights_azel.detector_pointing = det_pointing_azel
    weights_radec.detector_pointing = det_pointing_radec

    #=============================#
    # Select Pixelization and weights for solve and final binning

    if pixels_wcs_azel.enabled:
        if scan_healpix_map.enabled:
            raise RuntimeError("Cannot scan from healpix map with WCS pointing")
        pixels_solve = pixels_wcs_azel
        weights_solve = weights_azel
    elif pixels_wcs_radec.enabled:
        if scan_healpix_map.enabled:
            raise RuntimeError("Cannot scan from healpix map with WCS pointing")
        pixels_solve = pixels_wcs_radec
        weights_solve = weights_radec
    else:
        if scan_wcs_map.enabled:
            raise RuntimeError("Cannot scan from WCS map with healpix pointing")
        pixels_solve = pixels_healpix_radec
        weights_solve = weights_radec

    weights_final = weights_solve
    pixels_final = pixels_solve
                
    #=============================#
    #Noise and Elevation Model

    # Construct a "perfect" noise model just from the focalplane parameters # after det pointing #after pwv handle
    default_model = toast.ops.DefaultNoiseModel(name="default_model", noise_model="noise_model")
    default_model.apply(data)

    # Create the Elevation modulated noise model
    elevation_model = toast.ops.ElevationNoise(name="elevation_model",
                                               out_model="el_noise_model")
    elevation_model.noise_model = default_model.noise_model
    elevation_model.detector_pointing = det_pointing_azel
    elevation_model.apply(data)

    #=============================#
    #Set up the pointing used in the binning operator

    binner_final = toast.ops.BinMap(name="binner_final", pixel_dist="pix_dist_final")
    binner_final.enabled = True
    binner_final.pixel_pointing = pixels_final
    binner_final.stokes_weights = weights_final
    log.info_rank(" Simulated telescope boresight pointing in", comm=world_comm, timer=timer)

    #=============================#

    # Atmospheric simulation
    log.info_rank(f"Atmospheric simulation...", world_comm)
    #Atmosphere set-up
    rand_realisation = random.randint(10000, 99999)
    tel_fov = 1.5* u.deg
    cache_dir = "./atm_cache"


    sim_atm_coarse =toast.ops.SimAtmosphere(
                    name="sim_atm_coarse",
                    add_loading=False,
                    lmin_center=300 * u.m,
                    lmin_sigma=30 * u.m,
                    lmax_center=10000 * u.m,
                    lmax_sigma=1000 * u.m,
                    xstep=50 * u.m,
                    ystep=50 * u.m,
                    zstep=50 * u.m,
                    zmax=2000 * u.m,
                    nelem_sim_max=30000,
                    gain=1e-5, #changed 04.02.2025 # 2e-5
                    realization=1000000,
                    wind_dist=10000 * u.m,
                    enabled=False,
                    cache_dir=cache_dir,
                )

    sim_atm_coarse.realization = 1000000 + rand_realisation
    sim_atm_coarse.field_of_view = tel_fov
    sim_atm_coarse.detector_pointing = det_pointing_azel
    sim_atm_coarse.enabled = args.sim_atm # Toggle to False to disable
    sim_atm_coarse.serial = False
    sim_atm_coarse.apply(data)
    log.info_rank(" Applied large-scale Atmosphere simulation in", comm=world_comm, timer=timer)

    #------------------------#

    sim_atm_fine= toast.ops.SimAtmosphere(
            name="sim_atm_fine",
            add_loading=True,
            lmin_center=0.001 * u.m,
            lmin_sigma=0.0001 * u.m,
            lmax_center=1 * u.m,
            lmax_sigma=0.1 * u.m,
            xstep=4 * u.m,
            ystep=4 * u.m,
            zstep=4 * u.m,
            zmax=200 * u.m, #changed 31.01.2025
            gain=1e-5, #Changed 04.02.2025 4e-5
            wind_dist=1000 * u.m,
            enabled=False,
            cache_dir=cache_dir,
        )

    sim_atm_fine.realization = rand_realisation
    sim_atm_fine.field_of_view = tel_fov
    
    sim_atm_fine.detector_pointing = det_pointing_azel
    sim_atm_fine.enabled = args.sim_atm  # Toggle to False to disable
    sim_atm_fine.serial = False
    sim_atm_fine.apply(data)
    #------------------------#

    log.info_rank("Applied full Atmosphere simulation in", comm=world_comm, timer=timer)

    #=============================#
    # Simulate sky signal from a map
    input_map = os.path.join("input_files/input_maps", args.input_map)
    #check if this file exists, else raise runtime error
    if not os.path.exists(input_map):
        raise RuntimeError(f"Input map file not found: {input_map}")

    if scan_healpix_map.enabled:
        log.info_rank(f"Loading Healpix Map {input_map}", world_comm)
        scan_healpix_map.file = input_map
        scan_healpix_map.pixel_dist = binner_final.pixel_dist
        scan_healpix_map.pixel_pointing = pixels_final
        scan_healpix_map.stokes_weights = weights_final
        scan_healpix_map.save_pointing = False
        scan_healpix_map.apply(data)

    elif scan_wcs_map.enabled:
        log.info_rank(f"Loading WCS Map {input_map}", world_comm)
        scan_wcs_map.file = input_map
        scan_wcs_map.pixel_dist = binner_final.pixel_dist
        scan_wcs_map.pixel_pointing = pixels_final
        scan_wcs_map.stokes_weights = weights_final
        scan_wcs_map.save_pointing = False
        scan_wcs_map.apply(data)

    mem = toast.utils.memreport(msg="(whole node)", comm=world_comm, silent=True)
    log.info_rank(f"After Scanning Input Map:  {mem}", world_comm)

    #=============================#
    # # Simulate detector noise
    # sim_noise = toast.ops.SimNoise(name="sim_noise")
    # sim_noise.noise_model = elevation_model.out_model
    # sim_noise.apply(data)

    #=============================#

    mem = toast.utils.memreport(msg="(whole node)", comm=world_comm, silent=True)
    log.info_rank(f"After generating detector timestreams:  {mem}", world_comm)

    field_name = (data.obs[0].name).split('-')[0]
    n_dets = telescope.focalplane.n_detectors

    #=============================#
    #Write to h5
    module_code = str(args.parsed_args.chnl)
    f_path = f"sim_eorspec_f{module_code}_{args.parsed_args.step}_d{n_dets}"
    save_dir = os.path.join(args.h5_outdir, f_path)
    os.makedirs(save_dir, exist_ok=True)

    log.info_rank(f"Writing timestream data to h5 files for \
Field {field_name} observed at {args.parsed_args.chnl} GHz channel\
with {n_dets} detectors", world_comm)
    log.info_rank(f"Writing h5 files to: {save_dir}", world_comm)

    detdata_tosave = ["signal", "flags"]

    for obs in data.obs:
        io.save_hdf5(
            obs=obs,
            dir=save_dir,
            detdata=detdata_tosave
        )

###==================================================### 

# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Simulate PrimeCam Timestream Data with 1 schedule",
        epilog="Note: Provide only the name of the schedule file, not the path. \n \
            The schedule file must be in the 'input_files/schedules' directory.")
    # Required argument for the schedule file
    parser.add_argument('-s','--sch', required=True, help="Name of the schedule file")
    parser.add_argument('-c', '--chnl',
                        type=int, 
                        default=350, 
                        help="Channel ID of EoR-Spec Channel (e.g. 350)")
    # parser.add_argument('--step',
    #                     type=str, 
    #                     default="step228", 
    #                     help="EoR-Spec Step")
    parser.add_argument('--step', 
                        type=str, 
                        default="step228", 
                        help="EoR-Spec Step. If not provided, compute for max ndets")
    parser.add_argument('-d', '--ndets',
                        type=int,
                        default=None,
                        help="Number of detectors to be selected")
    parser.add_argument('-g','--grp_size', default=None, type=int, help="Group size (optional)")

    parsed_args = parser.parse_args()
    
    args = Args(parsed_args)
    
    #Set up logger and timer
    log_global = toast.utils.Logger.get()
    global_timer = toast.timing.Timer()
    timer = toast.timing.Timer()
    global_timer.start()
    timer.start()

    # Initialize the communicator
    comm, procs, rank = toast.get_world()
    
    # Initialize the TOAST logger
    if "OMP_NUM_THREADS" in os.environ:
        nthread = os.environ["OMP_NUM_THREADS"]
    else:
        nthread = "unknown number of"
        
    log_global.info_rank(
        f"Executing PrimeCam workflow with {procs} MPI tasks, each with "
        f"{nthread} OpenMP threads at {datetime.now()}",
        comm,
    )

    log_global.info_rank(
        f"Using TOAST version: {toast.__version__}", comm)

    log_global.info_rank(
        f"Starting timesteam simulation...", comm)
    if rank == 0:
        sim_start_time = t.time()

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log_global.info_rank(f"Start of the workflow:  {mem}", comm)
    
    log_global.info_rank(
        f"Begin set-up and monitors for Simulating timestream data for PrimeCam/FYST",
        comm)
    
    # Set up Detector file
    # If --step is not given, compute it from the files in the channel directory
    fchl_dir = "input_files/fp_files/fchl_h5"
    if parsed_args.step is None:
        try:
            max_step, max_ndets, count_steps = get_max_ndets_step(fchl_dir, parsed_args.chnl)
            if max_step is None:
                log_global.info_rank(
                    f"No valid files found for channel chnl_{parsed_args.chnl} in '{fchl_dir}'",
                    comm)
                return
            parsed_args.step = f"step{max_step}"
            log_global.info_rank(
                f"Channel chnl_{parsed_args.chnl}: found in {count_steps} FPI steps,"
                f"\t max ndets = {max_ndets} at {parsed_args.step}",
                comm)
        except Exception as e:
            msg = f"Error computing max ndets: {e}"
            raise RuntimeError(msg)
    
    # Focalplane file
    try:
        focalplane_dir = f"input_files/fp_files/fchl_h5/chnl_{parsed_args.chnl}/{parsed_args.step}"
        fp_filename = os.path.join(focalplane_dir, os.listdir(focalplane_dir)[0])
        with h5py.File(fp_filename, 'r') as dets_h5file:
            path = list(dets_h5file.keys())[0]
        det_table_full = QTable.read(fp_filename, path)
    except Exception as e:
        msg = f"Failed to load focalplane file: {fp_filename}. Check chnl and step!. Error: {e}"
        # log_global.error_rank(f"Failed to load focalplane file: {fp_filename}. Error: {e}", comm)
        raise RuntimeError(msg)
    
    log_global.info_rank(f"Loading focalplane: {fp_filename}", comm)
    if parsed_args.ndets is None:
        det_table = det_table_full
    elif parsed_args.ndets < len(det_table_full):
        det_table = det_table_full[:parsed_args.ndets]
    else:
        det_table = det_table_full
    
    # instantiate a TOAST focalplane instance
    focalplane = toast.instrument.Focalplane(
        detector_data=det_table,
        sample_rate=args.sample_rate,
        field_of_view=args.fov,
    )

    # Load the schedule file and instantiate the schedule object
    # schedule_file = os.path.join("input_files/schedules",parsed_args.sch)
    sch_dir = os.path.join("input_files/step_schedules", parsed_args.step)
    schedule_file = os.path.join(sch_dir,parsed_args.sch)
    schedule = toast.schedule.GroundSchedule()
    schedule.read(schedule_file, comm=comm)
    
    # Run the simulation pipeline
    eorspec_mockdata_pipeline(args, comm, focalplane, schedule, parsed_args.grp_size)
    log_global.info_rank(f"Wrote timestream data for {schedule_file} to disk", comm=comm)

    
    log_global.info_rank("Full mock data generated in", comm=comm, timer=global_timer)
    
    # Synchronize all ranks, so every process reaches this point before proceeding
    comm.barrier()
    if rank == 0:
        sim_end_time = t.time()
        sim_elapsed_time = sim_end_time - sim_start_time
        log_global.info_rank(
            f"Timestream Simulation completed. Elapsed Time: {sim_elapsed_time/60.0:.2f} minutes",
            comm
            )
    
if __name__ == "__main__":
    world, procs, rank = toast.mpi.get_world()
    with toast.mpi.exception_guard(comm=world):
        main()