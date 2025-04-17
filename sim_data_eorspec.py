###
#Timestream Simulation Script for Prime-cam
###
###Last updated: March 22, 2024
###
#Author: Ankur Dev, adev@astro.uni-bonn-de
###
###Logbook###
###
###Personal Notes and Updates:
#ToDo: Implement config and CLI
#ToDo: Implement CAR
#ToDo: Verify and validate sequential steps
#ToDo: Check max PWV
#ToDo: Clean up
###
###[Self]Updates Log:
#20-02-2024: Scraping TOAST2, Begin migration
#14-03-2024: Upgraded to TOAST3
#19-03-2024: Checked write and read to h5
#21-03-2024: Implement dual-regime Atm sim
#22-03-2024: Begin integrated script implemention
#23-03-2024: Implemented multiple schedules
#23-03-2024: Modified schedule field name; must be %field_%dd_%mm for uid
#02-04-2024: Updated Atmosphere implementationi
#10-05-2024: Implemented CAR and Healpy capabilities
#10-05-2024: Major changes to primecam_mockdata_pipeline()
#10-05-2024: Few changes to args Class, toast.mpi.get_world() call
#13-05-2024: Changes to schedule files, input CAR files
#13-05-2024: Implemented PWV limit handling
#14-05-2024: Added changes to SimGround, with median and max_pwv truncate
#17-05-2024: Implemented El Nods in SimGround
#19-05-2024: Updated reformat_dets() to be compatible with EoRSpec
#24-05-2024: sim_atm Boolean Flag added to Class args
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
from toast.instrument_coords import quat_to_xieta

import yaml
from toast.schedule_sim_ground import run_scheduler

import astropy.units as u
from astropy.table import QTable, Column
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
import random
import h5py

### Timing Imports ###
import time as t
import psutil as ps
import logging

### Settting up Logger ###
# ANSI escape sequences for colors
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': '\033[93m',  # Yellow
        'INFO': '\033[94m',     # Blue
        'DEBUG': '\033[92m',    # Green
        'CRITICAL': '\033[91m', # Red
        'ERROR': '\033[91m',    # Red
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, '\033[0m')  # Default to no color
        record.msg = color + str(record.msg) + '\033[0m'  # Reset to default after message
        return logging.Formatter.format(self, record)

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info(f"Using TOAST version: {toast.__version__}")
logger.info(f"Starting timesteam simulation...")
sim_start_time = t.time()

class args:
    weather = 'atacama'
    sim_atm = True #Bool
    pwv_limit = 1.27 #1.27 #mm #None
    sample_rate = 244 * u.Hz #400 * u.Hz # Hz
    scan_rate_az = 0.5  * (u.deg / u.s) #on sky rate #0.5
    #fix_rate_on_sky (bool):  If True, `scan_rate_az` is given in sky coordinates and azimuthal
    #rate on mount will be adjusted to meet it.
    #If False, `scan_rate_az` is used as the mount azimuthal rate. (default = True)

    scan_accel_az = 1  * (u.deg / u.s**2) #1

    fov = 1.3 * u.deg # Field-of-view in degrees
    h5_outdir = "./ccat_datacenter_mock/data_COSMOSf351_newdets"

    #Healpy
    input_map = "./input_files/fullsky_CII_map_nside2048_nest.fits"
    #input_map = "./input_files/CII_Healpy_nside2048_nest.fits"
    mode = "I"

    #CAR
    #input_map = "./input_files/pysm3_map_nside2048_car.fits"
    #mode = "I" #Gives an error if input_map mode is inconsistent with args.mode
    #input_map = "./input_files/pysm3_map_nside2048_allStokes_car.fits"
    #mode = "IQU"

    nside = 2048 #1024
    freq = 351 * u.GHz
    fwhm = 0.62 *u.arcmin

# def reformat_dets(dets_pck):
#     # extract values for each column from detector dictionary
#     det_names = list(dets_pck.keys())
#     n_det = len(dets_pck)

#     quats = [dets_pck[k]['quat'] for k in det_names]
#     epsilon = 0.0 #The cross-polar response for all detectors.
#     fwhm = dets_pck[det_names[0]]['fwhm'] * u.arcmin
#     psd_net = dets_pck[det_names[0]]['NET'] * u.K * np.sqrt(1 * u.second)
#     psd_fmin = dets_pck[det_names[0]]['fmin'] * u.Hz
#     psd_fknee = dets_pck[det_names[0]]['fknee'] * u.Hz
#     psd_alpha = dets_pck[det_names[0]]['alpha']
#     bandcenter = dets_pck[det_names[0]]['bandcenter_ghz'] * u.GHz
#     bandwidth = dets_pck[det_names[0]]['bandwidth_ghz'] * u.GHz
#     fwhm_sigma = 0.0 * u.arcmin # Draw random detector FWHM values from a normal distribution with this width.
#     bandcenter_sigma = 0.0 * u.GHz # Draw random bandcenter values from a normal distribution with this width.
#     bandwidth_sigma = 0.0 * u.GHz # Draw random bandwidth values from a normal distribution with this width.
#     wafer_slots = [dets_pck[k]['wafer_slot'] for k in det_names]
#     bands = [dets_pck[k]['band'] for k in det_names]
#     indexes = [dets_pck[k]['index'] for k in det_names]

#     gamma = []
#     for quat in quats:
#         _, _, temp_gamma = quat_to_xieta(quat)
#         gamma.append(temp_gamma * u.rad)

#     # Creating QTable
#     det_table = QTable([
#         Column(name="name", data=det_names),
#         Column(name="quat", data=quats),
#         Column(name="pol_leakage", length=n_det, unit=None),
#         Column(name="psi_pol", length=n_det, unit=u.rad),
#         Column(name="gamma", length=n_det, unit=u.rad),
#         Column(name="fwhm", length=n_det, unit=u.arcmin),
#         Column(name="psd_fmin", length=n_det, unit=u.Hz),
#         Column(name="psd_fknee", length=n_det, unit=u.Hz),
#         Column(name="psd_alpha", length=n_det, unit=None),
#         Column(name="psd_net", length=n_det, unit=(u.K * np.sqrt(1.0 * u.second))),
#         Column(name="bandcenter", length=n_det, unit=u.GHz),
#         Column(name="bandwidth", length=n_det, unit=u.GHz),
#         Column(name="wafer_slot", data=wafer_slots),
#         Column(name="band", data=bands),
#         Column(name="index", data=indexes),
#     ])

#     det_table['gamma'] = gamma
#     for idet, det in enumerate(dets_pck.keys()):
#         det_table[idet]["pol_leakage"] = epsilon
#         # psi_pol is the rotation from the PXX beam frame to the polarization
#         # sensitive direction.
#         det_table[idet]["psi_pol"] = 0 * u.rad
#         det_table[idet]["fwhm"] = fwhm * (
#             1 + np.random.randn() * fwhm_sigma.to_value(fwhm.unit)
#         )
#         det_table[idet]["bandcenter"] = bandcenter * (
#             1 + np.random.randn() * bandcenter_sigma.to_value(bandcenter.unit)
#         )
#         det_table[idet]["bandwidth"] = bandwidth * (
#             1 + np.random.randn() * bandwidth_sigma.to_value(bandcenter.unit)
#         )
#         det_table[idet]["psd_fmin"] = psd_fmin
#         det_table[idet]["psd_fknee"] = psd_fknee
#         det_table[idet]["psd_alpha"] = psd_alpha
#         det_table[idet]["psd_net"] = psd_net
    
#     return det_table


def primecam_mockdata_pipeline(focalplane, schedule_file):
    
    # Begin Pipeline
    #=============================# 
    #Load schedule
    schedule = toast.schedule.GroundSchedule()
    schedule.read(schedule_file, comm=comm)

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

    logger.info(f"Telescope metadata: \n {telescope}")

    # Create the (initially empty) data
    data = toast.Data(comm=toast_comm)

    #Load SimGround

    sim_ground = toast.ops.SimGround(weather=args.weather)
    sim_ground.telescope = telescope
    sim_ground.schedule = schedule
    sim_ground.scan_rate_az =  args.scan_rate_az
    sim_ground.scan_accel_az = args.scan_accel_az
    sim_ground.max_pwv = 1.45 * u.mm #Truncate PWV
    sim_ground.median_weather = False
   
    #=============================#
    ### El Nod Tests ###

    sim_ground.scan_rate_el = 1.5 * (u.deg / u.s) #Allowed by mount #1.6
    sim_ground.el_mod_amplitude = 1.0 * u.deg #1.2
    sim_ground.el_mod_rate = 0.1 * u.Hz
    sim_ground.el_mod_sine = True
    
    sim_ground.elnod_every_scan = False
    #=============================#
    sim_ground.apply(data)
    
    logger.info(f"Number of Observations loaded: {len(data.obs)}")

    #=============================#
    # Detector Pointing
    pixels_wcs_radec = toast.ops.PixelsWCS(
                                    name="pixels_wcs_radec",
                                    projection="CAR",
                                    center=(0*u.degree,0*u.degree),
                                    resolution=(0.02 * u.degree, 0.02 * u.degree),
                                    dimensions=(14400, 7201),
                                    auto_bounds=False,
                                    )

    ### Pointing Matrix
    pixels_wcs_azel = toast.ops.PixelsWCS(
                                    name="pixels_wcs_azel",
                                    projection="CAR",
                                    resolution=(0.02 * u.degree, 0.02 * u.degree),
                                    auto_bounds=True,
                                    )


    pixels_healpix_radec = toast.ops.PixelsHealpix(name="pixels_healpix_radec",
                                                   nside=args.nside)

    #Toggel for CAR / Healpy formats
    pixels_wcs_radec.enabled = False
    pixels_wcs_azel.enabled = False
    pixels_healpix_radec.enabled = True

    scan_wcs_map = toast.ops.ScanWCSMap(name="scan_wcs_map")
    scan_wcs_map.enabled = False

    scan_healpix_map = toast.ops.ScanHealpixMap(name="scan_healpix_map")
    scan_healpix_map.enabled = True

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
    weights_azel.hwp_angle = sim_ground.hwp_angle

    weights_radec.detector_pointing = det_pointing_radec
    weights_radec.hwp_angle = sim_ground.hwp_angle

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
    # Limit Observation PWV

    null_obs = False
    if args.pwv_limit is not None:
        iobs = 0
        ngood = 0
        nbad = 0
        while iobs < len(data.obs):
            pwv = data.obs[iobs].telescope.site.weather.pwv.to_value(u.mm)
            print(f"PWV for Obs {data.obs[iobs].session.name} is {pwv:.2f} mm")
            if pwv <= args.pwv_limit:
                ngood += 1
                iobs += 1
            else:
                nbad += 1
                logger.warning(f"Exceeding PWV detected {pwv:.2f} mm, "
                      f"Obs {data.obs[iobs].session.name} rejected")
                del data.obs[iobs]
                if len(data.obs) == 0:
                    msg = (
                        f"PWV limit = {args.pwv_limit} mm rejected all "
                        f"{nbad} observations assigned to this process"
                    )
                    # raise RuntimeError(msg)
                    logger.critical(msg)
                    null_obs = True
                    break
        log.info_rank(
             f"  Discarded {nbad} / {ngood + nbad} observations "
             f"with PWV > {args.pwv_limit} mm in",
             comm=world_comm,
             timer=timer,
         )

        if null_obs==True:
            print(f"Num data obs {len(data.obs)}")
            print(f"Skipping {schedule_file}")
            return None
                
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
    logger.info(f"Atmospheric simulation...")
    #Atmosphere set-up
    rand_realisation = random.randint(10000, 99999)
    tel_fov = 4* u.deg
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
                    gain=2e-5, #2e-5 changed 25.05.2024
                    realization=1000000,
                    wind_dist=10000 * u.m,
                    enabled=False,
                    cache_dir=cache_dir,
                )

    sim_atm_coarse.realization = 1000000 + rand_realisation
    sim_atm_coarse.field_of_view = tel_fov
    # telescope.focalplane.field_of_view * 1.3 #5* u.deg () for 100 dets
    #5* u.deg for 10 dets
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
            zmax=100 * u.m,
            gain=4e-5, #4e-5 changed 25.05.2024
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

    if scan_healpix_map.enabled:
        print(f"Loading Healpix Map {args.input_map}")
        scan_healpix_map.file = args.input_map
        scan_healpix_map.pixel_dist = binner_final.pixel_dist
        scan_healpix_map.pixel_pointing = pixels_final
        scan_healpix_map.stokes_weights = weights_final
        scan_healpix_map.save_pointing = False
        scan_healpix_map.apply(data)

    elif scan_wcs_map.enabled:
        print(f"Loading WCS Map {args.input_map}")
        scan_wcs_map.file = args.input_map
        scan_wcs_map.pixel_dist = binner_final.pixel_dist
        scan_wcs_map.pixel_pointing = pixels_final
        scan_wcs_map.stokes_weights = weights_final
        scan_wcs_map.save_pointing = False
        scan_wcs_map.apply(data)

    mem = toast.utils.memreport(msg="(whole node)", comm=world_comm, silent=True)
    log.info_rank(f"After Scanning Input Map:  {mem}", world_comm)

    #=============================#
    # Simulate detector noise
    sim_noise = toast.ops.SimNoise(name="sim_noise")
    sim_noise.noise_model = elevation_model.out_model
    sim_noise.apply(data)

    #=============================#

    mem = toast.utils.memreport(msg="(whole node)", comm=world_comm, silent=True)
    log.info_rank(f"After generating detector timestreams:  {mem}", world_comm)

    field_name = (data.obs[0].name).split('-')[0]
    n_dets = telescope.focalplane.n_detectors

    #=============================#
    #Write to h5
    output_dir = args.h5_outdir
    module_code = args.freq.to_string().split('.0')[0]
    f_path = f"sim_PCAM{module_code}_h5_{field_name}_d{n_dets}"
    save_dir = os.path.join(args.h5_outdir, f_path)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Writing timestream data to h5 files for \
Field {field_name} observed at {args.freq.to_string()} \
with {n_dets} detectors")
    logger.info(f"Writing h5 files to: {save_dir}")

    detdata_tosave = ["signal", "flags"]

    for obs in data.obs:
        io.save_hdf5(
            obs=obs,
            dir=save_dir,
            detdata=detdata_tosave
        )

    logger.info(f"Wrote timestream data for {schedule_file} to disk")
###==================================================###    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Simulate PrimeCam Timestream Data")
    parser.add_argument('--test-run', action='store_true', help="Test run with 1 schedule")
    parsed_args = parser.parse_args()

    world, procs, rank = toast.mpi.get_world()
    comm, procs, rank = toast.get_world()
    toast_comm = toast.Comm(world=comm, groupsize=1)
    # Shortcut for the world communicator
    world_comm = toast_comm.comm_world

    log = toast.utils.Logger.get()
    global_timer = toast.timing.Timer()
    timer = toast.timing.Timer()
    global_timer.start()
    timer.start()

    mem = toast.utils.memreport(msg="(whole node)", comm=comm, silent=True)
    log.info_rank(f"Start of the workflow:  {mem}", comm)
    
    if parsed_args.test_run:
        logger.info("Begin test run with 1 schedule...")
    log.info_rank("Begin set-up and monitors for Simulating timestream data for PrimeCam/FYST", comm=world_comm)

    #FP from https://github.com/ccatobs/sotodlib
    #fp_filename = "./input_files/fp_files/t3dets_eorspecFP_A6_step210_LFA_288pix.pkl" #f222
    # fp_filename = "./input_files/fp_files/t3dets_eorspecFP_A11_step210_HFA_96pix.pkl" #f351
    #fp_filename = "./input_files/fp_files/noatm_t3dets_eorspecFP_A11_step210_HFA_96pix.pkl"     #f351 no atm alpha 0
    #fp_filename = "./input_files/fp_files/testparam_t3dets_eorspecFP_A11_step210_HFA_96pix.pkl" #Tests params
    
    fp_filename = "./input_files/fp_files/fchl_h5/chnl_350/step210/f350_step210_d159_dettable.h5"
    logger.info(f"Loading focalplane: {fp_filename}")

    # with open(fp_filename, "rb") as f:
    #     dets_pck = pkl.load(f)

    # #Re-formatting FP
    # det_table = reformat_dets(dets_pck)
    
        # Focalplane file
    try:
        with h5py.File(fp_filename, 'r') as dets_h5file:
            # Get the first path (assuming there's only one)
            path = list(dets_h5file.keys())[0]
        
        # focalplane_file = f"dets_FP_PC280_{parsed_args.dets}_w2.h5"  
        # fp_filename = os.path.join("input_files/fp_files", focalplane_file)
        det_table = QTable.read(fp_filename, path)
    except Exception as e:
        log.error(f"Failed to load focalplane file: {fp_filename}. Error: {e}", comm)
        raise  
    
    log.info_rank(f"Loading focalplane: {fp_filename}", comm)
    
    # instantiate a TOAST focalplane instance 
    width = args.fov
    focalplane = toast.instrument.Focalplane(
        detector_data=det_table,
        sample_rate=args.sample_rate,
        field_of_view=1.1 * (width + 2 * args.fwhm),
    )

    sch_list = './input_files/schedule_list.txt'
    sch_rel_path = "./input_files/schedules/"
    filesnames = []

    with open(sch_list, 'r') as file:
        content = file.read()
        filenames = content.split(',')

    filenames = [name.strip() for name in filenames if name.strip()]

    logger.info(f"Indexed {len(filenames)} schedule files")

    if not parsed_args.test_run:
        for file_count, schedule_file in enumerate(filenames):
            #if file_count>1:
            #    break
            sch_file_path = os.path.join(sch_rel_path,schedule_file)
            logger.info(f"Loading schedule #{file_count+1}: {sch_file_path}")
            primecam_mockdata_pipeline(focalplane, sch_file_path)
    else:
        schedule_file = "./input_files/schedule_COSMOS_05_01.txt"
        primecam_mockdata_pipeline(focalplane, schedule_file)

    log.info_rank("Full mock data generated in", comm=world_comm, timer=global_timer)
    
    sim_end_time = t.time()
    sim_elapsed_time = sim_end_time - sim_start_time
    logger.info(f"Timestream Simulation completed. Elapsed Time: {sim_elapsed_time/60.0:.2f} minutes")
    
