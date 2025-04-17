import toast
import toast.io as io
import toast.ops

import numpy as np
import os

import astropy.units as u

comm, procs, rank = toast.get_world()

toast_comm = toast.Comm(world=comm, groupsize=1)

# Create the (empty) data
data = toast.Data(comm=toast_comm)
sim_ground = toast.ops.SimGround()

#---------------------------------------#
#maps_outdir = "./ccat_datacenter_mock/outmaps_toast_maps"
#savemaps_dir = os.path.join(maps_outdir, "tests_noatm_v1") 
#os.makedirs(savemaps_dir, exist_ok=True)
#print(savemaps_dir)
#
##Dir with h5 files
#obs_dir_h5 = \
#"ccat_datacenter_mock/no_atm_alphanull/sim_PCAM351_h5_COSMOS_05_01_d96"
#
#print("Loading h5 dir...")
#for h5files in os.listdir(obs_dir_h5):
#    file_path = os.path.join(obs_dir_h5, h5files)
#    # Load h5 file and make path
#    #print(file_path)
#    if os.path.isfile(file_path) and h5files.endswith(".h5"):
#        # Load the observation from the HDF5 file
#        obs = io.load_hdf5(path=file_path, comm=toast_comm)
#        # Append the observation
#        data.obs.append(obs)
#

#obs_dir_h5 = \
#"ccat_datacenter_mock/data_COSMOSf351_d96_3deg15_filt_all_comap/sim_PCAM351_h5_COSMOS_07_10_d96"
#
#print("Loading h5 dir...")
#for h5files in os.listdir(obs_dir_h5):
#    file_path = os.path.join(obs_dir_h5, h5files)
#    # Load h5 file and make path
#    #print(file_path)
#    if os.path.isfile(file_path) and h5files.endswith(".h5"):
#        # Load the observation from the HDF5 file
#        obs = io.load_hdf5(path=file_path, comm=toast_comm)
#        # Append the observation
#        data.obs.append(obs)
#
#=========================================#

maps_outdir = "./ccat_datacenter_mock/outmaps_toast_maps"
savemaps_dir = os.path.join(maps_outdir, "tests_v7_upd")
os.makedirs(savemaps_dir, exist_ok=True)
print(savemaps_dir)

def load_h5_dir(obs_dir_h5):
    print(f"Loading h5 dir {os.path.basename(obs_dir_h5)} ...")
    for h5files in os.listdir(obs_dir_h5):
        file_path = os.path.join(obs_dir_h5, h5files)
        # Load h5 file and make path
        #print(file_path)
        if os.path.isfile(file_path) and h5files.endswith(".h5"):
            # Load the observation from the HDF5 file
            obs = io.load_hdf5(path=file_path, comm=toast_comm)
            # Append the observation
            data.obs.append(obs)
#sim_PCAM351_h5_COSMOS_06_09_d96
data_parent_dir = "ccat_datacenter_mock/data_COSMOSf351_d96_3deg15_upd_filtchain_upd"

h5_dir1 = os.path.join(data_parent_dir,"sim_PCAM351_h5_COSMOS_05_01_d96")
load_h5_dir(h5_dir1)

#h5_dir2 = os.path.join(data_parent_dir,"sim_PCAM351_h5_COSMOS_07_10_d96")
#load_h5_dir(h5_dir2)
#
#h5_dir3 = os.path.join(data_parent_dir,"sim_PCAM351_h5_COSMOS_07_15_d96")
#load_h5_dir(h5_dir3)
#
#h5_dir4 = os.path.join(data_parent_dir,"sim_PCAM351_h5_COSMOS_10_15_d96")
#load_h5_dir(h5_dir4)






#======================#
#=============================#
#CAR
# 0.6 arcmin pixes
res = 0.6 * u.arcmin
mode = "I"
print("Generate Pointing...")
#center = RA, DEC in DEG
#bounds = RA_MAX, RA_MIN, DEC_MIN, DEC_MAX
pixels_wcs_radec = toast.ops.PixelsWCS(
                                name="pixels_wcs_radec",
                                projection="CAR",
                                resolution=(res.to(u.deg), res.to(u.deg)),
                                center=(150.0*u.degree,2.0*u.degree),
                                bounds=(154*u.degree,147*u.degree,-2*u.degree,5*u.degree),
                                auto_bounds=False,
                                )
pixels_wcs_radec.enabled = True

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
#Set det pointing
pixels_wcs_radec.detector_pointing = det_pointing_radec
#=============================#
### Pointing Weights
weights_azel = toast.ops.StokesWeights(name="weights_azel", 
                                       weights="weights_azel",  mode=mode)
weights_radec = toast.ops.StokesWeights(name="weights_radec", mode=mode)
weights_azel.enabled = True
weights_radec.enabled = True

weights_azel.detector_pointing = det_pointing_azel
weights_azel.hwp_angle = sim_ground.hwp_angle

weights_radec.detector_pointing = det_pointing_radec
weights_radec.hwp_angle = sim_ground.hwp_angle

#=============================#

# Construct a "perfect" noise model just from the focalplane parameters # after det pointing #after pwv handle
default_model = toast.ops.DefaultNoiseModel(name="default_model", noise_model="noise_model")
default_model.apply(data)

# Create the Elevation modulated noise model
elevation_model = toast.ops.ElevationNoise(name="elevation_model",
                                           out_model="el_noise_model")
elevation_model.noise_model = default_model.noise_model
elevation_model.detector_pointing = det_pointing_azel
elevation_model.apply(data)
#==============================#
#==============================#
print("Making Maps...")

#Set up the pointing used in the binning operator
binner_final = toast.ops.BinMap(name="binner_final", pixel_dist="pix_dist_final")
binner_final.enabled = True
binner_final.pixel_pointing = pixels_wcs_radec
binner_final.stokes_weights = weights_radec
binner_final.noise_model = elevation_model.out_model

mapmaker = toast.ops.MapMaker(name="cosmos_f351")
mapmaker.weather = "vacuum"
mapmaker.write_hdf5 = False
mapmaker.binning = binner_final
mapmaker.map_binning = binner_final

#If writing only hits
mapmaker.write_hits = True
mapmaker.write_binmap = True
mapmaker.write_cov = False
mapmaker.write_invcov = True
mapmaker.write_map = True
mapmaker.write_noiseweighted_map = True
mapmaker.write_rcond = False
mapmaker.write_solver_products = False


# No templates for now (will just do the binning)
#mapmaker.template_matrix = toast.ops.TemplateMatrix(templates=[])

#if solving templates:
mapmaker.template_matrix = toast.ops.TemplateMatrix(templates=[toast.templates.Offset(step_time=20*u.second)])

mapmaker.output_dir = savemaps_dir
mapmaker.enabled = True
mapmaker.apply(data)
