from .filters import remove_slope_1d, az_el_correction, pca_component_removal, remove_global_poly
from .helper_scripts.convert_units import convert_K_jysr

from toast.ops import Operator
from toast.observation import default_values as defaults
from toast.timing import function_timer
from toast.accelerator import ImplementationType
from toast.utils import Logger
from toast.traits import Int, Unicode, trait_docs
from numpy import asarray

## Deslope
@trait_docs
class Deslope(Operator):
    """Operator which Removes a linear slope and mean from the TOD data."""
    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key apply filtering to"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, use_accel=None, **kwargs):
        log = Logger.get()
        
        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)
        for obs in data.obs:
            # Only operate on local detectors for this process
            local_dets = obs.local_detectors
            # log.info(f"Deslope: Processing observation {obs.name} with {len(local_dets)} " + \
            #          f"dets at rank {obs.comm.group_rank}")

            for det in local_dets:
                local_detdata = asarray(obs.detdata[self.det_data][det])
                processed_signal = remove_slope_1d(local_detdata, w=10000)
                obs.detdata[self.det_data][det] = processed_signal

            # Synchronize after processing
            if obs.comm.comm_group is not None:
                obs.comm.comm_group.barrier()
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),  # Does not provide new detdata
        }
        return prov

## Az-El Template correction
@trait_docs
class Template_azel(Operator):
    """
    Operator which models the Az-El Template from the Az-El pointing info
    and subtracts the model
    """
    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key apply filtering to"
    )
    azimuth = Unicode(
        defaults.azimuth, help="Observation shared azimuth key"
    )
    elevation = Unicode(
        defaults.elevation, help="Observation shared elevation key"
    )
    times = Unicode(
        defaults.times, help="Observation shared times key"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, use_accel=None, **kwargs):
        log = Logger.get()
        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)
        for obs in data.obs:
            # Only operate on local detectors for this process
            local_dets = obs.local_detectors
            az_obs = obs.shared[self.azimuth]
            el_obs = obs.shared[self.elevation]
            times_obs = obs.shared[self.times]

            # log.info(f"AzEl Template: Processing observation {obs.name} with {len(local_dets)} " + \
            #          f"dets at rank {obs.comm.group_rank}")

            for det in local_dets:
                local_detdata = asarray(obs.detdata[self.det_data][det])
                processed_signal = az_el_correction(times_obs, 
                                                    local_detdata, 
                                                    el_obs, az_obs)
                obs.detdata[self.det_data][det] = processed_signal

            # Synchronize after processing
            if obs.comm.comm_group is not None:
                obs.comm.comm_group.barrier()

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),  # Does not provide new detdata
        }
        return prov

## PCA component removal
@trait_docs
class PCAComp_removal(Operator):
    """
    Operator which removes the specified leading components from the TOD data.
    """
    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key apply filtering to"
                    )
    n_components = Int(None, allow_none=True,  help="Number of leading PCA components to be subtracted"
                      )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, use_accel=None, **kwargs):
        log = Logger.get()
        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)
        for obs in data.obs:
            
            local_dets = obs.local_detectors
            local_data = asarray(obs.detdata[self.det_data][:])
            
            ## Operate on all local detectors for this process as 2D array
            # log.info(f"PCA: Processing observation {obs.name} with {len(local_dets)} " + \
            #          f"detectors on rank {obs.comm.group_rank} having shape {local_data.shape}")
            # log.info(f"Type of data: {type(local_data)}")           

            pcaprocessed_data, explained_var, n_components = pca_component_removal(
                                               local_data,
                                               pca_axis=1,
                                               n_components=self.n_components
            )
            percent_var = [round(float(x) * 100,4) for x in explained_var[:6]]
            
            
            # log.info(f"PCA: Processing observation {obs.name} with {len(local_dets)} detectors " + \
            #          f"on rank {obs.comm.group_rank} ; " + \
            #          f"Explained percent variance {percent_var} and " +  \
            #          f"{n_components} leading comp. removed")

            obs.detdata[self.det_data][:] = pcaprocessed_data

            # Synchronize after processing
            if obs.comm.comm_group is not None:
                obs.comm.comm_group.barrier()
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        return prov
    
    
## Poly Detrend
@trait_docs
class PolyDetrend(Operator):
    """Operator which Removes a global polynomial trend"""
    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key apply filtering to"
    )
    times = Unicode(
        defaults.times, help="Observation shared times key"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, use_accel=None, **kwargs):
        log = Logger.get()
        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)
        for obs in data.obs:
            # Only operate on local detectors for this process
            local_dets = obs.local_detectors
            times_obs = obs.shared[self.times]
            local_data = asarray(obs.detdata[self.det_data][:])

            # Operate on all local detectors for this process as 2D array
            # log.info(f"PolyDetrend: Processing observation {obs.name} with {len(local_dets)} " + \
            #          f"detectors on rank {obs.comm.group_rank}")

            processed_signal = remove_global_poly(local_data, times_obs)
            obs.detdata[self.det_data][:] = processed_signal
            
            # Synchronize after processing
            if obs.comm.comm_group is not None:
                obs.comm.comm_group.barrier()

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),  # Does not provide new detdata
        }
        return prov
    
    
from traitlets import Int, Unicode, Bool
from toast.timing import function_timer, Timer
import numpy as np

## PCA component removal v2
@trait_docs
class PCAComp_removal2(Operator):
    """
    Remove the leading principal-components from detector time-streams.

    If ``redistribute`` is ``True`` the observation is duplicated and
    redistributed so every MPI rank owns *all* detectors for a slice of
    samples.  This lets the PCA basis capture correlations across the
    whole focal-plane.  After filtering, data are re-distributed back to
    the original detector-major layout.
    """

    #------------------------------------------------------------
    # traits
    #------------------------------------------------------------
    API = Int(0, help="Internal interface version for this operator")

    times = Unicode(
        defaults.times,
        help="Observation shared key for timestamps",
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key apply filtering to"
    )

    n_components = Int(
        None,
        allow_none=True,
        help="Number of leading PCA components to be subtracted"
    )

    redistribute = Bool(
        True,
        help="If True, redistribute data before PCA and restore afterwards"
    )

    #------------------------------------------------------------
    # constructor
    #------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    #------------------------------------------------------------
    # internal helpers
    #------------------------------------------------------------
    @function_timer
    def _redistribute(self, data, obs, timer, log):
        """
        Duplicate *obs* (det_data only) and redistribute it by samples.

        Returns
        -------
        (comm, temp_ob)
            *comm* is the column communicator (``None`` if redistributed),
            *temp_ob* is the (possibly duplicated) observation to operate on.
        """
        if not self.redistribute:
            return obs.comm_col, obs

        # Duplicate the minimal set of fields we need.
        temp_ob = obs.duplicate(
            times=self.times,              
            meta=list(),
            shared=list(),
            detdata=[self.det_data],
            intervals=list(),
        )

        # Redistribute so each rank holds all detectors for some samples.
        temp_ob.redistribute(1, times=self.times, override_sample_sets=None)
        return None, temp_ob
    
    ### --- TEST ---###
    ### We test to hold all data in 1 Rank
    # @function_timer
    # def _redistribute(self, data, obs, timer, log):
    #     """
    #     Duplicate *obs* (det_data only) and gather *all detectors, all samples*
    #     onto the first column-rank.  Other ranks end up with zero samples.

    #     Returns
    #     -------
    #     (comm, temp_ob)
    #         comm   : column communicator (None because we now act as one-rank)
    #         temp_ob: duplicated / redistributed observation to run PCA on
    #     """
    #     if not self.redistribute:
    #         return obs.comm_col, obs          # original detector-major path

    #     # ---- duplicate the minimal payload we need ----
    #     temp_ob = obs.duplicate(
    #         times=self.times,
    #         meta=[],
    #         shared=[],
    #         detdata=[self.det_data],
    #         intervals=[],
    #     )

    #     # ---- build sample_sets so only column-0 owns data ----
    #     ncols = temp_ob.comm.comm_group.size          # #processes after rows=1
    #     full  = (0, obs.n_all_samples)                # whole observation
    #     empty = (0, 0)

    #     sample_sets = [full] + [empty] * (ncols - 1)  # len == ncols

    #     # ---- redistribute: 1 row ⇒ all dets everywhere; sample_sets ⇒ col-0 only ----
    #     temp_ob.redistribute(
    #         1,
    #         times=self.times,
    #         override_sample_sets=sample_sets,
    #     )

    #     log.debug_rank(
    #         f"{data.comm.group:4} : Gathered TOD on column-0 (shape "
    #         f"{temp_ob.detdata[self.det_data].data.shape})",
    #         comm=temp_ob.comm.comm_group,
    #         timer=timer,
    #     )
    #     return None, temp_ob


    @function_timer
    def _re_redistribute(self, data, obs, timer, log, temp_ob):
        """Undo the redistribution and copy filtered data back."""
        if not self.redistribute:
            return

        temp_ob.redistribute(
            obs.dist.process_rows,
            times=self.times,
            override_sample_sets=obs.dist.sample_sets,
        )

        # Copy filtered signal to the original observation.
        # obs.detdata[self.det_data][:] = temp_ob.detdata[self.det_data][:]
        obs.detdata[self.det_data].data[...] = temp_ob.detdata[self.det_data].data
        return

    # #------------------------------------------------------------
    # # main execution
    # #------------------------------------------------------------
    @function_timer
    def _exec(self, data, **kwargs):
        """
        Apply PCA filtering to each observation.
        Currently we re-distributed and collect all data to 1 rank and do PCA on that rank 0
        Post-PCA we re-distribute data back to original configuration        
        """
        log = Logger.get()
        timer = Timer()
        timer.start()

        for i_obs, obs in enumerate(data.obs):
            # orig = obs.detdata[self.det_data].data.copy()   # <- this is “cache_copy
            # log.info(
            #         f"[pre] {obs.name} rank {obs.comm.group_rank}: "
            #         f"dets={len(obs.local_detectors)}  samples={obs.n_local_samples}"
            #     )
            comm, temp_ob = self._redistribute(data, obs, timer, log)
            # log.info(
            #         f"[post] {obs.name} rank {temp_ob.comm.group_rank}: "
            #         f"dets={len(temp_ob.local_detectors)}  samples={temp_ob.n_local_samples}"
            #     )
            
            # --- run PCA on redistributed data ---
            # local_data = asarray(temp_ob.detdata[self.det_data][:])  # (ndet, nsamp_local)
            local_data = asarray(temp_ob.detdata[self.det_data].data[...])  # (ndet, nsamp_local)
            # if local_data.shape[1] != 0:
            # ranks with zero samples skip PCA

            pcaprocessed_data, explained_var, ncomp = pca_component_removal(
                                                local_data, pca_axis=1, 
                                                n_components=self.n_components
                                            )
            # temp_ob.detdata[self.det_data][:] = pcaprocessed_data
            temp_ob.detdata[self.det_data].data[...] = pcaprocessed_data
            
            percent_var = [round(float(x) * 100,4) for x in explained_var[:6]]
            log.info(f"PCA: Obs. {obs.name} with {len(temp_ob.local_detectors)} " + \
                    f"dets on rank {temp_ob.comm.group_rank} with shape {local_data.shape}; " + \
                    f"Expl. %var. {percent_var}; {ncomp} comp. rem.")
            #--- DEBUG  ---#
            # diff_norm = np.linalg.norm(local_data - pcaprocessed_data)
            # log.info(
            #     f"[PCA-debug] rank {temp_ob.comm.group_rank}: "
            #     f"n_comp={ncomp}  ||Δ||={diff_norm:.3e}"
            # )
            #--------------#

            if obs.comm.comm_group is not None:
                obs.comm.comm_group.barrier()
            
            # --- restore original distribution / copy back ---
            # log.info(f"[Before re-redist {obs.name}] Rank {temp_ob.comm.group_rank} "
            #          f"shape={temp_ob.detdata[self.det_data].data.shape}")
            # self._re_redistribute(data, obs, timer, log, temp_ob)
            # log.info(f"[After re-redist {obs.name}] Rank {obs.comm.group_rank} "
            #          f"shape={obs.detdata[self.det_data].data.shape}")


            #--- DEBUG  ---#
            # after = obs.detdata[self.det_data].data
            # log.info(
            #     f"[copy-debug] rank {obs.comm.group_rank}: "
            #     f"max|Δ|={np.max(np.abs(after - local_data)):.3e}"
            # )
            #--------------#

            # if self.redistribute:
            #     if temp_ob.comm.comm_group is not None:
            #         temp_ob.comm.comm_group.barrier()
                
            #     # 2.  patch _raw on ranks that own no samples
            #     # clear() on ranks that still own a valid buffer
            #     if temp_ob.n_local_samples == 0:
            #         print('This ...')
            #         for wrapper_dict in (temp_ob.detdata, temp_ob.shared):
            #             for obj in wrapper_dict.values():
            #                 if getattr(obj, "_raw", None) is None:
            #                     obj._raw = []
            #                     print(obj)
            #     # temp_ob.clear()
            #     # del temp_ob
            if self.redistribute:
                temp_ob.clear()
                del temp_ob

            # if comm is not None:
            #     comm.Barrier()

            
            #--- DEBUG  ---#
            # # --- after _re_redistribute(), once the observation contains the filtered TOD ---
            # delta_max = np.max(np.abs(obs.detdata[self.det_data].data - orig))
            # log.info(f"[pipeline] max|Δ after PCA| = {delta_max:.3e}")
            # # On one rank, before PCA:
            # rms_before = np.sqrt(np.mean(orig**2, axis=1))

            # # After PCA:
            # rms_after  = np.sqrt(np.mean(obs.detdata[self.det_data].data**2, axis=1))

            # log.info(f"median RMS drop = {np.median(rms_before - rms_after):.3e}")
            #--------------#
            if i_obs==0:
                break # Debug test
        return
    
    #------------------------------------------------------------
    # TEST main execution
    #------------------------------------------------------------
    # @function_timer
    # def _exec(self, data, **kwargs):
    #     """
    #     Apply PCA filtering to each observation.
    #     Currently we re-distributed and collect all data to 1 rank and do PCA on that rank 0
    #     Post-PCA we re-distribute data back to original configuration        
    #     """
    #     log = Logger.get()
    #     timer = Timer()
    #     timer.start()

        # for i_obs, obs in enumerate(data.obs):
        #     # orig = obs.detdata[self.det_data].data.copy()   # <- this is “cache_copy
        #     log.info(
        #             f"[pre] {obs.name} rank {obs.comm.group_rank}: "
        #             f"dets={len(obs.local_detectors)}  samples={obs.n_local_samples}"
        #         )
        #     comm, temp_ob = self._redistribute(data, obs, timer, log)
        #     log.info(
        #             f"[post] {obs.name} rank {temp_ob.comm.group_rank}: "
        #             f"dets={len(temp_ob.local_detectors)}  samples={temp_ob.n_local_samples}"
        #         )
            
        #     # --- run PCA on redistributed data ---

        #     local_data = asarray(temp_ob.detdata[self.det_data].data[...])  # (ndet, nsamp_local)
        #     if local_data.shape[1] != 0:
        #         # ranks with zero samples skip PCA

        #         pcaprocessed_data, explained_var, ncomp = pca_component_removal(
        #                                             local_data, pca_axis=1, 
        #                                             n_components=self.n_components
        #                                         )
        #         # temp_ob.detdata[self.det_data][:] = pcaprocessed_data
        #         temp_ob.detdata[self.det_data].data[...] = pcaprocessed_data
                
        #         percent_var = [round(float(x) * 100,4) for x in explained_var[:6]]
        #         log.info(f"PCA: Obs. {obs.name} with {len(temp_ob.local_detectors)} " + \
        #                 f"dets on rank {temp_ob.comm.group_rank} with shape {local_data.shape}; " + \
        #                 f"Expl. %var. {percent_var}; {ncomp} comp. rem.")
        #         #--- DEBUG  ---#
        #         # diff_norm = np.linalg.norm(local_data - pcaprocessed_data)
        #         # log.info(
        #         #     f"[PCA-debug] rank {temp_ob.comm.group_rank}: "
        #         #     f"n_comp={ncomp}  ||Δ||={diff_norm:.3e}"
        #         # )
        #         #--------------#

    #         if obs.comm.comm_group is not None:
    #             obs.comm.comm_group.barrier()
            
    #         # --- restore original distribution / copy back ---
    #         log.info(f"[Before re-redist {obs.name}] Rank {temp_ob.comm.group_rank} "
    #                  f"shape={temp_ob.detdata[self.det_data].data.shape}")
    #         self._re_redistribute(data, obs, timer, log, temp_ob)
    #         log.info(f"[After re-redist {obs.name}] Rank {obs.comm.group_rank} "
    #                  f"shape={obs.detdata[self.det_data].data.shape}")


    #         #--- DEBUG  ---#
    #         # after = obs.detdata[self.det_data].data
    #         # log.info(
    #         #     f"[copy-debug] rank {obs.comm.group_rank}: "
    #         #     f"max|Δ|={np.max(np.abs(after - local_data)):.3e}"
    #         # )
    #         #--------------#
            
    #         # Clear memory and temp_ob
    #         if self.redistribute:
    #             if temp_ob.comm.comm_group is not None:
    #                 temp_ob.comm.comm_group.barrier()
                
    #             # 2.  patch _raw on ranks that own no samples
    #             # clear() on ranks that still own a valid buffer
    #             if temp_ob.n_local_samples == 0:
    #                 print('This ...')
    #                 for wrapper_dict in (temp_ob.detdata, temp_ob.shared):
    #                     for obj in wrapper_dict.values():
    #                         if getattr(obj, "_raw", None) is None:
    #                             obj._raw = []
    #                             print(obj)
    #             # temp_ob.clear()
    #             del temp_ob

    #         # if comm is not None:
    #         #     comm.Barrier()

            
    #         #--- DEBUG  ---#
    #         # # --- after _re_redistribute(), once the observation contains the filtered TOD ---
    #         # delta_max = np.max(np.abs(obs.detdata[self.det_data].data - orig))
    #         # log.info(f"[pipeline] max|Δ after PCA| = {delta_max:.3e}")
    #         # # On one rank, before PCA:
    #         # rms_before = np.sqrt(np.mean(orig**2, axis=1))

    #         # # After PCA:
    #         # rms_after  = np.sqrt(np.mean(obs.detdata[self.det_data].data**2, axis=1))

    #         # log.info(f"median RMS drop = {np.median(rms_before - rms_after):.3e}")
    #         #--------------#
    #         if i_obs==0:
    #             break # Debug test
    #     return

    #------------------------------------------------------------
    # bookkeeping
    #------------------------------------------------------------
    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        return {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
        }

    def _provides(self):
        return {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
        }
        
        
#########################################
#---------------------------------------#


### Units converter: K to jy/sr
@trait_docs
class convert_to_jysr(Operator):
    """Operator which the TOD data from K units to jy/sr units"""
    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key apply filtering to"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    @function_timer
    def _exec(self, data, use_accel=None, **kwargs):
        log = Logger.get()
        
        # Kernel selection
        implementation, use_accel = self.select_kernels(use_accel=use_accel)
        for obs in data.obs:
            # Only operate on local detectors for this process
            local_dets = obs.local_detectors
            freq_GHz = float(obs["freq"].to_value()) # in GHz
            # log.info(f"Converting to Jy/sr: Obs {obs.name} at {freq_GHz} GHz")

            for det in local_dets:
                local_detdata_K = asarray(obs.detdata[self.det_data][det])
                arr_jysr = convert_K_jysr(local_detdata_K, freq_GHz)
                
                obs.detdata[self.det_data][det] = arr_jysr
                del arr_jysr, local_detdata_K

            # Synchronize after processing
            if obs.comm.comm_group is not None:
                obs.comm.comm_group.barrier()
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),  # Does not provide new detdata
        }
        return prov 