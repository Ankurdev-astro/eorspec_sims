from .filters import remove_slope_1d, az_el_correction, pca_component_removal, remove_global_poly

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